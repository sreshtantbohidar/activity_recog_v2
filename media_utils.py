"""Small, dependency-light media helpers used by the Flask analysis service."""

import json
import math
import os
import re
import subprocess
import wave
from typing import Any, Dict, Iterable, List

ACTIVITY_LABELS = {
    "deployment", "mobilization", "movement", "disposition", "training", "infra",
    "patrolling", "visit", "internal security", "miscellaneous", "elint",
    "procurement", "technology", "political diplomacy", "joint exercise",
    "foreign relation", "operation",
}
# Confidence is a numeric probability in [0, 1] reported by the model. The old
# low/medium/high labels are no longer reported directly: they are derived from
# the numeric confidence only to build the per-video distribution histogram.
CONFIDENCE_LEVELS = ("low", "medium", "high")
CONFIDENCE_MEDIUM_MIN = 0.4
CONFIDENCE_HIGH_MIN = 0.7
LOCATION_TYPES = {"urban", "rural", "border", "military base", "unknown"}

# Single shared classification prompt for the API service and the vLLM pipeline.
# Each label carries a short definition: several labels are near-neighbours
# (deployment vs disposition vs mobilization; elint vs technology), and without
# the definitions the model collapses them onto whichever is most familiar.
ACTIVITY_LABEL_DEFINITIONS = """- deployment: forces or operational equipment arriving at a destination and being established or
  set up there. Loading routine cargo onto transport before departure is not deployment.
- mobilization: forces being activated, assembled, or readied before they deploy.
- movement: forces or vehicles travelling or transiting between places (columns, convoys, transport).
- disposition: how forces or equipment are arranged, laid out, or positioned in a place
  (formations, spacing, placement, order of battle) rather than arriving or travelling.
- training: drills, rehearsals, instruction, mock combat, or exercises. Numbered mock structures,
  range/simulation overlays, repeated rehearsed actions, or recording markers support training
  even when weapons, gunfire, and tactical commands are present.
- infra: construction, roads, buildings, or fixed facilities.
- patrolling: patrols, guards, or security sweeps along routes.
- visit: inspections or official visits within one's own forces or establishments
  (a domestic visit that does not involve a foreign delegation).
- internal security: policing, checkpoints, crowd control, or internal law and order.
- elint: electronic or signal intelligence collection or monitoring, radar or signal analysis.
- operation: a real active mission or combat action. Do not choose operation merely because armed
  personnel use tactical movement or commands in a controlled training environment.
- procurement: purchasing, contracts, receiving, inventory, warehousing, delivery, or logistics
  handling of supplies and equipment, including loading or unloading cargo for transport.
- technology: technical systems, equipment demonstrations, or research and development.
- political diplomacy: political or diplomatic meetings, talks, negotiations, or agreements.
- joint exercise: a coordinated combined-arms, multi-service, or multinational exercise. Large
  simulated formations containing different force types, exercise/HUD overlays, and coordinated
  maneuver support this label even when flags or national identities are not visible.
- foreign relation: engagements with other nations or foreign representatives - foreign dignitaries
  arriving, international meetings, talks, agreements, or ceremonies attended by foreign officials.
- miscellaneous: use only when none of the above fit."""

# The label rubric is shared by the per-frame classifier and the whole-video
# reasoning pass so both always judge against the same definitions.
ACTIVITY_PROMPT = """Analyze only directly visible evidence. Choose one or more labels from this exact
list, using each short definition to tell similar labels apart:
""" + ACTIVITY_LABEL_DEFINITIONS + """
Judge what is visibly happening now and prefer the most specific fitting label. Do not infer intent,
identity, nationality, or unseen events.

Return ONLY a JSON object with exactly these fields and types:
{
  "scene_summary": "<string: one sentence describing what is visible>",
  "key_objects": ["<string>", ...],
  "location_type": "<one of: urban, rural, border, military base, unknown>",
  "detected_activity": ["<one or more labels from the list above>"],
  "confidence": <number between 0.00 and 1.00: your certainty that the visible evidence matches detected_activity>,
  "reasoning": "<string: short justification grounded in what is visible>"
}
Example of the exact reply shape (values are illustrative):
{"scene_summary": "Two soldiers fire rifles from behind a sandbag wall", "key_objects": ["rifle", "sandbag", "soldier"], "location_type": "military base", "detected_activity": ["training"], "confidence": 0.82, "reasoning": "Range flags and repeated firing match a drill"}"""


class MediaProbeError(ValueError):
    """Raised when ffprobe cannot read a media file or its metadata."""


class VideoTooLongError(ValueError):
    """Raised when a video is longer than the configured processing limit."""


def parse_json_text(text: str) -> Dict[str, Any]:
    """Parse a JSON object from a model response, including fenced responses."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            return {"error": "No JSON found", "raw": text}
        try:
            value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            return {"error": "Parsing failed", "raw": text}
    return value if isinstance(value, dict) else {"error": "Expected a JSON object", "raw": text}


def normalize_confidence(value: Any, default: float = 0.0) -> float:
    """Coerce a model-reported confidence into a float in the range [0, 1].

    The prompt asks for a number between 0.00 and 1.00, but a model may instead
    answer with a percentage ("45%") or a plain 0-100 number; both are scaled
    down. Anything unparseable falls back to ``default``.
    """
    if isinstance(value, str):
        text = value.strip()
        percent = text.endswith("%")
        try:
            numeric = float(text.rstrip("%").strip())
        except (TypeError, ValueError):
            return default
        if percent:
            numeric /= 100.0
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
    # A bare whole number above 1 (e.g. 45) is a percentage; a fractional value
    # such as 1.5 is treated as out of range and clamped rather than rescaled,
    # so a confident answer is never silently turned into a low one.
    if 1.0 < numeric <= 100.0 and numeric.is_integer():
        numeric /= 100.0
    return max(0.0, min(1.0, numeric))


def confidence_label(value: Any) -> str:
    """Bucket a numeric confidence into low/medium/high for the report histogram."""
    numeric = normalize_confidence(value)
    if numeric >= CONFIDENCE_HIGH_MIN:
        return "high"
    if numeric >= CONFIDENCE_MEDIUM_MIN:
        return "medium"
    return "low"


def normalize_activity_result(value: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize activity output to the documented schema and approved label set."""
    if "error" in value:
        return value

    raw_activities = value.get("detected_activity", [])
    if isinstance(raw_activities, str):
        raw_activities = [raw_activities]
    if not isinstance(raw_activities, list):
        raw_activities = []
    activities = []
    for activity in raw_activities:
        normalized = str(activity).strip().lower().replace("_", " ")
        if normalized in ACTIVITY_LABELS and normalized not in activities:
            activities.append(normalized)
    if not activities:
        activities = ["miscellaneous"]

    raw_objects = value.get("key_objects", [])
    if isinstance(raw_objects, str):
        raw_objects = [raw_objects]
    if not isinstance(raw_objects, list):
        raw_objects = []

    confidence = normalize_confidence(value.get("confidence"))
    location = str(value.get("location_type", "unknown")).strip().lower()
    return {
        "scene_summary": str(value.get("scene_summary", "")),
        "key_objects": [str(item) for item in raw_objects if str(item).strip()],
        "location_type": location if location in LOCATION_TYPES else "unknown",
        "detected_activity": activities,
        "confidence": confidence,
        "reasoning": str(value.get("reasoning", "")),
    }


def compute_frame_change_scores(frame_paths: Iterable[str]) -> List[float]:
    """Per-frame grayscale change score vs the previous sampled frame.

    Decodes each saved frame at thumbnail size, so the cost is negligible
    next to the VLM calls that already processed these frames.
    """
    import numpy as np
    from PIL import Image

    scores: List[float] = []
    previous = None
    for path in frame_paths:
        gray = None
        try:
            with Image.open(path) as source:
                gray = np.asarray(source.convert("L").resize((96, 54)), dtype=np.float32)
        except Exception:
            gray = None
        if gray is None or previous is None:
            scores.append(0.0)
            previous = gray
            continue
        scores.append(round(float(np.mean(np.abs(gray - previous)) / 255.0), 4))
        previous = gray
    return scores


def analyze_motion_burstiness(scores: Iterable[float], score_threshold: float = 0.02) -> Dict[str, Any]:
    """Classify temporal motion rhythm from ordered frame change scores.

    Steady motion across sampled frames matches sustained activity such as an
    engagement, while quiet gaps between motion peaks match range drills with
    pauses between bursts of firing.
    """
    import numpy as np

    values = [float(value) for value in (scores or []) if value is not None]

    def _insufficient(reason: str) -> Dict[str, Any]:
        return {"classification": "insufficient_motion_evidence", "suggested_activity_bias": None,
            "frame_count": len(values), "score_mean": None, "score_cv": None,
            "active_fraction": None, "note": reason}

    if len(values) < int(os.getenv("MOTION_MIN_FRAMES", "4")):
        return _insufficient("too few sampled frames")
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if mean <= 0:
        return _insufficient("no inter-frame motion measured")
    cv = float(arr.std() / mean)
    active_fraction = float((arr > score_threshold).mean())

    classification, bias, note = "ambiguous", None, None
    consistent_cv_max = float(os.getenv("MOTION_CONSISTENT_CV_MAX", "0.45"))
    bursty_cv_min = float(os.getenv("MOTION_BURSTY_CV_MIN", "0.75"))
    steady_min_active = float(os.getenv("MOTION_STEADY_MIN_ACTIVE_FRACTION", "0.6"))
    bursty_max_active = float(os.getenv("MOTION_BURSTY_MAX_ACTIVE_FRACTION", "0.4"))
    if cv <= consistent_cv_max and active_fraction >= steady_min_active:
        classification, bias = "consistent", "operation"
    elif cv >= bursty_cv_min and active_fraction <= bursty_max_active:
        classification, bias = "bursty", "training"
    else:
        note = "motion rhythm does not clearly match sustained or bursty firing patterns"
    return {"classification": classification, "suggested_activity_bias": bias,
        "frame_count": len(values), "score_mean": round(mean, 4), "score_cv": round(cv, 2),
        "active_fraction": round(active_fraction, 2), "note": note}


def motion_activity_bias(motion) -> str | None:
    """Map a motion-burstiness result to a supplementary activity vote."""
    if not isinstance(motion, dict) or motion.get("classification") not in {"consistent", "bursty"}:
        return None
    return {"consistent": "operation", "bursty": "training"}[motion["classification"]]


def probe_media(video_path: str, ffprobe_path: str = "ffprobe") -> Dict[str, Any]:
    """Return the duration and stream types reported by ffprobe."""
    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v", "error",
                "-show_entries", "format=duration:stream=codec_type",
                "-of", "json",
                video_path,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        metadata = json.loads(result.stdout)
        duration = float(metadata.get("format", {}).get("duration"))
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise MediaProbeError(f"Unable to read media metadata: {exc}") from exc

    if not math.isfinite(duration) or duration < 0:
        raise MediaProbeError("Media duration is missing or invalid")

    streams = metadata.get("streams", [])
    return {
        "duration_sec": duration,
        "has_audio": any(stream.get("codec_type") == "audio" for stream in streams),
    }


def validate_video_duration(
    video_path: str,
    max_duration_seconds: float = 300,
    ffprobe_path: str = "ffprobe",
) -> Dict[str, Any]:
    """Probe a video and reject it when it exceeds the inclusive duration limit."""
    metadata = probe_media(video_path, ffprobe_path=ffprobe_path)
    duration = metadata["duration_sec"]
    if duration > max_duration_seconds:
        raise VideoTooLongError(
            f"Video duration is {duration:.2f} seconds; the maximum allowed is "
            f"{max_duration_seconds} seconds (5 minutes)."
        )
    return metadata


def extract_audio_track(
    video_path: str,
    audio_path: str,
    ffmpeg_path: str = "ffmpeg",
) -> None:
    """Extract the first audio stream as mono, 16 kHz, signed 16-bit PCM WAV."""
    try:
        subprocess.run(
            [
                ffmpeg_path,
                "-nostdin", "-v", "error", "-y",
                "-i", video_path,
                "-map", "0:a:0",
                "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le",
                audio_path,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MediaProbeError(f"Unable to extract video audio: {exc}") from exc


def read_pcm16_wav(audio_path: str):
    """Read mono 16 kHz PCM WAV audio as float32 samples."""
    import numpy as np

    try:
        with wave.open(audio_path, "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                raise MediaProbeError("Audio WAV must be mono 16-bit PCM")
            if wav_file.getframerate() != 16000:
                raise MediaProbeError("Audio WAV must have a 16 kHz sample rate")
            samples = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    except (OSError, wave.Error) as exc:
        raise MediaProbeError(f"Unable to read extracted WAV audio: {exc}") from exc
    return samples.astype(np.float32) / 32768.0


# Cross-modal summary prompt sizing. The vLLM server runs with an 8k context
# (VLLM_MAX_MODEL_LEN=8192), so a long video's evidence timeline must be clipped
# and character-budgeted or the request is rejected before it reaches the model.
_SUMMARY_FRAME_LIMIT = 200
_SUMMARY_FRAME_TEXT_LIMIT = 220
_SUMMARY_SPEECH_LIMIT = 300
_SUMMARY_SPEECH_TEXT_LIMIT = 400
_SUMMARY_SOUND_LIMIT = 200
_SUMMARY_SOUND_TEXT_LIMIT = 180
_SUMMARY_SECTION_BUDGET_CHARS = {"visual": 8000, "speech": 4500, "sound": 4500}


def _clip_summary_text(value: Any, limit: int) -> str:
    """Collapse whitespace and hard-cap a single evidence field."""
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 1)].rstrip() + "\u2026"


def _join_within_budget(lines: List[str], budget_chars: int) -> str:
    """Join timeline lines while staying under a character budget."""
    kept: List[str] = []
    used = 0
    for line in lines:
        cost = len(line) + 1
        if kept and used + cost > budget_chars:
            remaining = len(lines) - len(kept)
            kept.append(
                "... {} more entr{} omitted for length".format(
                    remaining, "y" if remaining == 1 else "ies")
            )
            break
        kept.append(line)
        used += cost
    return "\n".join(kept)


def build_multimodal_summary_prompt(
    frame_results: Iterable[Dict[str, Any]],
    speech_segments: Iterable[Dict[str, Any]],
    sound_events: Iterable[Dict[str, Any]],
) -> str:
    """Build an evidence-grounded summary request from visual and audio timelines."""
    visual_lines: List[str] = []
    for frame in list(frame_results)[:_SUMMARY_FRAME_LIMIT]:
        activities = frame.get("detected_activity", [])
        if isinstance(activities, str):
            activities = [activities]
        visual_lines.append(
            "[{:.1f}s] Activities: {} | Scene: {} | Evidence: {}".format(
                float(frame.get("timestamp_sec", 0)),
                ", ".join(str(item) for item in activities) or "not determined",
                _clip_summary_text(frame.get("scene_summary", ""), _SUMMARY_FRAME_TEXT_LIMIT),
                _clip_summary_text(frame.get("reasoning", ""), _SUMMARY_FRAME_TEXT_LIMIT),
            )
        )

    speech_lines = [
        "[{:.1f}-{:.1f}s] {}".format(
            float(segment.get("start_sec", 0)),
            float(segment.get("end_sec", segment.get("start_sec", 0))),
            _clip_summary_text(segment.get("text", ""), _SUMMARY_SPEECH_TEXT_LIMIT),
        )
        for segment in list(speech_segments)[:_SUMMARY_SPEECH_LIMIT]
    ]
    sound_lines = [
        "[{:.1f}-{:.1f}s] {} (confidence {:.2f})".format(
            float(event.get("start_sec", 0)),
            float(event.get("end_sec", event.get("start_sec", 0))),
            _clip_summary_text(event.get("label", "unknown sound"), _SUMMARY_SOUND_TEXT_LIMIT),
            float(event.get("confidence", 0)),
        )
        for event in list(sound_events)[:_SUMMARY_SOUND_LIMIT]
    ]

    visual_block = _join_within_budget(
        visual_lines, _SUMMARY_SECTION_BUDGET_CHARS["visual"]
    ) or "No visual observations available."
    speech_block = _join_within_budget(
        speech_lines, _SUMMARY_SECTION_BUDGET_CHARS["speech"]
    ) or "No speech transcript available."
    sound_block = _join_within_budget(
        sound_lines, _SUMMARY_SECTION_BUDGET_CHARS["sound"]
    ) or "No sound-event predictions available."

    return f"""You are an evidence-grounded activity analyst. Synthesize the video and its audio as one chronological analysis. Give the frame-by-frame observations primary weight for visible actions. Use speech only as a transcript (not as proof that the described event is visible), and treat audio event labels as uncertain acoustic cues. Resolve agreement and disagreement between modalities explicitly. Do not infer identities, intent, or events that are not supported by the observations. If audio is missing or unavailable, say so; do not invent audio evidence. Return a concise but specific overall summary, sequence of activity, and any cross-modal corroboration or uncertainty.

VISUAL TIMELINE:
{visual_block}

SPEECH TRANSCRIPT (timestamped):
{speech_block}

NON-SPEECH AUDIO EVENTS (model predictions; uncertain):
{sound_block}
"""


def build_video_activity_prompt(
    frame_results: Iterable[Dict[str, Any]],
    activity_counts: Dict[str, int],
    transcript: str = "",
    sound_events: Iterable[Dict[str, Any]] = (),
) -> str:
    """Build a whole-video classification request from all aggregated evidence.

    Per-frame voting loses mission context: a video of an active operation is
    dominated by frames that only show vehicles moving, so the vote reports the
    generic label. This prompt asks for one decision over the whole sequence,
    with the audio evidence included, and returns a single label.
    """
    visual_lines: List[str] = []
    for frame in list(frame_results)[:60]:
        activities = frame.get("detected_activity", [])
        if isinstance(activities, str):
            activities = [activities]
        visual_lines.append(
            "[{:.1f}s] frame_labels: {} | scene: {} | evidence: {}".format(
                float(frame.get("timestamp_sec", 0)),
                ", ".join(str(item) for item in activities) or "unclassified",
                (frame.get("scene_summary") or "").strip(),
                (frame.get("reasoning") or "").strip(),
            )
        )
    votes = ", ".join(
        "{}={}".format(label, count)
        for label, count in sorted(activity_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    sounds = "; ".join(sorted({
        str(event.get("label", "")).strip()
        for event in list(sound_events) if str(event.get("label", "")).strip()
    }))
    return f"""You are the final reviewer for ONE video. The frame-by-frame labels below each
only describe an instant, so an operation, exercise, or patrol is usually dominated by frames that
merely show vehicles moving. Judge the video as a whole and choose the ONE label that best
describes its overall purpose.

Labels and definitions:
{ACTIVITY_LABEL_DEFINITIONS}

Guidance:
- Prefer the most specific label the whole sequence supports over a generic one.
- Choose movement only when transit is all the video shows; if the sequence shows an active mission,
  engagement, or mission-specific activity, choose operation instead.
- Choose patrolling when the sequence shows a patrol, guard detail, or security sweep along a route
  rather than plain transit.
- Choose disposition for how forces or equipment are arranged in place; deployment when they are
  arriving and being established there; mobilization when they are being activated or assembled
  before deploying.
- Choose joint exercise for coordinated exercise or simulation involving distinct force types,
  services, or nations. A combined tank-and-infantry formation maneuvering under a simulation/HUD
  may be a joint exercise even when flags are not visible. Choose disposition for a genuinely static
  arrangement with no exercise or coordinated-maneuver context.
- Choose training when mock structures, range or recording overlays, repeated rehearsed movement,
  or other controlled-environment cues show practice. Tactical commands, weapons, and simulated
  gunfire alone do not make it a real operation.
- Choose procurement when the sequence centers on supplies, inventory, warehouses, deliveries, or
  cargo logistics. Loading cargo onto an aircraft is procurement/logistics unless the visuals show
  operational forces arriving and being established at their destination.
- Use the transcript and sound events as supporting evidence alongside the visuals, never as proof
  of something that is not visible or audible.

FRAME OBSERVATIONS:
{chr(10).join(visual_lines) or "No frame observations available."}

PER-FRAME LABEL VOTE: {votes or "none"}

SPEECH TRANSCRIPT: {transcript.strip() or "No speech transcript available."}

NON-SPEECH SOUND EVENTS: {sounds or "No sound events detected."}

Return ONLY a JSON object with exactly these fields and types:
{{
  "activity": "<string: exactly one label from the list above>",
  "confidence": <number between 0.00 and 1.00: your certainty that the whole sequence matches activity>,
  "reasoning": "<string: short justification grounded in the frames and audio evidence>"
}}
Example of the exact reply shape (values are illustrative):
{{"activity": "operation", "confidence": 0.74, "reasoning": "Sustained engagement with supporting vehicles, not plain transit"}}
"""
