"""Unified audio understanding with Qwen2-Audio-7B-Instruct.

One audio-language model performs both speech transcription (with
utterance-level timestamps) and non-speech sound description in a single
pass per chunk, prompted as: "Transcribe the speech and include sound
descriptions in brackets." The output is parsed into the same schema the
service has always exposed (transcript + sound_events), so cadence fusion,
cross-modal summaries, and PDF reports keep working unchanged.

The model runs on the one GPU this service owns. Vision (vLLM) and audio never
run together: the audio stage must be entered under AUDIO_STAGE_LOCK after all
frame inference has completed AND after the vLLM engine has been put to sleep,
so the audio model gets the whole GPU to itself. Callers release_qwen_audio()
before waking vLLM again.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from threading import Lock
from typing import Any, Dict, List, Tuple

from media_utils import extract_audio_track, read_pcm16_wav

logger = logging.getLogger(__name__)

# Model + runtime configuration (environment-overridable).
QWEN_AUDIO_MODEL_ID = os.environ.get("QWEN_AUDIO_MODEL_ID", "Qwen/Qwen2-Audio-7B-Instruct")
# int4 NF4 keeps the 7B audio model around 5-6 GB of VRAM, matching the budget
# for the audio stage. bitsandbytes >= 0.46 is required for a CUDA 12.8 build.
QWEN_AUDIO_LOAD_IN_4BIT = os.environ.get("QWEN_AUDIO_LOAD_IN_4BIT", "1") == "1"
QWEN_AUDIO_MAX_NEW_TOKENS = int(os.environ.get("QWEN_AUDIO_MAX_NEW_TOKENS", "512"))
QWEN_AUDIO_CHUNK_SECONDS = float(os.environ.get("QWEN_AUDIO_CHUNK_SECONDS", "30"))
QWEN_AUDIO_MAX_SOUNDS_PER_WINDOW = int(os.environ.get("QWEN_AUDIO_MAX_SOUNDS_PER_WINDOW", "10"))

# Retained for schema compatibility with earlier releases.
AUDIO_EVENT_WINDOW_SECONDS = 10
AUDIO_EVENT_HOP_SECONDS = 5
AUDIO_EVENT_CONFIDENCE_THRESHOLD = float(os.environ.get("AUDIO_EVENT_CONFIDENCE_THRESHOLD", "0.35"))

# Labels reported by earlier AST runs that indicate weapon discharge; now
# matched against Qwen2-Audio's free-text sound descriptions.
FIRING_LABEL_KEYWORDS = ("gun", "gunshot", "gunfire", "shot", "firearm", "bullet", "artillery", "explosion")

# A strict, example-driven JSON contract: naming each field and its type keeps the
# reply shape stable, so transcript segments and sound events always parse. The
# earlier two-sentence wording made the model answer "none" even on clips that
# audibly contain radio chatter and engine noise, losing every sound event.
UNIFIED_AUDIO_PROMPT = (
    "Listen to this audio clip and return ONE JSON object describing everything audible in it, "
    "with exactly this structure:\n"
    "{\n"
    "  \"transcript\": [\n"
    "    {\"start_sec\": <number>, \"end_sec\": <number>, \"text\": \"<string>\"}\n"
    "  ],\n"
    "  \"sounds\": [\n"
    "    {\"start_sec\": <number>, \"end_sec\": <number>, \"label\": \"<string>\"}\n"
    "  ]\n"
    "}\n"
    "\n"
    "Field rules:\n"
    "- start_sec and end_sec are numbers: decimal seconds from the clip start (example: 3.2). "
    "end_sec must be greater than start_sec.\n"
    "- transcript[].text is a string: the spoken words of one human utterance; one entry per utterance.\n"
    "- sounds[].label is a string: one short description of one audible non-speech sound, for "
    "example \"heavy vehicle engine\", \"radio chatter\", \"gunfire\".\n"
    "- Use an empty array ([]) when there is no speech or no non-speech sound.\n"
    "\n"
    "Worked example of the exact reply shape (not the content of this clip):\n"
    "{\"transcript\": [{\"start_sec\": 0.8, \"end_sec\": 2.1, \"text\": \"move up\"}], "
    "\"sounds\": [{\"start_sec\": 0.0, \"end_sec\": 3.5, \"label\": \"radio chatter\"}, "
    "{\"start_sec\": 3.0, \"end_sec\": 6.0, \"label\": \"heavy vehicle engine\"}]}\n"
    "\n"
    "Rules:\n"
    "- Always return the JSON object, even when both arrays are empty. Never reply 'none', with "
    "prose, or with markdown fences, and add no text outside the JSON.\n"
    "- Report every sound you actually hear, but do not invent speech or sounds that are not audible.\n"
    "- The worked example shows the shape only: never reuse its words as your answer.\n"
    "- If the clip is truly silent, return {\"transcript\": [], \"sounds\": []}."
)

# Serializes audio-stage GPU work across threads. Callers in app.py also hold
# this around the whole audio stage so it can never overlap vLLM vision work.
AUDIO_STAGE_LOCK = threading.Lock()

_model_lock = Lock()
_audio_components = None


def _load_qwen_audio():
    """Load the unified audio-language model (4-bit NF4, ~5-6 GB VRAM).

    The vLLM vision engine is asleep for the whole audio stage, so the entire
    GPU is free. 4-bit keeps the audio footprint inside its 5-6 GB budget; if the
    bitsandbytes GPU build is unavailable we fall back to bf16, which still fits
    because vLLM is asleep.
    """
    global _audio_components
    if _audio_components is None:
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        local_only = os.getenv("HF_HUB_OFFLINE", "0") == "1"
        logger.info("Loading %s (4bit=%s)", QWEN_AUDIO_MODEL_ID, QWEN_AUDIO_LOAD_IN_4BIT)
        model_kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True, "local_files_only": local_only}
        if QWEN_AUDIO_LOAD_IN_4BIT:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
                model_kwargs["device_map"] = {"": 0}
            except Exception as exc:  # bitsandbytes missing: fall back to bf16
                logger.warning("4-bit load unavailable (%s); using bf16", exc)
                model_kwargs.pop("quantization_config", None)
                model_kwargs.pop("device_map", None)
        if "quantization_config" not in model_kwargs:
            model_kwargs.update(torch_dtype=torch.bfloat16, device_map=None)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(QWEN_AUDIO_MODEL_ID, **model_kwargs)
        model.eval()
        if model_kwargs.get("device_map") is None:
            model.to("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoProcessor.from_pretrained(QWEN_AUDIO_MODEL_ID, local_files_only=local_only)
        _audio_components = (processor, model)
    return _audio_components


def release_qwen_audio() -> None:
    """Unload the audio model to reclaim VRAM for other stages when needed."""
    global _audio_components
    with _model_lock:
        if _audio_components is None:
            return
        import torch
        processor, model = _audio_components
        try:
            model.to("cpu")
        except Exception:
            pass
        del processor, model
        _audio_components = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _cuda_scrub() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _as_number(value: Any, default: float) -> float:
    """Coerce a model-reported number, falling back when it is missing or garbage."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _looks_like_json(text: str) -> bool:
    """True when the reply is shaped like a JSON object (fences and whitespace ignored)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned.startswith("{")


def _load_audio_json(text: str):
    """Parse an audio reply into a dict, tolerating fences and Python-style literals.

    Qwen2-Audio sometimes answers with single-quoted Python dict syntax instead of
    JSON, so ``ast.literal_eval`` salvages that shape. Returns None when the reply
    is not an object.
    """
    import ast
    import json as _json

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    if not cleaned.startswith("{"):
        return None
    try:
        payload = _json.loads(cleaned)
    except ValueError:
        try:
            payload = ast.literal_eval(cleaned)
        except (ValueError, SyntaxError):
            return None
    return payload if isinstance(payload, dict) else None


def _parse_json_audio(text: str, window_start: float, window_end: float):
    """Parse the JSON reply shape when the model answers with JSON instead of lines.

    Returns None when the reply is not a JSON object, so the line parser still runs.
    """
    payload = _load_audio_json(text)
    if payload is None:
        return None
    transcript: List[Dict[str, Any]] = []
    sounds: List[Dict[str, Any]] = []
    for segment in payload.get("transcript") or []:
        if not isinstance(segment, dict):
            continue
        body = str(segment.get("text") or segment.get("transcript") or "").strip()
        if not body:
            continue
        start = _as_number(segment.get("start_sec", 0.0), 0.0)
        end = _as_number(segment.get("end_sec", start), start)
        transcript.append({"start_sec": round(window_start + max(0.0, start), 2),
            "end_sec": round(window_start + max(start + 0.01, end), 2), "text": body})
    for event in payload.get("sounds") or payload.get("sound_events") or []:
        confidence = 1.0
        if isinstance(event, dict):
            label = str(event.get("label") or event.get("text") or event.get("description")
                or event.get("name") or "").strip()
            if "start_sec" in event or "end_sec" in event:
                start = _as_number(event.get("start_sec", 0.0), 0.0)
                end = _as_number(event.get("end_sec", start), start)
                start_abs = round(window_start + max(0.0, start), 2)
                end_abs = round(window_start + max(start + 0.01, end), 2)
            else:
                start_abs, end_abs = round(window_start, 2), round(window_end, 2)
            confidence = _as_number(event.get("confidence", 1.0), 1.0)
        else:
            label = str(event).strip()
            start_abs, end_abs = round(window_start, 2), round(window_end, 2)
        if not label:
            continue
        start_abs = max(window_start, min(start_abs, window_end))
        end_abs = max(start_abs, min(end_abs, window_end))
        sounds.append({"start_sec": round(start_abs, 2), "end_sec": round(end_abs, 2),
            "label": label, "confidence": max(0.0, min(1.0, confidence))})
    return transcript, sounds[:QWEN_AUDIO_MAX_SOUNDS_PER_WINDOW]


def _parse_unified_output(text: str, window_start: float, window_end: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract transcript segments and bracketed sound events from one reply."""
    from_json = _parse_json_audio(text, window_start, window_end)
    if from_json is not None:
        return from_json
    # A JSON-shaped reply that could not be parsed must not be read as bracketed
    # sound names: that turns the raw object into a garbage label. Return empty
    # rather than emit nonsense.
    if _looks_like_json(text):
        logger.warning("Audio reply looked like JSON but could not be parsed: %.200r", text)
        return [], []
    transcript: List[Dict[str, Any]] = []
    sounds: List[Dict[str, Any]] = []
    in_sounds = False
    timestamp_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*:\s*(.+)")
    bracket_pattern = re.compile(r"\[([^\[\]]+)\]")
    # The model often numbers its two sections ("1) TRANSCRIPT: ...", "2) SOUNDS: ...")
    # and sometimes omits the colon, so headers are matched after stripping any
    # leading list marker. Without this the SOUNDS section is never entered and
    # every sound event is silently discarded.
    # A decimal timestamp ("0.5-2.4: ...") must never be mistaken for a "1." marker,
    # so a dotted marker is only stripped when followed by a space and a letter.
    marker_pattern = re.compile(r"^\s*(?:\d+\s*[)\]]\s*|\d+\.\s+(?=[A-Za-z])|[-*\u2022]\s+)")
    empty_values = {"none", "no speech", "n/a", "na", "(none)", "-"}
    saw_header = False
    for raw_line in text.splitlines():
        line = marker_pattern.sub("", raw_line.strip()).strip()
        if not line:
            continue
        header = re.match(r"^(TRANSCRIPT|TRANSCRIPTION|SPEECH|SOUNDS?|SOUND EVENTS?|NON-SPEECH SOUNDS?)\b\s*:?\s*",
                          line, flags=re.IGNORECASE)
        if header:
            saw_header = True
            is_sounds_header = header.group(1).upper().startswith("SOUND")
            in_sounds = is_sounds_header
            line = line[header.end():].strip()
            if not line or line.lower() in empty_values:
                continue
        timestamp_match = timestamp_pattern.match(line)
        if timestamp_match and not in_sounds:
            start, end, body = float(timestamp_match.group(1)), float(timestamp_match.group(2)), timestamp_match.group(3).strip()
            if body:
                transcript.append({"start_sec": round(window_start + max(0.0, start), 2),
                    "end_sec": round(window_start + max(start + 0.01, end), 2), "text": body})
            continue
        # Bracketed text is the model's sound notation; capture it even when the
        # reply forgets the SOUNDS: header entirely.
        brackets = bracket_pattern.findall(line)
        if in_sounds or brackets:
            for description in brackets:
                description = description.strip()
                if description:
                    sounds.append({"start_sec": round(window_start, 2), "end_sec": round(window_end, 2),
                        "label": description, "confidence": 1.0})
            if in_sounds and not brackets:
                sounds.append({"start_sec": round(window_start, 2), "end_sec": round(window_end, 2),
                    "label": line, "confidence": 1.0})
    # Last resort: an unstructured prose reply ("I heard radio chatter and beeps")
    # still carries acoustic evidence, so keep it rather than reporting silence.
    if not transcript and not sounds and not saw_header:
        prose = " ".join(part for part in (text or "").splitlines() if part.strip()).strip()
        if prose and prose.lower() not in empty_values:
            sounds.append({"start_sec": round(window_start, 2), "end_sec": round(window_end, 2),
                "label": prose[:200], "confidence": 1.0})
    # Cap sound spam per window; keep the earliest entries (model-ordered by time).
    return transcript, sounds[:QWEN_AUDIO_MAX_SOUNDS_PER_WINDOW]


def _understand_chunk(processor, model, chunk, device: str, window_start: float = 0.0,
                      window_end: float | None = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run one unified inference over an audio chunk and parse the reply.

    ``window_start`` is the chunk's offset within the full track. It is added to
    every parsed timestamp so results stay on the video timeline instead of
    restarting at zero on each chunk.
    """
    import torch

    conversation = [{"role": "user", "content": [
        {"type": "audio", "audio_url": "placeholder.wav"},
        {"type": "text", "text": UNIFIED_AUDIO_PROMPT}]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=[chunk], sampling_rate=16000, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=QWEN_AUDIO_MAX_NEW_TOKENS, do_sample=False, num_beams=1)
    generated = generated[:, inputs["input_ids"].shape[1]:] if "input_ids" in inputs else generated
    reply = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    del inputs, generated
    if window_end is None:
        window_end = window_start + len(chunk) / 16000
    return _parse_unified_output(reply, window_start, round(window_end, 2))


def _analyze_samples(samples, device: str) -> Dict[str, Any]:
    """Full-timeline unified analysis: sequential 30s chunks, no gaps."""
    import torch

    processor, model = _load_qwen_audio()
    if device == "cuda" and next(model.parameters()).device.type != "cuda":
        model.to(device)
    chunk_samples = int(QWEN_AUDIO_CHUNK_SECONDS * 16000)
    min_chunk_samples = int(0.5 * 16000)
    transcript: List[Dict[str, Any]] = []
    sound_events: List[Dict[str, Any]] = []
    try:
        for offset in range(0, len(samples), chunk_samples):
            chunk = samples[offset:offset + chunk_samples]
            if len(chunk) < min_chunk_samples:
                break
            window_start = offset / 16000
            window_end = min(offset + len(chunk), len(samples)) / 16000
            try:
                # window_start keeps each chunk's timestamps on the video timeline.
                pieces, sounds = _understand_chunk(processor, model, chunk, device, window_start, window_end)
            except RuntimeError:
                _cuda_scrub()
                raise
            for piece in pieces:
                if piece["end_sec"] > window_end:
                    piece["end_sec"] = round(window_end, 2)
                transcript.append(piece)
            for event in sounds:
                event["start_sec"] = round(min(event["start_sec"], window_end), 2)
                sound_events.append(event)
    finally:
        _cuda_scrub()
    return {"transcript": transcript, "sound_events": sound_events}


def analyze_firing_cadence(samples, sample_rate: int = 16000) -> Dict[str, Any]:
    """Classify weapon-fire rhythm from the raw waveform.

    Sustained firing (short, evenly spaced discharges) is characteristic of an
    operation, while bursts separated by long regular pauses match training
    range drills. Pure numpy so it stays cheap and always available.
    """
    import numpy as np

    duration = len(samples) / float(sample_rate)

    def _insufficient(reason: str) -> Dict[str, Any]:
        return {"classification": "insufficient_firing_evidence", "suggested_activity_bias": None,
            "burst_count": 0, "bursts_sec": [], "truncated": False,
            "duration_sec": round(duration, 2), "firing_coverage_pct": 0.0,
            "median_gap_sec": None, "gap_stddev_sec": None, "gap_cv": None, "note": reason}

    hop = int(0.010 * sample_rate)  # 10 ms energy hop
    if duration < 1.0 or len(samples) < hop:
        return _insufficient("audio shorter than one second")

    frames = len(samples) // hop
    energy = np.sqrt(np.mean(np.asarray(samples[:frames * hop], dtype=np.float64).reshape(frames, hop) ** 2, axis=1))
    smooth = np.convolve(energy, np.ones(5) / 5.0, mode="same")  # ~50 ms smoothing
    noise_floor = float(np.median(smooth))
    threshold = max(noise_floor * float(os.getenv("CADENCE_IMPULSE_THRESHOLD_FACTOR", "3.5")), noise_floor + 1e-6)
    above = smooth > threshold

    close_quiet_frames = max(1, int(float(os.getenv("CADENCE_CLOSE_QUIET_SEC", "0.1")) * sample_rate / hop))
    bursts: List[Tuple[int, int]] = []
    in_burst, start_index, quiet = False, 0, 0
    for index, is_loud in enumerate(above):
        if is_loud:
            if not in_burst:
                in_burst, start_index = True, index
            quiet = 0
        elif in_burst:
            quiet += 1
            if quiet >= close_quiet_frames:
                bursts.append((start_index, index - quiet + 1))
                in_burst = False
    if in_burst:
        bursts.append((start_index, frames))

    merge_gap = float(os.getenv("CADENCE_MERGE_GAP_SEC", "0.12"))
    merged: List[List[int]] = []
    for start, end in bursts:
        if merged and (start - merged[-1][1]) * 0.010 < merge_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    burst_times = [(round(start * 0.010, 2), round(end * 0.010, 2)) for start, end in merged]

    gaps = [round(burst_times[i + 1][0] - burst_times[i][1], 2) for i in range(len(burst_times) - 1)]
    coverage = 100.0 * sum(end - start for start, end in burst_times) / duration if duration else 0.0
    median_gap = float(np.median(gaps)) if gaps else None
    gap_std = float(np.std(gaps)) if gaps else None
    gap_cv = (gap_std / median_gap) if median_gap and gap_std is not None else None

    classification, bias, note = "insufficient_firing_evidence", None, None
    min_bursts = int(os.getenv("CADENCE_MIN_BURSTS", "3"))
    if len(burst_times) < min_bursts or median_gap is None:
        note = "fewer than " + str(min_bursts) + " discharge bursts detected"
    else:
        consistent_max = float(os.getenv("CADENCE_CONSISTENT_GAP_MAX_SEC", "1.5"))
        gapped_min = float(os.getenv("CADENCE_GAPPED_GAP_MIN_SEC", "2.0"))
        cv_max = float(os.getenv("CADENCE_GAP_CV_MAX", "0.6"))
        min_coverage = float(os.getenv("CADENCE_MIN_COVERAGE_PCT", "8"))
        if median_gap <= consistent_max and gap_cv is not None and gap_cv <= cv_max and coverage >= min_coverage:
            classification, bias = "consistent", "operation"
        elif median_gap >= gapped_min and gap_cv is not None and gap_cv <= cv_max * 2:
            classification, bias = "gapped", "training"
        elif median_gap <= consistent_max or median_gap >= gapped_min:
            classification, note = "ambiguous", "gap rhythm matches firing but timing is irregular"
        else:
            classification, note = "ambiguous", "inter-burst gaps fall between consistent and gapped ranges"

    return {"classification": classification, "suggested_activity_bias": bias,
        "burst_count": len(burst_times), "bursts_sec": burst_times[:50],
        "truncated": len(burst_times) > 50, "duration_sec": round(duration, 2),
        "firing_coverage_pct": round(coverage, 1),
        "median_gap_sec": None if median_gap is None else round(median_gap, 2),
        "gap_stddev_sec": None if gap_std is None else round(gap_std, 2),
        "gap_cv": None if gap_cv is None else round(gap_cv, 2), "note": note}


def cadence_activity_bias(cadence) -> str | None:
    """Map a firing-cadence result to a supplementary activity vote.
    Consistent sustained firing supports operation; regular long gaps support training."""
    if not isinstance(cadence, dict) or cadence.get("classification") not in {"consistent", "gapped"}:
        return None
    return {"consistent": "operation", "gapped": "training"}[cadence["classification"]]


def analyze_sound_event_cadence(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Derive an operation-vs-training bias from timestamped sound events.

    Works with Qwen2-Audio's described events: tight overlap suggests
    sustained firing; separated, evenly spaced events suggest drill pauses.
    """
    import numpy as np

    valid = []
    for event in events or []:
        try:
            start, end = float(event["start_sec"]), float(event["end_sec"])
        except (TypeError, ValueError, KeyError):
            continue
        if end > start:
            valid.append((start, end, str(event.get("label", ""))))
    valid.sort()

    def _insufficient(reason: str) -> Dict[str, Any]:
        return {"classification": "insufficient_sound_event_evidence", "suggested_activity_bias": None,
            "event_count": len(valid), "covered_sec": 0.0, "coverage_ratio": 0.0,
            "median_gap_sec": None, "gap_cv": None, "note": reason}

    min_events = int(os.getenv("AST_CADENCE_MIN_EVENTS", "3"))
    if len(valid) < min_events:
        return _insufficient("fewer than " + str(min_events) + " timestamped events")

    covered = sum(end - start for start, end, _ in valid)
    span = valid[-1][1] - valid[0][0]
    gaps = [round(valid[i + 1][0] - valid[i][1], 2) for i in range(len(valid) - 1)]
    median_gap = float(np.median(gaps)) if gaps else None
    gap_cv = (float(np.std(gaps)) / median_gap) if median_gap else None
    coverage_ratio = covered / span if span > 0 else 0.0

    classification, bias, note = "ambiguous", None, None
    consistent_gap_max = float(os.getenv("AST_CADENCE_CONSISTENT_GAP_MAX_SEC", "1.5"))
    gapped_gap_min = float(os.getenv("AST_CADENCE_GAPPED_GAP_MIN_SEC", "2.0"))
    cv_max = float(os.getenv("AST_CADENCE_GAP_CV_MAX", "0.6"))
    if median_gap <= consistent_gap_max and gap_cv <= cv_max:
        classification, bias = "consistent", "operation"
    elif median_gap >= gapped_gap_min and gap_cv <= cv_max * 2:
        classification, bias = "gapped", "training"
    else:
        note = "event spacing is irregular; no reliable rhythm"
    return {"classification": classification, "suggested_activity_bias": bias,
        "event_count": len(valid), "covered_sec": round(covered, 2),
        "coverage_ratio": round(coverage_ratio, 2),
        "median_gap_sec": None if median_gap is None else round(median_gap, 2),
        "gap_cv": None if gap_cv is None else round(gap_cv, 2), "note": note}


def sound_event_cadence_bias(result) -> str | None:
    """Map a sound-event cadence result to a supplementary activity vote."""
    if not isinstance(result, dict) or result.get("classification") not in {"consistent", "gapped"}:
        return None
    return {"consistent": "operation", "gapped": "training"}[result["classification"]]


def analyze_video_audio(video_path: str, device: str = "cpu") -> Dict[str, Any]:
    """Analyze the full audio timeline with the unified Qwen2-Audio model.

    Must be called under AUDIO_STAGE_LOCK (strictly after all vision
    inference for the request has finished) so GPU work never overlaps.
    """
    audio_path = None
    result: Dict[str, Any] = {
        "has_audio": False,
        "transcript": [],
        "sound_events": [],
        "firing_cadence": {},
        "background_noise_analysis": {"detected_sounds": [], "analysis_window_sec": AUDIO_EVENT_WINDOW_SECONDS,
            "hop_sec": AUDIO_EVENT_HOP_SECONDS, "overlap_sec": max(0, AUDIO_EVENT_WINDOW_SECONDS - AUDIO_EVENT_HOP_SECONDS),
            "confidence_threshold": AUDIO_EVENT_CONFIDENCE_THRESHOLD},
        "warnings": [],
    }

    try:
        with __import__("tempfile").NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
            audio_path = audio_file.name
        extract_audio_track(video_path, audio_path)
        samples = read_pcm16_wav(audio_path)
        if not len(samples):
            return result
        result["has_audio"] = True

        # Waveform impulse cadence runs on CPU; it never touches the GPU.
        try:
            result["firing_cadence"] = analyze_firing_cadence(samples)
        except Exception as exc:
            logger.exception("Firing cadence analysis failed")
            result["warnings"].append(f"Firing cadence analysis unavailable: {exc}")

        with _model_lock:
            try:
                unified = _analyze_samples(samples, device)
                result["transcript"] = unified["transcript"]
                result["sound_events"] = unified["sound_events"]
            except Exception as exc:  # Keep visual analysis available if the model is not provisioned.
                logger.exception("Unified audio understanding failed")
                result["warnings"].append(f"Qwen2-Audio analysis unavailable: {exc}")

        try:
            result["sound_event_cadence"] = analyze_sound_event_cadence(result["sound_events"])
        except Exception as exc:
            logger.exception("Sound-event cadence analysis failed")
            result["warnings"].append(f"Sound-event cadence unavailable: {exc}")

        try:
            support = [event for event in result["sound_events"]
                if any(keyword in event["label"].lower() for keyword in FIRING_LABEL_KEYWORDS)]
            result["firing_cadence"]["ast_gunfire_support"] = {
                "event_count": len(support),
                "total_sec": round(sum(event["end_sec"] - event["start_sec"] for event in support), 2),
                "labels": sorted({event["label"] for event in support})}
        except Exception as exc:
            logger.exception("Gunfire corroboration failed")
            result["warnings"].append(f"Gunfire corroboration unavailable: {exc}")

        try:
            grouped: Dict[str, Dict[str, Any]] = {}
            for event in result["sound_events"]:
                label = event["label"]
                item = grouped.setdefault(label, {"label": label, "occurrences": 0,
                    "max_confidence": 0.0, "intervals": []})
                item["occurrences"] += 1
                item["max_confidence"] = max(item["max_confidence"], event.get("confidence", 1.0))
                item["intervals"].append({"start_sec": event["start_sec"], "end_sec": event["end_sec"]})
            result["background_noise_analysis"]["detected_sounds"] = list(grouped.values())
        except Exception as exc:
            logger.exception("Sound grouping failed")
            result["warnings"].append(f"Sound grouping unavailable: {exc}")

        return result
    except Exception as exc:
        logger.info("Video audio track unavailable: %s", exc)
        result["warnings"].append(f"Audio analysis unavailable: {exc}")
        return result
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                logger.warning("Could not remove temporary audio file %s", audio_path)
