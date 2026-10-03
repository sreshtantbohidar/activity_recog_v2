"""Thermal-efficient sequential video pipeline backed by vLLM.

Slices uploads into fixed time windows, infers each window with Qwen2.5-VL
under tight GPU/memory parameters, and inserts a cool-down pause between
segments so sustained 5-minute jobs never generate sustained thermal spikes.

Run standalone:  python vllm_video_pipeline.py <video> [output.json]
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from media_utils import ACTIVITY_PROMPT, normalize_activity_result, parse_json_text, validate_video_duration

logger = logging.getLogger("vllm-pipeline")

# ---- Engine / segmentation settings (environment-overridable) ---------------
# int4 AWQ build of Qwen3-VL-8B: ~6 GB of weights instead of ~17.5 GB, with
# the vision tower left unquantized by the checkpoint.
VLLM_MODEL_ID = os.getenv("VLLM_MODEL_ID", "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit")
WINDOW_SECONDS = float(os.getenv("VLLM_WINDOW_SECONDS", "15"))
WINDOW_OVERLAP_SECONDS = float(os.getenv("VLLM_WINDOW_OVERLAP_SECONDS", "0"))
SEGMENT_DIR = os.getenv("VLLM_SEGMENT_DIR", tempfile.gettempdir())
COOLDOWN_SECONDS = float(os.getenv("VLLM_COOLDOWN_SECONDS", "0.25"))
MAX_NEW_TOKENS = int(os.getenv("VLLM_MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.getenv("VLLM_TEMPERATURE", "0.1"))
GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.82"))
MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "16384"))
MM_MAX_PIXELS = int(os.getenv("VLLM_MM_MAX_PIXELS", str(448 * 448)))
MM_NFRAMES = int(os.getenv("VLLM_MM_NFRAMES", "8"))
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")


def plan_windows(duration_sec: float, window_sec: float = WINDOW_SECONDS,
                 overlap_sec: float = WINDOW_OVERLAP_SECONDS) -> List[Dict[str, float]]:
    """Split [0, duration] into consecutive windows with optional overlap."""
    if duration_sec <= 0:
        return []
    step = max(0.1, window_sec - overlap_sec)
    windows: List[Dict[str, float]] = []
    start = 0.0
    while start < duration_sec - 1e-6:
        end = min(start + window_sec, duration_sec)
        windows.append({"start_sec": round(start, 3), "end_sec": round(end, 3)})
        if end >= duration_sec:
            break
        start += step
    return windows


def extract_window_clip(video_path: str, start_sec: float, end_sec: float) -> str:
    """Cut [start_sec, end_sec) into a small re-encoded MP4 via ffmpeg."""
    window_path = os.path.join(SEGMENT_DIR, f"vllm_win_{int(start_sec * 1000):09d}.mp4")
    command = [
        FFMPEG_PATH, "-nostdin", "-v", "error", "-y",
        "-ss", f"{start_sec:.3f}", "-i", video_path,
        "-t", f"{max(0.1, end_sec - start_sec):.3f}",
        "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "23", "-pix_fmt", "yuv420p", "-an",
        window_path,
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    return window_path


def sample_window_frames(window_path: str, nframes: int = MM_NFRAMES) -> Any:
    """Decode at most `nframes` keyframes (RGB, pixel-bounded) with OpenCV."""
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(window_path)
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"No decodable frames in {window_path}")
        indices = np.unique(np.linspace(0, total - 1, max(1, nframes), dtype=np.int64))
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            scale = (MM_MAX_PIXELS / float(height * width)) ** 0.5
            if scale < 1.0:
                frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not frames:
            raise ValueError(f"Could not decode frames from {window_path}")
        return np.stack(frames)
    finally:
        capture.release()


def build_engine() -> Any:
    """Instantiate the vLLM engine with the thermal-tight configuration."""
    os.environ.setdefault("VLLM_TORCH_COMPILE_LEVEL", "0")
    from vllm import LLM

    return LLM(
        model=VLLM_MODEL_ID,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        limit_mm_per_prompt={"video": 1},
        enforce_eager=True,  # required: Qwen3-VL dynamic shapes break CUDA graphs
        mm_processor_kwargs={
            "max_pixels": MM_MAX_PIXELS,  # bounds the video grid token footprint
            "nframes": MM_NFRAMES,
        },
    )


def build_sampling_params() -> Any:
    from vllm import SamplingParams
    return SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS)


def infer_window(llm: Any, sampling_params: Any, window_path: str,
                 prompt: str = ACTIVITY_PROMPT) -> Dict[str, Any]:
    """Run one video-window inference and normalize the JSON verdict."""
    frames = sample_window_frames(window_path)
    # Qwen3-VL expands the single <|video_pad|> placeholder to the grid
    # token count computed from the provided frames and mm_processor_kwargs.
    inputs = {"prompt": f"<|vision_start|><|video_pad|><|vision_end|>{prompt}",
        "multi_modal_data": {"video": frames}}
    outputs = llm.generate(inputs, sampling_params)
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    return normalize_activity_result(parse_json_text(text))


def analyze_video_sequential(video_path: str, llm: Any = None, sampling_params: Any = None,
                             prompt: str = ACTIVITY_PROMPT) -> Dict[str, Any]:
    """Walk the video chronologically window-by-window with cool-down pauses."""
    metadata = validate_video_duration(video_path, float(os.getenv("MAX_VIDEO_DURATION_SECONDS", "300")))
    owned_engine = llm is None
    if owned_engine:
        llm = build_engine()
        sampling_params = build_sampling_params()
    windows = plan_windows(metadata["duration_sec"])
    results: List[Dict[str, Any]] = []
    try:
        for index, window in enumerate(windows):
            window_path = extract_window_clip(video_path, window["start_sec"], window["end_sec"])
            try:
                verdict = infer_window(llm, sampling_params, window_path, prompt)
            finally:
                Path(window_path).unlink(missing_ok=True)
            results.append({"window_index": index, "start_sec": window["start_sec"],
                "end_sec": window["end_sec"], "analysis": verdict})
            logger.info("Window %d/%d (%.1f-%.1fs) done", index + 1, len(windows),
                window["start_sec"], window["end_sec"])
            # Physical pause after every individual inference step: drops GPU
            # power draw to zero for a split second so fans can expel heat.
            if COOLDOWN_SECONDS > 0:
                time.sleep(COOLDOWN_SECONDS)
    finally:
        if owned_engine:
            import gc

            import torch
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {"metadata": metadata, "windows_analyzed": len(results), "window_results": results}


def main(argv: List[str]) -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if not 1 <= len(argv) <= 3:
        print("Usage: python vllm_video_pipeline.py <video> [output.json]", file=sys.stderr)
        return 2
    video_path = argv[1]
    analysis = analyze_video_sequential(video_path)
    output_path = argv[2] if len(argv) == 3 else str(
        Path(video_path).with_name(Path(video_path).stem + "_vllm_analysis.json"))
    Path(output_path).write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    logger.info("Wrote %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
