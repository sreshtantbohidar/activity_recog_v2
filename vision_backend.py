"""Vision inference client for the single in-container vLLM engine.

Only one vision model is supported: the int4 AWQ build of Qwen3-VL-8B-Instruct
served by the vLLM engine that runs inside this same container. There is no
fallback backend.

The vLLM engine and the audio model share one GPU. The engine is capped
so both fit in VRAM at once, and the caller (app.py) serializes the two stages
with locks so they never *execute* together.

When the service is idle the engine is put to sleep and the GPU is released for
other containers (see engine_power.py), which uses the control endpoints below.
"""

import base64
import json
import os
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen

SUPPORTED_VISION_BACKENDS = {"vllm"}
DEFAULT_VISION_BACKEND = "vllm"
# int4 AWQ build of Qwen3-VL-8B-Instruct (~6 GB weights, vision tower left
# unquantized); this is the only model the in-container vLLM engine serves.
DEFAULT_VLLM_VISION_MODEL = "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"


class VisionBackendError(RuntimeError):
    """Raised when the vLLM engine is misconfigured or cannot analyze an image."""


def get_vision_backend(value: Optional[str] = None) -> str:
    """Read and validate ACTIVITY_VISION_BACKEND (vLLM only)."""
    backend = (value if value is not None else os.environ.get(
        "ACTIVITY_VISION_BACKEND", DEFAULT_VISION_BACKEND))
    backend = backend.strip().lower()
    if backend not in SUPPORTED_VISION_BACKENDS:
        raise VisionBackendError(
            f"Unsupported ACTIVITY_VISION_BACKEND={backend!r}; "
            f"the only supported backend is 'vllm'."
        )
    return backend


def vllm_base_url(host: Optional[str] = None) -> str:
    """Resolve the in-container vLLM OpenAI-compatible server base URL."""
    return (host or os.environ.get(
        "VLLM_VISION_HOST", "http://127.0.0.1:8001"
    )).rstrip("/")


def check_vllm_server(
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 3,
    opener=urlopen,
) -> Dict[str, Any]:
    """Check the vLLM engine and whether the vision model is served."""
    model_name = model or os.environ.get("VLLM_VISION_MODEL", DEFAULT_VLLM_VISION_MODEL)
    request = Request(vllm_base_url(host) + "/v1/models", method="GET")
    try:
        with opener(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        served = [item.get("id", "") for item in result.get("data", []) if isinstance(item, dict)]
        return {"reachable": True, "model": model_name, "model_available": model_name in served}
    except Exception as exc:
        return {"reachable": False, "model": model_name, "model_available": False, "error": str(exc)}


def analyze_image_with_vllm(
    image_path: str,
    prompt: str,
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    opener=urlopen,
) -> str:
    """Send one image to the in-container vLLM engine (OpenAI chat format)."""
    timeout = timeout or float(os.environ.get("VLLM_VISION_TIMEOUT_SECONDS", "180"))
    model_name = model or os.environ.get("VLLM_VISION_MODEL", DEFAULT_VLLM_VISION_MODEL)
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("ascii")
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                {"type": "text", "text": prompt},
            ]}],
            "temperature": 0.1,
            "max_tokens": int(os.environ.get("VLLM_MAX_NEW_TOKENS", "256")),
            "stream": False,
        }
        request = Request(
            vllm_base_url(host) + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        text = (result.get("choices") or [{}])[0].get("message", {}).get("content")
        if not isinstance(text, str) or not text.strip():
            raise VisionBackendError("vLLM server returned no message content.")
        return text.strip()
    except VisionBackendError:
        raise
    except Exception as exc:
        raise VisionBackendError(f"vLLM image analysis failed: {exc}") from exc


def complete_text_with_vllm(
    prompt: str,
    host: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 320,
    timeout: Optional[float] = None,
    opener=urlopen,
) -> str:
    """Run a text-only completion on the same vLLM engine (cross-modal summary)."""
    timeout = timeout or float(os.environ.get("VLLM_VISION_TIMEOUT_SECONDS", "180"))
    model_name = model or os.environ.get("VLLM_VISION_MODEL", DEFAULT_VLLM_VISION_MODEL)
    payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": max_tokens, "stream": False}
    try:
        request = Request(vllm_base_url(host) + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with opener(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        text = (result.get("choices") or [{}])[0].get("message", {}).get("content")
        if not isinstance(text, str) or not text.strip():
            raise VisionBackendError("vLLM returned no cross-modal summary content.")
        return text.strip()
    except VisionBackendError:
        raise
    except Exception as exc:
        raise VisionBackendError(f"vLLM text completion failed: {exc}") from exc


def _vllm_control(path: str, host: Optional[str] = None,
                  timeout: Optional[float] = None, opener=urlopen) -> Dict[str, Any]:
    """POST one vLLM sleep/wake control endpoint and return the decoded reply."""
    timeout = timeout or float(os.environ.get("VLLM_CONTROL_TIMEOUT_SECONDS", "120"))
    request = Request(vllm_base_url(host) + path, data=b"", method="POST")
    try:
        with opener(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
        return json.loads(body) if body.strip() else {}
    except Exception as exc:
        raise VisionBackendError(f"vLLM control call {path} failed: {exc}") from exc


def sleep_vllm(host: Optional[str] = None, level: int = 1,
               timeout: Optional[float] = None, opener=urlopen) -> Dict[str, Any]:
    """Park the engine: offload weights to CPU RAM and discard the KV cache.

    Releases most of the engine's VRAM without reloading the checkpoint; wake
    with ``wake_vllm``. Used when the service is idle.
    """
    return _vllm_control(f"/sleep?level={int(level)}", host=host, timeout=timeout, opener=opener)


def wake_vllm(host: Optional[str] = None,
              timeout: Optional[float] = None, opener=urlopen) -> Dict[str, Any]:
    """Restore the engine after a sleep, before serving vision work."""
    return _vllm_control("/wake_up", host=host, timeout=timeout, opener=opener)
