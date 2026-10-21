"""Model file store for operator-uploaded weights (Flask-free, unit-testable).

Operators can push model files through the dashboard's Configuration page; each
upload lands in the directory the runtime actually reads from:

* ``*.pt`` / ``*.onnx``  -> ``<dir>/yolo/<name>``      (object detector)
* zip/tar of a HF model  -> ``<dir>/hf/<name>/``       (config.json + safetensors)
  with the intended runtime recorded as ``target``: ``omni`` (audio
  understanding, activated in-process) or ``vllm`` (vision engine; activation
  writes the marker file the entrypoint honours on the next container start).

Every upload is registered in ``<dir>/registry.json`` so the active models
survive container restarts (app.py replays them on boot) and the dashboard can
list what is deployed. The directory should live on the activity-data volume.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

YOLO_SUFFIXES = (".pt", ".onnx")
ARCHIVE_SUFFIXES = (".zip", ".tar", ".gz", ".tgz")


def classify_kind(filename: str) -> Optional[str]:
    """Map an uploaded filename to a storage kind (None = not a model file)."""
    name = (filename or "").strip().lower()
    if name.endswith(YOLO_SUFFIXES):
        return "yolo"
    if name.endswith(ARCHIVE_SUFFIXES):
        return "archive"
    return None


def _safe_name(filename: str) -> str:
    """Strip any directory components and keep a conservative on-disk name."""
    name = Path(filename.replace("\\", "/")).name.strip()
    keep = [ch if (ch.isalnum() or ch in "._-") else "_" for ch in name]
    cleaned = "".join(keep).strip("._")
    return cleaned or "model"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


class ModelStore:
    """Owns the upload directory, the registry file, and activation state."""

    def __init__(self, root: Path, default_yolo_path: Optional[Path] = None):
        self.root = Path(root)
        self.registry_path = self.root / "registry.json"
        self.default_yolo_path = default_yolo_path
        self.root.mkdir(parents=True, exist_ok=True)

    # -- registry ------------------------------------------------------------
    def _read(self) -> List[Dict[str, Any]]:
        try:
            entries = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return entries if isinstance(entries, list) else []

    def _write(self, entries: List[Dict[str, Any]]) -> None:
        self.registry_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def list(self) -> List[Dict[str, Any]]:
        return [entry for entry in self._read() if Path(entry.get("path", "")).exists()]

    def active(self, kind: str = "yolo") -> Optional[Dict[str, Any]]:
        for entry in self._read():
            if entry.get("kind") == kind and entry.get("active") and Path(entry["path"]).exists():
                return entry
        return None

    def replay_active(self, activate: Callable[[Path], None]) -> Optional[Dict[str, Any]]:
        """Point the runtime at the recorded active model (call on startup)."""
        entry = self.active()
        if entry is None:
            return None
        activate(Path(entry["path"]))
        return entry

    # -- upload ----------------------------------------------------------------
    def save_stream(self, stream, filename: str, target: str = "") -> Dict[str, Any]:
        kind = classify_kind(filename)
        if kind is None:
            raise ValueError("Unsupported model file; upload .pt, .onnx, or a .zip/.tar model archive")
        name = _safe_name(filename)
        if kind == "yolo":
            target_dir = self.root / "yolo"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / name
            counter = 1
            while target.exists():
                target = target_dir / f"{target.stem}_{counter}{target.suffix}"
                counter += 1
            with target.open("wb") as handle:
                shutil.copyfileobj(stream, handle)
            entry = {"id": f"{int(time.time())}-{target.name}", "name": target.name,
                "kind": "yolo", "target": "yolo", "path": str(target), "size": target.stat().st_size,
                "sha256": sha256_of(target), "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "active": False}
        else:
            wanted = target if target in ("omni", "vllm") else ""
            extract_dir = self.root / "hf" / Path(name).stem.replace(".", "_")
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True)
            _extract_archive(stream, name, extract_dir)
            architecture = _read_architectures(extract_dir)
            if not wanted:
                # Infer the runtime from the checkpoint itself when the operator
                # did not pick one explicitly.
                wanted = "omni" if any("Omni" in a for a in architecture) else \
                    "vllm" if any("Qwen3VL" in a or "VL" in a for a in architecture) else ""
            files = [str(p.relative_to(extract_dir)) for p in extract_dir.rglob("*") if p.is_file()]
            entry = {"id": f"{int(time.time())}-{extract_dir.name}", "name": extract_dir.name,
                "kind": "hf", "target": wanted, "path": str(extract_dir), "size": sum(
                    (extract_dir / f).stat().st_size for f in files),
                "sha256": None, "architecture": architecture, "files": files[:200],
                "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "active": False}
        entries = self._read()
        entries.append(entry)
        self._write(entries)
        return entry

    def delete(self, entry_id: str) -> Optional[Dict[str, Any]]:
        entries = self._read()
        keep, removed = [], None
        for entry in entries:
            if entry.get("id") == entry_id:
                removed = entry
                continue
            keep.append(entry)
        if removed is None:
            return None
        self._write(keep)
        shutil.rmtree(removed["path"], ignore_errors=True) if Path(removed["path"]).is_dir() \
            else Path(removed["path"]).unlink(missing_ok=True)
        return removed

    def set_active(self, entry_id: str, activate: Callable[[Path], None]) -> Dict[str, Any]:
        entries = self._read()
        chosen = None
        for entry in entries:
            if entry.get("id") == entry_id:
                if entry.get("kind") == "hf" and not entry.get("target"):
                    raise ValueError("This archive has no detected runtime; upload it again choosing a target")
                if not Path(entry["path"]).exists():
                    raise FileNotFoundError(f"model missing: {entry['path']}")
                entry["active"] = True
                chosen = entry
            elif entry.get("target") in ("yolo", "omni", "vllm") and entry.get("kind") == entry_id_kind(entry_id, entries):
                entry["active"] = False
        if chosen is None:
            raise KeyError(entry_id)
        self._write(entries)
        activate(Path(chosen["path"]))
        return chosen


def entry_id_kind(entry_id: str, entries: List[Dict[str, Any]]) -> str:
    for entry in entries:
        if entry.get("id") == entry_id:
            return str(entry.get("target") or entry.get("kind") or "")
    return ""


def _read_architectures(model_dir: Path) -> List[str]:
    """Read config.json's architectures list so uploads can be auto-classified."""
    config = model_dir / "config.json"
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
        archs = payload.get("architectures") or []
        return [str(a) for a in archs]
    except (OSError, ValueError):
        return []


def _extract_archive(stream, name: str, destination: Path) -> None:
    """Extract an uploaded archive, refusing entries that escape the target."""
    staging = destination.parent / (destination.name + ".uploading")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    local = staging.parent / (staging.name + Path(name).suffix)
    if local.exists():
        local.unlink()
    with local.open("wb") as handle:
        shutil.copyfileobj(stream, handle)
    try:
        if zipfile.is_zipfile(local):
            with zipfile.ZipFile(local) as bundle:
                for member in bundle.namelist():
                    _assert_inside(member, destination)
                bundle.extractall(staging)
        elif tarfile.is_tarfile(local):
            with tarfile.open(local) as bundle:
                for member in bundle.getmembers():
                    _assert_inside(member.name, destination)
                bundle.extractall(staging, filter="data")
        else:
            raise ValueError(f"{name} is not a readable zip/tar archive")
        for child in staging.iterdir():
            shutil.move(str(child), str(destination / child.name))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        local.unlink(missing_ok=True)


def _assert_inside(member: str, destination: Path) -> None:
    """Refuse archive members that would write outside the extraction folder."""
    target = (destination / member).resolve()
    if not str(target).startswith(str(destination.resolve())):
        raise ValueError(f"archive entry escapes the model directory: {member}")
