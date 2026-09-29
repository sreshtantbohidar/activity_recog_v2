"""SFTP delivery of annotated object-detection artefacts.

The DB-driven /detect-objects mode must place its artefact on the shared file
server before the URL is recorded in media_analysis.output_file_path. paramiko
is imported lazily so a missing wheel degrades only this path instead of the
whole API.
"""
from __future__ import annotations

import io
import logging
import os
import posixpath
from pathlib import Path

logger = logging.getLogger("activity-recognition")


class SftpUploadError(RuntimeError):
    """Raised when an artefact cannot be published to the file server."""


def _config():
    host = os.getenv("SFTP_HOST", "").strip()
    return {
        "host": host,
        "port": int(os.getenv("SFTP_PORT", "22")),
        "user": os.getenv("SFTP_USER", "").strip(),
        "key_path": os.getenv("SFTP_PRIVATE_KEY_PATH", "").strip(),
        "remote_dir": os.getenv("SFTP_REMOTE_DIR", "").strip(),
        # The legacy service served artefacts off the same host that accepted the
        # upload, so the public base defaults to that host on port 9000.
        "public_base": os.getenv("OUTPUT_PUBLIC_BASE", "").strip().rstrip("/")
            or (f"https://{host}:{os.getenv('OUTPUT_PUBLIC_PORT', '9000')}" if host else ""),
        "public_path": os.getenv("OUTPUT_PUBLIC_PATH", "/media_analytics/output").strip().rstrip("/"),
    }


def _require_upload_config(config):
    required = (
        ("SFTP_HOST", config["host"]),
        ("SFTP_USER", config["user"]),
        ("SFTP_PRIVATE_KEY_PATH", config["key_path"]),
        ("SFTP_REMOTE_DIR", config["remote_dir"]),
    )
    missing = [name for name, value in required if not value]
    if missing:
        raise SftpUploadError("SFTP delivery is not configured; missing " + ", ".join(missing))


def _private_key(config):
    try:
        import paramiko
    except ImportError as exc:  # pragma: no cover - depends on the image build
        raise SftpUploadError("paramiko is not installed in this image") from exc
    path = Path(config["key_path"])
    if not path.exists():
        raise SftpUploadError(f"SFTP private key not found: {path}")
    # The shipped key file starts with a blank line, which OpenSSH's own parser
    # rejects; stripping it keeps the file usable without hand-editing it.
    text = path.read_text(encoding="utf-8").lstrip()
    for loader in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
        try:
            return paramiko, loader.from_private_key(io.StringIO(text))
        except paramiko.SSHException:
            continue
    raise SftpUploadError(f"Unsupported private key format: {path}")


def _ensure_remote_dir(sftp, remote_dir):
    current = ""
    for folder in remote_dir.strip("/").split("/"):
        current += f"/{folder}"
        try:
            sftp.stat(current)
        except FileNotFoundError:
            logger.info("Creating remote directory %s", current)
            sftp.mkdir(current)


def upload_to_sftp(local_file_path, relative_path):
    """Upload `local_file_path` to <SFTP_REMOTE_DIR>/<relative_path>."""
    config = _config()
    _require_upload_config(config)
    paramiko, key = _private_key(config)
    remote_path = posixpath.join(config["remote_dir"], relative_path)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=config["host"], port=config["port"], username=config["user"],
            pkey=key, timeout=15)
        with client.open_sftp() as sftp:
            _ensure_remote_dir(sftp, posixpath.dirname(remote_path))
            sftp.put(str(local_file_path), remote_path)
    except SftpUploadError:
        raise
    except Exception as exc:
        raise SftpUploadError(f"SFTP upload failed for {remote_path}: {exc}") from exc
    finally:
        client.close()
    logger.info("Published %s -> %s", local_file_path, remote_path)
    return remote_path


def publish(local_file_path, media_type):
    """Upload an artefact under its media_type folder and return its public URL."""
    config = _config()
    if not config["public_base"]:
        raise SftpUploadError("OUTPUT_PUBLIC_BASE (or SFTP_HOST) is required to build the output URL")
    relative_path = f"{media_type}/{Path(local_file_path).name}"
    upload_to_sftp(local_file_path, relative_path)
    return f"{config['public_base']}{config['public_path']}/{relative_path}"
