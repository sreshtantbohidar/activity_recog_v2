import hashlib, json, os, sys

ROOT = r"C:\Users\Hp\projects\activity_recog_v2"
EXCLUDE_DIRS = {"hf_models", "deliverable", "__pycache__", ".git", "ubuntu",
                "venv", ".venv", "node_modules", "segments", "reports", "logs",
                "testvideos", "build"}
EXCLUDE_EXT = {".mp4", ".avi", ".mkv", ".tar", ".gz", ".tgz", ".pyc", ".log",
               ".vhdx", ".zip", ".whl"}

out = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d.lower() not in EXCLUDE_DIRS]
    for fn in filenames:
        ext = os.path.splitext(fn)[1].lower()
        if ext in EXCLUDE_EXT:
            continue
        p = os.path.join(dirpath, fn)
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        try:
            h = hashlib.md5()
            sz = os.path.getsize(p)
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            out.append({"path": rel, "size": sz, "md5": h.hexdigest()})
        except OSError as e:
            out.append({"path": rel, "error": str(e)})

json.dump(out, sys.stdout, indent=0)
