import json, os, subprocess, sys

hashes = json.load(open("_remote_work/_hashes_dot16.json"))
EXCLUDE_PREFIXES = ("secrets/", "config/", ".docker-anonymous/", ".docker-codex/")
EXCLUDE_NAMES = {".env.bak-before-db-config", ".env.new-host", "build_log.txt",
                 "_hash_tree.py"}

# Paths on .16 that must land in the PR branch
fetch = []
for x in hashes:
    if "error" in x:
        continue
    p = x["path"]
    if p in EXCLUDE_NAMES or p.startswith(EXCLUDE_PREFIXES):
        continue
    fetch.append(p)

# Local-only files that are NOT on .16 -> remove from the branch
local_only_remove = [
    "_audio_probe.py", "_probe_omni.py", "_push_clips.py", "_v3_extract.py",
    "_xmodal_check.py", "alljobs.json", "audio_analysis.py.bak",
    "deploy_ubuntu.sh", "docker-compose.ubuntu.yml", "export_deliverable.cmd",
    "job466.json", "job97.json", "jobfix2.json", "jobfix3.json",
    "model_store.py", "test_audio_analysis.py.bak", "v3_audio_analysis.py",
    "UBUNTU_DEPLOY.md", ".env.ubuntu",
]

failed = []
for rel in fetch:
    dest = os.path.join("_remote_work", rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    r = subprocess.run(["python", "/tmp/sftp.py", "get",
                        f"projects/activity_recog_v2/{rel}", dest],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        failed.append((rel, (r.stderr or r.stdout).strip()[:120]))

for rel in local_only_remove:
    p = os.path.join("_remote_work", rel.replace("/", os.sep))
    if os.path.exists(p):
        os.remove(p)
        print("removed local-only:", rel)

print(f"fetched {len(fetch) - len(failed)}/{len(fetch)}")
for rel, err in failed:
    print("FAILED:", rel, "->", err)
