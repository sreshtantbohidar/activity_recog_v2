"""Push the 17 test clips from the Windows host (.16) to the Ubuntu target (.125).

The SFTP key shipped with the project has a leading blank line that OpenSSH's
own scp rejects, so this uses paramiko with password auth (the same library the
service itself uses for delivery uploads).
"""
import os

import paramiko

SRC = r"C:\Users\Hp\projects\activity_videos_for_testing"
HOST, USER, PW = "192.168.1.125", "demo", "Tec#2309@Tum"
DEST = "/home/demo/Music/activity-code-v3/activity_recog_v2/testvideos"

t = paramiko.Transport((HOST, 22))
t.connect(username=USER, password=PW)
sftp = paramiko.SFTPClient.from_transport(t)
try:
    sftp.mkdir(DEST)
except IOError:
    pass  # already exists
clips = sorted(f for f in os.listdir(SRC) if f.endswith(".mp4"))
print("clips:", len(clips))
for name in clips:
    local = os.path.join(SRC, name)
    remote = f"{DEST}/{name}"
    try:
        size = sftp.stat(remote).st_size
    except IOError:
        size = -1
    if size == os.path.getsize(local):
        print("skip (already there):", name)
        continue
    print("put:", name, flush=True)
    sftp.put(local, remote)
print("ALL-CLIPS-PUSHED")
t.close()
