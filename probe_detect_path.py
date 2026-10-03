"""Run the deployed ObjectDetector.detect on the same input the failing job used.

The queue paths are redirected so importing app does not touch the live queue
database or purge its retained uploads.
"""
import glob
import os
import shutil
import traceback

os.environ["JOB_DB_PATH"] = "/tmp/probe-jobs.sqlite3"
os.environ["JOB_FILES_DIR"] = "/tmp/probe-uploads"

import app  # noqa: E402

source = sorted(glob.glob("/app/data/uploads/*activity_*"))[-1]
target = "/tmp/probe-copy.png"
shutil.copy(source, target)
print("source:", source)
print("staged target:", target, "exists:", os.path.exists(target))
try:
    result = app.detector.detect(target)
    print("detect returned keys:", sorted(result))
    print("counts:", result["detected_object_counts"])
except Exception:
    traceback.print_exc()
