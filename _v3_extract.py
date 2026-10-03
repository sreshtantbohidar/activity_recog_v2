import os
import zipfile

ZP = r"C:\Users\Hp\projects\activity-code-v3.zip"
OUT = r"C:\Users\Hp\projects\_v3_extract"

z = zipfile.ZipFile(ZP)
names = z.namelist()
print("total entries", len(names))
os.makedirs(OUT, exist_ok=True)
targets = {"audio_analysis.py", "app.py", "media_utils.py", "vllm_video_pipeline.py",
           "response_schema.py", "test_audio_analysis.py"}
for n in names:
    base = os.path.basename(n)
    if base in targets:
        safe = n.replace("/", "__").replace("\\", "__")
        with open(os.path.join(OUT, safe), "wb") as f:
            f.write(z.read(n))
        print("extracted", n)
