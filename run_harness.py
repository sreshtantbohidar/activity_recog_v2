"""Run a harness page in headless Chrome and print the renderer's own report."""
import html as html_mod
import json
import re
import subprocess
import sys
from pathlib import Path

CHROME = Path(r"C:/Program Files/Google/Chrome/Application/chrome.exe")
REPORT = re.compile(r'<div id="__report">(.*?)</div>', re.S)


def run(harness):
    url = "file:///" + str(Path(harness).resolve()).replace("\\", "/")
    proc = subprocess.run(
        [str(CHROME), "--headless=new", "--disable-gpu", "--no-sandbox",
         "--no-first-run", "--disable-extensions", "--virtual-time-budget=5000",
         "--dump-dom", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
    )
    match = REPORT.search(proc.stdout or "")
    if not match:
        print("no report element found; chrome stderr tail:")
        print((proc.stderr or "")[-2000:])
        return None
    return json.loads(html_mod.unescape(match.group(1)))


def main():
    for harness in sys.argv[1:]:
        report = run(harness)
        if report is None:
            continue
        print(f"===== {harness}")
        print(json.dumps(report, indent=2)[:6000])


if __name__ == "__main__":
    main()
