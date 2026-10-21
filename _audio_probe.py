"""Isolated probe for the audio stage: run each chunk and report timing/failures."""
import sys
import time
import traceback
import numpy as np

from media_utils import extract_audio_track, read_pcm16_wav
import audio_analysis as A


def main():
    path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    audio_path = "/tmp/probe_audio.wav"

    t0 = time.time()
    extract_audio_track(path, audio_path)
    samples = read_pcm16_wav(audio_path)
    print("extract+read %.1fs  samples=%d duration=%.1fs" % (
        time.time() - t0, len(samples), len(samples) / 16000.0), flush=True)

    t0 = time.time()
    proc, model = A._load_qwen_audio()
    print("model loaded %.1fs" % (time.time() - t0), flush=True)

    chunk_samples = int(A.QWEN_AUDIO_CHUNK_SECONDS * 16000)
    min_samples = int(0.5 * 16000)
    n = 0
    failures = 0
    for offset in range(0, len(samples), chunk_samples):
        chunk = samples[offset:offset + chunk_samples]
        if len(chunk) < min_samples:
            break
        ws = offset / 16000.0
        we = min(offset + len(chunk), len(samples)) / 16000.0
        n += 1
        t0 = time.time()
        try:
            tr, so = A._understand_chunk(proc, model, chunk, device, ws, we)
            print("chunk %d %.1f-%.1fs ok transcript=%d sounds=%d dt=%.1fs" % (
                n, ws, we, len(tr), len(so), time.time() - t0), flush=True)
            for item in tr:
                print("    SPEECH:", repr(item.get("text"))[:300], flush=True)
            for item in so:
                print("    SOUND :", repr(item.get("label"))[:200], flush=True)
        except Exception as exc:
            failures += 1
            print("chunk %d %.1f-%.1fs FAILED %s: %s" % (
                n, ws, we, type(exc).__name__, exc), flush=True)
            traceback.print_exc()
    print("DONE chunks=%d failures=%d" % (n, failures), flush=True)


if __name__ == "__main__":
    main()
