# Detector weights

`./models` is mounted read-only at `/app/models` in the container. The object
detector loads whatever `OBJECT_MODEL_PATH` points at, and class names are read
from the checkpoint itself, so swapping the model needs **no code change and no
image rebuild**:

1. drop the new checkpoint here, e.g. `models/detector_2026.pt`;
2. point `OBJECT_MODEL_PATH` at it — in `.env` or the shell that runs
   `docker compose`:

   ```sh
   OBJECT_MODEL_PATH=/app/models/detector_2026.pt docker compose \
     -f docker-compose.yml -f docker-compose.ubuntu.yml up -d
   ```

   (or add `OBJECT_MODEL_PATH=/app/models/detector_2026.pt` to `.env`);
3. recreate the container. The detector loads lazily on the first
   `/detect-objects` request, so a restart is all that is needed.

Leaving `OBJECT_MODEL_PATH` unset keeps the checkpoint baked into the image
(`/app/object_detect_model_256_class.pt`). A path that does not resolve fails
loudly — `Object model not found: <path>` — rather than silently falling back to
the baked weights.

`README.md` and the `Dockerfile.ubuntu` health check also read this path, so a
mount that is missing on another host is reported instead of being ignored.
