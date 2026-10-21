#!/bin/sh
# Flatten HF cache duplication: snapshot files that are real copies of blobs
# become relative symlinks (the layout HF Hub itself writes), so the image and
# its docker save tar carry each weight exactly once.
set -e
HUB=/models/huggingface/hub
echo "before: $(du -sh $HUB | cut -f1)"
saved=0
for repo in "$HUB"/models--*; do
  [ -d "$repo/blobs" ] || continue
  for snap in "$repo"/snapshots/*/; do
    [ -d "$snap" ] || continue
    for f in "$snap"*; do
      [ -f "$f" ] || continue
      [ -L "$f" ] && continue
      fname=$(basename "$f")
      size=$(stat -c %s "$f")
      blob=$(sha256sum "$f" | cut -d' ' -f1)
      target="$repo/blobs/$blob"
      if [ -f "$target" ]; then
        # An identical blob already exists; just link and drop the copy.
        same=yes
      else
        if mv "$f" "$target" 2>/dev/null; then same=no; else
          # Cross-filesystem fallback: copy then remove.
          cp "$f" "$target" && rm "$f" && same=no
        fi
        size=0
      fi
      ln -s "$target" "$f"
      [ "$same" = "yes" ] && saved=$((saved + size))
    done
  done
done
echo "relinked duplicate bytes: $((saved / 1000000)) MB"
echo "after: $(du -sh $HUB | cut -f1)"
echo "DEDUPE-DONE"
