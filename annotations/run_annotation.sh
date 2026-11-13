#!/bin/bash
# 4-hour young-adult annotation server.
# Images live in protocol_4h/images.tar.gz. This script unpacks them next to
# the assignment and serves those files. The full diffusion pool is not required.

cd "$(dirname "$0")"

ARCHIVE="protocol_4h/images.tar.gz"
DEST="protocol_4h/images"
STAMP="$DEST/.archive-stamp"

if [ ! -f "$ARCHIVE" ]; then
    echo "Missing $ARCHIVE. From this directory: python protocol_4h.py --export-images" >&2
    exit 1
fi
if [ ! -f "protocol_4h/assignment.json" ]; then
    echo "Missing protocol_4h/assignment.json" >&2
    exit 1
fi

mkdir -p "$DEST"
NEED=$(sha256sum "$ARCHIVE" | awk '{print $1}')
HAVE=""
if [ -f "$STAMP" ]; then
    HAVE=$(cat "$STAMP")
fi
if [ "$NEED" != "$HAVE" ]; then
    echo "Extracting annotation images..."
    rm -rf "$DEST/by_id"
    tar -xzf "$ARCHIVE" -C "$DEST"
    echo "$NEED" > "$STAMP"
    echo "Extracted into $DEST/by_id"
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    ./venv/bin/pip install "Flask>=2.3.0"
fi
# shellcheck disable=SC1091
source venv/bin/activate

python protocol_4h.py
echo "Starting annotation server on http://localhost:5003"
python annotation_app.py
