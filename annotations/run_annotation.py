#!/usr/bin/env python3
"""4-hour young-adult annotation server.

Same steps as run_annotation.sh, for Windows, macOS, and Linux.
Images live in protocol_4h/images.tar.gz. This script unpacks them next to
the assignment and serves those files. The full diffusion pool is not required.

    python run_annotation.py
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "protocol_4h" / "images.tar.gz"
DEST = ROOT / "protocol_4h" / "images"
STAMP = DEST / ".archive-stamp"
ASSIGNMENT = ROOT / "protocol_4h" / "assignment.json"
VENV = ROOT / "venv"


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_images() -> None:
    if not ARCHIVE.is_file():
        print(
            f"Missing {ARCHIVE.relative_to(ROOT)}. From this directory: "
            "python protocol_4h.py --export-images",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not ASSIGNMENT.is_file():
        print(f"Missing {ASSIGNMENT.relative_to(ROOT)}", file=sys.stderr)
        raise SystemExit(1)

    DEST.mkdir(parents=True, exist_ok=True)
    need = sha256(ARCHIVE)
    have = STAMP.read_text(encoding="utf-8").strip() if STAMP.is_file() else ""
    if need == have:
        return

    print("Extracting annotation images...")
    by_id = DEST / "by_id"
    if by_id.exists():
        shutil.rmtree(by_id)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        # Python 3.12 warns unless extraction is filtered.
        if sys.version_info >= (3, 12):
            archive.extractall(DEST, filter="data")
        else:
            archive.extractall(DEST)
    STAMP.write_text(need + "\n", encoding="utf-8")
    print(f"Extracted into {by_id.relative_to(ROOT)}")


def ensure_venv(py: Path) -> None:
    if not py.is_file():
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
    probe = subprocess.run(
        [str(py), "-c", "import flask"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode != 0:
        subprocess.check_call([str(py), "-m", "pip", "install", "Flask>=2.3.0"])


def main() -> None:
    ensure_images()
    py = venv_python()
    ensure_venv(py)
    subprocess.check_call([str(py), "protocol_4h.py"], cwd=ROOT)
    print("Starting annotation server on http://localhost:5003")
    raise SystemExit(subprocess.call([str(py), "annotation_app.py"], cwd=ROOT))


if __name__ == "__main__":
    main()
