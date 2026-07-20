"""Download all required model files into models/.

Fetches / copies:
- silero_vad.onnx — Silero VAD (MIT), copied from the bundled silero_vad pip package
- kokoro-v1.0.onnx + voices-v1.0.bin — Kokoro TTS (Apache 2.0), fetched from GitHub releases

Idempotent: skips files that already exist with the expected size.
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

KOKORO_FILES: list[tuple[str, str]] = [
    (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        "kokoro-v1.0.onnx",
    ),
    (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        "voices-v1.0.bin",
    ),
]


def _download(url: str, dest: Path) -> None:
    print(f"⬇️  Downloading {url}")
    print(f"    → {dest}")
    urllib.request.urlretrieve(url, dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"    ✓ {size_mb:.1f} MiB")


def _copy_silero_from_package() -> bool:
    """Copy silero_vad.onnx from the installed silero_vad package.

    Returns True on success, False if the package or file isn't found.
    """
    try:
        import silero_vad
    except ImportError:
        print("❌ silero_vad package not installed. Run `uv sync` first.")
        return False

    pkg_dir = Path(silero_vad.__file__).parent
    src = pkg_dir / "data" / "silero_vad.onnx"
    if not src.exists():
        print(f"❌ Bundled silero_vad.onnx not found at {src}")
        return False

    dest = MODELS_DIR / "silero_vad.onnx"
    print(f"📄 Copying silero_vad.onnx from pip package")
    print(f"    {src} → {dest}")
    shutil.copy2(src, dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"    ✓ {size_mb:.1f} MiB")
    return True


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {MODELS_DIR}")

    # Silero: copy from bundled package
    silero_dest = MODELS_DIR / "silero_vad.onnx"
    if silero_dest.exists() and silero_dest.stat().st_size > 1024 * 1024:
        print(f"✓ Already present: silero_vad.onnx "
              f"({silero_dest.stat().st_size / 1e6:.1f} MB)")
    elif not _copy_silero_from_package():
        return 1

    # Kokoro: download
    for url, filename in KOKORO_FILES:
        dest = MODELS_DIR / filename
        if dest.exists() and dest.stat().st_size > 1024 * 1024:
            print(f"✓ Already present: {filename} "
                  f"({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        try:
            _download(url, dest)
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}", file=sys.stderr)
            return 1

    print("\n✅ All models ready.")
    print("Next: `uv run python scripts/generate_test_fixtures.py` "
          "to build VAD test fixtures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

