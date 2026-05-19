#!/usr/bin/env python3
"""
Build script — package Watermark Remover into a standalone Windows .exe.
Requires: pip install pyinstaller
Usage:    python build.py
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = PROJECT_DIR / "dist"
SPEC_FILE = PROJECT_DIR / "watermark_remover.spec"


def build():
    # Ensure PyInstaller is available
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Build command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "WatermarkRemover",
        "--onefile",           # Single .exe
        "--windowed",          # No console window (GUI only)
        "--clean",
        "--noconfirm",
        # Data files: include model downloader, not the model itself (88MB)
        # Add ONNX runtime — PyInstaller usually picks it up automatically
        "--add-data", f"requirements.txt{os.pathsep}.",
        "--hidden-import", "onnxruntime",
        "--hidden-import", "PIL",
        "--hidden-import", "PIL._tkinter_finder",
        "--collect-all", "onnxruntime",
    ]

    # Icon (optional)
    icon_path = PROJECT_DIR / "icon.ico"
    if icon_path.exists():
        cmd.extend(["--icon", str(icon_path)])

    cmd.append(str(PROJECT_DIR / "watermark_remover.py"))

    print("=" * 60)
    print("Building WatermarkRemover.exe ...")
    print("=" * 60)
    subprocess.check_call(cmd)

    exe_path = OUTPUT_DIR / "WatermarkRemover.exe"
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ Build complete: {exe_path} ({size_mb:.1f} MB)")
        print("\nNOTE: The first run will download the AI model (88 MB).")
    else:
        print("\n❌ Build failed — check output above.")


if __name__ == "__main__":
    build()
