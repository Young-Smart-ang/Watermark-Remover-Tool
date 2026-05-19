#!/usr/bin/env python3
"""Wrapper that imports and runs the main application (Chinese filename)."""
import sys
import os

# Ensure the script's directory is on the path
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from watermark_remover import main

if __name__ == "__main__":
    main()
