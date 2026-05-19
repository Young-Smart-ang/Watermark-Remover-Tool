# Watermark Remover Tool

An AI-powered desktop application for removing watermarks from images using the **LaMa (Large Mask Inpainting)** model. Select unwanted regions with rectangle or brush tools, and the AI fills them with realistic content.

## Features

- **Rectangle Selection** — Drag to select watermark areas precisely
- **Brush Tool** — Freehand mask for irregular regions
- **Before/After Preview** — Toggle to compare original and result
- **Zoom & Pan** — Scroll-wheel zoom, drag to pan
- **Undo** — Revert to previous state
- **Batch Processing** — Process multiple images at once
- **中文 / English** — One-click language switch

## Installation

```bash
pip install -r requirements.txt
python watermark_remover.py
```

## How to Use

1. **Open an image** — Click "Open Image"
2. **Select watermark area** — Use rectangle or brush tool
3. **Remove** — Click "Remove" to start AI inpainting
4. **Save** — Click "Save Result" to export

## License

MIT
