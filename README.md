# Watermark Remover Tool

An AI-powered desktop application for removing watermarks from images. Supports **AI inpainting (LaMa)** and **traditional OpenCV methods**, batch processing, multi-language interface, and standalone .exe packaging.

## Features

| Feature | Description |
|---------|-------------|
| 🎨 **3 Models** | LaMa AI (high quality), OpenCV Telea (fast), OpenCV NS (smooth) |
| ✏️ **Rect & Brush** | Rectangle selection + freehand brush for any watermark shape |
| 👁 **Before/After** | Toggle to compare original and result instantly |
| 🔍 **Zoom & Pan** | Scroll-wheel zoom, drag to pan on large images |
| ↩️ **Multi-step Undo** | Full undo history (up to 50 steps) |
| 📦 **Batch Processing** | Process multiple images at once with auto water-mark detection |
| 🌐 **中文 / English** | One-click language switch |
| ⏳ **Progress Dialog** | Non-blocking inference with progress feedback |
| ⌨️ **Keyboard Shortcuts** | `Ctrl+O` open, `Ctrl+S` save, `Ctrl+R` remove, `Ctrl+Z` undo |
| 📦 **Standalone .exe** | Package with PyInstaller — no Python needed to run |

## Screenshots

*(Coming soon)*

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
git clone https://github.com/Young-Smart-ang/Watermark-Remover-Tool.git
cd Watermark-Remover-Tool
pip install -r requirements.txt
```

### Launch

```bash
python watermark_remover.py
```

Or double-click `启动去水印.bat` on Windows.

## How to Use

1. **Open an image** — Click "Open Image" or press `Ctrl+O`
2. **Select watermark area** — Use rectangle tool (default) or brush tool
3. **Choose a model** — LaMa (best quality), Telea (fastest), or NS (smooth)
4. **Remove** — Click "Remove!" or press `Ctrl+R` to start inpainting
5. **Save** — Click "Save Result" or press `Ctrl+S` to export

**Pro tips:**
- Switch to English via the 🌐 button in the toolbar
- For large watermarks, process in multiple small selections for better quality
- Use zoom (`Ctrl+Scroll`) for pixel-level precision
- Batch mode is great for images with watermarks in similar positions

## Models

| Model | Quality | Speed | Download Needed | Best For |
|-------|---------|-------|-----------------|----------|
| **LaMa AI** | ⭐⭐⭐ Best | 🐢 Slow | 88 MB | Complex watermarks, backgrounds |
| **Telea** | ⭐⭐ Good | ⚡ Instant | No | Simple text watermarks |
| **NS** | ⭐⭐ Good | ⚡ Instant | No | Smooth edge blending |

## Batch Processing

The batch dialog lets you:
1. Add multiple images
2. Choose output folder
3. Select model (LaMa/Telea/NS)
4. Process all with a single click
5. Per-image progress log

## Building .exe (Windows)

```bash
pip install pyinstaller
python build.py
```

The standalone `.exe` will be in the `dist/` folder — no Python installation required on the target machine.

## How It Works

- **LaMa**: Uses ONNX Runtime to run the LaMa (Large Mask Inpainting) deep learning model. Trained by Samsung Research, it can fill large missing regions with coherent, realistic content.
- **Telea / NS**: Traditional OpenCV inpainting algorithms — fast, lightweight, no model needed. Best for simple watermarks on uniform backgrounds.
- All processing runs **100% locally** — no data is ever sent to any server.

## Tech Stack

- **Python** — Core language
- **Tkinter** — Desktop GUI framework
- **ONNX Runtime** — Deep learning inference engine
- **OpenCV** — Image processing and traditional inpainting
- **Pillow** — Image format support (PNG, JPEG, BMP, TIFF, WebP)
- **NumPy** — Array operations
- **LaMa** — State-of-the-art inpainting model (via OpenCV Model Zoo)
- **Hugging Face Hub** — Model distribution

## License

MIT License — see [LICENSE](LICENSE) for details.
