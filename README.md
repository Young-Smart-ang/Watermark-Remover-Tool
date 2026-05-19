# Watermark Remover Tool

An AI-powered desktop application for removing watermarks from images using the **LaMa (Large Mask Inpainting)** model. Select unwanted regions with rectangle or brush tools, and the AI fills them with realistic content. All processing runs **100% locally** — no data is sent to any server.

## Features

- **Rectangle Selection** — Drag to select watermark areas precisely
- **Brush Tool** — Freehand mask for irregular regions
- **Before / After Preview** — Toggle to compare original and result
- **Zoom & Pan** — Scroll-wheel zoom, drag to pan on large images
- **Undo** — Revert to previous selection state
- **Batch Processing** — Remove watermarks from multiple images at once
- **中文 / English** — One-click language switch

## Prerequisites

- **Python 3.10 or higher**
- **pip** (Python package installer)

## Installation

### Step 1: Clone or download

```bash
git clone https://github.com/Young-Smart-ang/Watermark-Remover-Tool.git
cd Watermark-Remover-Tool
```

Or download and extract the ZIP from GitHub.

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `onnxruntime`, `pillow`, `numpy`, `opencv-python-headless`, `huggingface-hub`

### Step 3: Launch

```bash
python watermark_remover.py
```

Or double-click `启动去水印.bat` on Windows.

> **Note:** The first launch will download the AI model (~88 MB). This happens automatically and only once.

## How to Use

### Single image

1. **Open an image** — Click "Open Image" or use the file dialog
2. **Select the watermark area**:
   - **Rectangle tool** (default): Click and drag to draw a rectangle around the watermark
   - **Brush tool**: Click the "Brush" radio button, then paint over the watermark
   - Adjust brush size with the "Size" spinner
3. **Remove the watermark** — Click "Remove" and wait for the AI to process
4. **Toggle view** — Click "Show Result" / "Show Original" to compare
5. **Save** — Click "Save Result" to export the cleaned image

### Tips

- For large watermarks, process in multiple small selections for better quality
- The brush tool works best for irregular shapes
- Use mouse scroll wheel to zoom in/out for precise editing

### Batch processing

1. Click the "Batch" button in the toolbar
2. **Add images** — Click "Add Images" and select multiple files
3. **Choose output folder** — Click "Browse" to select where to save results
4. **Start** — Click "Start Batch" to process all images
5. Each image will be saved as `{original_name}_wm_removed.png`

> The batch mode automatically targets the bottom-center area (common watermark position). For images with watermarks in different positions, process them individually.

### Switch language

Click **"EN"** in the toolbar to switch to English. Click **"中文"** to switch back to Chinese.

## Project Files

| File | Purpose |
|------|---------|
| `watermark_remover.py` | Main application (English name) |
| `去水印工具.py` | Wrapper that launches the app (Chinese name) |
| `启动去水印.bat` | Windows batch launcher (double-click to run) |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Git ignore rules |
| `LICENSE` | MIT License |

## How It Works

The application uses **ONNX Runtime** to run the **LaMa (Large Mask Inpainting)** model, a state-of-the-art image inpainting model developed by Samsung Research. It can fill large missing regions with coherent, realistic content.

Pipeline:
1. User draws a mask over the watermark
2. The image and mask are resized to 512×512 (LaMa input size)
3. ONNX Runtime runs inference using the CPU
4. The output is resized back to the original image dimensions
5. The result is displayed and available for saving

## Dependencies

- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — Deep learning inference engine
- [Pillow](https://python-pillow.org/) — Image I/O and format support
- [NumPy](https://numpy.org/) — Array operations
- [OpenCV](https://opencv.org/) — Image processing
- [Hugging Face Hub](https://huggingface.co/) — Model distribution
- [LaMa](https://github.com/saic-mdal/lama) — Inpainting model via OpenCV Model Zoo
- Tkinter — Desktop GUI (built into Python)

## License

MIT License — see [LICENSE](LICENSE) for details.
