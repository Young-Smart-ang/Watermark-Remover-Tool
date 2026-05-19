# Watermark Remover Tool

An AI-powered desktop application for removing watermarks from images using the **LaMa (Large Mask Inpainting)** model. Select unwanted regions with rectangle or brush tools, and the AI fills them with realistic content.

## Features

- **Rectangle Selection** — Drag to select watermark areas precisely
- **Brush Tool** — Freehand mask for irregular regions
- **Before/After Preview** — Toggle to compare original and result
- **Zoom & Pan** — Convenient for detailed editing
- **Undo Support** — Revert to previous state
- **GPU Acceleration** — Supports CUDA, DirectML, and CoreML via ONNX Runtime providers

## Screenshots

*(Coming soon)*

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/Young-Smart-ang/Watermark-Remover-Tool.git
cd Watermark-Remover-Tool

# Install dependencies
pip install -r requirements.txt
```

### Launch

```bash
python watermark_remover.py
```

Or double-click `去水印工具.bat` (Chinese) / `启动去水印.bat` on Windows.

## How to Use

1. **Open an image** — Click "打开图片" or press `Ctrl+O`
2. **Select watermark area** — Use rectangle tool (default) or brush tool
3. **Remove** — Click "去除水印" or press `Ctrl+R` to start inpainting
4. **Save** — Click "保存" or press `Ctrl+S` to export the result

### Tips

- For large watermarks, process in multiple small selections for better quality
- The brush tool works best for irregular shapes
- Use zoom (`Ctrl+滚轮`) for pixel-level precision

## How It Works

The application uses ONNX Runtime to run the **LaMa (Large Mask Inpainting)** model, a state-of-the-art image inpainting model that can handle large missing regions with coherent, realistic content. The model runs entirely locally — no data is sent to any server.

## Dependencies

- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — Model inference engine
- [LaMa](https://github.com/saic-mdal/lama) — Inpainting model by Samsung Research (via OpenCV model zoo)
- OpenCV — Image processing
- Pillow — Image I/O and format support
- Tkinter — Desktop GUI

## License

MIT License — see [LICENSE](LICENSE) for details.
