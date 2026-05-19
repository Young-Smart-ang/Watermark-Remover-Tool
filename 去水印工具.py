#!/usr/bin/env python3
"""
去水印工具 — 基于 AI (LaMa) 的图像修复
======================================
用矩形或画笔标记水印区域，LaMa 深度学习模型会自动生成
真实感内容填补，保持画面清晰。

依赖:
  pip install onnxruntime pillow numpy opencv-python-headless huggingface-hub
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import urllib.request
import ssl
import shutil

# LaMa inference
import onnxruntime as ort

TOOL_RECT = "rect"
TOOL_BRUSH = "brush"
MODE_ORIGINAL = "original"
MODE_RESULT = "result"

# Direct download URLs (no auth needed)
LAMA_URLS = [
    "https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx",
    "https://hf-mirror.com/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx",
]
LAMA_FILENAME = "inpainting_lama_2025jan.onnx"


def _get_model_path():
    """Return path to model file (downloaded if needed)."""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".models")
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, LAMA_FILENAME)


def _download_model(progress_cb=None):
    """Download LaMa ONNX model (88 MB) from best available source.

    progress_cb: optional callable(current_bytes, total_bytes)
    """
    dest = _get_model_path()
    if os.path.exists(dest) and os.path.getsize(dest) > 1e6:
        return dest

    # Try each URL in order
    tmp = dest + ".tmp"
    for url in LAMA_URLS:
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; WatermarkRemover/1.0)"
            })
            with urllib.request.urlopen(req, context=ctx, timeout=300) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_cb and total:
                            progress_cb(downloaded, total)
            os.replace(tmp, dest)
            return dest
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            continue
    raise RuntimeError(
        "无法下载 AI 模型。可能是网络问题，请尝试:\n"
        "1. 检查网络连接\n"
        "2. 使用 VPN 或代理\n"
        "3. 手动下载后放到 .models 文件夹:\n"
        f"   {LAMA_URLS[0]}"
    )


def _lama_inpaint(image_rgb, mask, model_path):
    """Run LaMa model inference on CPU."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    h_orig, w_orig = image_rgb.shape[:2]

    # Resize to 512x512
    img_512 = cv2.resize(image_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    mask_512 = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    # Preprocess: image to [0,1] float CHW, mask to {0,1} float
    img_norm = img_512.astype(np.float32) / 255.0
    img_blob = np.transpose(img_norm, (2, 0, 1))[np.newaxis, :, :, :]
    mask_blob = ((mask_512 > 0).astype(np.float32))[np.newaxis, np.newaxis, :, :]

    # Run
    output = session.run(None, {"image": img_blob, "mask": mask_blob})[0]

    # Postprocess: back to HWC uint8
    result_512 = np.transpose(output[0], (1, 2, 0)).clip(0, 255).astype(np.uint8)

    # Resize back to original size
    result = cv2.resize(result_512, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return result


class WatermarkRemover:
    def __init__(self, root):
        self.root = root
        self.root.title("去水印工具 (AI 版)")
        self.root.geometry("960x720")
        self.root.minsize(640, 480)

        # Image state
        self.original = None  # numpy (RGB)
        self.mask = None      # numpy (uint8), 255 = selected
        self.result = None    # numpy (RGB)
        self.filepath = None
        self.canvas_img_id = None

        # Model
        self._model_path = None
        self._model_ready = False

        # Display mode
        self._display_mode = MODE_ORIGINAL

        # Drawing state
        self.tool = TOOL_RECT
        self.brush_size = 10
        self._start_x = None
        self._start_y = None
        self._last_x = None
        self._last_y = None
        self._rect_preview = None
        self._strokes = []
        self._current_stroke = None

        # Zoom
        self.scale = 1.0

        self.setup_ui()
        self._init_model()

    def _init_model(self):
        """Download LaMa model (or use cache) with progress shown in UI."""
        self._model_path = _get_model_path()
        self._dl_done = threading.Event()
        self._dl_error = None
        self._dl_current = 0
        self._dl_total = 0

        # Already cached?
        if os.path.exists(self._model_path) and os.path.getsize(self._model_path) > 1e6:
            self._model_ready = True
            self.status.config(text="✅ AI 模型就绪 — 打开图片开始去水印")
            return

        # UI: progress bar + status
        self._dl_progress = ttk.Progressbar(self.root, mode="determinate", length=200)
        self._dl_progress.pack(before=self.status, fill=tk.X, padx=10, pady=(0, 2))

        def _on_progress(curr, total):
            self._dl_current = curr
            self._dl_total = total

        def _load():
            try:
                path = _download_model(progress_cb=_on_progress)
                self._model_path = path
                self._model_ready = True
            except Exception as e:
                self._dl_error = str(e)
            finally:
                self._dl_done.set()

        threading.Thread(target=_load, daemon=True).start()
        self._poll_dl()

    def _poll_dl(self):
        """Poll download progress from main thread (thread-safe)."""
        if self._dl_done.is_set():
            if self._dl_progress:
                self._dl_progress.destroy()
                self._dl_progress = None
            if self._dl_error:
                self.status.config(text=f"❌ 模型下载失败")
                messagebox.showerror("下载失败", self._dl_error)
            else:
                self.status.config(text="✅ AI 模型就绪 — 打开图片开始去水印")
            return

        # Update progress
        if self._dl_total > 0:
            pct = int(self._dl_current / self._dl_total * 100)
            mb = self._dl_current / (1024 * 1024)
            total_mb = self._dl_total / (1024 * 1024)
            self._dl_progress.configure(value=pct)
            self.status.config(text=f"⏳ 下载 AI 模型中 {mb:.0f}/{total_mb:.0f} MB ({pct}%)")
        else:
            self.status.config(text="⏳ 正在连接下载服务器...")

        self.root.after(200, self._poll_dl)

    # ── UI ─────────────────────────────────────────────────

    def setup_ui(self):
        style = ttk.Style()
        style.configure("Accent.TButton", font=("", 10, "bold"), foreground="#2e7d32")
        style.configure("Toggle.TButton", font=("", 9))

        tb = ttk.Frame(self.root, padding=6)
        tb.pack(fill=tk.X)

        ttk.Button(tb, text="📂 打开图片", command=self.open_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text="💾 保存结果", command=self.save_result).pack(side=tk.LEFT, padx=2)
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self.tool_var = tk.StringVar(value=TOOL_RECT)
        ttk.Radiobutton(tb, text="矩形", variable=self.tool_var, value=TOOL_RECT,
                        command=self._on_tool_change).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(tb, text="画笔", variable=self.tool_var, value=TOOL_BRUSH,
                        command=self._on_tool_change).pack(side=tk.LEFT, padx=2)

        ttk.Label(tb, text="大小:").pack(side=tk.LEFT, padx=(8, 2))
        self.size_spin = ttk.Spinbox(tb, from_=3, to=80, width=4, command=self._on_size_change)
        self.size_spin.set(10)
        self.size_spin.pack(side=tk.LEFT)
        self.size_spin.bind("<KeyRelease>", self._on_size_change)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(tb, text="↩ 撤销", command=self.undo).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text="🗑 清除选区", command=self.clear_mask).pack(side=tk.LEFT, padx=2)
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(tb, text="✨ 去水印!", command=self.inpaint, style="Accent.TButton").pack(side=tk.LEFT, padx=4)
        self.toggle_btn = ttk.Button(tb, text="👁 查看结果", command=self.toggle_view,
                                     style="Toggle.TButton", state=tk.DISABLED)
        self.toggle_btn.pack(side=tk.LEFT, padx=4)

        # Canvas
        cframe = ttk.Frame(self.root)
        cframe.pack(fill=tk.BOTH, expand=True)

        hsb = ttk.Scrollbar(cframe, orient=tk.HORIZONTAL)
        vsb = ttk.Scrollbar(cframe, orient=tk.VERTICAL)
        self.canvas = tk.Canvas(cframe, bg="#2b2b2b", cursor="cross",
                                xscrollcommand=hsb.set, yscrollcommand=vsb.set)
        hsb.config(command=self.canvas.xview)
        vsb.config(command=self.canvas.yview)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

        # Status bar
        self.status = ttk.Label(self.root, text="正在初始化...",
                                relief=tk.SUNKEN, anchor=tk.W, padding=4)
        self.status.pack(fill=tk.X)

    def _on_tool_change(self):
        self.tool = self.tool_var.get()

    def _on_size_change(self, event=None):
        try:
            self.brush_size = max(1, int(self.size_spin.get()))
        except ValueError:
            pass

    # ── Image IO ───────────────────────────────────────────

    def open_image(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                       ("所有文件", "*.*")])
        if not path:
            return
        self.filepath = path
        pil = Image.open(path).convert("RGB")
        self.original = np.array(pil)
        h, w = self.original.shape[:2]
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self.result = None
        self._strokes.clear()
        self._display_mode = MODE_ORIGINAL
        self.toggle_btn.config(state=tk.DISABLED, text="👁 查看结果")
        self.scale = 1.0
        self._fit_to_canvas()
        self._render()
        self.status.config(text=f"{os.path.basename(path)} ({w}×{h}) — 涂抹水印区域，点击「去水印」")

    def save_result(self):
        if self.result is None:
            messagebox.showinfo("提示", "还没有去水印的结果，请先点击「去水印」")
            return
        path = filedialog.asksaveasfilename(
            title="保存结果", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"),
                       ("BMP", "*.bmp"), ("所有文件", "*.*")])
        if not path:
            return
        Image.fromarray(self.result).save(path)
        self.status.config(text=f"✅ 已保存: {os.path.basename(path)}")

    # ── Rendering ──────────────────────────────────────────

    def _fit_to_canvas(self):
        if self.original is None:
            return
        cw = max(self.canvas.winfo_width(), 200)
        ch = max(self.canvas.winfo_height(), 200)
        h, w = self.original.shape[:2]
        self.scale = min((cw - 40) / w, (ch - 40) / h, 2.0)

    def _render(self):
        self.canvas.delete("all")
        if self.original is None:
            return

        if self._display_mode == MODE_RESULT and self.result is not None:
            img = self.result.copy()
        else:
            img = self.original.copy()
            if self.mask is not None and self.mask.any():
                alpha = 0.45
                mask_bool = self.mask > 0
                for c in range(3):
                    ch = img[:, :, c]
                    ch[mask_bool] = (ch[mask_bool] * (1 - alpha) + 255 * alpha * (1 if c == 0 else 0.3)).astype(np.uint8)

        h, w = img.shape[:2]
        nw = max(1, int(w * self.scale))
        nh = max(1, int(h * self.scale))
        pil = Image.fromarray(img).resize((nw, nh), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(pil)
        self.canvas.config(scrollregion=(0, 0, nw, nh))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)

    def _img_coords(self, cx, cy):
        canvas_x = self.canvas.canvasx(cx)
        canvas_y = self.canvas.canvasy(cy)
        return int(canvas_x / self.scale), int(canvas_y / self.scale)

    # ── Mouse handlers ────────────────────────────────────

    def _on_mouse_down(self, event):
        if self.original is None:
            return
        if self._display_mode == MODE_RESULT:
            self._display_mode = MODE_ORIGINAL
            self.toggle_btn.config(text="👁 查看结果")

        self._start_x = event.x
        self._start_y = event.y
        self._last_x = event.x
        self._last_y = event.y

        if self.tool == TOOL_RECT:
            self._rect_preview = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline="#ff4444", width=2, dash=(4, 2))
        elif self.tool == TOOL_BRUSH:
            ix, iy = self._img_coords(event.x, event.y)
            self._current_stroke = [(ix, iy)]
            cv2.circle(self.mask, (ix, iy), self.brush_size, 255, -1)
            self._render()

    def _on_mouse_move(self, event):
        if self.original is None or self._start_x is None:
            return

        if self.tool == TOOL_RECT:
            self.canvas.coords(self._rect_preview,
                               self._start_x, self._start_y, event.x, event.y)
        elif self.tool == TOOL_BRUSH:
            ix, iy = self._img_coords(event.x, event.y)
            lx, ly = self._img_coords(self._last_x, self._last_y)
            self._current_stroke.append((ix, iy))
            self._draw_line_on_mask(self.mask, lx, ly, ix, iy, self.brush_size)
            self._last_x = event.x
            self._last_y = event.y
            self._render()

    def _on_mouse_up(self, event):
        if self.original is None or self._start_x is None:
            return

        if self._rect_preview is not None:
            self.canvas.delete(self._rect_preview)
            self._rect_preview = None

        if self.tool == TOOL_RECT:
            x1, y1 = self._img_coords(self._start_x, self._start_y)
            x2, y2 = self._img_coords(event.x, event.y)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            self._strokes.append(("rect", x1, y1, x2, y2))
            self.mask[y1:y2 + 1, x1:x2 + 1] = 255
            self._render()
        elif self.tool == TOOL_BRUSH and self._current_stroke:
            self._strokes.append(self._current_stroke)
            self._current_stroke = None

        self._start_x = None
        self._start_y = None
        self._last_x = None
        self._last_y = None

    def _on_mousewheel(self, event):
        if self.original is None:
            return
        self.scale *= (1.1 ** (-1 if event.delta < 0 else 1))
        self.scale = max(0.1, min(self.scale, 10.0))
        self._render()

    @staticmethod
    def _draw_line_on_mask(mask, x0, y0, x1, y1, radius):
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        steps = max(dx, dy, 1)
        for i in range(steps + 1):
            t = i / steps
            cv2.circle(mask, (int(round(x0 + (x1 - x0) * t)),
                              int(round(y0 + (y1 - y0) * t))), radius, 255, -1)

    def _replay_strokes(self):
        self.mask.fill(0)
        for stroke in self._strokes:
            if stroke[0] == "rect":
                _, x1, y1, x2, y2 = stroke
                self.mask[y1:y2 + 1, x1:x2 + 1] = 255
            else:
                pts = stroke
                for i in range(len(pts) - 1):
                    self._draw_line_on_mask(self.mask, pts[i][0], pts[i][1],
                                            pts[i + 1][0], pts[i + 1][1], self.brush_size)
                if pts:
                    cv2.circle(self.mask, pts[-1][0], pts[-1][1], self.brush_size, 255, -1)

    def undo(self):
        if not self._strokes:
            return
        self._strokes.pop()
        self._replay_strokes()
        self._render()

    def clear_mask(self):
        self.mask.fill(0)
        self._strokes.clear()
        self._render()

    # ── AI Inpainting ──────────────────────────────────────

    def inpaint(self):
        if self.original is None:
            return
        if not self.mask.any():
            messagebox.showinfo("提示", "请先在要消除的水印上绘制选区")
            return
        if not self._model_ready:
            messagebox.showinfo("提示", "AI 模型还在下载中，请稍等片刻...")
            return

        self.status.config(text="⏳ AI 处理中 (LaMa 模型推理)...")
        self.root.update()

        try:
            result_rgb = _lama_inpaint(self.original, self.mask, self._model_path)
        except Exception as e:
            messagebox.showerror("处理出错", str(e))
            self.status.config(text="❌ 处理失败")
            return

        self.result = result_rgb
        self._display_mode = MODE_RESULT
        self.toggle_btn.config(state=tk.NORMAL, text="👁 查看原图")
        self._render()
        self.status.config(text="✅ AI 去水印完成！可保存结果或继续涂抹其他区域")

    # ── View toggle ────────────────────────────────────────

    def toggle_view(self):
        if self._display_mode == MODE_RESULT:
            self._display_mode = MODE_ORIGINAL
            self.toggle_btn.config(text="👁 查看结果")
        else:
            self._display_mode = MODE_RESULT
            self.toggle_btn.config(text="👁 查看原图")
        self._render()


def main():
    root = tk.Tk()
    WatermarkRemover(root)
    root.mainloop()


if __name__ == "__main__":
    main()
