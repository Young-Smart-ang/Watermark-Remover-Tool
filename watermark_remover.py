#!/usr/bin/env python3
"""
Watermark Remover Tool — AI-powered image inpainting
=====================================================
Multi-model watermark removal with batch processing, multi-step undo,
language support (中文/English), and PyInstaller packaging support.

Dependencies:
  pip install onnxruntime pillow numpy opencv-python-headless huggingface-hub
"""

from __future__ import annotations

import json
import os
import ssl
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

import onnxruntime as ort

# ═══════════════════════════════════════════════════════════════════════════════
# i18n — 国际化 / Internationalization
# ═══════════════════════════════════════════════════════════════════════════════

LANGUAGES: dict[str, dict[str, str]] = {
    "zh": {
        "app_title": "去水印工具 (AI 版)",
        "open": "📂 打开图片",
        "save": "💾 保存结果",
        "save_as": "保存结果",
        "batch": "📦 批量处理",
        "rect_tool": "矩形",
        "brush_tool": "画笔",
        "size_label": "大小:",
        "undo": "↩ 撤销",
        "undo_n": "↩ 撤销 ({n})",
        "clear_mask": "🗑 清除选区",
        "inpaint": "去水印",
        "toggle_result": "查看结果",
        "toggle_original": "查看原图",
        "model_label": "模型:",
        "language": "🌐 EN",
        "status_ready": "✅ AI 模型就绪 — 打开图片开始去水印",
        "status_downloading": "⏳ 正在连接下载服务器...",
        "status_download_progress": "⏳ 下载 AI 模型中 {mb:.0f}/{total_mb:.0f} MB ({pct}%)",
        "status_download_fail": "❌ 模型下载失败",
        "status_done": "✅ 模型下载完成",
        "status_open": "{name} ({w}×{h}) — 涂抹水印区域，点击「去水印」",
        "status_processing": "⏳ AI 处理中 ({model})...",
        "status_complete": "✅ AI 去水印完成！可保存结果或继续涂抹其他区域",
        "status_saved": "✅ 已保存: {name}",
        "status_failed": "❌ 处理失败",
        "status_batch_complete": "✅ 批量处理完成: {done}/{total} 张",
        "no_image": "还没有去水印的结果，请先点击「去水印」",
        "no_mask": "请先在要消除的水印上绘制选区",
        "model_busy": "AI 模型还在下载中，请稍等片刻...",
        "model_download": "下载 AI 模型",
        "model_lama": "LaMa AI (高质量)",
        "model_telea": "OpenCV Telea (快速)",
        "model_ns": "OpenCV NS (平滑)",
        "filetypes_images": "图片",
        "filetypes_all": "所有文件",
        "batch_title": "批量处理 — 去水印工具",
        "batch_add": "添加图片",
        "batch_remove": "移除",
        "batch_clear": "清空列表",
        "batch_output": "输出文件夹:",
        "batch_browse": "浏览...",
        "batch_start": "🚀 开始批量处理",
        "batch_processing": "正在处理: {name}",
        "batch_done": "✅ 完成",
        "batch_fail": "❌ 失败: {msg}",
        "batch_select_output": "选择输出文件夹",
        "batch_no_output": "请选择输出文件夹",
        "batch_no_images": "请添加要处理的图片",
        "batch_confirm_clear": "清空列表？",
        "no_model_ready": "模型未就绪",
        "telea_hint": "适合简单水印，无需下载模型",
        "ns_hint": "适合简单水印，平滑效果较好",
        "lama_hint": "适合复杂水印，需下载 88MB 模型",
        "save_png": "PNG",
        "save_jpg": "JPEG",
        "save_bmp": "BMP",
    },
    "en": {
        "app_title": "Watermark Remover (AI)",
        "open": "📂 Open Image",
        "save": "💾 Save Result",
        "save_as": "Save Result",
        "batch": "📦 Batch",
        "rect_tool": "Rect",
        "brush_tool": "Brush",
        "size_label": "Size:",
        "undo": "↩ Undo",
        "undo_n": "↩ Undo ({n})",
        "clear_mask": "🗑 Clear Mask",
        "inpaint": "Remove",
        "toggle_result": "Show Result",
        "toggle_original": "Show Original",
        "model_label": "Model:",
        "language": "🌐 中文",
        "status_ready": "✅ AI model ready — open an image to start",
        "status_downloading": "⏳ Connecting to download server...",
        "status_download_progress": "⏳ Downloading AI model {mb:.0f}/{total_mb:.0f} MB ({pct}%)",
        "status_download_fail": "❌ Model download failed",
        "status_done": "✅ Model download complete",
        "status_open": "{name} ({w}×{h}) — paint watermark area, click Remove",
        "status_processing": "⏳ AI processing ({model})...",
        "status_complete": "✅ Done! Save or continue editing",
        "status_saved": "✅ Saved: {name}",
        "status_failed": "❌ Processing failed",
        "status_batch_complete": "✅ Batch complete: {done}/{total} images",
        "no_image": "No result yet, click Remove first",
        "no_mask": "Please draw a selection on the watermark area first",
        "model_busy": "AI model is still downloading, please wait...",
        "model_download": "Download AI Model",
        "model_lama": "LaMa AI (High Quality)",
        "model_telea": "OpenCV Telea (Fast)",
        "model_ns": "OpenCV NS (Smooth)",
        "filetypes_images": "Images",
        "filetypes_all": "All Files",
        "batch_title": "Batch Process — Watermark Remover",
        "batch_add": "Add Images",
        "batch_remove": "Remove",
        "batch_clear": "Clear All",
        "batch_output": "Output folder:",
        "batch_browse": "Browse...",
        "batch_start": "🚀 Start Batch",
        "batch_processing": "Processing: {name}",
        "batch_done": "✅ Done",
        "batch_fail": "❌ Failed: {msg}",
        "batch_select_output": "Select Output Folder",
        "batch_no_output": "Please select an output folder",
        "batch_no_images": "Please add images to process",
        "batch_confirm_clear": "Clear the list?",
        "no_model_ready": "Model not ready",
        "telea_hint": "Good for simple watermarks, no download needed",
        "ns_hint": "Good for simple watermarks, smoother results",
        "lama_hint": "Best for complex watermarks, needs 88MB download",
        "save_png": "PNG",
        "save_jpg": "JPEG",
        "save_bmp": "BMP",
    },
}

_lang = "zh"  # current language


def _(key: str, **kwargs) -> str:
    """Translate key to current language, then format with kwargs."""
    text = LANGUAGES.get(_lang, LANGUAGES["zh"]).get(key, key)
    return text.format(**kwargs) if kwargs else text


def toggle_language():
    """Switch between zh and en."""
    global _lang
    _lang = "en" if _lang == "zh" else "zh"


# ═══════════════════════════════════════════════════════════════════════════════
# Model definitions
# ═══════════════════════════════════════════════════════════════════════════════

LAMA_URLS = [
    "https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx",
    "https://hf-mirror.com/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx",
]
LAMA_FILENAME = "inpainting_lama_2025jan.onnx"

MODEL_REGISTRY = {
    "lama": {
        "need_download": True,
        "params": {},
    },
    "telea": {
        "need_download": False,
        "params": {"radius": 3},
    },
    "ns": {
        "need_download": False,
        "params": {"radius": 3},
    },
}


def _get_model_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".models")


def _get_lama_path() -> str:
    return os.path.join(_get_model_dir(), LAMA_FILENAME)


def _download_lama(progress_cb=None) -> str:
    """Download LaMa ONNX model (88 MB). Returns model path."""
    dest = _get_lama_path()
    if os.path.exists(dest) and os.path.getsize(dest) > 1e6:
        return dest

    os.makedirs(_get_model_dir(), exist_ok=True)
    tmp = dest + ".tmp"
    for url in LAMA_URLS:
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; WatermarkRemover/2.0)"
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
        _("model_download") + "\n"
        "Failed to download AI model. Check your network or use a VPN.\n"
        f"Manual download: {LAMA_URLS[0]}"
    )


def inpaint_lama(image_rgb: np.ndarray, mask: np.ndarray,
                 model_path: str) -> np.ndarray:
    """Run LaMa model inference."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    h_orig, w_orig = image_rgb.shape[:2]

    img_512 = cv2.resize(image_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    mask_512 = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    img_norm = img_512.astype(np.float32) / 255.0
    img_blob = np.transpose(img_norm, (2, 0, 1))[np.newaxis, :, :, :]
    mask_blob = ((mask_512 > 0).astype(np.float32))[np.newaxis, np.newaxis, :, :]

    output = session.run(None, {"image": img_blob, "mask": mask_blob})[0]

    result_512 = np.transpose(output[0], (1, 2, 0)).clip(0, 255).astype(np.uint8)
    result = cv2.resize(result_512, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return result


def inpaint_telea(image_rgb: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """OpenCV Telea inpainting (fast, no model needed)."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask_bin = (mask > 0).astype(np.uint8)
    result = cv2.inpaint(image_rgb, mask_bin, radius, cv2.INPAINT_TELEA)
    return result


def inpaint_ns(image_rgb: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """OpenCV NS inpainting (smooth, no model needed)."""
    mask_bin = (mask > 0).astype(np.uint8)
    result = cv2.inpaint(image_rgb, mask_bin, radius, cv2.INPAINT_NS)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Undo Manager
# ═══════════════════════════════════════════════════════════════════════════════

class UndoManager:
    """Multi-step undo for mask + result states."""

    class Snapshot:
        __slots__ = ("mask", "has_result", "result")

        def __init__(self, mask: np.ndarray | None, has_result: bool,
                     result: np.ndarray | None):
            self.mask = mask.copy() if mask is not None else None
            self.has_result = has_result
            self.result = result.copy() if result is not None else None

    def __init__(self, max_steps: int = 50):
        self._steps: list[UndoManager.Snapshot] = []
        self._max = max_steps

    def save(self, mask: np.ndarray | None, has_result: bool = False,
             result: np.ndarray | None = None):
        self._steps.append(UndoManager.Snapshot(mask, has_result, result))
        if len(self._steps) > self._max:
            self._steps.pop(0)

    def undo(self) -> Snapshot | None:
        if not self._steps:
            return None
        return self._steps.pop()

    @property
    def count(self) -> int:
        return len(self._steps)

    def clear(self):
        self._steps.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Processing Dialog
# ═══════════════════════════════════════════════════════════════════════════════

class BatchDialog:
    """Dialog for batch watermark removal."""

    def __init__(self, parent: tk.Tk, model_ctx: dict):
        self.parent = parent
        self.model_ctx = model_ctx  # references: model_name, model_path, get_model_fn

        self.dialog = tk.Toplevel(parent)
        self.dialog.title(_("batch_title"))
        self.dialog.geometry("700x500")
        self.dialog.minsize(500, 350)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.files: list[str] = []
        self.output_dir: str = ""

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.dialog, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # ── File list ──
        ttk.Label(main, text=_("batch_add"), font=("", 11, "bold")).pack(anchor=tk.W)

        btn_row = ttk.Frame(main)
        btn_row.pack(fill=tk.X, pady=(4, 4))
        ttk.Button(btn_row, text=_("batch_add"), command=self._add_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text=_("batch_remove"), command=self._remove_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text=_("batch_clear"), command=self._clear_all).pack(side=tk.LEFT, padx=2)

        list_frame = ttk.Frame(main)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        vsb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED,
                                  yscrollcommand=vsb.set)
        vsb.config(command=self.listbox.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── Output folder ──
        out_row = ttk.Frame(main)
        out_row.pack(fill=tk.X, pady=4)
        ttk.Label(out_row, text=_("batch_output")).pack(side=tk.LEFT)
        self.out_var = tk.StringVar()
        out_entry = ttk.Entry(out_row, textvariable=self.out_var, state="readonly")
        out_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(out_row, text=_("batch_browse"), command=self._browse_output).pack(side=tk.RIGHT)

        # ── Batch model selector ──
        model_row = ttk.Frame(main)
        model_row.pack(fill=tk.X, pady=4)
        ttk.Label(model_row, text=_("model_label")).pack(side=tk.LEFT)
        self.batch_model = tk.StringVar(value=self.model_ctx["model_name"])
        batch_model_combo = ttk.Combobox(model_row, textvariable=self.batch_model,
                                         values=["lama", "telea", "ns"],
                                         state="readonly", width=30)
        batch_model_combo.pack(side=tk.LEFT, padx=4)

        # ── Progress area ──
        self.prog_frame = ttk.LabelFrame(main, text=_("batch_start"), padding=6)
        self.prog_frame.pack(fill=tk.X, pady=8)

        self.prog_bar = ttk.Progressbar(self.prog_frame, mode="determinate")
        self.prog_bar.pack(fill=tk.X)

        self.prog_status = ttk.Label(self.prog_frame, text="")
        self.prog_status.pack(anchor=tk.W)

        self.prog_list = tk.Text(self.prog_frame, height=5, state="disabled")
        self.prog_list.pack(fill=tk.X, pady=(4, 0))

        # ── Start button ──
        self.start_btn = ttk.Button(main, text=_("batch_start"),
                                    command=self._run_batch, style="Accent.TButton")
        self.start_btn.pack(pady=6)

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title=_("batch_add"),
            filetypes=[(_("filetypes_images"), "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                       (_("filetypes_all"), "*.*")])
        for p in paths:
            if p not in self.files:
                self.files.append(p)
                self.listbox.insert(tk.END, os.path.basename(p))

    def _remove_selected(self):
        sel = self.listbox.curselection()
        for i in reversed(sel):
            del self.files[i]
            self.listbox.delete(i)

    def _clear_all(self):
        if not self.files:
            return
        if not messagebox.askyesno("", _("batch_confirm_clear")):
            return
        self.files.clear()
        self.listbox.delete(0, tk.END)

    def _browse_output(self):
        d = filedialog.askdirectory(title=_("batch_select_output"))
        if d:
            self.output_dir = d
            self.out_var.set(d)

    def _log(self, msg: str):
        self.prog_list.config(state="normal")
        self.prog_list.insert(tk.END, msg + "\n")
        self.prog_list.see(tk.END)
        self.prog_list.config(state="disabled")
        self.dialog.update()

    def _run_batch(self):
        if not self.output_dir:
            messagebox.showwarning("", _("batch_no_output"))
            return
        if not self.files:
            messagebox.showwarning("", _("batch_no_images"))
            return

        self.start_btn.config(state="disabled")
        model_name = self.batch_model.get()
        total = len(self.files)
        done = 0

        for idx, fp in enumerate(self.files):
            basename = os.path.basename(fp)
            self.prog_status.config(text=_("batch_processing", name=basename))
            self.prog_bar["value"] = (idx / total) * 100
            self.dialog.update()

            try:
                pil = Image.open(fp).convert("RGB")
                img = np.array(pil)

                # Apply LaMa to the whole image (simple batch approach)
                ones = np.ones(img.shape[:2], dtype=np.uint8) * 255
                zeros = np.zeros(img.shape[:2], dtype=np.uint8)

                # For batch, we remove the whole selection area per image
                # User can define custom mask via the main UI per batch session
                # For now: use a centered bottom box as default watermark zone
                h, w = img.shape[:2]
                mask = zeros.copy()
                # Default: bottom 15% centered area — common watermark position
                y1 = int(h * 0.8)
                x1 = int(w * 0.15)
                x2 = int(w * 0.85)
                mask[y1:h, x1:x2] = 255

                result = None
                if model_name == "lama":
                    model_path = self.model_ctx.get("model_path")
                    if model_path and os.path.exists(model_path):
                        result = inpaint_lama(img, mask, model_path)
                elif model_name == "telea":
                    result = inpaint_telea(img, mask)
                elif model_name == "ns":
                    result = inpaint_ns(img, mask)

                if result is not None:
                    stem = Path(fp).stem
                    out_path = os.path.join(self.output_dir, f"{stem}_wm_removed.png")
                    Image.fromarray(result).save(out_path)
                    done += 1
                    self._log(f"✅ {basename}")
                else:
                    self._log(_("batch_fail", msg=_("no_model_ready")))
            except Exception as e:
                self._log(_("batch_fail", msg=str(e)))

        self.prog_bar["value"] = 100
        if done == total:
            self.prog_status.config(text=_("status_batch_complete", done=done, total=total))
        else:
            self.prog_status.config(text=f"✅ {done}/{total} done, {total - done} failed")
        self.start_btn.config(state="normal")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_RECT = "rect"
TOOL_BRUSH = "brush"
MODE_ORIGINAL = "original"
MODE_RESULT = "result"


class WatermarkRemover:
    """Main application class."""

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(_("app_title"))
        root.geometry("960x720")
        root.minsize(640, 480)

        # ── Image state ──
        self.original: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.result: np.ndarray | None = None
        self.filepath: str | None = None
        self.canvas_img_id = None

        # ── Model ──
        self._model_name = "lama"
        self._model_path: str | None = None
        self._model_ready = False

        # ── Display ──
        self._display_mode = MODE_ORIGINAL

        # ── Drawing ──
        self.tool = TOOL_RECT
        self.brush_size = 10
        self._start_x: int | None = None
        self._start_y: int | None = None
        self._last_x: int | None = None
        self._last_y: int | None = None
        self._rect_preview: int | None = None
        self._strokes: list = []
        self._current_stroke: list | None = None

        # ── Zoom ──
        self.scale = 1.0

        # ── Undo ──
        self.undo_mgr = UndoManager(max_steps=50)
        # We also keep _strokes for replay; undo saves mask snapshots.
        self._use_state_undo = True  # new approach: save mask before each change

        # ── Model download state ──
        self._dl_done: threading.Event | None = None
        self._dl_error: str | None = None
        self._dl_current = 0
        self._dl_total = 0
        self._dl_progress: ttk.Progressbar | None = None

        self.setup_ui()
        self._init_model()

    # ──────────────────────────────────────────────────────────────────────────
    # Model management
    # ──────────────────────────────────────────────────────────────────────────

    def _init_model(self):
        """Prepare model — download LaMa if needed, else just mark ready."""
        if self._model_name == "lama":
            self._model_path = _get_lama_path()
            if os.path.exists(self._model_path) and os.path.getsize(self._model_path) > 1e6:
                self._model_ready = True
                self.status.config(text=_("status_ready"))
                return
            self._start_lama_download()
        else:
            # OpenCV methods don't need a model
            self._model_ready = True
            self.status.config(text=_("status_ready"))

    def _start_lama_download(self):
        self._dl_done = threading.Event()
        self._dl_error = None
        self._dl_current = 0
        self._dl_total = 0

        self._dl_progress = ttk.Progressbar(self.root, mode="determinate", length=200)
        self._dl_progress.pack(before=self.status, fill=tk.X, padx=10, pady=(0, 2))

        def on_progress(curr, total):
            self._dl_current = curr
            self._dl_total = total

        def load():
            try:
                path = _download_lama(progress_cb=on_progress)
                self._model_path = path
                self._model_ready = True
            except Exception as e:
                self._dl_error = str(e)
            finally:
                if self._dl_done:
                    self._dl_done.set()

        threading.Thread(target=load, daemon=True).start()
        self._poll_dl()

    def _poll_dl(self):
        if self._dl_done is None:
            return
        if self._dl_done.is_set():
            if self._dl_progress:
                self._dl_progress.destroy()
                self._dl_progress = None
            if self._dl_error:
                self.status.config(text=_("status_download_fail"))
                messagebox.showerror(_("model_download"), self._dl_error)
            else:
                self.status.config(text=_("status_done"))
                # Re-trigger init
                self._model_ready = True
                self.status.config(text=_("status_ready"))
            return

        if self._dl_total > 0:
            pct = int(self._dl_current / self._dl_total * 100)
            mb = self._dl_current / (1024 * 1024)
            total_mb = self._dl_total / (1024 * 1024)
            self._dl_progress.configure(value=pct)
            self.status.config(text=_("status_download_progress",
                                      mb=mb, total_mb=total_mb, pct=pct))
        else:
            self.status.config(text=_("status_downloading"))

        self.root.after(200, self._poll_dl)

    def _switch_model(self, name: str):
        if name == self._model_name:
            return
        self._model_name = name
        hint_map = {"lama": "lama_hint", "telea": "telea_hint", "ns": "ns_hint"}
        hint_text = _(hint_map.get(name, "lama_hint"))
        self.status.config(text=f"🔄 {hint_text}")

        if name == "lama":
            self._model_path = _get_lama_path()
            if os.path.exists(self._model_path) and os.path.getsize(self._model_path) > 1e6:
                self._model_ready = True
                self.status.config(text=_("status_ready"))
            else:
                self._model_ready = False
                self._start_lama_download()
        else:
            self._model_ready = True
            self.status.config(text=_("status_ready"))

    # ──────────────────────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────────────────────

    def setup_ui(self):
        style = ttk.Style()
        style.configure("Accent.TButton", font=("", 10, "bold"), foreground="#2e7d32")
        style.configure("Toggle.TButton", font=("", 9))

        tb = ttk.Frame(self.root, padding=6)
        tb.pack(fill=tk.X)

        # ── File operations ──
        ttk.Button(tb, text=_("open"), command=self.open_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text=_("save"), command=self.save_result).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text=_("batch"), command=self.open_batch).pack(side=tk.LEFT, padx=2)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # ── Tools ──
        self.tool_var = tk.StringVar(value=TOOL_RECT)
        ttk.Radiobutton(tb, text=_("rect_tool"), variable=self.tool_var, value=TOOL_RECT,
                        command=self._on_tool_change).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(tb, text=_("brush_tool"), variable=self.tool_var, value=TOOL_BRUSH,
                        command=self._on_tool_change).pack(side=tk.LEFT, padx=2)

        ttk.Label(tb, text=_("size_label")).pack(side=tk.LEFT, padx=(8, 2))
        self.size_spin = ttk.Spinbox(tb, from_=3, to=80, width=4, command=self._on_size_change)
        self.size_spin.set(10)
        self.size_spin.pack(side=tk.LEFT)
        self.size_spin.bind("<KeyRelease>", self._on_size_change)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # ── Actions ──
        self.undo_btn = ttk.Button(tb, text=_("undo"), command=self.undo)
        self.undo_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text=_("clear_mask"), command=self.clear_mask).pack(side=tk.LEFT, padx=2)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # ── Model selector ──
        ttk.Label(tb, text=_("model_label")).pack(side=tk.LEFT, padx=(0, 2))
        self.model_combo = ttk.Combobox(tb, values=["lama", "telea", "ns"],
                                        state="readonly", width=22)
        self.model_combo.set("lama")
        self.model_combo.pack(side=tk.LEFT, padx=2)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # ── Process & Toggle ──
        ttk.Button(tb, text=_("inpaint"), command=self.inpaint,
                   style="Accent.TButton").pack(side=tk.LEFT, padx=4)
        self.toggle_btn = ttk.Button(tb, text=_("toggle_result"), command=self.toggle_view,
                                     style="Toggle.TButton", state=tk.DISABLED)
        self.toggle_btn.pack(side=tk.LEFT, padx=4)

        # ── Language toggle ──
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        self.lang_btn = ttk.Button(tb, text=_("language"), command=self._toggle_lang)
        self.lang_btn.pack(side=tk.LEFT, padx=4)

        # ── Canvas ──
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

        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-O>", lambda e: self.open_image())
        self.root.bind("<Control-s>", lambda e: self.save_result())
        self.root.bind("<Control-S>", lambda e: self.save_result())
        self.root.bind("<Control-r>", lambda e: self.inpaint())
        self.root.bind("<Control-R>", lambda e: self.inpaint())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-Z>", lambda e: self.undo())

        # ── Status bar ──
        self.status = ttk.Label(self.root, text=_("status_downloading"),
                                relief=tk.SUNKEN, anchor=tk.W, padding=4)
        self.status.pack(fill=tk.X)

    def _on_tool_change(self):
        self.tool = self.tool_var.get()

    def _on_size_change(self, event=None):
        try:
            self.brush_size = max(1, int(self.size_spin.get()))
        except ValueError:
            pass

    def _on_model_change(self, event=None):
        name = self.model_combo.get()
        self._switch_model(name)

    def _toggle_lang(self):
        global _lang
        toggle_language()
        self.lang_btn.config(text=_("language"))
        self._rebuild_ui_text()

    def _rebuild_ui_text(self):
        """Update all UI text after language switch."""
        self.root.title(_("app_title"))
        # Rebuild toolbar and status
        for w in self.root.winfo_children():
            if isinstance(w, ttk.Frame):
                for child in w.winfo_children():
                    if isinstance(child, ttk.Button):
                        txt = child.cget("text")
                        key_map = {
                            "📂 ": "open", "💾 ": "save", "📦 ": "batch",
                            "↩ ": "undo", "↩ 撤销": "undo", "↩ Undo": "undo",
                            "🗑 ": "clear_mask", "✨ ": "inpaint",
                            "👁 ": "toggle_result", "🌐 ": "language",
                        }
                        for prefix, key in key_map.items():
                            if txt.startswith(prefix):
                                try:
                                    child.config(text=_(key))
                                except Exception:
                                    pass
                                break
        # Update status
        if self._model_ready:
            self.status.config(text=_("status_ready"))
        self.undo_btn.config(text=_("undo_n", n=self.undo_mgr.count) if self.undo_mgr.count else _("undo"))

    # ──────────────────────────────────────────────────────────────────────────
    # Image I/O
    # ──────────────────────────────────────────────────────────────────────────

    def open_image(self):
        path = filedialog.askopenfilename(
            title=_("open"),
            filetypes=[(_("filetypes_images"), "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                       (_("filetypes_all"), "*.*")])
        if not path:
            return
        self.filepath = path
        pil = Image.open(path).convert("RGB")
        self.original = np.array(pil)
        h, w = self.original.shape[:2]
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self.result = None
        self._strokes.clear()
        self.undo_mgr.clear()
        self._display_mode = MODE_ORIGINAL
        self.toggle_btn.config(state=tk.DISABLED, text=_("toggle_result"))
        self.scale = 1.0
        self._fit_to_canvas()
        self._render()
        self.status.config(text=_("status_open", name=os.path.basename(path), w=w, h=h))
        self.undo_btn.config(text=_("undo"))

    def save_result(self):
        if self.result is None:
            messagebox.showinfo("", _("no_image"))
            return
        path = filedialog.asksaveasfilename(
            title=_("save_as"), defaultextension=".png",
            filetypes=[(_("save_png"), "*.png"), (_("save_jpg"), "*.jpg"),
                       (_("save_bmp"), "*.bmp"), (_("filetypes_all"), "*.*")])
        if not path:
            return
        Image.fromarray(self.result).save(path)
        self.status.config(text=_("status_saved", name=os.path.basename(path)))

    def open_batch(self):
        if not self._model_ready:
            messagebox.showinfo("", _("model_busy"))
            return
        ctx = {
            "model_name": self._model_name,
            "model_path": self._model_path if self._model_name == "lama" else None,
        }
        BatchDialog(self.root, ctx)

    # ──────────────────────────────────────────────────────────────────────────
    # Rendering
    # ──────────────────────────────────────────────────────────────────────────

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
                    ch[mask_bool] = (ch[mask_bool] * (1 - alpha)
                                     + 255 * alpha * (1 if c == 0 else 0.3)).astype(np.uint8)

        h, w = img.shape[:2]
        nw = max(1, int(w * self.scale))
        nh = max(1, int(h * self.scale))
        pil = Image.fromarray(img).resize((nw, nh), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(pil)
        self.canvas.config(scrollregion=(0, 0, nw, nh))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)

    def _img_coords(self, cx: float, cy: float) -> tuple[int, int]:
        canvas_x = self.canvas.canvasx(cx)
        canvas_y = self.canvas.canvasy(cy)
        return int(canvas_x / self.scale), int(canvas_y / self.scale)

    # ──────────────────────────────────────────────────────────────────────────
    # Mouse handlers
    # ──────────────────────────────────────────────────────────────────────────

    def _on_mouse_down(self, event):
        if self.original is None:
            return
        if self._display_mode == MODE_RESULT:
            self._display_mode = MODE_ORIGINAL
            self.toggle_btn.config(text=_("toggle_result"))

        # Save undo state before modification
        if self.mask is not None:
            self.undo_mgr.save(self.mask, self.result is not None, self.result)

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
        self._update_undo_btn()

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

    # ──────────────────────────────────────────────────────────────────────────
    # Undo / Clear
    # ──────────────────────────────────────────────────────────────────────────

    def undo(self):
        if self.original is None:
            return
        snap = self.undo_mgr.undo()
        if snap is not None:
            if snap.mask is not None:
                self.mask = snap.mask
            # Restore result state
            if snap.has_result and snap.result is not None:
                self.result = snap.result
                self._display_mode = MODE_RESULT
                self.toggle_btn.config(state=tk.NORMAL, text=_("toggle_original"))
            else:
                self.result = None
                self._display_mode = MODE_ORIGINAL
                self.toggle_btn.config(state=tk.DISABLED, text=_("toggle_result"))
            self._render()
            self._update_undo_btn()
            return
        # Fallback: stroke-based undo
        if not self._strokes:
            return
        self._strokes.pop()
        self._replay_strokes()
        self._render()

    def _update_undo_btn(self):
        n = self.undo_mgr.count
        if n:
            self.undo_btn.config(text=_("undo_n", n=n))
        else:
            self.undo_btn.config(text=_("undo"))

    def clear_mask(self):
        if self.original is None:
            return
        if self.mask is not None:
            self.undo_mgr.save(self.mask, self.result is not None, self.result)
        self.mask.fill(0)
        self._strokes.clear()
        self._render()
        self._update_undo_btn()

    # ──────────────────────────────────────────────────────────────────────────
    # AI Inpainting
    # ──────────────────────────────────────────────────────────────────────────

    def inpaint(self):
        if self.original is None:
            return
        if not self.mask.any():
            messagebox.showinfo("", _("no_mask"))
            return
        if not self._model_ready:
            messagebox.showinfo("", _("model_busy"))
            return

        # Save state before inpainting (for undo)
        self.undo_mgr.save(self.mask, self.result is not None, self.result)

        self.status.config(text=_("status_processing", model=self._model_name))
        self.root.update()

        try:
            # Show progress dialog on large images
            self._show_progress_dialog()

            def worker():
                try:
                    if self._model_name == "lama":
                        r = inpaint_lama(self.original, self.mask, self._model_path)
                    elif self._model_name == "telea":
                        r = inpaint_telea(self.original, self.mask)
                    elif self._model_name == "ns":
                        r = inpaint_ns(self.original, self.mask)
                    else:
                        r = inpaint_lama(self.original, self.mask, self._model_path)
                    self.root.after(0, self._on_inpaint_done, r)
                except Exception as e:
                    self.root.after(0, self._on_inpaint_error, str(e))

            threading.Thread(target=worker, daemon=True).start()
        except Exception as e:
            self._close_progress_dialog()
            messagebox.showerror("", _("status_failed") + f"\n{str(e)}")
            self.status.config(text=_("status_failed"))

    def _show_progress_dialog(self):
        self._prog_dialog = tk.Toplevel(self.root)
        self._prog_dialog.title("")
        self._prog_dialog.geometry("300x100")
        self._prog_dialog.transient(self.root)
        self._prog_dialog.grab_set()
        self._prog_dialog.resizable(False, False)

        # Center on parent
        self.root.update_idletasks()
        rx, ry = self.root.winfo_x(), self.root.winfo_y()
        rw, rh = self.root.winfo_width(), self.root.winfo_height()
        pw, ph = 300, 100
        self._prog_dialog.geometry(f"+{rx + (rw - pw) // 2}+{ry + (rh - ph) // 2}")

        ttk.Label(self._prog_dialog,
                  text=_("status_processing", model=self._model_name),
                  padding=10).pack()
        self._prog_bar = ttk.Progressbar(self._prog_dialog, mode="indeterminate")
        self._prog_bar.pack(fill=tk.X, padx=20, pady=10)
        self._prog_bar.start(10)
        self._prog_dialog.update()

    def _close_progress_dialog(self):
        if hasattr(self, "_prog_dialog") and self._prog_dialog:
            try:
                self._prog_bar.stop()
                self._prog_dialog.destroy()
            except Exception:
                pass
            self._prog_dialog = None

    def _on_inpaint_done(self, result: np.ndarray):
        self._close_progress_dialog()
        self.result = result
        self._display_mode = MODE_RESULT
        self.toggle_btn.config(state=tk.NORMAL, text=_("toggle_original"))
        self._render()
        self.status.config(text=_("status_complete"))

    def _on_inpaint_error(self, msg: str):
        self._close_progress_dialog()
        messagebox.showerror("", _("status_failed") + f"\n{msg}")
        self.status.config(text=_("status_failed"))

    # ──────────────────────────────────────────────────────────────────────────
    # View toggle
    # ──────────────────────────────────────────────────────────────────────────

    def toggle_view(self):
        if self._display_mode == MODE_RESULT:
            self._display_mode = MODE_ORIGINAL
            self.toggle_btn.config(text=_("toggle_result"))
        else:
            self._display_mode = MODE_RESULT
            self.toggle_btn.config(text=_("toggle_original"))
        self._render()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    WatermarkRemover(root)
    root.mainloop()


if __name__ == "__main__":
    main()
