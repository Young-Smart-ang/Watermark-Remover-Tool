#!/usr/bin/env python3
"""
去水印工具 — 基于 AI (LaMa) 的图像修复
======================================
用矩形或画笔标记水印区域，LaMa 深度学习模型会自动生成
真实感内容填补，保持画面清晰。

支持中英文切换和批量处理。

依赖:
  pip install onnxruntime pillow numpy opencv-python-headless huggingface-hub
"""

from __future__ import annotations

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
# i18n
# ═══════════════════════════════════════════════════════════════════════════════

LANGUAGES = {
    "zh": {
        "app_title": "去水印工具 (AI 版)",
        "open": "打开图片",
        "save": "保存结果",
        "batch": "批量处理",
        "rect": "矩形",
        "brush": "画笔",
        "size": "大小:",
        "undo": "撤销",
        "clear": "清除选区",
        "inpaint": "去水印",
        "view_result": "查看结果",
        "view_orig": "查看原图",
        "lang_btn": "EN",
        "status_ready": "AI 模型就绪 — 打开图片开始去水印",
        "status_dl": "正在连接下载服务器...",
        "status_dl_prog": "下载 AI 模型中 {mb:.0f}/{total_mb:.0f} MB ({pct}%)",
        "status_dl_fail": "模型下载失败",
        "status_open": "{name} ({w}x{h}) — 涂抹水印区域，点下去水印",
        "status_work": "AI 处理中...",
        "status_done": "去水印完成！可保存或继续涂抹",
        "status_saved": "已保存: {name}",
        "status_fail": "处理失败",
        "no_result": "还没有结果，请先去水印",
        "no_mask": "请先在要消除的水印上绘制选区",
        "model_busy": "AI 模型还在下载中，请稍等...",
        "dl_title": "下载 AI 模型",
        "batch_title": "批量处理",
        "batch_add": "添加图片",
        "batch_del": "移除",
        "batch_clear": "清空",
        "batch_out": "输出文件夹:",
        "batch_browse": "浏览...",
        "batch_start": "开始批量处理",
        "batch_proc": "处理中: {name}",
        "batch_sel_out": "选择输出文件夹",
        "batch_no_out": "请选择输出文件夹",
        "batch_no_img": "请添加要处理的图片",
        "batch_ask_clear": "清空列表？",
        "batch_done": "批量处理完成: {done}/{total} 张",
    },
    "en": {
        "app_title": "Watermark Remover (AI)",
        "open": "Open Image",
        "save": "Save Result",
        "batch": "Batch",
        "rect": "Rect",
        "brush": "Brush",
        "size": "Size:",
        "undo": "Undo",
        "clear": "Clear Mask",
        "inpaint": "Remove",
        "view_result": "Show Result",
        "view_orig": "Show Original",
        "lang_btn": "中文",
        "status_ready": "AI model ready — open an image to start",
        "status_dl": "Connecting to server...",
        "status_dl_prog": "Downloading model {mb:.0f}/{total_mb:.0f} MB ({pct}%)",
        "status_dl_fail": "Model download failed",
        "status_open": "{name} ({w}x{h}) — paint watermark, click Remove",
        "status_work": "AI processing...",
        "status_done": "Done! Save or continue editing",
        "status_saved": "Saved: {name}",
        "status_fail": "Processing failed",
        "no_result": "No result yet, click Remove first",
        "no_mask": "Please draw a selection on the watermark area first",
        "model_busy": "Model is still downloading, please wait...",
        "dl_title": "Download AI Model",
        "batch_title": "Batch Process",
        "batch_add": "Add Images",
        "batch_del": "Remove",
        "batch_clear": "Clear All",
        "batch_out": "Output folder:",
        "batch_browse": "Browse...",
        "batch_start": "Start Batch",
        "batch_proc": "Processing: {name}",
        "batch_sel_out": "Select Output Folder",
        "batch_no_out": "Please select an output folder",
        "batch_no_img": "Please add images to process",
        "batch_ask_clear": "Clear the list?",
        "batch_done": "Batch complete: {done}/{total} images",
    },
}

_lang = "zh"


def _(key: str, **kw) -> str:
    t = LANGUAGES.get(_lang, LANGUAGES["zh"]).get(key, key)
    return t.format(**kw) if kw else t


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════

LAMA_URLS = [
    "https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx",
    "https://hf-mirror.com/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx",
]
LAMA_FILENAME = "inpainting_lama_2025jan.onnx"


def _get_model_path():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".models")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, LAMA_FILENAME)


def _download_model(progress_cb=None):
    dest = _get_model_path()
    if os.path.exists(dest) and os.path.getsize(dest) > 1e6:
        return dest

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
        f"3. 手动下载后放到 .models 文件夹:\n   {LAMA_URLS[0]}"
    )


def _lama_inpaint(image_rgb, mask, model_path):
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


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Dialog
# ═══════════════════════════════════════════════════════════════════════════════

class BatchDialog:
    """Dialog for batch watermark removal."""

    def __init__(self, parent: tk.Tk, model_path: str):
        self.parent = parent
        self.model_path = model_path

        self.dialog = tk.Toplevel(parent)
        self.dialog.title(_("batch_title"))
        self.dialog.geometry("650x460")
        self.dialog.minsize(480, 300)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.files: list[str] = []
        self.output_dir: str = ""
        self._build_ui()

    def _build_ui(self):
        mf = ttk.Frame(self.dialog, padding=10)
        mf.pack(fill=tk.BOTH, expand=True)

        ttk.Label(mf, text=_("batch_title"), font=("", 11, "bold")).pack(anchor=tk.W)

        br = ttk.Frame(mf)
        br.pack(fill=tk.X, pady=4)
        ttk.Button(br, text=_("batch_add"), command=self._add).pack(side=tk.LEFT, padx=2)
        ttk.Button(br, text=_("batch_del"), command=self._remove).pack(side=tk.LEFT, padx=2)
        ttk.Button(br, text=_("batch_clear"), command=self._clear).pack(side=tk.LEFT, padx=2)

        lf = ttk.Frame(mf)
        lf.pack(fill=tk.BOTH, expand=True, pady=4)
        sb = ttk.Scrollbar(lf, orient=tk.VERTICAL)
        self.lb = tk.Listbox(lf, selectmode=tk.EXTENDED, yscrollcommand=sb.set)
        sb.config(command=self.lb.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        orow = ttk.Frame(mf)
        orow.pack(fill=tk.X, pady=4)
        ttk.Label(orow, text=_("batch_out")).pack(side=tk.LEFT)
        self.out_var = tk.StringVar()
        e = ttk.Entry(orow, textvariable=self.out_var, state="readonly")
        e.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(orow, text=_("batch_browse"), command=self._browse).pack(side=tk.RIGHT)

        # Progress
        pf = ttk.LabelFrame(mf, text="", padding=6)
        pf.pack(fill=tk.X, pady=6)
        self.pbar = ttk.Progressbar(pf, mode="determinate")
        self.pbar.pack(fill=tk.X)
        self.pstat = ttk.Label(pf, text="")
        self.pstat.pack(anchor=tk.W)
        self.plog = tk.Text(pf, height=4, state="disabled")
        self.plog.pack(fill=tk.X, pady=(4, 0))

        self.start_btn = ttk.Button(mf, text=_("batch_start"), command=self._run)
        self.start_btn.pack(pady=4)

    def _add(self):
        paths = filedialog.askopenfilenames(
            title=_("batch_add"),
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                       ("All", "*.*")])
        for p in paths:
            if p not in self.files:
                self.files.append(p)
                self.lb.insert(tk.END, os.path.basename(p))

    def _remove(self):
        sel = self.lb.curselection()
        for i in reversed(sel):
            del self.files[i]
            self.lb.delete(i)

    def _clear(self):
        if not self.files:
            return
        if not messagebox.askyesno("", _("batch_ask_clear")):
            return
        self.files.clear()
        self.lb.delete(0, tk.END)

    def _browse(self):
        d = filedialog.askdirectory(title=_("batch_sel_out"))
        if d:
            self.output_dir = d
            self.out_var.set(d)

    def _log(self, msg):
        self.plog.config(state="normal")
        self.plog.insert(tk.END, msg + "\n")
        self.plog.see(tk.END)
        self.plog.config(state="disabled")
        self.dialog.update()

    def _run(self):
        if not self.output_dir:
            messagebox.showwarning("", _("batch_no_out"))
            return
        if not self.files:
            messagebox.showwarning("", _("batch_no_img"))
            return

        self.start_btn.config(state="disabled")
        total = len(self.files)
        done = 0

        for i, fp in enumerate(self.files):
            name = os.path.basename(fp)
            self.pstat.config(text=_("batch_proc", name=name))
            self.pbar["value"] = (i / total) * 100
            self.dialog.update()

            try:
                img = np.array(Image.open(fp).convert("RGB"))
                h, w = img.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                # Default: bottom 15% centered — common watermark position
                y1 = int(h * 0.8)
                x1 = int(w * 0.15)
                x2 = int(w * 0.85)
                mask[y1:h, x1:x2] = 255

                result = _lama_inpaint(img, mask, self.model_path)
                out = os.path.join(self.output_dir, Path(fp).stem + "_wm_removed.png")
                Image.fromarray(result).save(out)
                done += 1
                self._log(f"OK  {name}")
            except Exception as e:
                self._log(f"FAIL {name}: {e}")

        self.pbar["value"] = 100
        self.pstat.config(text=_("batch_done", done=done, total=total))
        self.start_btn.config(state="normal")


# ═══════════════════════════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_RECT = "rect"
TOOL_BRUSH = "brush"
MODE_ORIGINAL = "original"
MODE_RESULT = "result"


class WatermarkRemover:
    def __init__(self, root):
        self.root = root
        root.title(_("app_title"))
        root.geometry("960x720")
        root.minsize(640, 480)

        self.original = None
        self.mask = None
        self.result = None
        self.filepath = None

        self._model_path = None
        self._model_ready = False
        self._display_mode = MODE_ORIGINAL

        self.tool = TOOL_RECT
        self.brush_size = 10
        self._start_x = None
        self._start_y = None
        self._last_x = None
        self._last_y = None
        self._rect_preview = None
        self._strokes = []
        self._current_stroke = None

        self.scale = 1.0

        self._dl_done = None
        self._dl_error = None
        self._dl_current = 0
        self._dl_total = 0
        self._dl_progress = None

        self.setup_ui()
        self._init_model()

    # ── Model ────────────────────────────────────────────────

    def _init_model(self):
        self._model_path = _get_model_path()
        if os.path.exists(self._model_path) and os.path.getsize(self._model_path) > 1e6:
            self._model_ready = True
            self.status.config(text=_("status_ready"))
            return

        self._dl_done = threading.Event()
        self._dl_error = None
        self._dl_current = 0
        self._dl_total = 0
        self._dl_progress = ttk.Progressbar(self.root, mode="determinate", length=200)
        self._dl_progress.pack(before=self.status, fill=tk.X, padx=10, pady=(0, 2))

        def _cb(curr, total):
            self._dl_current = curr
            self._dl_total = total

        def _load():
            try:
                self._model_path = _download_model(progress_cb=_cb)
                self._model_ready = True
            except Exception as e:
                self._dl_error = str(e)
            finally:
                self._dl_done.set()

        threading.Thread(target=_load, daemon=True).start()
        self._poll_dl()

    def _poll_dl(self):
        if not self._dl_done or not self._dl_done.is_set():
            if self._dl_total > 0:
                p = int(self._dl_current / self._dl_total * 100)
                mb = self._dl_current / (1024 * 1024)
                tm = self._dl_total / (1024 * 1024)
                self._dl_progress.configure(value=p)
                self.status.config(text=_("status_dl_prog", mb=mb, total_mb=tm, pct=p))
            else:
                self.status.config(text=_("status_dl"))
            self.root.after(200, self._poll_dl)
            return

        if self._dl_progress:
            self._dl_progress.destroy()
            self._dl_progress = None
        if self._dl_error:
            self.status.config(text=_("status_dl_fail"))
            messagebox.showerror(_("dl_title"), self._dl_error)
        else:
            self._model_ready = True
            self.status.config(text=_("status_ready"))

    # ── UI ──────────────────────────────────────────────────

    def setup_ui(self):
        style = ttk.Style()
        style.configure("Accent.TButton", font=("", 10, "bold"))

        tb = ttk.Frame(self.root, padding=6)
        tb.pack(fill=tk.X)

        # Buttons stored as attributes for i18n rebuild
        self._btn_open = ttk.Button(tb, text=_("open"), command=self.open_image)
        self._btn_open.pack(side=tk.LEFT, padx=2)
        self._btn_save = ttk.Button(tb, text=_("save"), command=self.save_result)
        self._btn_save.pack(side=tk.LEFT, padx=2)
        self._btn_batch = ttk.Button(tb, text=_("batch"), command=self.open_batch)
        self._btn_batch.pack(side=tk.LEFT, padx=2)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self.tool_var = tk.StringVar(value=TOOL_RECT)
        ttk.Radiobutton(tb, text=_("rect"), variable=self.tool_var, value=TOOL_RECT,
                        command=self._on_tool_change).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(tb, text=_("brush"), variable=self.tool_var, value=TOOL_BRUSH,
                        command=self._on_tool_change).pack(side=tk.LEFT, padx=2)

        ttk.Label(tb, text=_("size")).pack(side=tk.LEFT, padx=(8, 2))
        self.size_spin = ttk.Spinbox(tb, from_=3, to=80, width=4, command=self._on_size_change)
        self.size_spin.set(10)
        self.size_spin.pack(side=tk.LEFT)
        self.size_spin.bind("<KeyRelease>", self._on_size_change)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self._btn_undo = ttk.Button(tb, text=_("undo"), command=self.undo)
        self._btn_undo.pack(side=tk.LEFT, padx=2)
        self._btn_clear = ttk.Button(tb, text=_("clear"), command=self.clear_mask)
        self._btn_clear.pack(side=tk.LEFT, padx=2)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self._btn_inpaint = ttk.Button(tb, text=_("inpaint"), command=self.inpaint,
                                       style="Accent.TButton")
        self._btn_inpaint.pack(side=tk.LEFT, padx=4)
        self._btn_toggle = ttk.Button(tb, text=_("view_result"), command=self.toggle_view,
                                      state=tk.DISABLED)
        self._btn_toggle.pack(side=tk.LEFT, padx=4)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        self._btn_lang = ttk.Button(tb, text=_("lang_btn"), command=self._toggle_lang)
        self._btn_lang.pack(side=tk.LEFT, padx=4)

        # Canvas
        cf = ttk.Frame(self.root)
        cf.pack(fill=tk.BOTH, expand=True)

        hsb = ttk.Scrollbar(cf, orient=tk.HORIZONTAL)
        vsb = ttk.Scrollbar(cf, orient=tk.VERTICAL)
        self.canvas = tk.Canvas(cf, bg="#2b2b2b", cursor="cross",
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

    def _toggle_lang(self):
        global _lang
        _lang = "en" if _lang == "zh" else "zh"
        self._rebuild_text()

    def _rebuild_text(self):
        self.root.title(_("app_title"))
        self._btn_open.config(text=_("open"))
        self._btn_save.config(text=_("save"))
        self._btn_batch.config(text=_("batch"))
        self._btn_undo.config(text=_("undo"))
        self._btn_clear.config(text=_("clear"))
        self._btn_inpaint.config(text=_("inpaint"))
        self._btn_lang.config(text=_("lang_btn"))
        if self._display_mode == MODE_RESULT and self.result is not None:
            self._btn_toggle.config(text=_("view_orig"))
        else:
            self._btn_toggle.config(text=_("view_result"))
        if self._model_ready:
            self.status.config(text=_("status_ready"))

    # ── Image I/O ───────────────────────────────────────────

    def open_image(self):
        path = filedialog.askopenfilename(
            title=_("open"),
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                       ("All", "*.*")])
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
        self._btn_toggle.config(state=tk.DISABLED, text=_("view_result"))
        self.scale = 1.0
        self._fit_to_canvas()
        self._render()
        self.status.config(text=_("status_open", name=os.path.basename(path), w=w, h=h))

    def save_result(self):
        if self.result is None:
            messagebox.showinfo("", _("no_result"))
            return
        path = filedialog.asksaveasfilename(
            title=_("save"), defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"),
                       ("BMP", "*.bmp"), ("All", "*.*")])
        if not path:
            return
        Image.fromarray(self.result).save(path)
        self.status.config(text=_("status_saved", name=os.path.basename(path)))

    def open_batch(self):
        if not self._model_ready:
            messagebox.showinfo("", _("model_busy"))
            return
        BatchDialog(self.root, self._model_path)

    # ── Render ──────────────────────────────────────────────

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
                mb = self.mask > 0
                for c in range(3):
                    ch = img[:, :, c]
                    ch[mb] = (ch[mb] * (1 - alpha) + 255 * alpha * (1 if c == 0 else 0.3)).astype(np.uint8)

        h, w = img.shape[:2]
        nw = max(1, int(w * self.scale))
        nh = max(1, int(h * self.scale))
        pil = Image.fromarray(img).resize((nw, nh), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(pil)
        self.canvas.config(scrollregion=(0, 0, nw, nh))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)

    def _img_coords(self, cx, cy):
        return int(self.canvas.canvasx(cx) / self.scale), int(self.canvas.canvasy(cy) / self.scale)

    # ── Mouse ───────────────────────────────────────────────

    def _on_mouse_down(self, event):
        if self.original is None:
            return
        if self._display_mode == MODE_RESULT:
            self._display_mode = MODE_ORIGINAL
            self._btn_toggle.config(text=_("view_result"))

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
        self.scale *= 1.1 ** (-1 if event.delta < 0 else 1)
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
        for s in self._strokes:
            if s[0] == "rect":
                _, x1, y1, x2, y2 = s
                self.mask[y1:y2 + 1, x1:x2 + 1] = 255
            else:
                pts = s
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

    # ── Inpainting ──────────────────────────────────────────

    def inpaint(self):
        if self.original is None:
            return
        if not self.mask.any():
            messagebox.showinfo("", _("no_mask"))
            return
        if not self._model_ready:
            messagebox.showinfo("", _("model_busy"))
            return

        self.status.config(text=_("status_work"))
        self.root.update()

        try:
            result_rgb = _lama_inpaint(self.original, self.mask, self._model_path)
        except Exception as e:
            messagebox.showerror("", _("status_fail") + "\n" + str(e))
            self.status.config(text=_("status_fail"))
            return

        self.result = result_rgb
        self._display_mode = MODE_RESULT
        self._btn_toggle.config(state=tk.NORMAL, text=_("view_orig"))
        self._render()
        self.status.config(text=_("status_done"))

    # ── View toggle ─────────────────────────────────────────

    def toggle_view(self):
        if self._display_mode == MODE_RESULT:
            self._display_mode = MODE_ORIGINAL
            self._btn_toggle.config(text=_("view_result"))
        else:
            self._display_mode = MODE_RESULT
            self._btn_toggle.config(text=_("view_orig"))
        self._render()


def main():
    root = tk.Tk()
    WatermarkRemover(root)
    root.mainloop()


if __name__ == "__main__":
    main()
