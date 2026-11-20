"""
锂电顶盖复合视觉检测工业上位机 (Industrial Vision HMI)
-------------------------------------------------------------------
特性:
- 纯 Python 标准库 Tkinter + PIL + OpenCV，无需安装重型 GUI 框架即可在 Windows/macOS/Linux 启动
- 工业暗色 HMI 界面：高清检测视野窗口、巨型 OK/NG 状态指示灯、实时尺寸公差表、良率统计看板
- 支持【单步检测】、【模拟 PLC 连续触发播放】、【在线修改 SIP 公差并即时重判】、【自动存图】
"""

import os
import sys
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "02_CoreAlgorithm"))

from battery_inspector import LithiumBatteryInspector


class LithiumVisionHMI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lithium Battery Top-Cover Hybrid Vision HMI  |  Sub-pixel Metrology + YOLO Zone-SIP")
        self.geometry("1420x820")
        self.configure(bg="#181c24")

        self.inspector = LithiumBatteryInspector()
        self.image_files = []
        self.current_idx = -1
        self.auto_running = False
        self.total_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self.tk_img_ref = None

        self._build_ui()
        self._load_default_folder()

    def _build_ui(self):
        # 顶部标题与引擎状态栏
        top_bar = tk.Frame(self, bg="#10131a", height=54)
        top_bar.pack(fill=tk.X, side=tk.TOP)
        tk.Label(
            top_bar,
            text="  LITHIUM BATTERY TOP-COVER VISION INSPECTION STATION",
            font=("Segoe UI", 14, "bold"),
            fg="#e6edf3",
            bg="#10131a",
        ).pack(side=tk.LEFT, pady=12)

        self.lbl_engine = tk.Label(
            top_bar,
            text=f"Engine: {self.inspector.engine_mode}  ",
            font=("Consolas", 10),
            fg="#58a6ff",
            bg="#10131a",
        )
        self.lbl_engine.pack(side=tk.RIGHT, pady=12)

        # 主体左右分栏
        main_frame = tk.Frame(self, bg="#181c24")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧：图像显示区 + 底部控制栏
        left_panel = tk.Frame(main_frame, bg="#181c24")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left_panel, bg="#0d1117", highlightthickness=1, highlightbackground="#30363d")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        ctrl_bar = tk.Frame(left_panel, bg="#21262d", height=56)
        ctrl_bar.pack(fill=tk.X, pady=(8, 0))

        btn_style = {"font": ("Segoe UI", 10, "bold"), "bd": 0, "padx": 14, "pady": 8, "cursor": "hand2"}
        tk.Button(ctrl_bar, text="打开图像目录", bg="#30363d", fg="#ffffff", command=self.select_folder, **btn_style).pack(side=tk.LEFT, padx=6, pady=8)
        tk.Button(ctrl_bar, text="单次软触发 (Next)", bg="#1f6feb", fg="#ffffff", command=self.step_next, **btn_style).pack(side=tk.LEFT, padx=6, pady=8)
        self.btn_auto = tk.Button(ctrl_bar, text="▶ 启动模拟产线连续触发", bg="#238636", fg="#ffffff", command=self.toggle_auto, **btn_style)
        self.btn_auto.pack(side=tk.LEFT, padx=6, pady=8)
        tk.Button(ctrl_bar, text="清零良率计数", bg="#6e7681", fg="#ffffff", command=self.reset_stats, **btn_style).pack(side=tk.RIGHT, padx=6, pady=8)

        # 右侧：状态灯 + 测量数据 + 良率看板 + SIP 参数调节
        right_panel = tk.Frame(main_frame, bg="#21262d", width=390)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_panel.pack_propagate(False)

        # 1. OK/NG 状态大灯
        self.lbl_status = tk.Label(
            right_panel,
            text="READY",
            font=("Segoe UI", 34, "bold"),
            fg="#ffffff",
            bg="#30363d",
            height=2,
        )
        self.lbl_status.pack(fill=tk.X, padx=12, pady=12)

        # 2. 亚像素测量与缺陷详情卡片
        self._add_section_header(right_panel, "实时精密量测与缺陷复判")
        info_frame = tk.Frame(right_panel, bg="#161b22", padx=12, pady=10)
        info_frame.pack(fill=tk.X, padx=12, pady=(0, 10))

        self.vars = {
            "file": tk.StringVar(value="-"),
            "pitch": tk.StringVar(value="- mm"),
            "hole": tk.StringVar(value="- mm"),
            "ng_defects": tk.StringVar(value="0"),
            "minor_defects": tk.StringVar(value="0"),
            "cycle_ms": tk.StringVar(value="- ms"),
        }
        rows = [
            ("当前电芯样本:", "file"),
            ("正负极柱间距 (Pitch):", "pitch"),
            ("注液孔直径 (Dia):", "hole"),
            ("NG 缺陷数 (含Zone-A):", "ng_defects"),
            ("B区微小擦痕放行:", "minor_defects"),
            ("单帧检测节拍:", "cycle_ms"),
        ]
        for r_idx, (label_txt, key) in enumerate(rows):
            tk.Label(info_frame, text=label_txt, font=("Segoe UI", 10), fg="#8b949e", bg="#161b22", anchor="w").grid(row=r_idx, column=0, sticky="w", pady=3)
            tk.Label(info_frame, textvariable=self.vars[key], font=("Consolas", 10, "bold"), fg="#e6edf3", bg="#161b22", anchor="e").grid(row=r_idx, column=1, sticky="e", padx=(10, 0), pady=3)
        info_frame.columnconfigure(1, weight=1)

        # 3. 产线良率统计看板
        self._add_section_header(right_panel, "产线良率统计看板 (Yield)")
        stat_frame = tk.Frame(right_panel, bg="#161b22", padx=12, pady=10)
        stat_frame.pack(fill=tk.X, padx=12, pady=(0, 10))

        self.stat_vars = {
            "total": tk.StringVar(value="0"),
            "ok": tk.StringVar(value="0"),
            "ng": tk.StringVar(value="0"),
            "yield": tk.StringVar(value="0.0%"),
        }
        stat_items = [("总检数", "total", "#e6edf3"), ("良品 OK", "ok", "#3fb950"), ("不良 NG", "ng", "#f85149"), ("良率", "yield", "#58a6ff")]
        for col, (title, key, color) in enumerate(stat_items):
            tk.Label(stat_frame, text=title, font=("Segoe UI", 9), fg="#8b949e", bg="#161b22").grid(row=0, column=col, padx=8)
            tk.Label(stat_frame, textvariable=self.stat_vars[key], font=("Consolas", 13, "bold"), fg=color, bg="#161b22").grid(row=1, column=col, padx=8, pady=(2, 0))
            stat_frame.columnconfigure(col, weight=1)

        # 4. 在线 SIP 工艺配方参数设置
        self._add_section_header(right_panel, "SIP 工艺公差在线配置")
        sip_frame = tk.Frame(right_panel, bg="#161b22", padx=12, pady=10)
        sip_frame.pack(fill=tk.X, padx=12, pady=(0, 10))

        self.ent_pitch_tol = self._add_param_row(sip_frame, 0, "极柱间距公差 (±mm):", str(self.inspector.cfg["metrology_spec"]["pole_pitch_tol_mm"]))
        self.ent_hole_tol = self._add_param_row(sip_frame, 1, "注液孔公差 (±mm):", str(self.inspector.cfg["metrology_spec"]["injection_hole_dia_tol_mm"]))
        self.ent_b_area = self._add_param_row(sip_frame, 2, "B区最小NG面积 (mm²):", str(self.inspector.cfg["surface_sip"]["zone_b_min_ng_area_mm2"]))

        tk.Button(
            sip_frame, text="应用 SIP 参数并重检当前帧", bg="#8957e5", fg="#ffffff",
            font=("Segoe UI", 9, "bold"), bd=0, pady=6, cursor="hand2", command=self.apply_sip_and_rerun
        ).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        # 5. 缺陷明细日志列表
        self._add_section_header(right_panel, "当前帧缺陷明细日志")
        self.log_box = tk.Text(right_panel, bg="#0d1117", fg="#c9d1d9", font=("Consolas", 9), height=7, bd=0, padx=8, pady=6)
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

    def _add_section_header(self, parent, title):
        tk.Label(parent, text=title, font=("Segoe UI", 10, "bold"), fg="#58a6ff", bg="#21262d", anchor="w").pack(fill=tk.X, padx=12, pady=(4, 4))

    def _add_param_row(self, parent, row, label_txt, default_val):
        tk.Label(parent, text=label_txt, font=("Segoe UI", 9), fg="#8b949e", bg="#161b22", anchor="w").grid(row=row, column=0, sticky="w", pady=3)
        ent = tk.Entry(parent, font=("Consolas", 10), bg="#0d1117", fg="#e6edf3", insertbackground="#ffffff", width=10, bd=1)
        ent.insert(0, default_val)
        ent.grid(row=row, column=1, sticky="e", pady=3)
        parent.columnconfigure(1, weight=1)
        return ent

    def _load_default_folder(self):
        default_dir = os.path.join(BASE_DIR, "data", "battery_images")
        if os.path.isdir(default_dir):
            self.image_files = sorted(glob.glob(os.path.join(default_dir, "cell_*.bmp")))
            if self.image_files:
                self.current_idx = -1
                self.after(200, self.step_next)

    def select_folder(self):
        folder = filedialog.askdirectory(initialdir=BASE_DIR, title="选择锂电顶盖测试图像目录")
        if folder:
            files = sorted(glob.glob(os.path.join(folder, "*.bmp")) + glob.glob(os.path.join(folder, "*.png")))
            files = [f for f in files if "template" not in os.path.basename(f).lower()]
            if not files:
                messagebox.showwarning("提示", "该目录下未找到 .bmp / .png 图像文件")
                return
            self.image_files = files
            self.current_idx = -1
            self.step_next()

    def apply_sip_and_rerun(self):
        try:
            p_tol = float(self.ent_pitch_tol.get())
            h_tol = float(self.ent_hole_tol.get())
            b_area = float(self.ent_b_area.get())
            self.inspector.update_sip_params(p_tol, h_tol, b_area)
            if 0 <= self.current_idx < len(self.image_files):
                self._run_single(self.image_files[self.current_idx], count_stats=False)
        except ValueError:
            messagebox.showerror("输入错误", "SIP 公差参数必须为有效数字")

    def step_next(self):
        if not self.image_files:
            messagebox.showinfo("提示", "未检测到测试图，请先运行 01_DataGenerator/generate_lithium_battery_data.py")
            return
        self.current_idx = (self.current_idx + 1) % len(self.image_files)
        self._run_single(self.image_files[self.current_idx], count_stats=True)

    def toggle_auto(self):
        self.auto_running = not self.auto_running
        if self.auto_running:
            self.btn_auto.configure(text="■ 停止连续触发", bg="#da3633")
            self._auto_loop()
        else:
            self.btn_auto.configure(text="▶ 启动模拟产线连续触发", bg="#238636")

    def _auto_loop(self):
        if self.auto_running:
            self.step_next()
            self.after(800, self._auto_loop)

    def reset_stats(self):
        self.total_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self._refresh_stats()

    def _refresh_stats(self):
        self.stat_vars["total"].set(str(self.total_count))
        self.stat_vars["ok"].set(str(self.ok_count))
        self.stat_vars["ng"].set(str(self.ng_count))
        rate = (self.ok_count / self.total_count * 100.0) if self.total_count else 0.0
        self.stat_vars["yield"].set(f"{rate:.1f}%")

    def _run_single(self, img_path, count_stats=True):
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return
        res = self.inspector.inspect_image(gray, filename=os.path.basename(img_path))

        if count_stats:
            self.total_count += 1
            if res.overall_ok:
                self.ok_count += 1
            else:
                self.ng_count += 1
            self._refresh_stats()

        # 更新 OK/NG 状态大灯
        if res.overall_ok:
            self.lbl_status.configure(text="OK  PASS", bg="#238636")
        else:
            self.lbl_status.configure(text="NG  REJECT", bg="#da3633")

        self.vars["file"].set(res.filename)
        self.vars["pitch"].set(f"{res.pitch_mm:.3f} mm ({'OK' if res.pitch_ok else 'NG'})")
        self.vars["hole"].set(f"{res.hole_dia_mm:.3f} mm ({'OK' if res.hole_ok else 'NG'})")
        self.vars["ng_defects"].set(str(len(res.ng_defects)))
        self.vars["minor_defects"].set(str(len(res.minor_ignored_defects)))
        self.vars["cycle_ms"].set(f"{res.cycle_time_ms:.1f} ms")

        # 更新日志区
        self.log_box.delete("1.0", tk.END)
        if not res.defects:
            self.log_box.insert(tk.END, "[PASS] 表面无异常缺陷检出。\n")
        else:
            for idx, d in enumerate(res.defects, 1):
                tag = "NG  " if d.is_ng else "PASS"
                self.log_box.insert(tk.END, f"{idx}. [{tag}] {d.cls_name} | {d.area_mm2:.3f}mm² | {d.zone_tag}\n")

        # 自适应缩放并渲染图像
        if res.vis_image is not None:
            rgb = cv2.cvtColor(res.vis_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            cw = max(600, self.canvas.winfo_width())
            ch = max(400, self.canvas.winfo_height())
            pil_img.thumbnail((cw - 16, ch - 16), Image.Resampling.LANCZOS)
            self.tk_img_ref = ImageTk.PhotoImage(pil_img)
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=self.tk_img_ref, anchor=tk.CENTER)


if __name__ == "__main__":
    app = LithiumVisionHMI()
    app.mainloop()
