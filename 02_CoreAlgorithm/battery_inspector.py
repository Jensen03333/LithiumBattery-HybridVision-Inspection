"""
锂电顶盖复合视觉检测核心引擎 (Hybrid Vision Engine)
-------------------------------------------------------------------
架构:
1. 刚性定位与极性防呆纠偏 (复刻 Halcon find_shape_model + affine_trans_image)
2. 三圆同步 1D 径向亚像素卡尺量测 (复刻 Halcon Metrology / 海康 VM 圆查找)
   - 正负极柱中心距 (Pitch) 精密测量
   - 注液孔内径 (Injection Hole Diameter) 精密测量
3. YOLO 深度学习表面缺陷检测 + 锂电 A/B 分区形态学复判 (防过杀 SIP)
   - Zone-A (中央防爆阀禁区): 零容忍
   - Zone-B (普通铝壳面): 依据物理面积阈值过滤微小无害擦痕
"""

import os
import json
import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class DefectItem:
    box: Tuple[int, int, int, int]   # ROI 内坐标 (x1, y1, x2, y2)
    cls_name: str
    area_mm2: float
    is_ng: bool
    zone_tag: str
    conf: float = 1.0


@dataclass
class InspectionResult:
    filename: str
    overall_ok: bool
    aligned_ok: bool
    polarity_flipped: bool
    pitch_mm: float
    pitch_ok: bool
    hole_dia_mm: float
    hole_ok: bool
    surface_ok: bool
    cycle_time_ms: float
    engine_mode: str
    defects: List[DefectItem] = field(default_factory=list)
    vis_image: Optional[np.ndarray] = None

    @property
    def ng_defects(self) -> List[DefectItem]:
        return [d for d in self.defects if d.is_ng]

    @property
    def minor_ignored_defects(self) -> List[DefectItem]:
        return [d for d in self.defects if not d.is_ng]


class LithiumBatteryInspector:
    def __init__(self, config_path: Optional[str] = None, model_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if config_path is None:
            config_path = os.path.join(base_dir, "config", "sip_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg: Dict[str, Any] = json.load(f)

        self.yolo_model = None
        self.engine_mode = "Traditional-Fallback + Subpixel-Metrology"
        if model_path is None:
            candidate = os.path.join(base_dir, "models", "best_defect.onnx")
            if os.path.exists(candidate):
                model_path = candidate

        if model_path and os.path.exists(model_path) and YOLO is not None:
            self.yolo_model = YOLO(model_path)
            self.engine_mode = f"YOLO ({os.path.basename(model_path)}) + Subpixel-Metrology"

    @property
    def pixel_scale(self) -> float:
        return float(self.cfg["calibration"]["pixel_scale_mm_per_px"])

    def update_sip_params(self, pitch_tol_mm: float, hole_tol_mm: float, zone_b_min_area_mm2: float):
        """支持上位机在线动态调整 SIP 工艺公差"""
        self.cfg["metrology_spec"]["pole_pitch_tol_mm"] = pitch_tol_mm
        self.cfg["metrology_spec"]["injection_hole_dia_tol_mm"] = hole_tol_mm
        self.cfg["surface_sip"]["zone_b_min_ng_area_mm2"] = zone_b_min_area_mm2

    def align_battery_cover(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], bool, bool]:
        """刚性仿射摆正 + 左下角极性防呆倒角校验"""
        _, binary = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, False, False

        max_cnt = max(contours, key=cv2.contourArea)
        (cx, cy), (w, h), angle = cv2.minAreaRect(max_cnt)
        if w < h:
            angle += 90.0
        # 长轴朝向以 180° 为周期，先取最小摆正角，再由倒角判断来料极性。
        angle = (angle + 90.0) % 180.0 - 90.0

        target_cx, target_cy = self.cfg["calibration"]["canvas_center"]
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        M[0, 2] += (target_cx - cx)
        M[1, 2] += (target_cy - cy)
        aligned = cv2.warpAffine(img, M, (2048, 2048), flags=cv2.INTER_LINEAR, borderValue=30)

        # 校验左下角防呆倒角，若装反 180° 则自动纠偏并记录标志
        bottom_left_mean = float(np.mean(aligned[1155:1200, 428:475]))
        top_right_mean = float(np.mean(aligned[848:895, 1570:1620]))
        flipped = False
        if bottom_left_mean > top_right_mean:
            M_flip = cv2.getRotationMatrix2D((target_cx, target_cy), 180.0, 1.0)
            aligned = cv2.warpAffine(aligned, M_flip, (2048, 2048), flags=cv2.INTER_LINEAR, borderValue=30)
            flipped = True

        return aligned, True, flipped

    def measure_circle_subpixel(
        self,
        aligned_img: np.ndarray,
        expected_center: Tuple[float, float],
        expected_r: float,
        search_half_len: int = 18,
        num_calipers: int = 36,
    ):
        """
        1D 径向亚像素卡尺测量 (由黑到白 positive 极性 + 三点抛物线插值 + 最小二乘圆拟合)
        """
        cx, cy = expected_center
        r_min, r_max = int(expected_r - search_half_len), int(expected_r + search_half_len)
        subpixel_pts = []
        caliper_lines = []

        for i in range(num_calipers):
            theta = 2.0 * math.pi * i / num_calipers
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            caliper_lines.append((
                int(cx + r_min * cos_t), int(cy + r_min * sin_t),
                int(cx + r_max * cos_t), int(cy + r_max * sin_t),
            ))

            radii = np.arange(r_min, r_max + 1, dtype=np.float32)
            sx = (cx + radii * cos_t).reshape(1, -1).astype(np.float32)
            sy = (cy + radii * sin_t).reshape(1, -1).astype(np.float32)
            profile = cv2.remap(aligned_img, sx, sy, cv2.INTER_LINEAR).flatten().astype(np.float64)

            smooth = cv2.GaussianBlur(profile.reshape(1, -1), (1, 5), 1.0).flatten()
            grad = np.gradient(smooth)

            peak_idx = int(np.argmax(grad))
            if 1 <= peak_idx < len(grad) - 1 and grad[peak_idx] > 15.0:
                g_prev, g_curr, g_next = grad[peak_idx - 1], grad[peak_idx], grad[peak_idx + 1]
                denom = g_prev - 2.0 * g_curr + g_next
                delta = 0.5 * (g_prev - g_next) / denom if abs(denom) > 1e-6 else 0.0
                r_sub = r_min + peak_idx + delta
                subpixel_pts.append([cx + r_sub * cos_t, cy + r_sub * sin_t])

        pts = np.array(subpixel_pts, dtype=np.float64)
        x, y = pts[:, 0], pts[:, 1]
        A = np.column_stack([x, y, np.ones_like(x)])
        b = -(x ** 2 + y ** 2)
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        fit_cx, fit_cy = -coeffs[0] / 2.0, -coeffs[1] / 2.0
        fit_r = math.sqrt(max(1e-6, fit_cx ** 2 + fit_cy ** 2 - coeffs[2]))

        return (fit_cx, fit_cy, fit_r), pts, caliper_lines

    def inspect_surface_zone_sip(self, roi_gray: np.ndarray) -> List[DefectItem]:
        """YOLO 缺陷推理 (或传统回退检测) + 锂电 A/B 分区形态学面积复判"""
        sip = self.cfg["surface_sip"]
        raw_detections = []

        if self.yolo_model is not None:
            roi_bgr = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
            res = self.yolo_model.predict(roi_bgr, conf=sip["yolo_conf_threshold"], verbose=False)[0]
            names = {0: "Weld_Spatter", 1: "Valve_Scratch", 2: "Shell_Dent"}
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                raw_detections.append((x1, y1, x2, y2, names.get(cls_id, "Defect"), conf))
        else:
            # 1) 提取高亮激光焊渣核，并记录焊渣抑制掩膜，防止焊渣黑色氧化圈被重复判成凹坑
            spatter_suppress_mask = np.ones_like(roi_gray, dtype=np.uint8) * 255
            _, spatter_bin = cv2.threshold(roi_gray, 225, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(spatter_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if 3 <= w <= 30 and 3 <= h <= 30:
                    raw_detections.append((x - 5, y - 5, x + w + 5, y + h + 5, "Weld_Spatter", 0.96))
                    cv2.circle(spatter_suppress_mask, (x + w // 2, y + h // 2), max(w, h) + 10, 0, -1)

            # 2) 构建安全搜索掩膜 (严格排除左右极柱绝缘圈、注液孔、防爆阀边框与刻痕环)
            mask_search = np.zeros_like(roi_gray)
            cv2.rectangle(mask_search, (50, 155), (590, 485), 255, -1)
            cv2.circle(mask_search, (-80, 320), 130, 0, -1)          # 排除左负极柱绝缘圈
            cv2.circle(mask_search, (720, 320), 130, 0, -1)          # 排除右正极柱绝缘圈
            cv2.circle(mask_search, (526, 320), 62, 0, -1)           # 排除注液孔台阶
            cv2.ellipse(mask_search, (320, 320), (118, 73), 0, 0, 360, 0, 14)  # 排除防爆阀外沿
            cv2.ellipse(mask_search, (320, 320), (85, 45), 0, 0, 360, 0, 12)   # 排除防爆刻痕环
            mask_search = cv2.bitwise_and(mask_search, spatter_suppress_mask)

            local_mean = cv2.GaussianBlur(roi_gray, (31, 31), 5.0)
            diff_dark = cv2.subtract(local_mean, roi_gray)
            _, dark_bin = cv2.threshold(diff_dark, 38, 255, cv2.THRESH_BINARY)
            dark_bin = cv2.bitwise_and(dark_bin, mask_search)
            dark_bin = cv2.morphologyEx(dark_bin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

            cnts_dark, _ = cv2.findContours(dark_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            vx, vy = sip["zone_a_valve_center_in_roi"]
            va, vb = sip["zone_a_valve_axes"]
            for c in cnts_dark:
                if cv2.contourArea(c) < 6:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                if w >= 4 or h >= 4:
                    cx_b, cy_b = x + w / 2.0, y + h / 2.0
                    in_valve = (((cx_b - vx) / va) ** 2 + ((cy_b - vy) / vb) ** 2) <= 1.0
                    cls_name = "Valve_Scratch" if in_valve else "Shell_Dent"
                    raw_detections.append((x - 4, y - 4, x + w + 4, y + h + 4, cls_name, 0.92))

        # 3) A/B 分区形态学面积复判
        evaluated: List[DefectItem] = []
        vx, vy = sip["zone_a_valve_center_in_roi"]
        va, vb = sip["zone_a_valve_axes"]
        min_b_area = float(sip["zone_b_min_ng_area_mm2"])

        for (x1, y1, x2, y2, cls_name, conf) in raw_detections:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(640, x2), min(640, y2)
            patch = roi_gray[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            if cls_name == "Weld_Spatter":
                area_px = int(np.sum((patch > 220) | (patch < 90)))
            else:
                med = float(np.median(patch))
                area_px = int(np.sum(patch < (med - 25)))
            if area_px < 4:
                continue
            area_mm2 = area_px * (self.pixel_scale ** 2)

            cx_b, cy_b = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            in_zone_a = (((cx_b - vx) / va) ** 2 + ((cy_b - vy) / vb) ** 2) <= 1.0

            if (in_zone_a and sip["zone_a_zero_tolerance"]) or (cls_name == "Weld_Spatter" and sip["spatter_zero_tolerance"]):
                is_ng = True
                zone_tag = "ZoneA-Critical" if in_zone_a else "Spatter-NG"
            else:
                is_ng = area_mm2 >= min_b_area
                zone_tag = "ZoneB-NG" if is_ng else "ZoneB-MinorPass"

            evaluated.append(DefectItem(
                box=(x1, y1, x2, y2),
                cls_name=cls_name,
                area_mm2=area_mm2,
                is_ng=is_ng,
                zone_tag=zone_tag,
                conf=conf,
            ))

        return evaluated

    def inspect_image(self, img_gray: np.ndarray, filename: str = "frame.bmp") -> InspectionResult:
        """单帧完整工业检测管线"""
        t0 = time.perf_counter()
        aligned, aligned_ok, flipped = self.align_battery_cover(img_gray)
        if not aligned_ok or aligned is None:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return InspectionResult(
                filename=filename, overall_ok=False, aligned_ok=False, polarity_flipped=False,
                pitch_mm=0.0, pitch_ok=False, hole_dia_mm=0.0, hole_ok=False,
                surface_ok=False, cycle_time_ms=dt_ms, engine_mode=self.engine_mode,
            )

        ms = self.cfg["metrology_spec"]
        nc = int(ms["num_calipers"])
        (lx, ly, lr), l_pts, l_cals = self.measure_circle_subpixel(
            aligned, tuple(ms["left_pole_expected_center"]), ms["pole_inner_radius_px"], 18, nc
        )
        (rx, ry, rr), r_pts, r_cals = self.measure_circle_subpixel(
            aligned, tuple(ms["right_pole_expected_center"]), ms["pole_inner_radius_px"], 18, nc
        )
        (hx, hy, hr), h_pts, h_cals = self.measure_circle_subpixel(
            aligned, tuple(ms["injection_hole_expected_center"]), ms["injection_hole_radius_px"], 14, nc
        )

        pitch_mm = math.hypot(rx - lx, ry - ly) * self.pixel_scale
        pitch_ok = abs(pitch_mm - ms["pole_pitch_std_mm"]) <= ms["pole_pitch_tol_mm"]

        hole_dia_mm = hr * 2.0 * self.pixel_scale
        hole_ok = abs(hole_dia_mm - ms["injection_hole_dia_std_mm"]) <= ms["injection_hole_dia_tol_mm"]

        rx1, ry1, rx2, ry2 = self.cfg["surface_sip"]["roi_crop_rect"]
        roi_surface = aligned[ry1:ry2, rx1:rx2]
        defects = self.inspect_surface_zone_sip(roi_surface)
        ng_defects = [d for d in defects if d.is_ng]
        surface_ok = len(ng_defects) == 0
        overall_ok = pitch_ok and hole_ok and surface_ok
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # 绘制工业 HMI 视觉叠加图
        vis = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
        cv2.ellipse(vis, (1024, 1024), (115, 70), 0, 0, 360, (255, 0, 255), 2)
        cv2.putText(vis, "Zone-A (Zero Tolerance)", (912, 938), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

        for cals, pts, (cx_f, cy_f, r_f), is_ok in [
            (l_cals, l_pts, (lx, ly, lr), pitch_ok),
            (r_cals, r_pts, (rx, ry, rr), pitch_ok),
            (h_cals, h_pts, (hx, hy, hr), hole_ok),
        ]:
            for (x1, y1, x2, y2) in cals:
                cv2.line(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)
            for (px, py) in pts:
                cv2.circle(vis, (int(round(px)), int(round(py))), 2, (0, 255, 255), -1)
            cv2.circle(vis, (int(round(cx_f)), int(round(cy_f))), int(round(r_f)), (0, 255, 0) if is_ok else (0, 0, 255), 2)

        pitch_color = (0, 255, 0) if pitch_ok else (0, 0, 255)
        cv2.line(vis, (int(round(lx)), int(round(ly))), (int(round(rx)), int(round(ry))), pitch_color, 2)
        cv2.putText(
            vis,
            f"Pole Pitch: {pitch_mm:.3f}mm (Std {ms['pole_pitch_std_mm']:.2f} +/- {ms['pole_pitch_tol_mm']:.3f})",
            (760, 1255), cv2.FONT_HERSHEY_SIMPLEX, 0.78, pitch_color, 2,
        )
        hole_color = (0, 255, 0) if hole_ok else (0, 0, 255)
        cv2.putText(
            vis,
            f"InjHole: {hole_dia_mm:.3f}mm",
            (1165, 955), cv2.FONT_HERSHEY_SIMPLEX, 0.58, hole_color, 2,
        )

        # 绘制缺陷框 (根据上下半区智能错开文字位置防止重叠)
        for idx, d in enumerate(defects):
            bx1, by1, bx2, by2 = d.box
            gx1, gy1, gx2, gy2 = bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1
            color = (0, 0, 255) if d.is_ng else (0, 210, 0)
            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), color, 3 if d.is_ng else 1)
            txt = f"{d.cls_name} {d.area_mm2:.3f}mm2 [{d.zone_tag}]"
            text_y = (gy1 - 10 - (idx % 2) * 18) if gy1 < 1024 else (gy2 + 20 + (idx % 2) * 18)
            cv2.putText(vis, txt, (max(380, gx1 - 20), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

        banner = (
            f"[{'OK' if overall_ok else 'NG'}] {filename} | Pitch:{pitch_mm:.3f}mm | "
            f"InjHole:{hole_dia_mm:.3f}mm | NG_Defects:{len(ng_defects)} | {dt_ms:.1f}ms"
        )
        cv2.putText(vis, banner, (380, 675), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0) if overall_ok else (0, 0, 255), 3)
        if flipped:
            cv2.putText(vis, "Polarity Auto-Corrected (180 deg)", (380, 715), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        vis_crop = vis[610:1320, 360:1690]

        return InspectionResult(
            filename=filename,
            overall_ok=overall_ok,
            aligned_ok=True,
            polarity_flipped=flipped,
            pitch_mm=pitch_mm,
            pitch_ok=pitch_ok,
            hole_dia_mm=hole_dia_mm,
            hole_ok=hole_ok,
            surface_ok=surface_ok,
            cycle_time_ms=dt_ms,
            engine_mode=self.engine_mode,
            defects=defects,
            vis_image=vis_crop,
        )
