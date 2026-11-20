"""
批量产线检测与 CSV 良率报表导出脚本
"""

import os
import csv
import glob
import cv2
from battery_inspector import LithiumBatteryInspector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    img_dir = os.path.join(BASE_DIR, "data", "battery_images")
    out_dir = os.path.join(BASE_DIR, "data", "inspection_results")
    os.makedirs(out_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(img_dir, "cell_*.bmp")))
    if not img_files:
        print("[!] 未找到测试图像，请先运行: python 01_DataGenerator/generate_lithium_battery_data.py")
        return

    inspector = LithiumBatteryInspector()
    print(f"[*] 启动检测引擎: {inspector.engine_mode}")

    csv_path = os.path.join(out_dir, "production_report.csv")
    total, ok_cnt = 0, 0

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Filename", "Overall_Result", "Pole_Pitch_mm", "Pitch_OK",
            "Injection_Hole_mm", "Hole_OK", "NG_Defects", "Minor_Ignored",
            "Polarity_Flipped", "Cycle_Time_ms"
        ])

        for p in img_files:
            name = os.path.basename(p)
            gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            res = inspector.inspect_image(gray, filename=name)
            total += 1
            if res.overall_ok:
                ok_cnt += 1

            if res.vis_image is not None:
                cv2.imwrite(os.path.join(out_dir, name.replace(".bmp", "_hmi.png")), res.vis_image)

            writer.writerow([
                res.filename,
                "OK" if res.overall_ok else "NG",
                f"{res.pitch_mm:.4f}",
                res.pitch_ok,
                f"{res.hole_dia_mm:.4f}",
                res.hole_ok,
                len(res.ng_defects),
                len(res.minor_ignored_defects),
                res.polarity_flipped,
                f"{res.cycle_time_ms:.2f}",
            ])

            print(
                f"[*] {name:12s} | Pitch:{res.pitch_mm:.3f}mm | Hole:{res.hole_dia_mm:.3f}mm | "
                f"NG:{len(res.ng_defects)} (Minor:{len(res.minor_ignored_defects)}) | "
                f"{res.cycle_time_ms:5.1f}ms | {'OK' if res.overall_ok else 'NG'}"
            )

    yield_rate = (ok_cnt / total * 100.0) if total else 0.0
    print("-" * 78)
    print(f"[+] 批测完成: 总计 {total} 片 | OK: {ok_cnt} | NG: {total - ok_cnt} | 良率: {yield_rate:.1f}%")
    print(f"[+] 可视化结果图与 CSV 报表已保存至: {out_dir}")


if __name__ == "__main__":
    main()
