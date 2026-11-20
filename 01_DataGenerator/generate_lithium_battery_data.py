"""
方形动力锂电池顶盖高保真工业样本与 YOLO 表面缺陷数据集生成器
-------------------------------------------------------------------
功能:
1. 生成无偏移标准模板图 (battery_template.bmp)，用于 Halcon/VM/VP 模板注册。
2. 生成带随机旋转(±18°)、平移、极柱间距(Pitch)波动、注液孔公差波动及三类缺陷的
   2048x2048 产线测试原图 (cell_00.bmp ~ cell_XX.bmp)。
3. 自动裁剪中心 640x640 检测区并输出标准 YOLO 格式标注 (0:焊渣, 1:防爆阀划伤, 2:铝壳凹坑)。
"""

import os
import sys
import math
import random
import argparse
import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def draw_lithium_battery_cover(pole_pitch_offset=0, hole_r=40):
    """绘制 2048x2048 画布上的标准方形动力锂电池顶盖 (中心在 1024, 1024)"""
    img = np.random.normal(30, 2.5, (2048, 2048)).astype(np.uint8)

    # 1. 铝合金顶盖主体 (宽 1200px, 高 360px: x:424~1624, y:844~1204)
    x1, y1, x2, y2 = 424, 844, 1624, 1204
    cv2.rectangle(img, (x1, y1), (x2, y2), 190, -1)
    noise = np.random.normal(0, 3.5, (y2 - y1, x2 - x1)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (15, 1), 3.0)
    shell_patch = np.clip(img[y1:y2, x1:x2].astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img[y1:y2, x1:x2] = shell_patch

    # 2. 左下角极性防呆倒角 (唯一确定正负极左右方向)
    pts_chamfer = np.array([[x1, y2 - 55], [x1 + 55, y2], [x1, y2]], np.int32)
    cv2.fillPoly(img, [pts_chamfer], 30)

    # 3. 左负极柱 (624, 1024) & 右正极柱 (1424 + offset, 1024)
    left_pole_cx = 624
    right_pole_cx = 1424 + pole_pitch_offset
    for cx in [left_pole_cx, right_pole_cx]:
        cv2.circle(img, (cx, 1024), 115, 75, -1)   # PPS 黑色绝缘圈
        cv2.circle(img, (cx, 1024), 85, 215, -1)   # 金属极柱端面
        cv2.circle(img, (cx, 1024), 60, 32, -1)    # 极柱定位内孔 (卡尺测量目标)

    # 4. 中央防爆阀 (Zone-A 核心禁区: 中心 1024,1024, 椭圆半轴 110x65)
    cv2.ellipse(img, (1024, 1024), (115, 70), 0, 0, 360, 130, 3)
    cv2.ellipse(img, (1024, 1024), (110, 65), 0, 0, 360, 172, -1)
    cv2.ellipse(img, (1024, 1024), (85, 45), 0, 0, 360, 140, 2)  # 防爆刻痕环

    # 5. 注液孔 (标准中心 1230, 1024, 标准半径 40px)
    cv2.circle(img, (1230, 1024), hole_r + 12, 160, -1)
    cv2.circle(img, (1230, 1024), hole_r, 30, -1)

    return img


def inject_battery_defects_on_roi(roi_img):
    """
    在中心 640x640 检测区注入三类典型锂电缺陷:
    0: weld_spatter  (激光焊渣飞溅)
    1: valve_scratch (防爆阀致命划伤)
    2: shell_dent    (铝壳磕碰凹坑 / 微小擦痕)
    """
    h, w = roi_img.shape
    labels = []

    # 1. 激光焊渣飞溅 (class 0)
    if random.random() > 0.25:
        for _ in range(random.randint(1, 2)):
            sx = random.randint(80, 560)
            sy = random.choice([random.randint(170, 235), random.randint(405, 470)])
            r_spatter = random.randint(5, 9)
            cv2.circle(roi_img, (sx, sy), r_spatter + 3, 60, -1)
            cv2.circle(roi_img, (sx, sy), r_spatter, 245, -1)
            box_w = (r_spatter + 6) * 2
            labels.append((0, sx / w, sy / h, box_w / w, box_w / h))

    # 2. 防爆阀致命划伤 (class 1)
    if random.random() > 0.40:
        vx = random.randint(270, 370)
        vy = random.randint(295, 345)
        length = random.randint(28, 52)
        angle = random.uniform(-0.6, 0.6)
        x1 = int(vx - (length / 2) * math.cos(angle))
        y1 = int(vy - (length / 2) * math.sin(angle))
        x2 = int(vx + (length / 2) * math.cos(angle))
        y2 = int(vy + (length / 2) * math.sin(angle))
        cv2.line(roi_img, (x1, y1), (x2, y2), random.randint(55, 80), thickness=2)
        bx1, bx2 = max(0, min(x1, x2) - 5), min(w, max(x1, x2) + 5)
        by1, by2 = max(0, min(y1, y2) - 5), min(h, max(y1, y2) + 5)
        labels.append((1, ((bx1 + bx2) / 2) / w, ((by1 + by2) / 2) / h, (bx2 - bx1) / w, (by2 - by1) / h))

    # 3. 铝壳表面凹坑 / 微小放行擦痕 (class 2)
    for _ in range(random.randint(1, 2)):
        dx = random.randint(70, 420)
        dy = random.choice([random.randint(165, 230), random.randint(410, 475)])
        is_minor = random.random() < 0.45
        if is_minor:
            rx, ry = random.randint(3, 4), random.randint(2, 4)
        else:
            rx, ry = random.randint(10, 16), random.randint(8, 13)
        cv2.ellipse(roi_img, (dx, dy), (rx, ry), random.randint(0, 180), 0, 360, random.randint(65, 95), -1)
        labels.append((2, dx / w, dy / h, (rx * 2 + 6) / w, (ry * 2 + 6) / h))

    return roi_img, labels


def generate_dataset(num_line_samples=20, num_yolo_samples=240, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    img_dir = os.path.join(BASE_DIR, "data", "battery_images")
    yolo_root = os.path.join(BASE_DIR, "03_YOLO_Workspace", "battery_dataset")
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(yolo_root, sub), exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # 1. 标准模板图
    tpl = draw_lithium_battery_cover(pole_pitch_offset=0, hole_r=40)
    cv2.imwrite(os.path.join(img_dir, "battery_template.bmp"), tpl)
    print(f"[+] 标准模板图已生成: {os.path.join(img_dir, 'battery_template.bmp')}")

    # 2. 产线大视野测试图
    for i in range(num_line_samples):
        if i % 4 == 0:
            # 良品（尺寸与表面均合格）
            pitch_offset = random.randint(-1, 1)
            hole_r = 40
            base = draw_lithium_battery_cover(pole_pitch_offset=pitch_offset, hole_r=hole_r)
        else:
            pitch_offset = random.randint(-4, 4)
            hole_r = random.randint(38, 42)
            base = draw_lithium_battery_cover(pole_pitch_offset=pitch_offset, hole_r=hole_r)
            roi = base[704:1344, 704:1344].copy()
            roi, _ = inject_battery_defects_on_roi(roi)
            base[704:1344, 704:1344] = roi

        angle = random.uniform(-18, 18)
        # 偶发 180° 反向来料，用于测试极性防呆纠偏
        if i == 3:
            angle += 180.0
        tx, ty = random.randint(-90, 90), random.randint(-90, 90)
        M = cv2.getRotationMatrix2D((1024, 1024), angle, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty
        rotated = cv2.warpAffine(base, M, (2048, 2048), borderValue=30)
        cv2.imwrite(os.path.join(img_dir, f"cell_{i:02d}.bmp"), rotated)
    print(f"[+] 已生成 {num_line_samples} 张产线实测大图 -> {img_dir}")

    # 3. YOLO 训练集
    if num_yolo_samples > 0:
        train_cutoff = int(num_yolo_samples * 0.8)
        for i in range(num_yolo_samples):
            split = "train" if i < train_cutoff else "val"
            base = draw_lithium_battery_cover(pole_pitch_offset=random.randint(-2, 2), hole_r=random.randint(39, 41))
            roi = base[704:1344, 704:1344].copy()
            roi, labels = inject_battery_defects_on_roi(roi)

            cv2.imwrite(os.path.join(yolo_root, "images", split, f"cell_roi_{i:03d}.bmp"), roi)
            with open(os.path.join(yolo_root, "labels", split, f"cell_roi_{i:03d}.txt"), "w", encoding="utf-8") as f:
                for cls_id, cx, cy, bw, bh in labels:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        yaml_content = f"""path: {yolo_root.replace(os.sep, '/')}
train: images/train
val: images/val
names:
  0: weld_spatter
  1: valve_scratch
  2: shell_dent
"""
        with open(os.path.join(yolo_root, "data.yaml"), "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"[+] 已生成 {num_yolo_samples} 张 YOLO 锂电缺陷数据集 -> {yolo_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成锂电顶盖工业视觉仿真数据")
    parser.add_argument("--line-samples", type=int, default=20, help="产线测试大图数量")
    parser.add_argument("--yolo-samples", type=int, default=240, help="YOLO 训练集图像数量")
    args = parser.parse_args()
    generate_dataset(args.line_samples, args.yolo_samples)
