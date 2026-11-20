"""
锂电顶盖表面微缺陷 YOLOv8/v11 工业定制化训练与静态 ONNX 导出脚本
"""

import os
import shutil
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_and_export():
    data_yaml = os.path.join(BASE_DIR, "03_YOLO_Workspace", "battery_dataset", "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError("请先运行 01_DataGenerator/generate_lithium_battery_data.py 生成数据集")

    model = YOLO("yolov8n.pt")
    model.train(
        data=data_yaml,
        epochs=60,
        imgsz=640,
        batch=16,
        device=0,             # 若无独显请改为 'cpu'
        close_mosaic=15,      # 末段关闭 Mosaic，强化微细划痕与焊渣边缘收敛
        degrees=5.0,          # 工件已经过刚性仿射摆正，仅需小角度扰动
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.0,            # 工业黑白相机禁用色相增强
        hsv_s=0.0,            # 禁用饱和度增强
        hsv_v=0.15,           # 仅模拟 15% 车间同轴光源衰减
        box=8.5,              # 提升小目标边界框回归权重
        project=os.path.join(BASE_DIR, "03_YOLO_Workspace", "runs"),
        name="lithium_defect_v1",
    )

    best_pt = os.path.join(BASE_DIR, "03_YOLO_Workspace", "runs", "lithium_defect_v1", "weights", "best.pt")
    trained = YOLO(best_pt)
    onnx_path = trained.export(
        format="onnx",
        imgsz=640,
        opset=12,
        dynamic=False,        # 静态维度 (1x3x640x640) 在 C# ONNX Runtime / TensorRT 下延迟最低
        simplify=True,
    )

    deploy_path = os.path.join(BASE_DIR, "models", "best_defect.onnx")
    os.makedirs(os.path.dirname(deploy_path), exist_ok=True)
    shutil.copy(onnx_path, deploy_path)
    print(f"[+] 锂电缺陷模型已成功导出并部署至: {deploy_path}")


if __name__ == "__main__":
    train_and_export()
