# 海康 VisionMaster (VM 4.x) 与康耐视 VisionPro (VP) 锂电顶盖移植指南

本项目的所有几何数学逻辑与工业三大视觉软件（Halcon、海康 VM、康耐视 VP）严格一一对应。安装软件后，直接导入 `data/battery_images/` 目录即可按以下流程复现：

## 一、 算子对照表

| 锂电工艺环节 | 本项目 Python 源码 | Halcon 算子 | 海康 VisionMaster (VM) | 康耐视 VisionPro (VP) |
| :--- | :--- | :--- | :--- | :--- |
| 顶盖定位与极性防呆 | `align_battery_cover()` | `find_shape_model` | 高精度模板匹配 | `CogPMAlignTool` (PatMax) |
| 坐标系跟随 / 摆正 | `cv2.warpAffine` | `vector_angle_to_rigid` + `affine_trans_image` | 位置修正 / 仿射变换 | `CogFixtureTool` |
| 正负极柱 & 注液孔测量 | `measure_circle_subpixel()` | `add_metrology_object_circle_measure` | 圆查找 (×3, 由黑到白) | `CogFindCircleTool` (×3, DarkToLight) |
| 正负极柱中心距 (Pitch) | `math.hypot(rx-lx, ry-ly)` | `distance_pp` | 点点测量 | `CogDistancePointPointTool` |
| 表面缺陷 + A/B 分区 | `inspect_surface_zone_sip()` | ONNX + `intersection` + `area_center` | 深度学习目标检测 + ROI 掩膜 + Blob 分析 | ViDi + `CogBlobTool` |

## 二、 海康 VisionMaster (VM) 拖拽实操步骤

1. **图像源：** 拖入【图像源】模块，指向 `data/battery_images/`。
2. **模板注册：** 拖入【高精度模板匹配】，选中 `battery_template.bmp` 框选顶盖外轮廓和左下角防呆倒角，角度范围设为 `-25° ~ +25°`。
3. **位置修正：** 拖入【位置修正】模块，订阅模板匹配的匹配点与角度。
4. **三圆测量：** 拖入 3 个【圆查找】模块，分别绑定上述位置修正，在左负极柱内孔、右正极柱内孔、注液孔边缘拉出圆环卡尺，极性选 **由黑到白**，边缘点选 **第一个点**。
5. **极柱距计算：** 拖入【点点测量】连接左右极柱圆心，拖入【脚本/格式化】模块乘以像素当量 `0.015`，使用【条件检测】卡控 `11.955 ~ 12.045 mm`。
6. **接入 YOLO ONNX：** 在 VM 中可通过【自定义 C#/C++ 模块】加载 `models/best_defect.onnx`（使用 `Microsoft.ML.OnnxRuntime`），或在 C# 上位机中通过 `VM.Core.VmSolution` 获取 VM 摆正后的图像直传 YOLO。
