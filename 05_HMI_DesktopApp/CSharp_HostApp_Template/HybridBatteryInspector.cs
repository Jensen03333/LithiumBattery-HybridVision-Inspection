using System;
using System.Collections.Generic;
using System.Linq;
using HalconDotNet;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LithiumBatteryVision.Core
{
    public record DefectBox(int X1, int Y1, int X2, int Y2, int ClassId, float Confidence);

    /// <summary>
    /// 工业级 C# (.NET) + HalconDotNet + ONNX Runtime 混合推理核心类
    /// 用于在真实产线 WPF/WinForms 上位机中实现零文件落地的高速检测 (< 50ms)
    /// </summary>
    public class HybridBatteryInspector : IDisposable
    {
        private readonly InferenceSession _yoloSession;
        private readonly HTuple _shapeModelId;
        private const double PixelScale = 0.015;
        private const double ZoneBMinNgAreaMm2 = 0.040;

        public HybridBatteryInspector(string shapeModelPath, string onnxModelPath, bool useGpu = true)
        {
            HOperatorSet.ReadShapeModel(shapeModelPath, out _shapeModelId);

            var options = new SessionOptions();
            if (useGpu)
            {
                options.AppendExecutionProvider_CUDA(0);
            }
            _yoloSession = new InferenceSession(onnxModelPath, options);
        }

        public bool InspectBatteryCover(
            HObject ho_RawImage,
            out double polePitchMm,
            out double injHoleDiaMm,
            out HObject ho_NgRegions)
        {
            polePitchMm = 0;
            injHoleDiaMm = 0;
            HOperatorSet.GenEmptyObj(out ho_NgRegions);

            // 1. Halcon 形状匹配定位 + 刚性仿射摆正
            HOperatorSet.FindShapeModel(
                ho_RawImage, _shapeModelId, -0.436, 0.872, 0.7, 1, 0.5,
                "least_squares", 0, 0.9,
                out HTuple row, out HTuple col, out HTuple angle, out HTuple score);

            if (score.Length == 0) return false;

            HOperatorSet.VectorAngleToRigid(row, col, angle, 1024.0, 1024.0, 0.0, out HTuple homMat2D);
            HOperatorSet.AffineTransImage(ho_RawImage, out HObject ho_Aligned, homMat2D, "constant", "false");

            // 2. 三圆同步亚像素卡尺测量 (左极柱、右极柱、注液孔)
            HOperatorSet.CreateMetrologyModel(out HTuple metrologyHandle);
            HTuple rows = new HTuple(1024.0, 1024.0, 1024.0);
            HTuple cols = new HTuple(624.0, 1424.0, 1230.0);
            HTuple radii = new HTuple(60.0, 60.0, 40.0);
            HOperatorSet.AddMetrologyObjectCircleMeasure(
                metrologyHandle, rows, cols, radii, 18, 5, 1.0, 25,
                new HTuple("measure_transition", "measure_select"),
                new HTuple("positive", "first"),
                out HTuple indices);

            HOperatorSet.ApplyMetrologyModel(ho_Aligned, metrologyHandle);
            HOperatorSet.GetMetrologyObjectResult(metrologyHandle, indices[0], "all", "result_type", new HTuple("row", "column"), out HTuple leftCenter);
            HOperatorSet.GetMetrologyObjectResult(metrologyHandle, indices[1], "all", "result_type", new HTuple("row", "column"), out HTuple rightCenter);
            HOperatorSet.GetMetrologyObjectResult(metrologyHandle, indices[2], "all", "result_type", "radius", out HTuple holeRadiusPx);
            HOperatorSet.ClearMetrologyModel(metrologyHandle);

            HOperatorSet.DistancePp(leftCenter[0], leftCenter[1], rightCenter[0], rightCenter[1], out HTuple pitchPx);
            polePitchMm = pitchPx.D * PixelScale;
            injHoleDiaMm = holeRadiusPx.D * 2.0 * PixelScale;

            bool sizeOk = Math.Abs(polePitchMm - 12.00) <= 0.045 && Math.Abs(injHoleDiaMm - 1.20) <= 0.035;

            // 3. 裁剪中心 640x640 ROI 内存直传 YOLO ONNX
            HOperatorSet.CropRectangle1(ho_Aligned, out HObject ho_Roi, 704, 704, 1343, 1343);
            var boxes = RunYoloOnnx(ho_Roi, confThreshold: 0.30f, iouThreshold: 0.45f);

            // 4. 锂电 A/B 分区形态学复判 (防过杀核心)
            HOperatorSet.GenEllipse(out HObject ho_ZoneAValve, 320.0, 320.0, 0.0, 115.0, 70.0);
            bool surfaceOk = true;

            foreach (var box in boxes)
            {
                HOperatorSet.GenRectangle1(out HObject ho_BoxReg, box.Y1, box.X1, box.Y2, box.X2);
                HOperatorSet.ReduceDomain(ho_Roi, ho_BoxReg, out HObject ho_Patch);
                HOperatorSet.BinaryThreshold(ho_Patch, out HObject ho_DefectReal, "max_separability", "dark", out _);

                // 检查是否侵入防爆阀 A 区
                HOperatorSet.Intersection(ho_DefectReal, ho_ZoneAValve, out HObject ho_OverlapZoneA);
                HOperatorSet.AreaCenter(ho_OverlapZoneA, out HTuple areaZoneA, out _, out _);
                HOperatorSet.AreaCenter(ho_DefectReal, out HTuple totalAreaPx, out _, out _);

                double areaMm2 = totalAreaPx.D * PixelScale * PixelScale;
                bool isNg = (areaZoneA.D > 0) || (box.ClassId == 0) || (areaMm2 >= ZoneBMinNgAreaMm2);

                if (isNg)
                {
                    surfaceOk = false;
                    HOperatorSet.ConcatObj(ho_NgRegions, ho_DefectReal, out ho_NgRegions);
                }

                ho_BoxReg.Dispose();
                ho_Patch.Dispose();
                ho_DefectReal.Dispose();
                ho_OverlapZoneA.Dispose();
            }

            ho_ZoneAValve.Dispose();
            ho_Roi.Dispose();
            ho_Aligned.Dispose();

            return sizeOk && surfaceOk;
        }

        private List<DefectBox> RunYoloOnnx(HObject ho_Roi640, float confThreshold, float iouThreshold)
        {
            HOperatorSet.GetImagePointer1(ho_Roi640, out HTuple ptr, out _, out HTuple wTuple, out HTuple hTuple);
            int w = wTuple.I, h = hTuple.I;
            var inputTensor = new DenseTensor<float>(new[] { 1, 3, h, w });

            unsafe
            {
                byte* data = (byte*)ptr.IP;
                for (int y = 0; y < h; y++)
                {
                    int rowOffset = y * w;
                    for (int x = 0; x < w; x++)
                    {
                        float val = data[rowOffset + x] / 255.0f;
                        inputTensor[0, 0, y, x] = val;
                        inputTensor[0, 1, y, x] = val;
                        inputTensor[0, 2, y, x] = val;
                    }
                }
            }

            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", inputTensor) };
            using var results = _yoloSession.Run(inputs);
            var output = results.First().AsTensor<float>(); // YOLOv8/v11: [1, 4 + numClasses, 8400]

            int numChannels = output.Dimensions[1];
            int numAnchors = output.Dimensions[2];
            int numClasses = numChannels - 4;

            var candidates = new List<DefectBox>();
            for (int i = 0; i < numAnchors; i++)
            {
                int bestCls = -1;
                float maxConf = 0f;
                for (int c = 0; c < numClasses; c++)
                {
                    float conf = output[0, 4 + c, i];
                    if (conf > maxConf)
                    {
                        maxConf = conf;
                        bestCls = c;
                    }
                }
                if (maxConf < confThreshold) continue;

                float cx = output[0, 0, i];
                float cy = output[0, 1, i];
                float bw = output[0, 2, i];
                float bh = output[0, 3, i];

                int x1 = Math.Clamp((int)(cx - bw / 2f), 0, w - 1);
                int y1 = Math.Clamp((int)(cy - bh / 2f), 0, h - 1);
                int x2 = Math.Clamp((int)(cx + bw / 2f), 0, w - 1);
                int y2 = Math.Clamp((int)(cy + bh / 2f), 0, h - 1);
                candidates.Add(new DefectBox(x1, y1, x2, y2, bestCls, maxConf));
            }

            return ApplyNms(candidates, iouThreshold);
        }

        private static List<DefectBox> ApplyNms(List<DefectBox> boxes, float iouThreshold)
        {
            var sorted = boxes.OrderByDescending(b => b.Confidence).ToList();
            var keep = new List<DefectBox>();

            while (sorted.Count > 0)
            {
                var current = sorted[0];
                keep.Add(current);
                sorted.RemoveAt(0);
                sorted.RemoveAll(b => b.ClassId == current.ClassId && ComputeIoU(current, b) > iouThreshold);
            }
            return keep;
        }

        private static float ComputeIoU(DefectBox a, DefectBox b)
        {
            int xx1 = Math.Max(a.X1, b.X1);
            int yy1 = Math.Max(a.Y1, b.Y1);
            int xx2 = Math.Min(a.X2, b.X2);
            int yy2 = Math.Min(a.Y2, b.Y2);
            int interW = Math.Max(0, xx2 - xx1);
            int interH = Math.Max(0, yy2 - yy1);
            float interArea = interW * interH;
            float unionArea = (a.X2 - a.X1) * (a.Y2 - a.Y1) + (b.X2 - b.X1) * (b.Y2 - b.Y1) - interArea;
            return unionArea > 0 ? interArea / unionArea : 0f;
        }

        public void Dispose()
        {
            _yoloSession?.Dispose();
            HOperatorSet.ClearShapeModel(_shapeModelId);
        }
    }
}
