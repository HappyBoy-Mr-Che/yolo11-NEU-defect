"""模型导出: PyTorch .pt -> ONNX / TensorRT engine。

用法:
    python tools/export.py --weights models/best.pt --format onnx --imgsz 640 --half
    python tools/export.py --weights models/best.pt --format engine --imgsz 640 --half

TensorRT 导出需要: GPU + pip install tensorrt (本机无 CUDA 时会自动跳过)。
"""
import argparse
import sys
from pathlib import Path

from common import PROJECT_ROOT, check_weight_compat

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="导出 ONNX / TensorRT 模型")
    parser.add_argument("--weights", required=True, help=".pt 权重路径")
    parser.add_argument("--format", default="onnx", choices=["onnx", "engine", "openvino", "tflite"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16 导出")
    parser.add_argument("--dynamic", action="store_true", help="动态 batch/尺寸")
    args = parser.parse_args()

    w = Path(args.weights)
    if not w.exists():
        raise SystemExit(f"权重不存在: {w}")
    if w.suffix != ".pt":
        raise SystemExit("请提供 .pt 权重作为导出源")

    if args.format == "engine":
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("[跳过] TensorRT 导出需要 NVIDIA GPU, 本机未检测到 CUDA。"
                             "可在 GPU 机器上运行本脚本, 或在有 TensorRT 的机器上 "
                             "先导出 ONNX 再用 trtexec 转换。")

    check_weight_compat(args.weights)

    size_before = w.stat().st_size
    model = YOLO(str(w))

    print(f"导出中: {w.name} -> {args.format} (imgsz={args.imgsz}, half={args.half})")
    out_path = model.export(format=args.format, imgsz=args.imgsz, half=args.half, dynamic=args.dynamic)

    out = Path(out_path)
    size_after = out.stat().st_size
    print(f"\n导出完成: {out}")
    print(f"  原始 .pt: {size_before / 1e6:.2f} MB")
    print(f"  导出模型: {size_after / 1e6:.2f} MB  ({(size_after / size_before) * 100:.1f}%)")


if __name__ == "__main__":
    main()
