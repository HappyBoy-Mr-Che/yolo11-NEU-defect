"""全链路推理基准: 预处理/推理/后处理分阶段计时 + 模型效率指标。

支持后端: .pt (PyTorch) / .onnx (ONNXRuntime) / .engine (TensorRT, 需 GPU)。

用法:
    python tools/benchmark.py --weights models/best.pt models/best.onnx --n 50
    python tools/benchmark.py --weights models/*.pt --source NEU-DET/test/images

输出: docs/benchmark_report.md (自动汇总所有模型的对比表)
"""
import argparse
import sys
import time
from pathlib import Path

from common import PROJECT_ROOT, check_weight_compat, get_device, model_efficiency, write_report

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DEFAULT_SOURCE = PROJECT_ROOT / "NEU-DET" / "test" / "images"


def collect_images(source, limit=64):
    src = Path(source)
    if not src.exists():
        src = DEFAULT_SOURCE
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if src.is_dir():
        imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in exts)[:limit]
    else:
        imgs = [src]
    if not imgs:
        raise SystemExit(f"未找到测试图片: {source}")
    return imgs


def benchmark_one(weights, imgs, n, device, imgsz=640):
    path = Path(weights)
    if not path.exists():
        print(f"[跳过] 文件不存在: {path}")
        return None
    if path.suffix == ".engine":
        try:
            import tensorrt  # noqa: F401
        except ImportError:
            print(f"[跳过] {path.name}: 未安装 tensorrt")
            return None

    if path.suffix == ".pt":
        check_weight_compat(path)

    model = YOLO(str(path))
    stages = {"preprocess": [], "inference": [], "postprocess": []}
    total = []
    n = min(n, len(imgs))

    for img in imgs[:n]:
        t0 = time.perf_counter()
        results = model.predict(source=str(img), imgsz=imgsz, device=device,
                                conf=0.25, verbose=False)
        total.append((time.perf_counter() - t0) * 1000)
        sp = results[0].speed
        for k in stages:
            stages[k].append(sp.get(k, 0.0))

    rows = {k: sum(v) / len(v) for k, v in stages.items()}
    rows["total"] = sum(total) / len(total)
    rows["fps"] = 1000.0 / rows["total"] if rows["total"] > 0 else 0.0

    rows["params_m"] = rows["gflops"] = None
    if path.suffix == ".pt":
        rows["params_m"], rows["gflops"] = model_efficiency(model)

    rows["size_mb"] = path.stat().st_size / 1e6

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        model.predict(source=str(imgs[0]), imgsz=imgsz, device=device, verbose=False)
        rows["gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6
    else:
        rows["gpu_mem_mb"] = None

    backend = {".pt": "PyTorch", ".onnx": "ONNXRuntime", ".engine": "TensorRT"}[path.suffix]
    rows.update({"name": path.name, "backend": backend, "device": device})
    return rows


def render_report(results):
    lines = [
        "# 全链路推理基准报告",
        "",
        f"- 测试硬件: `{get_device()}` (GPU 型号未检测到则为 CPU 环境)",
        "- 测试配置: imgsz=640, batch=1, conf=0.25, 每模型平均 N 次推理",
        "- 全链路延迟 = 预处理(letterbox+归一化) + 推理 + 后处理(NMS)",
        "",
        "| 模型 | 后端 | 预处理(ms) | 推理(ms) | 后处理(ms) | 总延迟(ms) | FPS | 参数量(M) | GFLOPs | 文件体积(MB) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['backend']} | {r['preprocess']:.2f} | {r['inference']:.2f} | "
            f"{r['postprocess']:.2f} | {r['total']:.2f} | {r['fps']:.1f} | "
            f"{r['params_m'] or '-'} | {r['gflops'] or '-'} | {r['size_mb']:.2f} |"
        )
    lines += [
        "",
        "> 注: PyTorch 后端延迟含 Python 调度开销; ONNX/TensorRT 后端更接近真实部署延迟。",
        "> 剪枝/蒸馏前后请分别跑一次本脚本, 用同一批图片和相同 N 对比。",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="全链路推理基准测试")
    parser.add_argument("--weights", nargs="+", required=True, help="一个或多个模型文件 (.pt/.onnx/.engine)")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="测试图片目录")
    parser.add_argument("--n", type=int, default=30, help="每个模型的推理次数")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="默认自动检测 cuda/cpu")
    args = parser.parse_args()

    device = args.device or get_device()
    imgs = collect_images(args.source, limit=max(args.n, 8))
    print(f"测试图片: {len(imgs)} 张, 设备: {device}, 每模型 {args.n} 次推理")

    results = []
    for w in args.weights:
        r = benchmark_one(w, imgs, args.n, device, args.imgsz)
        if r is not None:
            results.append(r)
            print(f"  {r['name']:24s} [{r['backend']:10s}] 总延迟 {r['total']:.2f} ms  "
                  f"FPS {r['fps']:.1f}  推理 {r['inference']:.2f} ms")

    if not results:
        raise SystemExit("没有可用的模型结果")

    report = render_report(results)
    write_report("benchmark_report.md", report)


if __name__ == "__main__":
    main()
