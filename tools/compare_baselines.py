"""跨模型对比实验: 统一超参训练 yolov8n / yolo11n / 改进版 / RT-DETR-l 于 NEU-DET。

用法:
    python tools/compare_baselines.py                 # 逐个训练(已训过自动跳过)+验证+测延迟
    python tools/compare_baselines.py --only-val      # 只用已有 best.pt 验证
    python tools/compare_baselines.py --epochs 10 --force  # 快速重训调试
    python tools/compare_baselines.py --batch 4       # RT-DETR-l 显存不足时减小 batch

说明:
  - yolov8n.yaml / yolo11n.yaml 由 ultralytics 自动映射到本地 yolo11.yaml + scale n
    (见 ultralytics/nn/tasks.py yaml_model_load), rtdetr-l.yaml 为本仓已复制的官方配置
  - 所有模型统一超参 (SGD lr0=1e-4, 与 train.py 一致), 保证对比公平

训练产物: ultralytics/runs/compare/<模型名>/weights/best.pt
输出: docs/compare_report.md
"""
import argparse
import sys
from pathlib import Path

from common import PROJECT_ROOT, build_train_kwargs, get_device, model_efficiency, write_report

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402

CFG_DIR = PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "11"
DATA = PROJECT_ROOT / "ultralytics" / "cfg" / "datasets" / "NUE_DET.yaml"
RUNS_DIR = PROJECT_ROOT / "ultralytics" / "runs" / "compare"
TEST_IMGS = PROJECT_ROOT / "NEU-DET" / "test" / "images"

# (名称, 模型标识, 说明)
MODELS = [
    ("yolov8n", "yolov8n.yaml", "YOLOv8n 基线"),
    ("yolo11n", "yolo11n.yaml", "YOLOv11n 基线"),
    ("yolo11_ours", str(CFG_DIR / "YOLOV11.yaml"), "改进版 YOLO11n: C3k2_IDC + SAFMNPP"),
    ("rtdetr_l", "rtdetr-l.yaml", "RT-DETR-l (Transformer 实时检测, 大模型对照)"),
]


def train_model(name, model_id, args):
    save_dir = RUNS_DIR / name
    best_pt = save_dir / "weights" / "best.pt"
    if best_pt.exists() and not args.force:
        print(f"[跳过训练] {name} 已有权重: {best_pt}")
        return best_pt
    if args.only_val:
        raise SystemExit(f"[错误] --only-val 模式下 {name} 无权重: {best_pt}")

    print(f"\n===== 训练 {name} ({model_id}) =====")
    kwargs = build_train_kwargs(str(DATA), args.epochs, args.batch, args.device, name, str(RUNS_DIR))
    YOLO(model_id).train(**kwargs)

    if not best_pt.exists():
        raise SystemExit(f"[错误] 训练产物不存在: {best_pt}")
    return best_pt


def val_model(best_pt, args):
    m = YOLO(str(best_pt))
    r = m.val(data=str(DATA), imgsz=640, device=args.device, plots=False, verbose=False)
    params_m, gflops = model_efficiency(m)
    return {
        "map50": r.box.map50,
        "map": r.box.map,
        "p": r.box.mp,
        "r": r.box.mr,
        "params_m": params_m,
        "gflops": gflops,
        "size_mb": best_pt.stat().st_size / 1e6,
    }


def measure_latency(best_pt, args):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    src = TEST_IMGS if TEST_IMGS.exists() else PROJECT_ROOT / "NEU-DET" / "val" / "images"
    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in exts)[:args.n_latency]
    if not imgs:
        return None
    m = YOLO(str(best_pt))
    times = []
    for img in imgs:
        res = m.predict(source=str(img), imgsz=640, device=args.device, conf=0.25, verbose=False)
        times.append(res[0].speed.get("inference", 0.0))
    return sum(times) / len(times)


def render_report(rows, args):
    lines = [
        "# 跨模型对比实验报告",
        "",
        f"- 数据集: NEU-DET (nc=6), imgsz=640, epochs={args.epochs}, batch={args.batch}, "
        "SGD lr0=1e-4, pretrained=False (统一超参公平对比)",
        f"- 延迟测试: {args.n_latency} 张测试图, batch=1, 取均值 (设备: {args.device})",
        "",
        "| 模型 | mAP@0.5 | mAP@0.5:0.95 | P | R | 参数量(M) | GFLOPs | 推理(ms) | FPS | 权重体积(MB) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, _, _ in MODELS:
        r = rows[name]
        fps = 1000.0 / r["latency"] if r["latency"] else None
        lines.append(
            f"| {name} | {r['map50']:.4f} | {r['map']:.4f} | {r['p']:.4f} | {r['r']:.4f} | "
            f"{r['params_m']:.3f} | {r['gflops'] or '-'} | "
            f"{r['latency']:.2f} | {fps:.1f} | {r['size_mb']:.2f} |"
        )
    lines += ["",
              "> 注: RT-DETR-l 参数量远超 YOLO11n 系列, 仅作精度上限参考; 推理延迟受设备影响, "
              "部署数据以 docs/benchmark_report.md (含 ONNX/TensorRT) 为准。"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="跨模型对比实验")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8, help="RT-DETR-l 显存不足可降到 4")
    parser.add_argument("--device", default=None, help="默认自动检测 cuda/cpu")
    parser.add_argument("--n-latency", type=int, default=16, help="延迟测试图片数")
    parser.add_argument("--force", action="store_true", help="已有权重也重新训练")
    parser.add_argument("--only-val", action="store_true", help="只验证已有权重, 不训练")
    args = parser.parse_args()

    args.device = args.device or get_device()

    rows = {}
    for name, model_id, _ in MODELS:
        best_pt = train_model(name, model_id, args)
        rows[name] = val_model(best_pt, args)
        rows[name]["latency"] = measure_latency(best_pt, args)
        r = rows[name]
        print(f"  {name}: mAP@0.5={r['map50']:.4f}  mAP@0.5:0.95={r['map']:.4f}  "
              f"params={r['params_m']:.3f}M  latency={r['latency']:.2f}ms")

    write_report("compare_report.md", render_report(rows, args))


if __name__ == "__main__":
    main()
