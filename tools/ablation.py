"""消融实验: 逐模块叠加训练 + 验证, 汇总 mAP / 参数量 / GFLOPs 对比。

变体链 (统一 scale=n, 超参与 ultralytics/train.py 一致):
  A baseline      = yolo11.yaml          (标准 C3k2, P3-P5)
  B +C3k2_IDC     = yolo11.yaml 全部 C3k2 -> C3k2_IDC (自动生成)
  C +SAFMNPP      = yolo11.yaml P4->P3 上采样替换为 SAFMNPP (自动生成)
  D +IDC+SAFMNPP  = YOLOV11.yaml
  E 完整版        = Yolov11_EMA.yaml     (+P2 小目标检测层 + SCSABlock 通道注意力)

用法:
    python tools/ablation.py                    # 生成 B/C YAML, 逐个训练(已训过自动跳过)+验证
    python tools/ablation.py --only-val         # 只用已有 best.pt 验证, 不训练
    python tools/ablation.py --epochs 10 --force  # 全部重训 10 轮 (快速调试)

训练产物: ultralytics/runs/ablation/<变体名>/weights/best.pt
输出: docs/ablation_report.md
"""
import argparse
import sys
from pathlib import Path

from common import PROJECT_ROOT, build_train_kwargs, get_device, model_efficiency, write_report

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402

CFG_DIR = PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "11"
ABLATION_DIR = CFG_DIR / "ablation"
DATA = PROJECT_ROOT / "ultralytics" / "cfg" / "datasets" / "NUE_DET.yaml"
RUNS_DIR = PROJECT_ROOT / "ultralytics" / "runs" / "ablation"

# (名称, YAML 路径或 None=自动生成, 说明)
VARIANTS = [
    ("A_baseline", str(CFG_DIR / "yolo11.yaml"), "baseline: 标准 YOLO11n (C3k2, P3-P5)"),
    ("B_c3k2_idc", str(ABLATION_DIR / "yolo11_idc.yaml"),
     "A + C3k2_IDC (DASConv 多尺度空洞非对称卷积 + iAFF 迭代注意力融合)"),
    ("C_safmnpp", str(ABLATION_DIR / "yolo11_safmnpp.yaml"),
     "A + SAFMNPP 空间自适应特征调制上采样"),
    ("D_idc_safmnpp", str(CFG_DIR / "YOLOV11.yaml"), "B + C (C3k2_IDC + SAFMNPP)"),
    ("E_full", str(CFG_DIR / "Yolov11_EMA.yaml"), "D + P2 小目标检测层 + SCSABlock 通道注意力"),
]

_NEW_BLOCK = "  - [-1, 1, SAFMNPP, [512]]\n  - [[-1, 4], 1, Concat, [1]] # cat backbone P3"


def generate_variants():
    """生成 B/C 变体 YAML 到 cfg/models/11/ablation/。"""
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    base = (CFG_DIR / "yolo11.yaml").read_text(encoding="utf-8")

    b = base.replace("C3k2,", "C3k2_IDC,")
    assert "C3k2_IDC" in b, "C3k2 -> C3k2_IDC 替换失败"
    b = "# 消融实验自动生成: B = baseline + C3k2_IDC (由 tools/ablation.py 生成, 勿手改)\n" + b
    (ABLATION_DIR / "yolo11_idc.yaml").write_text(b, encoding="utf-8")

    block = ('  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]\n'
             '  - [[-1, 4], 1, Concat, [1]] # cat backbone P3')
    assert block in base, "未找到 P4->P3 上采样块"
    c = base.replace(block, _NEW_BLOCK)
    c = "# 消融实验自动生成: C = baseline + SAFMNPP (由 tools/ablation.py 生成, 勿手改)\n" + c
    (ABLATION_DIR / "yolo11_safmnpp.yaml").write_text(c, encoding="utf-8")
    print(f"已生成变体 YAML: {ABLATION_DIR / 'yolo11_idc.yaml'}, {ABLATION_DIR / 'yolo11_safmnpp.yaml'}")


def train_variant(name, yaml_path, args):
    save_dir = RUNS_DIR / name
    best_pt = save_dir / "weights" / "best.pt"
    if best_pt.exists() and not args.force:
        print(f"[跳过训练] {name} 已有权重: {best_pt}")
        return best_pt
    if args.only_val:
        raise SystemExit(f"[错误] --only-val 模式下 {name} 无权重: {best_pt}")

    print(f"\n===== 训练 {name}: {yaml_path} =====")
    kwargs = build_train_kwargs(str(DATA), args.epochs, args.batch, args.device, name, str(RUNS_DIR))
    YOLO(yaml_path).train(**kwargs)

    if not best_pt.exists():
        raise SystemExit(f"[错误] 训练产物不存在: {best_pt}")
    return best_pt


def val_variant(best_pt, args):
    m = YOLO(str(best_pt))
    r = m.val(data=str(DATA), imgsz=640, device=args.device, plots=False, verbose=False)
    params_m, gflops = model_efficiency(m)
    names = [r.names.get(i, str(i)) for i in range(len(r.box.ap50))]
    return {
        "map50": r.box.map50,
        "map": r.box.map,
        "p": r.box.mp,
        "r": r.box.mr,
        "ap50_cls": dict(zip(names, r.box.ap50)),
        "params_m": params_m,
        "gflops": gflops,
    }


def render_report(rows, args):
    base = rows[0]
    classes = sorted({c for row in rows for c in row["ap50_cls"]})
    lines = [
        "# 消融实验报告",
        "",
        f"- 数据集: NEU-DET (nc=6), imgsz=640, epochs={args.epochs}, batch={args.batch}, "
        "SGD lr0=1e-4, pretrained=False (与 train.py 统一超参)",
        "- 变体链: A(基线) -> B(+C3k2_IDC) -> C(+SAFMNPP) -> D(B+C) -> E(+P2+SCSABlock)",
        "",
        "| 变体 | 说明 | mAP@0.5 | ΔmAP@0.5 | mAP@0.5:0.95 | P | R | 参数量(M) | GFLOPs |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, _, desc in VARIANTS:
        row = rows[name]
        d50 = row["map50"] - base["map50"]
        lines.append(
            f"| {name} | {desc} | {row['map50']:.4f} | {d50:+.4f} | {row['map']:.4f} | "
            f"{row['p']:.4f} | {row['r']:.4f} | {row['params_m']:.3f} | {row['gflops'] or '-'} |"
        )

    lines += ["", "## 每类 AP@0.5", "",
              "| 变体 | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
    for name, _, _ in VARIANTS:
        row = rows[name]
        lines.append("| " + name + " | " + " | ".join(f"{row['ap50_cls'].get(c, 0):.4f}" for c in classes) + " |")

    lines += ["",
              "> 注: 训练超参与 `ultralytics/train.py` 完全一致 (SGD, lr0=1e-4, cache=disk, "
              "close_mosaic=64)。",
              "> 最终模型建议选择精度/参数量折中最优的变体, 再走剪枝/蒸馏/导出流程。"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="消融实验: 改进模块逐项叠加对比")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="默认自动检测 cuda/cpu")
    parser.add_argument("--force", action="store_true", help="已有权重也重新训练")
    parser.add_argument("--only-val", action="store_true", help="只验证已有权重, 不训练")
    args = parser.parse_args()

    args.device = args.device or get_device()
    generate_variants()

    rows = {}
    for name, yaml_path, _ in VARIANTS:
        best_pt = train_variant(name, yaml_path, args)
        rows[name] = val_variant(best_pt, args)
        r = rows[name]
        print(f"  {name}: mAP@0.5={r['map50']:.4f}  mAP@0.5:0.95={r['map']:.4f}  "
              f"params={r['params_m']:.3f}M  GFLOPs={r['gflops']}")

    write_report("ablation_report.md", render_report(rows, args))


if __name__ == "__main__":
    main()
