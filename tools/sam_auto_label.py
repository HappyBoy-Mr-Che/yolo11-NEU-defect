"""SAM 辅助预标注: 检测粗框 -> SAM 精修 mask -> YOLO 标签 + 可视化审核。

流程:
  1. 改进版检测模型对无标注图片推理, 得到粗框 + 类别
  2. 粗框作为 box prompt 送入 ultralytics SAM (sam_b.pt, 首次运行自动下载)
  3. SAM 输出精修 mask, 取 mask 外接框生成 YOLO 格式标签
  4. 生成 mask 叠加可视化图供人工审核 (out/vis/), 审核通过后直接入库训练

用法:
    python tools/sam_auto_label.py --source dataset/unlabeled --det-model models/best.pt
    python tools/sam_auto_label.py --source dataset/unlabeled --conf 0.35 --det-only  # 只用检测框

输出: <out>/labels/*.txt (YOLO 格式) + <out>/vis/*.jpg (叠加可视化)
      以及 docs/sam_auto_label_report.md (含"无检测"图片清单, 需人工重点检查)
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from common import PROJECT_ROOT, get_device, write_report

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_COLORS = [(255, 60, 60), (60, 180, 255), (255, 200, 60), (120, 220, 90),
                (200, 90, 255), (90, 255, 200), (255, 120, 200), (150, 150, 255)]


def mask_to_box(mask):
    """mask: [H, W] bool numpy -> 归一化 (cx, cy, w, h)。"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    H, W = mask.shape
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    return cx, cy, (x2 - x1) / W, (y2 - y1) / H


def draw_vis(img, masks, classes, names):
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for mask, cls in zip(masks, classes):
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        d.rectangle([x1, y1, x2, y2], outline=color + (255,), width=2)
        d.text((x1, max(0, y1 - 14)), f"{names[cls] if cls < len(names) else cls}", fill=color + (255,))
        pts = list(zip(xs, ys))
        for pt in random.sample(pts, min(800, len(pts))):
            d.point(pt, fill=color + (110,))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="SAM 辅助预标注")
    parser.add_argument("--source", required=True, help="无标注图片目录")
    parser.add_argument("--det-model", default=str(PROJECT_ROOT / "models" / "best.pt"),
                        help="检测模型权重 (提供粗框)")
    parser.add_argument("--sam-model", default="sam_b.pt", help="SAM 权重 (缺失时自动下载)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "auto_label"))
    parser.add_argument("--det-only", action="store_true", help="跳过 SAM, 直接用检测框生成标签")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or get_device()
    src = Path(args.source)
    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise SystemExit(f"[错误] {src} 中没有图片")

    out_lab = Path(args.out) / "labels"
    out_vis = Path(args.out) / "vis"
    out_lab.mkdir(parents=True, exist_ok=True)
    out_vis.mkdir(parents=True, exist_ok=True)

    from ultralytics import SAM, YOLO

    print(f"加载检测模型: {args.det_model}")
    det = YOLO(args.det_model)
    names = det.names
    sam = None if args.det_only else SAM(args.sam_model)
    print(f"SAM 模型: {'跳过 (--det-only)' if sam is None else args.sam_model}")

    per_class = Counter()
    empty_imgs = []
    n_masks = 0
    for i, p in enumerate(imgs):
        det_res = det.predict(source=str(p), imgsz=args.imgsz, device=device,
                              conf=args.conf, verbose=False)[0]
        boxes = det_res.boxes.xyxy.cpu().numpy().tolist()
        classes = det_res.boxes.cls.int().cpu().numpy().tolist()
        if not boxes:
            empty_imgs.append(p.name)
            (out_lab / (p.stem + ".txt")).write_text("", encoding="utf-8")
            continue

        with Image.open(p) as im:
            W, H = im.size

        masks, out_boxes = [], []
        if sam is not None:
            try:
                sam_res = sam(p, bboxes=boxes, device=device, verbose=False)[0]
                mask_data = sam_res.masks.data.cpu().numpy().astype(bool)
                if mask_data.shape[0] != len(boxes):
                    print(f"  [提示] {p.name}: SAM 返回 {mask_data.shape[0]} 个 mask, "
                          f"期望 {len(boxes)} 个, 按顺序截取")
                masks = [mask_data[i] for i in range(min(len(boxes), mask_data.shape[0]))]
            except Exception as e:
                print(f"  [警告] {p.name}: SAM 失败 ({e}), 退回检测框")
                masks = []
        if len(masks) != len(boxes):
            masks = []

        lines = []
        for bi, (box, cls) in enumerate(zip(boxes, classes)):
            if masks:
                b = mask_to_box(masks[bi])
                n_masks += 1
            else:
                x1, y1, x2, y2 = box
                b = ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)
            if b is None:
                continue
            lines.append(f"{cls} {' '.join(f'{v:.6f}' for v in b)}\n")
            per_class[cls] += 1

        (out_lab / (p.stem + ".txt")).write_text("".join(lines), encoding="utf-8")

        if masks:
            with Image.open(p) as im:
                vis = draw_vis(im.convert("RGB"), masks, classes, names)
            vis.save(out_vis / p.name)

        if (i + 1) % 20 == 0:
            print(f"  进度: {i + 1}/{len(imgs)}")

    report = ["# SAM 辅助预标注报告", "",
              f"- 检测模型: `{args.det_model}`, SAM: `{'跳过' if sam is None else args.sam_model}`",
              f"- 图片总数: {len(imgs)}, 生成标签: {len(list(out_lab.glob('*.txt')))} 个", "",
              "| 类别 | 预标注框数 |", "|---|---|"]
    for cls, n in sorted(per_class.items()):
        report.append(f"| {names[cls] if cls < len(names) else cls} | {n} |")
    report += ["", f"## 无检测图片 ({len(empty_imgs)} 张, 需人工重点检查)", ""]
    for name in empty_imgs:
        report.append(f"- `{name}`")
    report += ["",
               f"> 输出: `{args.out}` (labels/ + vis/)",
               "> 审核流程: 对照 vis/ 叠加图逐张检查 mask 贴合度, 修正后把 images 与 labels 一并入库训练。",
               "> 首次运行需联网下载 sam_b.pt (~375MB), 也可提前放置到 models/ 并改 --sam-model 路径。"]
    write_report("sam_auto_label_report.md", "\n".join(report))


if __name__ == "__main__":
    main()
