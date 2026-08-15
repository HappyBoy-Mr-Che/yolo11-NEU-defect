"""难例挖掘 (badcase 回流闭环): 漏检/误检/低置信度样本自动裁剪归档。

流程:
  1. model.val(save_json=True) 得到全量预测
  2. 预测框与真值按 IoU>=0.5 (同类) 贪心匹配
  3. 分类归档:
     - 漏检 FN: 真值无预测匹配 -> badcase/fn/<类别>/
     - 误检 FP: 预测无真值匹配 -> badcase/fp/<类别>/
     - 低置信 TP: 匹配成功但 conf < 阈值 -> badcase/low_conf/<类别>/
  4. 同时保存整图标注可视化 (绿=GT, 红=预测) 到 badcase/overview/
  5. 输出 badcase/badcase_summary.csv + docs/badcase_report.md

用法:
    python tools/badcase_mining.py --weights models/best.pt --split val
    python tools/badcase_mining.py --weights models/best.pt --low-conf 0.35 --limit 200

人工复核后把裁剪图放回训练集重训, 形成数据回流闭环。
"""
import argparse
import csv
import glob
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from common import PROJECT_ROOT, get_device, write_report

DEFAULT_DATA = PROJECT_ROOT / "ultralytics" / "cfg" / "datasets" / "NUE_DET.yaml"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

GREEN, RED, BLUE = (0, 200, 0), (220, 40, 40), (30, 90, 220)


def box_iou(a, b):
    """a, b 为 [x1, y1, x2, y2]。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def yolo_to_xyxy(cx, cy, w, h, W, H):
    x1 = max(0.0, (cx - w / 2) * W)
    y1 = max(0.0, (cy - h / 2) * H)
    x2 = min(float(W), (cx + w / 2) * W)
    y2 = min(float(H), (cy + h / 2) * H)
    return [x1, y1, x2, y2]


def load_gt(lab_path, W, H):
    if not lab_path.exists():
        return []
    gts = []
    for line in lab_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        box = yolo_to_xyxy(*[float(x) for x in parts[1:]], W, H)
        gts.append((cls, box))
    return gts


def match(gts, preds, iou_thr):
    """贪心匹配, 返回 (matched_pairs, unmatched_gt_idx, unmatched_pred_idx)。"""
    matched = []
    used_p = set()
    for gi, (gc, gb) in enumerate(gts):
        best = None
        for pi, (pc, pb, _) in enumerate(preds):
            if pi in used_p or pc != gc:
                continue
            iou = box_iou(gb, pb)
            if iou >= iou_thr and (best is None or iou > best[0]):
                best = (iou, pi)
        if best:
            used_p.add(best[1])
            matched.append((gi, best[1]))
    unmatched_gt = [i for i in range(len(gts)) if i not in {m[0] for m in matched}]
    unmatched_pd = [i for i in range(len(preds)) if i not in {m[1] for m in matched}]
    return matched, unmatched_gt, unmatched_pd


def crop_and_save(img, box, out_path, margin=0.1):
    W, H = img.size
    x1, y1, x2, y2 = box
    mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
    box_c = (max(0, x1 - mw), max(0, y1 - mh), min(W, x2 + mw), min(H, y2 + mh))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.crop(box_c).save(out_path)
    return out_path


def draw_overview(img, gts, preds):
    d = ImageDraw.Draw(img)
    for cls, box in gts:
        d.rectangle(box, outline=GREEN, width=2)
        d.text((box[0], max(0, box[1] - 12)), f"GT:{cls}", fill=GREEN)
    for cls, box, conf in preds:
        d.rectangle(box, outline=RED, width=2)
        d.text((box[0], max(0, box[3])), f"P:{cls}({conf:.2f})", fill=RED)
    return img


def find_predictions_json():
    cands = sorted(glob.glob(str(PROJECT_ROOT / "ultralytics" / "runs" / "detect" / "**" / "predictions.json"),
                             recursive=True), key=lambda p: Path(p).stat().st_mtime)
    return Path(cands[-1]) if cands else None


def main():
    parser = argparse.ArgumentParser(description="难例挖掘: 漏检/误检/低置信样本归档")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--split", default="val")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--low-conf", type=float, default=0.35, help="低于该置信度的 TP 记为低置信难例")
    parser.add_argument("--limit", type=int, default=0, help="只分析前 N 张图, 0=全部")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "badcase"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or get_device()
    out = Path(args.out)
    d = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    names = list(d.get("names", []))
    split_dir = Path(d[args.split])

    print(f"模型验证 (save_json) ...")
    from ultralytics import YOLO
    ts = time.strftime("%Y%m%d_%H%M%S")
    m = YOLO(args.weights)
    m.val(data=args.data, split=args.split, imgsz=640, device=device, conf=args.conf,
          save_json=True, plots=False, project="ultralytics/runs/badcase", name=f"mine_{ts}", exist_ok=True)

    pred_json = find_predictions_json()
    if pred_json is None or ts not in str(pred_json):
        raise SystemExit("[错误] 未找到本次 predictions.json, 请确认 save_json 已生效")
    print(f"预测文件: {pred_json}")

    import json
    preds_by_img = defaultdict(list)
    for p in json.loads(pred_json.read_text(encoding="utf-8")):
        preds_by_img[p["image_id"]].append((int(p["category_id"]), p["bbox"], float(p["score"])))

    imgs = sorted(x for x in split_dir.iterdir() if x.suffix.lower() in IMG_EXTS)
    if args.limit:
        imgs = imgs[:args.limit]
    lab_dir = split_dir.parent / "labels"

    rows = []
    stats = defaultdict(Counter)
    for i, img_path in enumerate(imgs):
        with Image.open(img_path) as im:
            W, H = im.size
        gts = load_gt(lab_dir / (img_path.stem + ".txt"), W, H)
        preds = [(c, b, s) for c, b, s in preds_by_img.get(img_path.stem, [])]
        matched, un_gt, un_pd = match(gts, preds, args.iou)

        if not (un_gt or un_pd) and not any(s < args.low_conf for _, _, s in
                                           [preds[j] for _, j in matched]):
            continue

        with Image.open(img_path) as im:
            overview = None
            for gi in un_gt:
                cls, box = gts[gi]
                name = names[cls] if cls < len(names) else str(cls)
                f = crop_and_save(im, box, out / "fn" / name / f"{img_path.stem}_{gi}.jpg")
                rows.append([img_path.name, "FN", name, "", box, str(f)])
                stats["FN"][name] += 1
            for pi in un_pd:
                cls, box, conf = preds[pi]
                name = names[cls] if cls < len(names) else str(cls)
                f = crop_and_save(im, box, out / "fp" / name / f"{img_path.stem}_{pi}.jpg")
                rows.append([img_path.name, "FP", name, f"{conf:.3f}", box, str(f)])
                stats["FP"][name] += 1
            for gi, pi in matched:
                cls, box, conf = preds[pi]
                if conf >= args.low_conf:
                    continue
                name = names[cls] if cls < len(names) else str(cls)
                f = crop_and_save(im, box, out / "low_conf" / name / f"{img_path.stem}_{pi}.jpg")
                rows.append([img_path.name, "LOW_CONF", name, f"{conf:.3f}", box, str(f)])
                stats["LOW_CONF"][name] += 1
            if un_gt or un_pd:
                overview = draw_overview(im, gts, preds)

        if overview is not None:
            ov_dir = out / "overview"
            ov_dir.mkdir(parents=True, exist_ok=True)
            overview.save(ov_dir / img_path.name)

        if (i + 1) % 50 == 0:
            print(f"  分析进度: {i + 1}/{len(imgs)}")

    csv_path = out / "badcase_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["原图", "类型", "类别", "置信度", "bbox_xyxy", "裁剪图路径"])
        w.writerows(rows)

    report = ["# 难例挖掘报告", "",
              f"- 模型: `{args.weights}`, 划分: {args.split}, conf={args.conf}, IoU 阈值={args.iou}",
              f"- 低置信阈值: {args.low_conf}, 分析图片数: {len(imgs)}", "",
              "| 类型 | 类别 | 数量 |", "|---|---|---|"]
    total = 0
    for t in ("FN", "FP", "LOW_CONF"):
        for c, n in sorted(stats[t].items()):
            report.append(f"| {t} | {c} | {n} |")
            total += n
    report += ["", f"**难例总数: {total}** (CSV: `{csv_path}`)", "",
               "> 复核建议: 先看 overview/ 整图确认标注正确性, 再决定裁剪图是否放回训练集。",
               "> 回流方式: 把确认的裁剪图+标签加入训练集目录重新训练即可。"]
    write_report("badcase_report.md", "\n".join(report))
    print(f"\n难例归档完成: {total} 个 -> {out}")


if __name__ == "__main__":
    main()
