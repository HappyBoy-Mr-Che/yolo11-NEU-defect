"""小缺陷 Copy-Paste 离线增强 (检测版)。

原理: 从源图按标签裁剪缺陷 patch (含随机边距), 随机旋转/缩放/亮度抖动后,
粘贴到目标图随机位置 (与现有标注框 IoU <= --max-iou 才接受), 生成新图 + 合并标签。
解决自建数据集小缺陷样本不足的问题 (ultralytics 8.3.2 的 copy_paste 参数仅支持分割任务)。

用法:
    python tools/copy_paste_augment.py --data dataset/data.yaml --multiplier 1
    python tools/copy_paste_augment.py --data dataset/data.yaml --multiplier 2 --patches 1 3 \
        --scale 0.5 1.5 --max-iou 0.1 --out dataset/augmented

输出: <out>/images/ + <out>/labels/ (YOLO 格式, 可直接合并进训练集)
      以及 docs/copy_paste_report.md
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageEnhance

from common import PROJECT_ROOT, write_report

DEFAULT_DATA = PROJECT_ROOT / "ultralytics" / "cfg" / "datasets" / "NUE_DET.yaml"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_img_labels(img_dir, lab_dir):
    out = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lab = lab_dir / (img.stem + ".txt")
        boxes = []
        if lab.exists():
            for line in lab.read_text(encoding="utf-8").strip().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    boxes.append((int(float(parts[0])), [float(x) for x in parts[1:]]))
        out.append((img, boxes))
    return out


def crop_patch(src_img, src_box, margin=0.3):
    """按 YOLO 框裁剪 patch (含随机边距), 返回 (patch, patch 相对原图归一化框)。"""
    W, H = src_img.size
    cls, (cx, cy, w, h) = src_box
    m = margin * random.uniform(0.5, 1.0)
    x1 = max(0.0, (cx - w / 2 - w * m) * W)
    y1 = max(0.0, (cy - h / 2 - h * m) * H)
    x2 = min(float(W), (cx + w / 2 + w * m) * W)
    y2 = min(float(H), (cy + h / 2 + h * m) * H)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, None
    patch = src_img.crop((x1, y1, x2, y2))
    return patch, (x1 / W, y1 / H, x2 / W, y2 / H)


def transform_patch(patch, scale_range):
    """随机旋转/缩放/亮度抖动, 返回 (patch, 目标框归一化宽高)。"""
    ang = random.uniform(-30, 30)
    patch = patch.rotate(ang, expand=True, resample=Image.Resampling.BICUBIC)
    ow, oh = patch.size
    rad = np.deg2rad(abs(ang))
    rw = ow * np.cos(rad) + oh * np.sin(rad)
    rh = ow * np.sin(rad) + oh * np.cos(rad)
    scale = random.uniform(*scale_range)
    fw, fh = rw * scale, rh * scale
    patch = patch.resize((max(2, int(fw)), max(2, int(fh))), Image.Resampling.BICUBIC)
    for name, lo, hi in (("Brightness", 0.85, 1.2), ("Contrast", 0.85, 1.15), ("Color", 0.8, 1.2)):
        patch = getattr(ImageEnhance, name)(patch).enhance(random.uniform(lo, hi))
    return patch, fw, fh


def normalized_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def try_paste(target, patch, boxes, max_iou, attempts=20):
    """随机找不与现有框严重重叠的位置, 返回 (成功, 新框)。"""
    W, H = target.size
    pw, ph = patch.size
    if pw >= W * 0.8 or ph >= H * 0.8:
        return False, None
    for _ in range(attempts):
        px = random.uniform(0, W - pw)
        py = random.uniform(0, H - ph)
        nb = (px / W, py / H, (px + pw) / W, (py + ph) / H)
        if all(normalized_iou(nb, b) <= max_iou for _, b in boxes):
            target.paste(patch, (int(px), int(py)))
            return True, nb
    return False, None


def main():
    parser = argparse.ArgumentParser(description="缺陷 Copy-Paste 离线增强")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--split", default="train")
    parser.add_argument("--multiplier", type=int, default=1, help="每张图生成的增强副本数")
    parser.add_argument("--patches", nargs=2, type=int, default=[1, 3], help="每张副本粘贴 patch 数范围")
    parser.add_argument("--scale", nargs=2, type=float, default=[0.5, 1.5], help="patch 缩放范围 (相对原大小)")
    parser.add_argument("--max-iou", type=float, default=0.1, help="与现有框允许的最大 IoU")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "dataset" / "augmented"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    d = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    img_dir = Path(d[args.split])
    lab_dir = img_dir.parent / "labels"
    pool = load_img_labels(img_dir, lab_dir)
    if len(pool) < 2:
        raise SystemExit("[错误] 源图不足 2 张, 无法 Copy-Paste")

    out_img = Path(args.out) / "images"
    out_lab = Path(args.out) / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    added = Counter()
    n_pasted = 0
    for ti, (timg_path, tboxes) in enumerate(pool):
        for rep in range(args.multiplier):
            with Image.open(timg_path) as im:
                target = im.convert("RGB")
            boxes = [b for b in tboxes]
            k = random.randint(*args.patches)
            ok = 0
            for _ in range(k * 2):
                if ok >= k:
                    break
                simg_path, sboxes = random.choice(pool)
                if simg_path == timg_path or not sboxes:
                    continue
                cls, sbox = random.choice(sboxes)
                with Image.open(simg_path) as im:
                    patch, _ = crop_patch(im, (cls, sbox))
                if patch is None:
                    continue
                patch, _, _ = transform_patch(patch, args.scale)
                done, nb = try_paste(target, patch, boxes, args.max_iou)
                if done:
                    boxes.append((cls, nb))
                    added[cls] += 1
                    ok += 1
                    n_pasted += 1
            if ok == 0:
                continue
            name = f"{timg_path.stem}_cp{rep}"
            target.save(out_img / f"{name}.jpg", quality=95)
            (out_lab / f"{name}.txt").write_text(
                "".join(f"{c} {' '.join(f'{v:.6f}' for v in b)}\n" for c, b in boxes), encoding="utf-8")
        if (ti + 1) % 50 == 0:
            print(f"  进度: {ti + 1}/{len(pool)} 张")

    names = list(d.get("names", []))
    report = ["# Copy-Paste 增强报告", "",
              f"- 源数据集: `{args.data}` ({args.split}, {len(pool)} 张)", "",
              f"- 生成图片数: {len(list(out_img.iterdir()))}, 粘贴 patch 总数: {n_pasted}", "",
              "| 类别 | 新增实例数 |", "|---|---|"]
    for cls, n in sorted(added.items()):
        report.append(f"| {names[cls] if cls < len(names) else cls} | {n} |")
    report += ["",
               f"> 输出: `{args.out}` (images/ + labels/)",
               "> 合并方式: 把 images/* 与 labels/* 复制进训练集对应目录, 重新训练即可。",
               "> 注意: 增强图保留原图全部标注框 + 新增粘贴框, 类别索引与原数据集一致。"]
    write_report("copy_paste_report.md", "\n".join(report))


if __name__ == "__main__":
    main()
