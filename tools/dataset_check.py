"""数据集健康检查: 标签/图片配对、格式合法性、类别分布、标注框统计。

检查项:
  1. 图片-标签一一配对 (缺标签/缺图片/多余文件)
  2. 标签格式: 字段数、数值合法性、类别索引越界
  3. bbox 合法性: 越界 (超出图像)、宽高 <= 0、中心点越界
  4. 空标签文件 (背景图) 统计
  5. 图片可读性 (PIL 校验) 与重复图片 (MD5)
  6. 每类实例数直方图 + bbox 宽高分布 (按 COCO 标准: 小 <32px / 中 <96px / 大)

用法:
    python tools/dataset_check.py                          # 检查 NEU-DET train/val
    python tools/dataset_check.py --data my_data.yaml --splits train val test
    python tools/dataset_check.py --img-dir imgs --label-dir labels --names a,b,c

输出: docs/dataset_check_report.md + 控制台摘要
"""
import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from common import PROJECT_ROOT, write_report

DEFAULT_DATA = PROJECT_ROOT / "ultralytics" / "cfg" / "datasets" / "NUE_DET.yaml"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_splits(data_yaml):
    """返回 {split: (images_dir, labels_dir)}。"""
    d = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    out = {}
    for k in ("train", "val", "test"):
        v = d.get(k)
        if not v:
            continue
        img_dir = Path(v)
        lab_dir = img_dir.parent / "labels"
        out[k] = (img_dir, lab_dir if lab_dir.exists() else None)
    return out, int(d.get("nc", 0)), d.get("names", [])


def parse_label(path, nc, errors):
    boxes = []
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
    except Exception as e:
        errors["读取失败"].append(f"{path}: {e}")
        return boxes
    for ln, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5:
            errors["字段数错误"].append(f"{path}:{ln} -> {line!r}")
            continue
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            errors["非数值"].append(f"{path}:{ln} -> {line!r}")
            continue
        cls, cx, cy, w, h = int(parts[0]), vals[1], vals[2], vals[3], vals[4]
        if nc and cls >= nc:
            errors["类别越界"].append(f"{path}:{ln} -> class {cls} >= nc {nc}")
            continue
        if w <= 0 or h <= 0:
            errors["宽高非法"].append(f"{path}:{ln} -> w={w} h={h}")
            continue
        if not (-0.0 <= cx <= 1.0 and -0.0 <= cy <= 1.0):
            errors["中心点越界"].append(f"{path}:{ln} -> cx={cx} cy={cy}")
        if cx - w / 2 < 0 or cx + w / 2 > 1 or cy - h / 2 < 0 or cy + h / 2 > 1:
            errors["bbox超出图像"].append(f"{path}:{ln} -> bbox 越界")
            continue
        boxes.append((cls, cx, cy, w, h))
    return boxes


def check_split(name, img_dir, lab_dir, nc, names, errors, stats):
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    img_stems = {p.stem for p in imgs}
    lab_files = sorted(lab_dir.glob("*.txt")) if lab_dir else []
    lab_stems = {p.stem for p in lab_files}

    for stem in sorted(lab_stems - img_stems):
        errors["有标签无图片"].append(f"{name}/{stem}.txt")
    for stem in sorted(img_stems - lab_stems):
        if lab_dir is None:
            break
        errors["有图片无标签"].append(f"{name}/{stem}.jpg")

    class_counts = Counter()
    n_empty = 0
    n_boxes = 0
    size_bins = Counter()
    seen_md5 = defaultdict(list)

    for i, p in enumerate(imgs):
        if i % 200 == 0:
            print(f"  {name}: 检查中 {i}/{len(imgs)} ...")
        try:
            with Image.open(p) as im:
                im.verify()
                with Image.open(p) as im2:
                    W, H = im2.size
        except Exception as e:
            errors["图片无法读取"].append(f"{p}: {e}")
            continue
        try:
            md5 = hashlib.md5(p.read_bytes()).hexdigest()
            seen_md5[md5].append(p.name)
        except OSError:
            pass

        lab = lab_dir / (p.stem + ".txt") if lab_dir else None
        if lab is None or not lab.exists():
            continue
        boxes = parse_label(lab, nc, errors)
        if not boxes:
            n_empty += 1
        n_boxes += len(boxes)
        for cls, cx, cy, w, h in boxes:
            class_counts[cls] += 1
            pw, ph = w * W, h * H
            size_bins["small" if pw < 32 and ph < 32 else "medium" if pw < 96 and ph < 96 else "large"] += 1

    dups = {md5: names_list for md5, names_list in seen_md5.items() if len(names_list) > 1}
    stats[name] = {
        "图片数": len(imgs),
        "标签文件数": len(lab_files) if lab_dir else 0,
        "空标签数": n_empty,
        "标注框总数": n_boxes,
        "类别实例数": {names[c] if names and c < len(names) else str(c): class_counts.get(c, 0)
                       for c in sorted({cc for cc in class_counts} | set(range(nc)))},
        "尺寸分布": dict(size_bins),
        "重复图片组数": len(dups),
    }
    if dups:
        for md5, lst in list(dups.items())[:5]:
            errors["重复图片"].append(f"{name}: {lst}")


def render_report(errors, stats):
    lines = ["# 数据集健康检查报告", ""]
    lines.append("## 概览\n")
    lines.append("| 划分 | 图片数 | 标签文件数 | 空标签 | 标注框总数 | 小目标(<32px) | 中目标 | 大目标 | 重复图片组 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, s in stats.items():
        d = s["尺寸分布"]
        lines.append(f"| {name} | {s['图片数']} | {s['标签文件数']} | {s['空标签数']} | "
                     f"{s['标注框总数']} | {d.get('small', 0)} | {d.get('medium', 0)} | "
                     f"{d.get('large', 0)} | {s['重复图片组数']} |")

    lines += ["", "## 每类实例数", ""]
    all_names = sorted({n for s in stats.values() for n in s["类别实例数"]})
    lines.append("| 类别 | " + " | ".join(stats) + " |")
    lines.append("|---|" + "---|" * len(stats))
    for c in all_names:
        lines.append("| " + c + " | " + " | ".join(str(stats[s]["类别实例数"].get(c, 0)) for s in stats) + " |")

    lines += ["", "## 问题清单", ""]
    total = 0
    for kind, items in errors.items():
        total += len(items)
        lines.append(f"### {kind}: {len(items)} 处")
        for it in items[:10]:
            lines.append(f"- `{it}`")
        if len(items) > 10:
            lines.append(f"- ... 其余 {len(items) - 10} 处省略")
    if total == 0:
        lines.append("未发现问题, 数据集健康。")
    lines += ["",
              "> 注: 小/中/大目标按 COCO 标准 (绝对像素 <32 / <96 / >=96) 统计, 缺陷检测更关注小目标占比。"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="数据集健康检查")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="data.yaml 路径")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="要检查的划分")
    parser.add_argument("--img-dir", default=None, help="直接指定图片目录 (跳过 data.yaml)")
    parser.add_argument("--label-dir", default=None, help="直接指定标签目录")
    parser.add_argument("--names", default=None, help="类别名, 逗号分隔 (img-dir 模式用)")
    args = parser.parse_args()

    errors = defaultdict(list)
    stats = {}

    if args.img_dir:
        names = args.names.split(",") if args.names else []
        check_split("custom", Path(args.img_dir), Path(args.label_dir) if args.label_dir else None,
                    len(names), names, errors, stats)
    else:
        splits, nc, names = load_splits(args.data)
        for split in args.splits:
            if split not in splits:
                raise SystemExit(f"[错误] data.yaml 中没有 {split} 划分, 可选: {list(splits)}")
            img_dir, lab_dir = splits[split]
            print(f"检查 {split}: {img_dir}")
            if lab_dir is None:
                print(f"  [提示] {split} 无 labels 目录, 跳过标签检查 (test 集常见)")
            check_split(split, img_dir, lab_dir, nc, names, errors, stats)

    total = sum(len(v) for v in errors.values())
    print(f"\n检查完成: 发现问题 {total} 处")
    for kind, items in errors.items():
        print(f"  {kind}: {len(items)}")
    write_report("dataset_check_report.md", render_report(errors, stats))


if __name__ == "__main__":
    main()
