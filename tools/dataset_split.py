"""数据集划分: 图片+标签按 train/val/test 分层抽样划分 (按类别实例数分层)。

分层策略: 贪心分配 — 每张图按"其包含类别中全局最稀有者"排序, 依次分配到
当前该类占比最不足的划分, 保证稀有类别在 val/test 中都有样本。

用法:
    python tools/dataset_split.py --src dataset/raw --out dataset/split --ratios 8 1 1
    python tools/dataset_split.py --src dataset/raw --data dataset/raw/data.yaml  # 从 data.yaml 取 nc/names
    python tools/dataset_split.py --src dataset/raw --ratios 8 2 --move          # 移动而非复制

输出: <out>/images/{train,val,test} + <out>/labels/{train,val,test} + <out>/data.yaml
      以及 docs/split_report.md (每划分每类实例数统计)
"""
import argparse
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from common import PROJECT_ROOT, write_report

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = ["train", "val", "test"]


def read_labels(img_dir, lab_dir):
    """返回 {stem: {cls: count}} 及全局类别集合。"""
    out = {}
    classes = set()
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lab = lab_dir / (img.stem + ".txt")
        counts = Counter()
        if lab.exists():
            for line in lab.read_text(encoding="utf-8").strip().splitlines():
                parts = line.split()
                if not parts:
                    continue
                cls = int(float(parts[0]))
                counts[cls] += 1
                classes.add(cls)
        out[img.stem] = counts
    return out, classes


def stratified_assign(imgs, ratios, split_names):
    """贪心分层: 稀有类优先, 分配到该类占比最不足的划分。"""
    global_counts = Counter()
    for stem in imgs:
        global_counts.update(imgs[stem])

    order = sorted(imgs, key=lambda s: min(global_counts[c] for c in imgs[s]) if imgs[s] else 10**9)

    assigned = {s: [] for s in split_names}
    current = {s: Counter() for s in split_names}
    targets = {s: r / sum(ratios) for s, r in zip(split_names, ratios)}

    for stem in order:
        vec = imgs[stem]
        best_s, best_score = split_names[0], -1e9
        for s in split_names:
            score = sum((targets[s] - current[s][c] / global_counts[c]) * n for c, n in vec.items())
            if score > best_score:
                best_s, best_score = s, score
        assigned[best_s].append(stem)
        current[best_s].update(vec)
    return assigned, current


def copy_files(stems, src_img, src_lab, dst_img, dst_lab, move):
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lab.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        img = next((p for p in src_img.iterdir() if p.stem == stem), None)
        lab = src_lab / (stem + ".txt")
        if img is None:
            print(f"[警告] 找不到图片 {stem}, 跳过")
            continue
        op = shutil.move if move else shutil.copy2
        op(img, dst_img / img.name)
        if lab.exists():
            op(lab, dst_lab / lab.name)


def render_report(out_dir, split_names, current, class_names):
    lines = ["# 数据集划分报告", "", f"- 输出目录: `{out_dir}`", "- 分层策略: 按类别实例数贪心分层 (稀有类优先)", "",
             "| 划分 | 图片数 | " + " | ".join(class_names) + " | 实例总数 |",
             "|---:|---:|" + "---:|" * len(class_names) + "---:|"]
    for s in split_names:
        n_img = len(list((out_dir / "images" / s).iterdir())) if (out_dir / "images" / s).exists() else 0
        cells = " | ".join(str(current[s].get(i, 0)) for i in range(len(class_names)))
        lines.append(f"| {s} | {n_img} | {cells} | {sum(current[s].values())} |")
    lines += ["", "> 注: 划分后的 data.yaml 位于输出目录根, 直接传给 train/val 使用。"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="数据集分层划分")
    parser.add_argument("--src", required=True, help="原始数据目录 (含 images/ 和 labels/)")
    parser.add_argument("--out", default=None, help="输出目录, 默认 <src>_split")
    parser.add_argument("--data", default=None, help="原始 data.yaml (取 nc/names), 可选")
    parser.add_argument("--ratios", nargs="+", type=int, default=[8, 1, 1], help="train/val/test 比例")
    parser.add_argument("--move", action="store_true", help="移动文件而非复制")
    args = parser.parse_args()

    src = Path(args.src)
    src_img, src_lab = src / "images", src / "labels"
    if not src_img.exists() or not src_lab.exists():
        raise SystemExit(f"[错误] {src} 下需有 images/ 和 labels/ 目录")
    if len(args.ratios) not in (2, 3):
        raise SystemExit("[错误] --ratios 需 2 个 (train/val) 或 3 个 (train/val/test) 数字")

    split_names = SPLIT_NAMES[:len(args.ratios)]
    out_dir = (Path(args.out) if args.out else src.parent / (src.name + "_split")).resolve()

    nc, names = 0, []
    if args.data:
        d = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
        nc, names = int(d.get("nc", 0)), list(d.get("names", []))
    else:
        names = [f"class{i}" for i in range(max(nc, 1))]
    if not nc:
        nc = len(names)

    imgs, classes = read_labels(src_img, src_lab)
    if not imgs:
        raise SystemExit(f"[错误] {src_img} 中没有图片")
    print(f"图片总数: {len(imgs)}, 类别数: {len(classes) or nc}")

    assigned, current = stratified_assign(imgs, args.ratios, split_names)

    for s in split_names:
        copy_files(assigned[s], src_img, src_lab, out_dir / "images" / s, out_dir / "labels" / s, args.move)
        print(f"  {s}: {len(assigned[s])} 张")

    data_yaml = {
        "path": str(out_dir).replace("\\", "/"),
        "train": str(out_dir / "images" / "train").replace("\\", "/"),
        "val": str(out_dir / "images" / "val").replace("\\", "/"),
        "nc": nc,
        "names": names[:nc] or [f"class{i}" for i in range(nc)],
    }
    if "test" in split_names:
        data_yaml["test"] = str(out_dir / "images" / "test").replace("\\", "/")
    (out_dir / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"data.yaml 已生成: {out_dir / 'data.yaml'}")

    write_report("split_report.md", render_report(out_dir, split_names, current, data_yaml["names"]))


if __name__ == "__main__":
    main()
