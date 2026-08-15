"""BN-L1 结构化通道剪枝 (自实现, 无外部依赖)。

原理:
1. 只剪 backbone/neck 中的独立 Conv 层 (下采样层、head 融合层), 不剪
   C3k2_IDC/C2PSA/SPPF/SAFMNPP/SCSABlock/Detect 内部结构 (避免残差/分组通道对齐问题)。
2. 收集所有可剪层的 BN gamma 绝对值, 按全局百分位阈值保留重要通道。
3. 通道 mask 沿前向图传播 (含 Concat 多分支), 下游模块的首个卷积按 mask 切输入通道。
4. 原位替换模块 + 逐层拷贝保留权重, 直接保存为可加载的 .pt。

用法:
    python tools/prune.py --weights models/best.pt --ratio 0.4
    python tools/prune.py --weights models/best.pt --ratio 0.4 --finetune-epochs 30
    python tools/prune.py --weights models/best.pt --ratio 0.4 --skip-val
"""
import argparse
import sys
import time
from pathlib import Path

from common import PROJECT_ROOT, check_weight_compat, get_device, write_report

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# 输入通道被剪后内部结构会失效的模块 (内部按固定通道数 split/groups)
FIXED_STRUCTURE = {"SCSABlock", "EMA", "iAFF", "InceptionDWConv2d", "StripPooling", "Detect"}

# 通道透传层: 输出通道 mask 直接继承输入 mask (Concat 拼接多分支, Upsample 不改变通道数)
PASSTHROUGH = {"Concat", "Upsample"}

# 各模块类型 -> 首个接收输入的卷积参数路径
FIRST_CONV = {
    "Conv": "conv",               # block.Conv.conv (nn.Conv2d)
    "C3k2": "cv1.conv",
    "C3k2_IDC": "cv1.conv",
    "C2f": "cv1.conv",
    "SPPF": "cv1.conv",
    "C2PSA": "cv1.conv",
    "C2fPSA": "cv1.conv",
    "SAFMNPP": "to_feat",         # SAFMNPP.to_feat (nn.Conv2d)
    "SCSABlock": "DConv.conv.conv",  # DASConv_1.conv 是 Conv 模块, 其 .conv 是 nn.Conv2d
    "EMA": None,                  # 固定结构, 不允许剪其上游
}


def unwrap_detection_model(wrapper):
    det = wrapper.model
    if hasattr(det, "ema"):  # ckpt 中保存的是 ModelEMA 包装
        det = det.ema
    if not hasattr(det, "yaml"):
        raise SystemExit("[错误] 无法从权重中提取模型 YAML 结构, 请用标准 YOLO 训练产物 (.pt)")
    return det


def get_yaml_rows(det):
    return det.yaml["backbone"] + det.yaml["head"]


def collect_out_channels(det, imgsz=320):
    """用前向 hook 记录每层输出通道数 (Detect 层输出为 list, 记 None)。"""
    shapes = {}
    hooks = []

    def make_hook(i):
        def hook(module, inp, outp):
            if isinstance(outp, (list, tuple)):
                shapes[i] = None
            elif torch.is_tensor(outp):
                shapes[i] = outp.shape[1]
        return hook

    for i, m in enumerate(det.model):
        hooks.append(m.register_forward_hook(make_hook(i)))
    det.eval()
    with torch.no_grad():
        # 必须走 det() 完整前向 (含 yaml 的 from 列表拼接逻辑), det.model 是 nn.Sequential 会丢失多输入关系
        det(torch.zeros(1, 3, imgsz, imgsz))
    for h in hooks:
        h.remove()
    return shapes


def compute_masks(det, ratio, keep_min=8, max_prune_frac=0.75):
    rows = get_yaml_rows(det)
    n = len(rows)
    out_ch = collect_out_channels(det)
    # Detect 层输出为 list 属预期 (mask 传播时按 None 处理), 其余层必须能确定输出通道数
    missing = [i for i in range(n) if out_ch.get(i) is None and type(det.model[i]).__name__ != "Detect"]
    if missing:
        raise SystemExit(f"[错误] 无法确定输出通道数的层: {missing}")

    # 1. 确定可剪层 (独立 Conv, 输出通道足够大)
    prunable = {}
    for i, m in enumerate(det.model):
        if type(m).__name__ == "Conv" and out_ch[i] >= 32:
            gamma = m.bn.weight.detach().abs()
            if gamma.numel() == out_ch[i]:
                prunable[i] = gamma
    if not prunable:
        raise SystemExit("[错误] 没有找到可剪枝的独立 Conv 层")

    # 2. 全局百分位阈值
    all_g = torch.cat(list(prunable.values()))
    thr = torch.quantile(all_g, ratio)
    print(f"可剪层: {len(prunable)}, BN gamma 阈值 (ratio={ratio}): {thr:.4f}")

    # 3. 每层保留 mask
    keep = {}
    for i, gamma in prunable.items():
        C = gamma.numel()
        mask = gamma >= thr
        k = int(mask.sum())
        if k < keep_min:  # 保底
            _, idx = gamma.topk(keep_min)
            mask = torch.zeros(C, dtype=torch.bool)
            mask[idx] = True
            k = keep_min
        if k > C * (1 - max_prune_frac):  # 单层最多剪 max_prune_frac
            k = max(int(C * (1 - max_prune_frac)), keep_min)
            _, idx = gamma.topk(k)
            mask = torch.zeros(C, dtype=torch.bool)
            mask[idx] = True
        keep[i] = mask
        print(f"  layer {i:>2} ({type(det.model[i]).__name__:<8}): {C} -> {int(mask.sum())} 通道")

    # 4. mask 沿前向图传播 (in_mask 描述模块输入通道, out_mask 描述模块输出通道)
    in_mask = [None] * n
    out_mask = [None] * n
    in_mask[0] = torch.ones(3, dtype=torch.bool)  # 第 0 层输入为原始图像
    for j in range(n):
        f = rows[j][0]
        froms = [f] if isinstance(f, int) else list(f)
        if j == 0:
            pass  # 第 0 层输入为原始图像, in_mask[0] 已预设
        else:
            froms = [j - 1 if x == -1 else x for x in froms]
            parts = []
            for x in froms:
                if out_mask[x] is None:
                    raise SystemExit(f"[错误] 层 {j} 的输入层 {x} mask 未定义")
                parts.append(out_mask[x])
            in_mask[j] = torch.cat(parts) if len(parts) > 1 else parts[0]

        if out_ch[j] is None:  # Detect 等输出非张量的层
            out_mask[j] = None
        elif j in keep:
            out_mask[j] = keep[j]  # 被剪 Conv: 输出 mask = 保留的输出通道
        elif type(det.model[j]).__name__ in PASSTHROUGH:
            out_mask[j] = in_mask[j]  # 透传层: 输出 mask = 输入 mask (拼接后的通道保留情况)
        else:
            out_mask[j] = torch.ones(out_ch[j], dtype=torch.bool)

        # 固定结构模块不允许输入被剪
        tname = type(det.model[j]).__name__
        if tname in FIXED_STRUCTURE and not bool(in_mask[j].all()):
            raise SystemExit(
                f"[错误] 固定结构模块 {tname} (layer {j}) 的输入通道被剪, 其内部按固定通道数 split 会失效。\n"
                f"  请降低 --ratio, 或该模块上游的 Conv 不适合剪枝。"
            )
    return keep, in_mask, out_ch


def slice_first_conv(module, mask):
    """按 mask 切模块首个卷积的输入通道, 原地修改。"""
    tname = type(module).__name__
    path = FIRST_CONV.get(tname)
    if path is None:
        return
    obj = module
    for attr in path.split("."):
        obj = getattr(obj, attr)
    w = obj.weight
    if mask.numel() != w.shape[1]:  # mask 与该卷积输入通道数不符, 跳过
        return
    keep_in = int(mask.sum())
    if keep_in == w.shape[1]:
        return
    new_w = w[:, mask, :, :].contiguous()
    obj.in_channels = keep_in
    obj.weight = torch.nn.Parameter(new_w)


def replace_pruned_conv(det, i, in_mask, out_mask):
    """把 layer i 的 Conv 替换为剪枝后的新 Conv 并拷贝保留权重。"""
    old = det.model[i]
    C_out = int(out_mask[i].sum())
    C_in = int(in_mask[i].sum())
    from ultralytics.nn.modules.block import Conv as BlockConv

    new = BlockConv(C_in, C_out, old.conv.kernel_size[0], old.conv.stride[0])
    sel_out = out_mask[i]  # 输出通道保留 mask (长度 = 旧输出通道数)
    new.conv.weight.data.copy_(old.conv.weight.data[sel_out][:, in_mask[i], :, :])
    if old.conv.bias is not None:
        new.conv.bias.data.copy_(old.conv.bias.data[sel_out])
    new.bn.weight.data.copy_(old.bn.weight.data[sel_out])
    new.bn.bias.data.copy_(old.bn.bias.data[sel_out])
    new.bn.running_mean.data.copy_(old.bn.running_mean.data[sel_out])
    new.bn.running_var.data.copy_(old.bn.running_var.data[sel_out])
    new.bn.num_batches_tracked.data.copy_(old.bn.num_batches_tracked.data)
    if hasattr(old, "i"):
        new.i, new.f, new.type = old.i, old.f, old.type
    det.model[i] = new


def model_params(det):
    return sum(p.numel() for p in det.parameters())


def main():
    parser = argparse.ArgumentParser(description="BN-L1 结构化通道剪枝")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--ratio", type=float, default=0.4, help="全局剪枝比例 (0~1)")
    parser.add_argument("--finetune-epochs", type=int, default=0, help="剪枝后微调轮数, 0 表示跳过")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "ultralytics" / "cfg" / "datasets" / "NUE_DET.yaml"))
    parser.add_argument("--skip-val", action="store_true", help="跳过剪枝前后精度对比验证")
    parser.add_argument("--out", default=None, help="输出路径, 默认与输入同目录")
    args = parser.parse_args()

    device = get_device()
    check_weight_compat(args.weights)

    print(f"加载模型: {args.weights}")
    wrapper = YOLO(args.weights)
    det = unwrap_detection_model(wrapper)
    det = det.to(device)
    params_before = model_params(det)

    keep, in_mask, out_ch = compute_masks(det, args.ratio)

    # 原位手术: 先切下游输入通道, 再替换被剪 Conv
    n = len(det.model)
    for j in range(n):
        slice_first_conv(det.model[j], in_mask[j])
    for i in keep:
        # 被剪 Conv 的输出 mask 即 keep[i]
        replace_pruned_conv(det, i, in_mask, keep)

    params_after = model_params(det)
    print(f"\n剪枝完成: 参数量 {params_before / 1e6:.3f} M -> {params_after / 1e6:.3f} M "
          f"(压缩 {100 * (1 - params_after / params_before):.1f}%)")

    # 保存
    out_path = Path(args.out) if args.out else Path(args.weights).with_name(
        Path(args.weights).stem + f"_pruned_r{args.ratio}.pt")
    ckpt = {
        "model": det,
        "train_args": dict(getattr(wrapper, "args", {}) or {}),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(ckpt, out_path)
    print(f"剪枝模型已保存: {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

    # 微调
    if args.finetune_epochs > 0:
        print(f"\n开始微调 {args.finetune_epochs} epochs ...")
        m = YOLO(str(out_path))
        m.train(data=args.data, imgsz=640, epochs=args.finetune_epochs, batch=8,
                device=device, optimizer="SGD", lr0=1e-4, lrf=0.2, momentum=0.937,
                weight_decay=5e-4, warmup_epochs=1, cache="disk", workers=4,
                name=f"prune_finetune_r{args.ratio}")
        # 用微调后的 best.pt 覆盖输出
        import glob
        bests = sorted(glob.glob(f"ultralytics/runs/detect/prune_finetune_r{args.ratio}/weights/best.pt"))
        if bests:
            import shutil
            shutil.copy(bests[-1], out_path)
            print(f"已用微调后权重覆盖: {out_path}")

    # 精度对比
    if not args.skip_val:
        print("\n验证剪枝前模型 ...")
        r_before = YOLO(args.weights).val(data=args.data, imgsz=640, device=device,
                                          verbose=False, plots=False)
        print("验证剪枝后模型 ...")
        r_after = YOLO(str(out_path)).val(data=args.data, imgsz=640, device=device,
                                         verbose=False, plots=False)
        m50_b = r_before.box.map50
        m50_a = r_after.box.map50
        m5095_b = r_before.box.map
        m5095_a = r_after.box.map
        drop50 = m50_b - m50_a
        drop5095 = m5095_b - m5095_a

        report = (
            "# 剪枝报告\n\n"
            f"- 原始模型: `{args.weights}`  ({params_before / 1e6:.3f} M 参数)\n"
            f"- 剪枝模型: `{out_path.name}`  ({params_after / 1e6:.3f} M 参数)\n"
            f"- 剪枝比例: {args.ratio} (全局 BN-gamma L1 百分位)\n"
            f"- 参数量压缩: {100 * (1 - params_after / params_before):.1f}%\n\n"
            "| 指标 | 剪枝前 | 剪枝后 | 变化 |\n|---|---|---|---|\n"
            f"| mAP@0.5 | {m50_b:.4f} | {m50_a:.4f} | {drop50:+.4f} |\n"
            f"| mAP@0.5:0.95 | {m5095_b:.4f} | {m5095_a:.4f} | {drop5095:+.4f} |\n"
            f"| 参数量 (M) | {params_before / 1e6:.3f} | {params_after / 1e6:.3f} | "
            f"{-100 * (1 - params_after / params_before):.1f}% |\n"
        )
        if args.finetune_epochs > 0:
            report += f"\n> 注: 以上为微调 {args.finetune_epochs} epochs 后的精度。\n"
        write_report("prune_report.md", report)


if __name__ == "__main__":
    main()
