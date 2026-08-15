"""共享工具函数：设备检测、权重兼容性检查、markdown 报告输出。"""
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DOCS_DIR = PROJECT_ROOT / "docs"

import torch


def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def check_weight_compat(weights_path, verbose=True):
    """检查权重文件与当前仓库代码构建的模型结构是否一致。

    返回 (missing, unexpected) 两个列表：missing 为权重中存在但模型没有的参数
    （加载时被静默丢弃，说明权重是旧架构训练的）。
    """
    from ultralytics import YOLO

    path = Path(weights_path)
    if path.suffix != ".pt":
        return [], []

    model = YOLO(str(path))
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    ckpt_model = ckpt.get("model")
    if ckpt_model is None:
        return [], []

    ckpt_keys = set(ckpt_model.state_dict().keys())
    cur_keys = set(model.model.state_dict().keys())
    missing = sorted(ckpt_keys - cur_keys)
    unexpected = sorted(cur_keys - ckpt_keys)

    if verbose and (missing or unexpected):
        print(f"[警告] {path.name} 与当前代码结构不一致:")
        print(f"  权重中缺失的层 ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        print(f"  当前模型多出的层 ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        print("  提示: 不匹配的层加载时被静默丢弃/随机初始化, 推理结果不可信。")
        print("        建议用当前仓库代码重新训练后再使用该模型。")
    return missing, unexpected


def write_report(filename, content):
    """把 markdown 内容写入 docs/ 目录。"""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / filename
    out.write_text(content, encoding="utf-8")
    print(f"报告已生成: {out}")
    return out


def fmt(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def model_efficiency(model):
    """返回 (参数量M, GFLOPs) 或 (None, None)。"""
    try:
        info = model.info(verbose=False)
        return info[1] / 1e6, info[2]
    except Exception:
        params = sum(p.numel() for p in model.model.parameters())
        return params / 1e6, None


def build_train_kwargs(data, epochs, batch, device, name, project):
    """与 ultralytics/train.py 一致的训练超参。"""
    return {
        "data": data,
        "imgsz": 640,
        "epochs": epochs,
        "batch": batch,
        "device": device,
        "cache": "disk",
        "workers": 4,
        "optimizer": "SGD",
        "lr0": 1e-4,
        "lrf": 0.2,
        "momentum": 0.937,
        "weight_decay": 5e-4,
        "pretrained": False,
        "close_mosaic": 64 if epochs > 64 else 0,
        "name": name,
        "project": project,
        "exist_ok": True,
    }
