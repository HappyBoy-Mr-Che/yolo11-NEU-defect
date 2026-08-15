"""特征级知识蒸馏: 大模型 (teacher) 蒸馏小模型 (student)。

原理: 在 YOLO 检测头输入特征 (P3/P4/P5, 按 stride 配对) 上做 MSE 蒸馏,
student 特征先经可学习 1x1 adapter 对齐通道数, 蒸馏损失加权叠加到检测损失。

用法:
    # teacher 用大尺寸权重, student 用小尺寸 YAML 从头训练
    python tools/distill.py --teacher models/yolo11_DSE.pt \
        --student ultralytics/cfg/models/11/YOLOV11.yaml \
        --data ultralytics/cfg/datasets/NUE_DET.yaml \
        --epochs 150 --kd-weight 0.5 --batch 8 --device cpu

    # 或 student 直接指定 scale: 在 YAML 中修改 scales 的 n/s 即可

训练结束后自动: 解包出纯净 student 权重 -> 验证 teacher/student -> 写 docs/distill_report.md
"""
import argparse
import sys
import time
from pathlib import Path

from common import PROJECT_ROOT, check_weight_compat, get_device, write_report

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import DEFAULT_CFG  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402


def unwrap_detection_model(wrapper):
    det = wrapper.model
    if hasattr(det, "ema"):
        det = det.ema
    return det


class KDStudentWrapper(nn.Module):
    """包装 student, 在训练前向中叠加与 teacher 的特征蒸馏损失。

    teacher 存在普通 dict 中 (不注册为子模块), 因此:
    - 不进 optimizer / EMA / named_parameters, 梯度恒为 None
    - 不受 trainer 的冻结层解冻逻辑影响
    """

    def __init__(self, student, teacher, kd_weight=0.5):
        super().__init__()
        self.student = student
        self._teacher = {"model": teacher}
        self.kd_weight = kd_weight

        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()

        self._s_feats = None
        self._t_feats = None
        self._mapping = None  # [(student_level, teacher_level, adapter)]
        self._adapters = nn.ModuleDict()
        self._h_s = student.model[-1].register_forward_hook(self._capture_student)
        self._h_t = teacher.model[-1].register_forward_hook(self._capture_teacher)

    @property
    def teacher(self):
        return self._teacher["model"]

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.student, name)

    def _capture_student(self, module, inp, out):
        self._s_feats = list(inp[0])

    def _capture_teacher(self, module, inp, out):
        self._t_feats = [f.detach() for f in inp[0]]

    def _run_teacher(self, img):
        if next(self.teacher.parameters()).device != img.device:
            self.teacher.to(img.device)
        with torch.no_grad():
            self.teacher(img)

    def _build_mapping(self, img):
        H = img.shape[2]
        s_strides = [H // f.shape[2] for f in self._s_feats]
        t_strides = [H // f.shape[2] for f in self._t_feats]
        mapping = []
        for i, st in enumerate(s_strides):
            for j, tt in enumerate(t_strides):
                if st == tt:
                    cs = self._s_feats[i].shape[1]
                    ct = self._t_feats[j].shape[1]
                    adapter = nn.Conv2d(cs, ct, 1, bias=False)
                    self._adapters[f"stride{st}"] = adapter
                    mapping.append((i, j, adapter))
        if not mapping:
            raise RuntimeError("student 与 teacher 检测层 stride 无交集, 无法蒸馏")
        self._mapping = mapping
        print(f"蒸馏特征层 (stride): {[H // self._s_feats[i].shape[2] for i, _, _ in mapping]}")

    def _kd_loss(self, img):
        if self._mapping is None:
            self._build_mapping(img)
        total = 0.0
        for i, j, adapter in self._mapping:
            total = total + F.mse_loss(adapter(self._s_feats[i]), self._t_feats[j])
        return total / len(self._mapping)

    def forward(self, x, *args, **kwargs):
        if not isinstance(x, dict):  # 推理路径: 直接走 student
            return self.student(x, *args, **kwargs)

        loss, loss_items = self.student(x)
        if self.training:
            self._run_teacher(x["img"])
            kd = self._kd_loss(x["img"])
            loss = loss + self.kd_weight * kd
            loss_items = torch.cat([loss_items, kd.detach().reshape(1)])
        return loss, loss_items


class DistillTrainer(DetectionTrainer):
    def __init__(self, teacher_weights=None, kd_weight=0.5, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.teacher_weights = teacher_weights
        self.kd_weight = kd_weight

    def get_model(self, cfg=None, weights=None, verbose=True):
        student = super().get_model(cfg, weights, verbose)
        tw = YOLO(self.teacher_weights)
        teacher = unwrap_detection_model(tw)
        return KDStudentWrapper(student, teacher, self.kd_weight)


def extract_clean_student(best_pt, out_path):
    """从训练产物中解包出纯净的 student DetectionModel 权重。"""
    ckpt = torch.load(str(best_pt), map_location="cpu", weights_only=False)
    wrapped = ckpt.get("model")
    if isinstance(wrapped, KDStudentWrapper):
        student = wrapped.student
    else:
        student = wrapped
    ckpt["model"] = student
    ckpt.pop("ema", None)
    torch.save(ckpt, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="特征级知识蒸馏训练")
    parser.add_argument("--teacher", required=True, help="teacher 权重 .pt")
    parser.add_argument("--student", required=True, help="student 模型 YAML (或 .pt)")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "ultralytics" / "cfg" / "datasets" / "NUE_DET.yaml"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="student 最终权重输出路径")
    args = parser.parse_args()

    device = args.device or get_device()
    if Path(args.teacher).suffix == ".pt":
        check_weight_compat(args.teacher)

    overrides = {
        "model": args.student,
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": device,
        "optimizer": "SGD",
        "lr0": 1e-4,
        "lrf": 0.2,
        "momentum": 0.937,
        "weight_decay": 5e-4,
        "cache": "disk",
        "workers": 4,
        "name": f"distill_kd{args.kd_weight}",
        "project": "ultralytics/runs/distill",
    }

    print(f"teacher: {args.teacher}\nstudent: {args.student}\nkd_weight: {args.kd_weight}")
    trainer = DistillTrainer(teacher_weights=args.teacher, kd_weight=args.kd_weight,
                             overrides=overrides)
    trainer.train()

    best_pt = Path(trainer.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        raise SystemExit(f"[错误] 训练产物不存在: {best_pt}")
    out_path = Path(args.out) if args.out else PROJECT_ROOT / "models" / (
        Path(args.student).stem + f"_distill_kd{args.kd_weight}.pt")
    extract_clean_student(best_pt, out_path)
    print(f"纯净 student 权重已保存: {out_path}")

    print("\n验证 teacher ...")
    r_t = YOLO(args.teacher).val(data=args.data, imgsz=640, device=device, verbose=False, plots=False)
    print("验证 student ...")
    r_s = YOLO(str(out_path)).val(data=args.data, imgsz=640, device=device, verbose=False, plots=False)

    p_t = sum(x.numel() for x in unwrap_detection_model(YOLO(args.teacher)).parameters())
    p_s = sum(x.numel() for x in unwrap_detection_model(YOLO(str(out_path))).parameters())

    report = (
        "# 知识蒸馏报告\n\n"
        f"- teacher: `{args.teacher}` ({p_t / 1e6:.3f} M 参数)\n"
        f"- student: `{out_path.name}` ({p_s / 1e6:.3f} M 参数)\n"
        f"- 蒸馏方式: 检测头输入特征 MSE (按 stride 配对, 可学习 1x1 adapter 对齐通道)\n"
        f"- kd_weight: {args.kd_weight}, epochs: {args.epochs}\n\n"
        "| 指标 | teacher | student(蒸馏) | 差距 |\n|---|---|---|---|\n"
        f"| mAP@0.5 | {r_t.box.map50:.4f} | {r_s.box.map50:.4f} | {r_s.box.map50 - r_t.box.map50:+.4f} |\n"
        f"| mAP@0.5:0.95 | {r_t.box.map:.4f} | {r_s.box.map:.4f} | {r_s.box.map - r_t.box.map:+.4f} |\n"
        f"| 参数量 (M) | {p_t / 1e6:.3f} | {p_s / 1e6:.3f} | "
        f"{-100 * (1 - p_s / p_t):.1f}% |\n"
    )
    write_report("distill_report.md", report)


if __name__ == "__main__":
    main()
