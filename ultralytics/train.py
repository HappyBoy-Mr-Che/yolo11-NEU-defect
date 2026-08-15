import os
import warnings
import torch
from ultralytics import YOLO

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # ==================== 模型选择 ====================
    # 方案A：使用改进的 YOLO11 模型（C3k2_IDC + SAFMNPP，3个检测头）
    model = YOLO('./cfg/models/11/YOLOV11.yaml')

    # 方案B：使用改进的 YOLO11 增强模型（C3k2_IDC + SCSABlock + 4个检测头，含P2小目标层）
    # model = YOLO('./cfg/models/11/Yolov11_EMA.yaml')

    # 方案C：使用标准 YOLO11 模型（未改进）
    # model = YOLO('./cfg/models/11/yolo11.yaml')

    # 可选：加载预训练权重加速收敛
    # model.load('yolo11n.pt')

    # ==================== 开始训练 ====================
    model.train(
        data='./cfg/datasets/NUE_DET.yaml',
        cache='disk',
        imgsz=640,
        epochs=150,
        batch=8,
        close_mosaic=64,
        workers=8,
        optimizer='SGD',
        lr0=0.0001,
        lrf=0.2,
        momentum=0.937,
        weight_decay=5e-4,
        pretrained=False,
    )

    # ==================== 保存最终模型 ====================
    save_dir = './runs/detect/train_final'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.model.state_dict(), f'{save_dir}/final_model.pt')
    print(f'模型已保存至: {save_dir}/final_model.pt')
