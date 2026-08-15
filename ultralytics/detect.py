import warnings
from ultralytics import YOLO

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # 加载最优的模型权重
    best_model = './runs/detect/train85/weights/best.pt'  # 修改为你保存模型的位置
    #best_model = r'F:\SWUST\2025\2025_UY\competition\fuchuang\Yolov11_defect_project\ultralytics-8.3.2\runs\yolo11_prams\yolo_EMA\weights\best.pt'
    model = YOLO(best_model)

    # 进行验证，使用真实标签
    results = model.val(
        data='./cfg/datasets/NUE_DET.yaml',  # 指定数据集配置文件
        save=True,  # 保存检测结果
        conf=0.25,  # 置信度阈值
        imgsz=640,  # 图像大小
        #iou=0.5,  # IoU 阈值，默认为0.5，可以调整
        batch=16,  # 验证批次大小
        #device='0',  # 使用GPU
        task='val'  # 验证任务
    )
# 输出精确度
    precision = results.results_dict['metrics/precision(B)']
    print(f"测试集精确度: {precision}")
'''
class：代表模型检测的类别名称；
Images:代表验证集图片总数；
Instances:代表每个类别目标所标注的总数；
P:代表精确率Precision=TP / (TP+FP), 在预测是Positive所有结果中，预测正确的比重
R:召回率recall=TP / (TP+FN), 在真实值为Positive的所有结果中，预测正确的比重
mAP50:表示IOU阈值大于0.5的平均精确度（Mean Average Precision, mAP）
mAP50-95:表示在不同IoU阈值（从0.5到0.95，步长0.05）（0.5、0.55、0.6、0.65、0.7、0.75、0.8、0.85、0.9、0.95）上的平均mAP
'''
    # 输出验证结果，包括精度（P）、召回率（R）、mAP 等指标
    # 打印验证结果的属性
    # print(f"Precision: {val_results.metrics.precision}")
    # print(f"Recall: {val_results.metrics.recall}")
    # print(f"mAP@50: {val_results.metrics.map50}")
    # print(f"mAP@50-95: {val_results.metrics.map50_95}")
