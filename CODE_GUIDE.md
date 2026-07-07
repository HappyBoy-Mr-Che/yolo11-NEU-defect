# 项目文件说明文档

## 项目概述

本项目是一个**基于改进 YOLO11 的钢材表面缺陷实时检测系统**，技术栈为 PyQt5（GUI）+ 改进的 ultralytics（模型）+ OpenCV（图像处理）。

---

## 一、项目根目录

### 入口和配置

| 文件 | 功能说明 |
|------|---------|
| `main.py` | **系统主入口**。PyQt5 图形界面主程序，负责：打开图片/视频/摄像头、加载模型权重、执行 YOLO 推理、显示检测结果（画框）、导出结果。点击运行此文件即可启动整个系统 |
| `requirements.txt` | **Python 依赖清单**。列出项目需要的第三方包（ultralytics、pyside6），使用 `pip install -r requirements.txt` 一键安装 |
| `README.md` | **项目说明文档**。包含项目简介、结构说明、模型改进详解、使用方法、环境配置等 |

### 资源目录

| 文件夹 | 说明 |
|--------|------|
| `models/` | **训练好的模型权重**。存放 `.pt` 文件（best.pt, yolo11_DSE.pt 等），GUI 从这里加载模型 |
| `img_video/` | **测试用的图片和视频样本**。供 GUI 快速测试检测效果 |
| `ui_images/` | **UI 图标资源**。按钮图标（摄像头、检测等）、背景图、登录界面图片 |
| `data/` | **数据库文件**。存放 `userDB.db`（SQLite），存储用户账号密码信息 |
| `Font/` | **字体文件**。`platech.ttf` 供界面显示使用 |
| `save_data/` | **导出结果保存目录**。检测完成的图片/视频导出到这里 |
| `NEU-DET/` | **钢材缺陷数据集**（6类缺陷图片+标注），用于训练模型 |
| `NEU-DET.rar` | 数据集的压缩包备份 |

---

## 二、utils/ — PyQt5 GUI 界面模块

这是整个 GUI 系统的界面和工具层。

### 界面核心文件

| 文件 | 功能说明 |
|------|---------|
| `main_window.ui` | **主窗口设计稿**。用 Qt Designer 可视化拖拽生成的 XML 文件，定义了主界面上所有按钮、标签、滑块、图片框的位置和属性。用 Qt Designer 打开可以直接看到界面长什么样 |
| `main_window.py` | **主窗口 UI 编译代码**。由 `pyside6-uic main_window.ui -o main_window.py` 命令自动生成。将所有按钮/控件转成 Python 类 `Ui_MainWindow`，`main.py` 继承它来使用 |
| `ui_login.ui` | **登录窗口设计稿**。定义了登录界面的布局（账号输入框、密码输入框、登录/注册按钮等） |
| `ui_login.py` | **登录窗口 UI 编译代码**。由 `.ui` 文件编译生成，包含 `Ui_Login` 类 |

### 图片/图标资源相关

| 文件 | 功能说明 |
|------|---------|
| `image.qrc` | **Qt 资源定义文件**。声明哪些图标文件被打包到程序中（如 `:/image/icons/image.png`） |
| `image_rc.py` | **资源编译文件**。由 `pyside6-rcc image.qrc -o image_rc.py` 生成，把 PNG 图标编码成 Python 可直接引用的二进制数据 |
| `camera.qrc` | 同上，摄像头相关图标资源定义 |
| `camera_rc.py` | 同上，摄像头图标编译文件 |

### 功能模块

| 文件 | 功能说明 |
|------|---------|
| `plot_mask.py` | **检测结果可视化**。核心函数 `draw_detections(img, boxes, confs, cls_ids, threshold)`：在原始图片上画检测框 + 类别标签 + 置信度分数。`main.py` 中每次检测完成后调用它 |
| `config.py` | **类别名称配置**。定义了缺陷类别的中文/英文名称列表，供界面显示检测到的目标类别时查找名称 |
| `database.py` | **数据库操作模块**。封装了 SQLite 的增删改查操作（连接 userDB.db、查询用户、注册新用户等），登录界面调用它 |

---

## 三、ultralytics/ — 改进的 YOLO11 模型库

这是项目的核心部分，基于 ultralytics v8.3.53 源码，并对模型结构进行了自研改进。

### 顶层入口

| 文件 | 功能说明 |
|------|---------|
| `__init__.py` | **包初始化**。导入 `YOLO` 类和版本信息，使得 `from ultralytics import YOLO` 能正常工作 |
| `train.py` | **训练脚本（你写的）**。调用 `YOLO(cfg).train(data=...)` 进行模型训练，指定了数据集路径、超参数（epochs、batch size、学习率等），运行此文件开始训练 |
| `detect.py` | **验证/测试脚本（你写的）**。加载训练好的权重，在测试集上运行验证，输出 Precision、Recall、mAP 等指标 |
| `pyproject.toml` | Python 包元数据（包名、版本、依赖） |
| `requirements.txt` | ultralytics 原版依赖 |
| `LICENSE` | 开源协议（AGPL-3.0） |
| `.gitignore` | Git 忽略规则 |

### cfg/ — 配置文件层

| 文件 | 功能说明 |
|------|---------|
| `default.yaml` | **全局默认参数**。YOLO 训练/验证/预测/导出所有环节的默认超参数（学习率、数据增强、NMS、损失权重等），`model.train()` 会读取这里的默认值，训练脚本中的参数会覆盖它 |
| `cfg/__init__.py` | 配置管理入口。提供 `entrypoint()` 函数解析命令行参数 |

**数据集配置 (`cfg/datasets/`)：**

| 文件 | 功能说明 |
|------|---------|
| `NUE_DET.yaml` | **NEU 钢材缺陷数据集配置（主配置）**。指定了三件套：①数据集路径（train/val/test 的 images 目录）；②类别数 `nc: 6`；③类别名列表 `['Crazing', ..., 'Scratches']`。训练脚本通过 `data='./cfg/datasets/NUE_DET.yaml'` 引用它 |
| `data.yaml` | Severstal 钢材数据集配置（备用，4类缺陷），另一个数据集的配置 |

**模型结构配置 (`cfg/models/11/`)：**

| 文件 | 功能说明 |
|------|---------|
| `YOLOV11.yaml` | **★ 改进模型 1（推荐）**。在标准 YOLO11 基础上做的改进：Backbone 和 Neck 中所有 `C3k2` 模块替换为自研的 `C3k2_IDC`（IDC卷积+条带池化+iAFF融合），Neck 上采样层加入 `SAFMNPP`（空间自适应特征调制）。定义了模型每一层的结构：`[from, repeats, module, args]` |
| `Yolov11_EMA.yaml` | **★ 改进模型 2（增强版）**。在改进模型 1 基础上进一步：加入 `SCSABlock`（空间-通道自注意力）、增加 **P2 小目标检测层**（共4个检测头 P2/P3/P4/P5），对小尺寸缺陷更敏感 |
| `yolo11.yaml` | 标准 YOLO11 检测模型（未改进），保留作为对比基准 |
| `yolo11-cls.yaml` | YOLO11 分类模型配置 |
| `yolo11-seg.yaml` | YOLO11 实例分割模型配置 |
| `yolo11-pose.yaml` | YOLO11 姿态估计模型配置 |
| `yolo11-obb.yaml` | YOLO11 旋转框检测模型配置 |

> **关键理解**：YAML 文件就是模型的"建筑图纸"。`train.py` 中 `YOLO('YOLOV11.yaml')` 就是告诉程序："按这张图纸搭建模型，然后训练"。图纸中写的 `C3k2_IDC`、`SAFMNPP` 等自定义模块在 `nn/modules/` 中实现。

**跟踪器配置 (`cfg/trackers/`)：**

| 文件 | 功能说明 |
|------|---------|
| `botsort.yaml` | BoT-SORT 多目标跟踪器参数 |
| `bytetrack.yaml` | ByteTrack 多目标跟踪器参数 |

### nn/ — 神经网络层（★ 核心改进所在）

| 文件 | 功能说明 |
|------|---------|
| `nn/tasks.py` | **★ 模型构建器（核心文件）**。这是整个模型系统的"黏合剂"：①解析 YAML 配置文件中的每一行 `[from, repeats, module, args]`；②根据 module 名称查找对应的 PyTorch 类；③按顺序搭建完整的模型计算图。你新增的自定义模块（C3k2_IDC、SAFMNPP 等）都在这个文件中注册导入 |
| `nn/__init__.py` | 神经网络包初始化 |
| `nn/autobackend.py` | **多格式模型加载器**。支持加载 `.pt`、`.onnx`、`.engine`、`.tflite` 等各种格式的模型文件，自动识别格式并创建对应的推理后端 |

**nn/modules/ — 网络模块层（★ 你的改进实现在这里）：**

在 YAML 配置中写的每一个模块名，都对应这里的 `.py` 文件中的一个同名类。

**🔴 你的核心改进模块（在 YAML 中实际使用）：**

| 文件 | 核心类 | 在哪个 YAML 中使用 | 功能说明 |
|------|--------|-------------------|---------|
| `IDC.py` | `C3k2_IDC` | YOLOV11, Yolov11_EMA | **★ 核心改进**。替换标准 C3k2 的增强 CSP 瓶颈模块，内含 `InceptionDWConv2d`（多分支深度可分离卷积，3×3方形+11×1条带+1×11条带）、`DASConv_1`（空洞非对称卷积，dilation=3/7/12）、`StripPooling`（条带池化）、`iAFF`（迭代注意力特征融合） |
| `SAFMNet.py` | `SAFMNPP` | YOLOV11 | **上采样增强模块**。基于 ICCV 2023 的 SAFM 论文，用空间自适应特征调制 + PixelShuffle 上采样替代传统插值上采样，保留更多细节用于小缺陷检测 |
| `SCSA_Bottleneckt.py` | `SCSABlock` | Yolov11_EMA | **空间-通道自注意力**。内含 `SCSA`（先空间注意力：多尺度1D条带卷积，再通道自注意力：Self-Attention建模通道关系）和 `DASConv_1`（空洞非对称卷积），将两者输出拼接融合 |
| `SPPF_CSPC.py` | `SPPFCSPC` | 可选用 | 跨阶段部分连接的空间金字塔池化，减少计算量同时保持多尺度特征 |
| `SPPF_LSKa.py` | `SPPF_LSKA` | 可选用 | 大核可分离注意力的 SPPF，用可分离卷积实现大感受野（等效 7×7 到 53×53），增强全局上下文建模 |
| `BiFPN.py` | `BiFPN_Concat2`, `BiFPN_Concat3` | 可选用 | 加权双向特征金字塔融合，给不同尺度的特征加可学习权重，比普通 Concat 更有效 |
| `SPD_Conv.py` | `SPDConv` | 可选用 | 空间到深度卷积，用切片+拼接替代 stride=2 的卷积下采样，避免细粒度信息丢失 |
| `AFF.py` | `iAFF` | 可选用 | 迭代注意力特征融合，两次局部-全局注意力迭代地融合两路特征 |

**🟡 辅助注意力/卷积模块（备用的改进组件）：**

| 文件 | 核心类 | 功能说明 |
|------|--------|---------|
| `AKConv.py` | `AKConv` | 可变核卷积，根据输入动态调整卷积核形状 |
| `C2PSA_EMA.py` | `C2PSA_EMA` | 带 EMA 注意力的 C2PSA 模块 |
| `Ca_attetion.py` | `CoordAtt` 等 | 坐标注意力机制（Coordinate Attention） |
| `EMA_attention.py` | `EMA` | 高效多头注意力（Efficient Multi-head Attention） |
| `DualConv.py` | `DualConv` | 双分支卷积（分组卷积+逐点卷积并行） |
| `EFF.py` | 高效特征融合模块 | 用于多尺度特征融合 |
| `FEM.py` | 特征增强模块 | 局部特征增强 |
| `FRMN.py` | 特征精炼模块 | 特征图精炼去噪 |
| `MSCA_attention.py` | 多尺度通道注意力 | 从多个尺度建模通道关系 |
| `SAFM.py` | 基础版 SAFM | SAFMNPP 的简化版 |
| `ScConv.py` | 空间/通道重构卷积 | 分开处理空间和通道维度的卷积 |

**🟢 标准模块（ultralytics 原生）：**

| 文件 | 核心类 | 功能说明 |
|------|--------|---------|
| `block.py` | `C2f`, `C3k2`, `Bottleneck`, `SPPF`, `C2PSA`, `DFL`, `SPP` | YOLO 标准构建块：C2f/C3k2（CSP瓶颈层）、SPPF（快速空间金字塔池化）、C2PSA（带注意力的C2f）、DFL（分布焦点损失头）等 |
| `conv.py` | `Conv`, `Concat`, `DWConv`, `GhostConv` | 基础卷积层和特征拼接 |
| `head.py` | `Detect`, `Segment`, `Pose`, `Classify` | 检测/分割/姿态/分类头，负责输出最终预测 |
| `transformer.py` | `AIFI`, `TransformerBlock`, `MLP` | Transformer 相关模块（注意力层、MLP） |
| `activation.py` | 各种激活函数 | SiLU、ReLU 等 |
| `utils.py` | 辅助工具函数 | 权重初始化、自适应输入尺寸等 |

**🔵 模块注册 (`nn/modules/__init__.py`)：**

| 功能 |
|------|
| 把 `nn/modules/` 下所有模块统一导入，使 `tasks.py` 能通过模块名找到对应的类。新增自定义模块时需要在 `tasks.py` 中增加 `from .modules.XXX import YYY` |

### engine/ — 训练/推理引擎

| 文件 | 功能说明 |
|------|---------|
| `model.py` | **模型核心类 `YOLO`**。你在 `main.py` 里写的 `YOLO(weights_path)` 和 `model.predict()` 就是调这个类。封装了训练/验证/预测/导出所有操作的统一入口 |
| `trainer.py` | **训练引擎**。管理完整训练流程：数据加载→前向传播→损失计算→反向传播→优化器更新→日志记录→模型保存。支持单GPU/多GPU/分布式训练 |
| `predictor.py` | **推理引擎**。管理预测流程：图像预处理→模型推理→后处理（NMS）→结果封装。`model.predict()` 底层就是调这个 |
| `validator.py` | **验证/测试引擎**。计算验证集上的 Precision、Recall、mAP@50、mAP@50-95 等指标 |
| `exporter.py` | **模型导出引擎**。把 .pt 模型导出为 ONNX、TensorRT、OpenVINO、CoreML 等格式，用于不同平台的部署 |
| `results.py` | **结果封装类 `Results`**。`model.predict()` 返回的结果对象，通过 `.boxes.xyxy` 取坐标、`.boxes.conf` 取置信度 |
| `tuner.py` | **超参数自动搜索**。用遗传算法自动寻找最优超参数组合 |

### models/ — 模型定义

| 目录/文件 | 功能说明 |
|-----------|---------|
| `yolo/detect/` | YOLO 检测任务的 train/val/predict 实现，是 trainer/predictor/validator 的 YOLO 特化版 |
| `yolo/model.py` | YOLO 模型类，解析 YAML 配置并构建网络结构 |
| `yolo/segment/` | 实例分割任务实现 |
| `yolo/classify/` | 图像分类任务实现 |
| `yolo/pose/` | 姿态估计任务实现 |
| `yolo/obb/` | 旋转框检测任务实现 |
| `yolo/world/` | YOLO-World 开放词汇检测 |
| `models/utils/loss.py` | 损失函数（VarifocalLoss、DFL loss 等） |
| `models/utils/ops.py` | 模型操作工具（匈牙利匹配、NMS、anchor 生成等） |
| `rtdetr/` | RT-DETR 模型（实时 Transformer 检测器，备选方案） |
| `sam/` | Segment Anything Model（Meta 的分割大模型，备选） |
| `fastsam/` | FastSAM（高速分割模型） |
| `nas/` | 神经架构搜索模型 |

### data/ — 数据加载与增强

| 文件 | 功能说明 |
|------|---------|
| `dataset.py` | **数据集类 `YOLODataset`**。读取 images+labels，返回预处理后的（图像，标签）对 |
| `build.py` | 数据集构建器，根据 data.yaml 自动构建 train/val/test 三者的 DataLoader |
| `augment.py` | **数据增强**。Mosaic（4张拼1张）、MixUp（混合）、翻转、缩放、HSV变换、随机裁剪等所有增强实现 |
| `base.py` | 数据集基类 |
| `loaders.py` | 各种数据源加载器（图片文件、视频流、摄像头、网络流等） |
| `utils.py` | 数据集辅助函数（下载数据、统计分布等） |

### utils/ — 工具函数库

| 文件 | 功能说明 |
|------|---------|
| `checks.py` | 环境检查（CUDA 是否可用、依赖版本是否正确等） |
| `loss.py` | 损失计算（分类损失、bbox 回归损失、DFL 损失） |
| `metrics.py` | 评估指标计算（Precision、Recall、mAP、混淆矩阵） |
| `ops.py` | 运算操作（NMS 非极大值抑制、坐标变换、缩放） |
| `plotting.py` | 可视化（画结果图、训练曲线、混淆矩阵热力图） |
| `torch_utils.py` | PyTorch 辅助（模型统计、参数计数、设备选择） |
| `tal.py` | 任务对齐学习（Task-Aligned Learning），负责正负样本分配 |
| `instance.py` | 实例分割结果对象 |
| `files.py` | 文件操作工具 |
| `downloads.py` | 模型权重下载工具 |
| `patches.py` | PyTorch 版本兼容补丁 |
| `benchmarks.py` | 性能基准测试（FPS、延迟） |
| `autobatch.py` | 自动批量大小估算 |
| `dist.py` | 分布式训练工具 |
| `errors.py` | 自定义异常处理 |
| `tuner.py` | 超参数调优工具 |
| `triton.py` | Triton 推理服务器客户端 |
| `callbacks/` | 回调函数（训练过程中插入自定义逻辑，如 TensorBoard 日志、W&B 上传等） |

### trackers/ — 目标跟踪

| 文件 | 功能说明 |
|------|---------|
| `track.py` | 多目标跟踪入口，对视频中每个检测框分配唯一 ID |
| `bot_sort.py` | BoT-SORT 跟踪器实现 |
| `byte_tracker.py` | ByteTrack 跟踪器实现 |
| `basetrack.py` | 跟踪器基类 |
| `utils/` | 卡尔曼滤波、匈牙利匹配、全局运动补偿等跟踪辅助算法 |

---

## 四、数据流全景图

```
训练阶段:
  train.py
    → YOLO('YOLOV11.yaml')  ─解析YAML─→  nn/tasks.py  ─构建模型─→  nn/modules/IDC.py (C3k2_IDC)
                                                                    nn/modules/SAFMNet.py (SAFMNPP)
    → model.train( data='NUE_DET.yaml' )
        → data/build.py ─读取数据集─→  NEU-DET/train/images/*.jpg + labels/*.txt
        → engine/trainer.py ─训练循环─→  前向→损失→反向→更新
        → 保存 best.pt

部署阶段(GUI):
  main.py
    → YOLO('models/best.pt')  ─加载训练好的权重─→  engine/model.py
    → model.predict(image)
        → engine/predictor.py  ─推理─→  返回 Results 对象
        → utils/plot_mask.py  ─画框─→  在图片上标注缺陷
        → 显示在 PyQt5 界面上
```

---

## 五、快速定位指南

| 你想做什么 | 去看哪个文件 |
|-----------|------------|
| 改训练参数（epochs、lr、batch） | `ultralytics/train.py` |
| 改模型结构（增减网络层） | `ultralytics/cfg/models/11/YOLOV11.yaml` |
| 加新的自定义模块 | `ultralytics/nn/modules/` 下新建 .py，然后在 `nn/tasks.py` 注册 |
| 改数据集路径或类别 | `ultralytics/cfg/datasets/NUE_DET.yaml` |
| 改 GUI 界面布局 | 用 Qt Designer 打开 `utils/main_window.ui` |
| 改 GUI 逻辑（按钮功能） | `main.py` |
| 改检测框的绘制样式 | `utils/plot_mask.py` |
| 改缺陷类别名 | `utils/config.py` |
| 换模型权重 | `main.py` 中 `openfile_name_model` 或 `train.py` 中 model 路径 |
| 导出模型到 ONNX/TensorRT | `engine/exporter.py` 或用 `YOLO.export()` |
