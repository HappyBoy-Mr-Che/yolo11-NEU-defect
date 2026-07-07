# 基于改进 YOLO11 的钢材表面缺陷实时检测系统

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO11-8.3.2-purple.svg)](https://github.com/ultralytics/ultralytics)
[![PyQt5](https://img.shields.io/badge/PyQt5-6.6-green.svg)](https://pypi.org/project/PySide6/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-lightgrey.svg)](LICENSE)

**钢材表面缺陷检测 | 改进 YOLO11 | PyQt5 桌面应用 | 实时推理**

</div>

---

## 📋 项目简介

本项目针对**钢材表面缺陷检测**任务，在 YOLO11 目标检测模型基础上进行了三点架构改进，并开发了完整的 PyQt5 桌面端检测系统，支持图片/视频/摄像头三种检测模式。

### 应用场景

钢材生产过程中，表面缺陷（裂纹、夹杂物、划痕等）直接影响产品质量。传统人工目检效率低、漏检率高。本系统通过深度学习自动识别和定位 6 类钢材表面缺陷，实现实时、高精度的质量检测。

### 数据集

使用 **NEU-DET** 钢材表面缺陷数据集，包含 6 类缺陷共 1800 张灰度图像：

| 类别 | 英文名 | 说明 |
|------|--------|------|
| 裂纹 | Crazing | 表面网状或线状裂纹 |
| 夹杂物 | Inclusion | 钢基体中混入的非金属杂质 |
| 斑块 | Patches | 表面块状缺陷 |
| 麻点 | Pitted Surface | 点状凹坑缺陷 |
| 轧制氧化皮 | Rolled-in Scale | 轧制过程中嵌入的氧化皮 |
| 划痕 | Scratches | 机械划伤痕迹 |

---

## 🔬 模型改进

本项目对 YOLO11 进行了以下**三点原创改进**，所有自定义模块均以即插即用方式集成，可在 YAML 配置文件中直接替换调用。

### 改进架构总览

```
                          ┌──────────────────────────────┐
                          │       YOLO11 Backbone        │
                          │  Conv → C3k2_IDC → SPPF     │
                          │           → C2PSA            │
                          └──────────────┬───────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
              ▼                          ▼                          ▼
        ┌──────────┐              ┌──────────┐              ┌──────────┐
        │  P5/32   │              │  P4/16   │              │  P3/8    │
        │ 1024 ch  │              │  512 ch  │              │  256 ch  │
        └────┬─────┘              └────┬─────┘              └────┬─────┘
             │                         │                         │
             │    ┌────────────────────┤                         │
             │    │                    ▼                         │
             │    │             ┌─────────────┐                  │
             │    │             │  SAFMNPP    │ ← 空间自适应上采样 │
             │    │             │  (PixelShuffle)                 │
             │    │             └──────┬──────┘                  │
             │    │                    │                         │
             │    │         ┌──────────┴──────────┐              │
             │    │         │     Concat + C3k2_IDC│              │
             │    │         └──────────┬──────────┘              │
             │    │                    │                         │
             │    │                    ▼                         │
             │    ▼             ┌─────────────┐                  │
             │  Concat          │  Concat     │                  │
             │  + C3k2_IDC      │  + C3k2_IDC │                  │
             │    │             └──────┬──────┘                  │
             │    │                    │                         │
             ▼    ▼                    ▼                         ▼
        ┌─────────────┐        ┌─────────────┐          ┌─────────────┐
        │  Detect P5  │        │  Detect P4  │          │  Detect P3  │
        │  (大缺陷)    │        │  (中缺陷)    │          │  (小缺陷)    │
        └─────────────┘        └─────────────┘          └─────────────┘
```

### 改进 1：C3k2_IDC — 增强型 CSP 瓶颈模块

**替换位置**：Backbone 和 Neck 中所有标准 C3k2 模块

**核心思路**：标准 C3k2 使用普通 3×3 卷积提取特征，对钢材缺陷这种尺度多变、形状不规则的检测目标表达不足。C3k2_IDC 融合了四种互补的特征提取策略：

| 子模块 | 功能 | 论文来源 |
|--------|------|---------|
| **InceptionDWConv2d** | 多分支深度可分离卷积：3×3 方形核 + 11×1 垂直条带核 + 1×11 水平条带核，兼顾方形缺陷和细长缺陷 | Inception 系列 |
| **DASConv_1** | 空洞非对称卷积，dilation rate = 3/7/12，在不增加参数的前提下将感受野扩大至 25×25 | DeepLab |
| **StripPooling** | 条带池化（1×W + H×1），沿水平和垂直方向分别池化后融合，保持长距离空间依赖 | Strip Pooling (CVPR 2021) |
| **iAFF** | 迭代注意力特征融合，两次局部+全局注意力迭代，自适应融合多分支输出 | iAFF (WACV 2022) |

**效果**：更丰富的多尺度特征 → 对不同大小、形状的缺陷都有更好的检测能力。

### 改进 2：SAFMNPP — 空间自适应特征调制上采样

**替换位置**：Neck 中 P4→P3 特征融合层的上采样操作

**核心思路**：YOLO11 默认使用最近邻插值进行上采样，容易丢失细节信息。SAFMNPP 使用 ICCV 2023 提出的空间自适应特征调制机制，在 PixelShuffle 上采样的同时保留更多空间细节：

```
输入 (H/2, W/2)
    │
    ├──→ SAFM 特征调制 ──→ PixelShuffle 上采样 ──┐
    │                                              ├──→ 输出 (H, W)
    └──→ 双线性插值上采样 (残差连接) ──────────────┘
```

**效果**：上采样后保留了更多纹理细节，对小缺陷（如麻点、细裂纹）检测提升明显。

### 改进 3：SCSABlock — 空间-通道自注意力增强模块

**使用位置**：Yolov11_EMA 增强模型的 Neck 各检测层输出前

**核心思路**：融合 DASConv（局部多尺度）和 SCSA（全局注意力）两路特征，兼顾局部细节和全局语义：

| 分支 | 构成 | 作用 |
|------|------|------|
| DASConv 分支 | 多尺度空洞卷积 (d=3/7/12) | 提取局部多尺度特征 |
| SCSA 分支 | 空间注意力（1D 条带卷积）+ 通道自注意力 | 建模全局上下文依赖 |
| 融合 | 两路拼接 + 1×1 卷积降维 | 局部+全局互补 |

**SCSA 注意力详解**：
1. **空间注意力**：将特征图沿 H 和 W 方向分别做均值池化，用 4 种不同尺度的 1D 卷积（kernel=3,5,7,9）建模不同粒度的空间关系
2. **通道自注意力**：对空间压缩后的特征图应用多头自注意力，自适应地学习通道重要性权重

### 改进 4（Yolov11_EMA）：P2 小目标检测层

在 Neck 中增加了一层 P2 特征图（4 倍下采样），专门检测极小缺陷。传统 3 检测头（P3/P4/P5）对大缺陷效果好但对小目标覆盖不足，P2 层保留了更多空间分辨率，使模型能够捕捉到微小的麻点、细裂纹等缺陷。

### 模型对照

| 配置文件 | Backbone | Neck | 检测头 | 适用场景 |
|---------|----------|------|--------|---------|
| `yolo11.yaml` | 标准 C3k2 | 标准 nn.Upsample | 3 个 (P3/P4/P5) | 基准对照 |
| **`YOLOV11.yaml`** | C3k2_IDC | SAFMNPP | 3 个 (P3/P4/P5) | 常规缺陷检测 |
| **`Yolov11_EMA.yaml`** | C3k2_IDC | SCSABlock + P2 层 | 4 个 (P2/P3/P4/P5) | 小缺陷+全面检测 |

---

## 🖥️ 系统功能

基于 PyQt5 开发的桌面端检测系统，界面友好，一键操作。

### 核心功能

| 功能 | 说明 |
|------|------|
| **图片检测** | 打开单张图片 → 一键检测 → 可视化结果 + 导出 |
| **视频检测** | 加载视频文件 → 逐帧检测 → 支持暂停/继续/停止 → 导出检测视频 |
| **摄像头检测** | 连接摄像头 → 实时画面 + 实时检测 → 双模式切换 |
| **多模型切换** | 支持加载不同权重文件 (.pt/.onnx/.engine) |
| **参数实时调节** | 滑动条调节置信度阈值和 IoU 阈值，实时生效 |
| **目标详情** | 点击检测到的目标 → 显示精确坐标 |
| **检测统计** | 实时显示检测耗时 + 目标数量 |

### 用户系统

- SQLite 数据库存储用户账号信息
- 支持注册/登录/游客模式
- 无边框窗口设计，支持拖拽移动

---

## 📦 安装与使用

### 环境要求

- Python 3.8+
- PyTorch 2.0+ (推荐 CUDA 12.1)
- Windows / Linux

### 安装

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd quexian_detect

# 2. 安装依赖
pip install -r requirements.txt

# 3. 以开发模式安装改进版 ultralytics
cd ultralytics
pip install -e .
cd ..

# 4. (可选) 安装 einops 依赖
pip install einops
```

### 使用

**启动 GUI 系统：**

```bash
python main.py
```

**训练模型：**

```bash
cd ultralytics
# 使用改进模型 YOLOV11 (C3k2_IDC + SAFMNPP)
python train.py

# 或使用增强模型 Yolov11_EMA (SCSABlock + P2层)
# 修改 train.py 中 model 路径后运行
```

**验证模型：**

```bash
cd ultralytics
python detect.py
```

### 自定义模块调用

所有改进模块已在 `ultralytics/nn/tasks.py` 中注册，可在 YAML 配置中直接使用：

```yaml
# 使用 C3k2_IDC 替换标准 C3k2
- [-1, 2, C3k2_IDC, [256, False, 0.25]]

# 使用 SAFMNPP 替换 nn.Upsample
- [-1, 1, SAFMNPP, [512]]

# 使用 SCSABlock 增强特征
- [-1, 1, SCSABlock, [8]]

# 使用 BiFPN 加权特征融合
- [[-1, 6], 1, BiFPN_Concat2, [1]]
```

---

## 📁 项目结构

```
quexian_detect/
│
├── main.py                      # 系统主入口 (PyQt5 GUI)
├── requirements.txt             # Python 依赖
├── README.md                    # 项目文档
├── CODE_GUIDE.md                # 逐文件代码说明
├── .gitignore                   # Git 忽略规则
│
├── utils/                       # GUI 界面模块
│   ├── main_window.ui/py        # 主窗口 (Qt Designer 设计)
│   ├── ui_login.ui/py           # 登录窗口
│   ├── plot_mask.py             # 检测框可视化
│   ├── config.py                # 缺陷类别配置
│   ├── database.py              # SQLite 用户数据库
│   └── *.qrc / *_rc.py         # Qt 图标资源文件
│
├── ultralytics/                 # 改进的 YOLO11 模型库
│   ├── train.py                 # 训练脚本
│   ├── detect.py                # 验证脚本
│   ├── cfg/
│   │   ├── datasets/            # 数据集路径 + 类别配置
│   │   │   ├── NUE_DET.yaml     # NEU 钢材缺陷 (6类)
│   │   │   └── data.yaml        # Severstal 数据集 (4类)
│   │   └── models/11/           # ★ 模型 YAML 设计图纸
│   │       ├── YOLOV11.yaml     # 改进模型 1 (C3k2_IDC + SAFMNPP)
│   │       ├── Yolov11_EMA.yaml # 改进模型 2 (SCSA + P2层)
│   │       └── yolo11.yaml      # 基准模型
│   ├── nn/
│   │   ├── tasks.py             # ★ 模型构建器 (注册自定义模块)
│   │   └── modules/             # ★ 自定义改进模块
│   │       ├── IDC.py           # C3k2_IDC (核心改进)
│   │       ├── SAFMNet.py       # SAFMNPP
│   │       ├── SCSA_Bottleneckt.py # SCSABlock
│   │       ├── BiFPN.py         # 加权双向特征金字塔
│   │       ├── SPD_Conv.py      # 空间到深度卷积
│   │       ├── SPPF_CSPC.py     # 跨阶段 SPPF
│   │       ├── SPPF_LSKa.py     # 大核注意力 SPPF
│   │       ├── AFF.py           # 注意力特征融合
│   │       ├── AKConv.py        # 可变核卷积
│   │       ├── EMA_attention.py # EMA 注意力
│   │       └── ... (更多备用模块)
│   ├── engine/                  # 训练/推理/导出引擎
│   ├── models/yolo/detect/      # YOLO 检测任务实现
│   ├── data/                    # 数据加载 + 增强
│   └── utils/                   # 工具函数库
│
├── models/                      # 训练好的 .pt 权重
├── img_video/                   # 测试图片/视频样本
├── ui_images/                   # UI 图标资源
├── data/                        # 用户数据库
└── NEU-DET/                     # 钢材缺陷数据集
```

---

## 🛠️ 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 深度学习框架 | PyTorch 2.0+ | 模型训练与推理 |
| 检测框架 | YOLO11 (ultralytics 8.3.2) | 基础检测模型 |
| GUI 框架 | PyQt5 / Qt Designer | 桌面应用程序 |
| 图像处理 | OpenCV | 图像读写与可视化 |
| 数据库 | SQLite | 用户账户管理 |
| 注意力模块 | einops | 张量重排（SCSA） |

---

## 🔗 参考资料

本项目改进参考了以下研究工作：

| 改进模块 | 参考论文 |
|---------|---------|
| SAFMNPP | Sun et al., "Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution", ICCV 2023 |
| SCSA | "SCSA: Spatial-Channel Self-Attention for Visual Recognition" |
| Strip Pooling | Hou et al., "Strip Pooling: Rethinking Spatial Pooling for Scene Parsing", CVPR 2021 |
| iAFF | Zhang et al., "iAFF: Iterative Attentional Feature Fusion", WACV 2022 |
| InceptionDWConv2d | Inspired by Inception Networks (Szegedy et al.) |

---

## 📄 License

本项目基于 [AGPL-3.0](LICENSE) 协议开源。ultralytics 原版代码版权归 Ultralytics 所有。
