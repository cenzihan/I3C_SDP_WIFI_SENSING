# 基于WiFi CSI的人体存在检测深度学习项目

本项目使用WiFi信道状态信息（CSI）来检测多个房间内的人体存在状态。


## 核心特性

- **多场景数据处理**: 支持同时处理多个原始数据集（例如 `Home_Scene1`, `Home_Scene2`），并将它们分离或合并用于训练。
- **可切换模型架构**: 在配置文件中可以选择使用两种不同的模型：一个经典的**Transformer编码器**或一个更先进的**Vision Transformer (ViT)**。
- **灵活的损失函数**: 支持多种损失函数（如 `BCEWithLogitsLoss`, `FocalLoss`）的任意加权组合，以应对多标签分类和类别不平衡问题。
- **并行的预处理**: 数据预处理流程使用多进程并行处理，能显著缩短在大型数据集上的处理时间。
- **集成的训练监控**: 使用 TensorBoard 全面监控训练过程，包括两种准确率指标（总体准确率和完全匹配率），帮助深入分析模型性能。

## 工作流程

请遵循以下步骤来设置环境、预处理数据及训练模型。

### 第1步：设置环境

首先，为所有脚本赋予可执行权限，然后运行环境设置脚本来创建Conda环境并安装所有依赖项。

```bash
chmod +x scripts/*.sh
./scripts/setup_env.sh
```
*注意: `setup_env.sh` 脚本会创建并激活名为 `sdp-wifi-sensing` 的Conda环境。对于后续步骤，核心脚本 (`preprocess.sh`, `start_training.sh`) 会自动激活此环境。*

### 第2步：数据预处理

此步骤将原始的 `.txt` 数据文件转换为模型可以直接使用的 `.pt` 张量文件。脚本会为配置文件 (`config.yml` 中 `scenes_to_process` 列表)中的每个场景，在 `datasets/predata/` 目录下创建一个独立的子目录来存放结果。

```bash
# 运行并行预处理脚本
./scripts/preprocess.sh
```
*例如，处理 `Home_Scene1` 和 `Home_Scene2` 后，您将得到 `datasets/predata/Home_Scene1` 和 `datasets/predata/Home_Scene2` 两个目录。*

### 第3步：计算类别权重 (关键步骤)

由于数据存在严重的类别不平衡（“无人”状态远多于“有人”状态），我们需要为损失函数计算精确的权重。此脚本会分析您选择用于训练的数据源，并计算出准确的 `pos_weight`。

```bash
# 运行权重计算脚本
python scripts/calculate_pos_weight.py
```
脚本运行后，会输出一行类似 `pos_weight: [4.4407, 3.969, 3.7598]` 的内容。**请将这行代码复制并更新到 `config.yml` 文件中的 `training.pos_weight` 字段。**

*注意：此步骤应在每次更改 `config.yml` 中的 `training_data_sources` 配置后重新运行。*

### 第4步：开始训练

完成以上所有步骤后，您就可以开始训练模型了。训练脚本会根据 `config.yml` 的配置（加载哪些数据、使用哪个模型等）来启动训练。

```bash
# 启动模型训练
./scripts/start_training.sh
```
您也可以在运行时从命令行覆盖配置参数，例如更改学习率：
`./scripts/start_training.sh --training.learning_rate 0.0005`

### 第5步：使用TensorBoard监控训练

训练过程中的所有指标（损失、学习率、准确率等）都会被记录在 `training/` 目录下。您可以使用TensorBoard来实时可视化这些数据。

```bash
# 在一个新的终端中运行此命令 (确保已激活Conda环境)
conda activate sdp-wifi-sensing
tensorboard --logdir training
```
然后在浏览器中打开显示的URL (通常是 `http://localhost:6006/`)。

## 文件结构简介

```
.
├── datasets/
│   ├── Home_Scene1/  # 原始数据场景1
│   ├── Home_Scene2/  # 原始数据场景2
│   └── predata/      # 预处理后的.pt文件将按场景存放在此
├── models/           # 训练好的模型检查点 (.pth) 将保存在此
├── scripts/          # 所有可执行脚本
├── src/              # 所有Python源代码
├── config.yml        # 关键！所有参数的主配置文件
├── dataset.md        # 数据集结构说明
└── README.md         # 本文档
``` 