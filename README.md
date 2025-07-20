# 基于WiFi CSI数据的多场景人体存在检测项目

本项目利用WiFi信道状态信息（CSI），通过深度学习模型判断多个房间内的人体存在情况。


注意：目前只对home_scene1和home_scene2两个场景进行训练，其他场景往后添加。

## 项目特点

- **多场景数据处理**: 支持对多个来源的原始数据集（如 `Home_Scene1`, `Home_Scene2`）进行统一的预处理，并可配置将其用于训练。
- **可配置的模型架构**: 用户可在配置文件中选择使用**Transformer编码器**或**Vision Transformer (ViT)** 模型。
- **自定义损失函数**: 支持将多种损失函数（如 `BCEWithLogitsLoss`, `FocalLoss`）进行加权组合，以适应多标签分类任务的需求。
- **并行化预处理**: 数据预处理脚本采用多进程并行处理，以提高在大型数据集上的处理效率。
- **训练过程监控**: 通过 TensorBoard 记录训练过程中的指标，包括两种准确率（总体准确率和完全匹配率），以供分析模型性能。

## 工作流程

请遵循以下步骤来配置环境、预处理数据及训练模型。

### 第1步：环境设置

首先，为所有脚本文件赋予执行权限，然后运行环境设置脚本以创建Conda环境并安装所需依赖。

```bash
chmod +x scripts/*.sh
./scripts/setup_env.sh
```
*注意: `setup_env.sh` 脚本将创建并激活一个名为 `sdp-wifi-sensing` 的Conda环境。后续的核心脚本 (`preprocess.sh`, `start_training.sh`) 会自动激活此环境。*

### 第2步：数据预处理

此步骤负责将原始的 `.txt` 数据文件转换为模型能够读取的 `.pt` 张量文件。脚本会根据配置文件 (`config.yml` 中 `scenes_to_process` 列表)指定的场景，在 `datasets/predata/` 目录下为每个场景创建独立的子目录来存放预处理结果。

```bash
# 运行并行预处理脚本
./scripts/preprocess.sh
```
*例如，在处理 `Home_Scene1` 和 `Home_Scene2` 后，会生成 `datasets/predata/Home_Scene1` 和 `datasets/predata/Home_Scene2` 两个目录。*

### 第3步：计算类别权重 (重要步骤)

由于数据集中可能存在类别不平衡（例如“无人”状态显著多于“有人”状态），建议为损失函数计算类别权重。此脚本会分析指定的训练数据源，并计算出相应的 `pos_weight`。

```bash
# 运行权重计算脚本
python scripts/calculate_pos_weight.py
```
脚本执行后会输出一行类似 `pos_weight: [4.4407, 3.969, 3.7598]` 的结果。**请将此结果复制并更新到 `config.yml` 文件的 `training.pos_weight` 字段。**

*注意：每当 `config.yml` 中的 `training_data_sources` 配置发生变化时，建议重新运行此步骤。*

### 第4步：开始训练

完成以上步骤后，即可开始训练模型。训练脚本将依据 `config.yml` 的配置（如加载的数据源、使用的模型等）来启动训练流程。

```bash
# 启动模型训练
./scripts/start_training.sh
```
在运行时，也可以通过命令行参数覆盖配置文件中的设置，例如更改学习率：
`./scripts/start_training.sh --training.learning_rate 0.0005`

### 第5步：使用TensorBoard监控训练

训练过程中的指标数据（如损失、学习率、准确率等）会被记录在 `training/` 目录下。可使用TensorBoard对这些数据进行可视化。

```bash
# 在新的终端中运行此命令 (请确保已激活对应的Conda环境)
conda activate sdp-wifi-sensing
tensorboard --logdir training
```
之后在浏览器中打开提示的URL (通常是 `http://localhost:6006/`) 即可查看。

## 文件结构

```
.
├── datasets/
│   ├── Home_Scene1/  # 原始数据场景1
│   ├── Home_Scene2/  # 原始数据场景2
│   └── predata/      # 预处理后的.pt文件将按场景存放于此
├── models/           # 训练好的模型检查点 (.pth) 将保存于此
├── scripts/          # 可执行脚本
├── src/              # Python源代码
├── config.yml        # 项目主配置文件
├── dataset.md        # 数据集结构与处理流程说明
└── README.md         # 本文档
``` 