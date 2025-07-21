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
chmod +x env/*.sh
./env/setup_env.sh
```
*注意: `setup_env.sh` 脚本将创建并激活一个名为 `sdp-wifi-sensing` 的Conda环境。后续的核心脚本 (`preprocess.sh`, `start_training.sh`) 会自动激活此环境。*

### 第2步：数据预处理

此步骤负责将原始的 `.txt` 数据文件转换为模型能够读取的 `.pt` 张量文件。脚本会根据配置文件 (`config.yml` 中 `scenes_to_process` 列表)指定的场景，在 `datasets/predata/` 目录下为每个场景创建独立的子目录来存放预处理结果。

```bash
# 运行并行预处理脚本
./scripts/preprocess.sh
```
*例如，在处理 `Home_Scene1` 和 `Home_Scene2` 后，会生成 `datasets/predata/Home_Scene1` 和 `datasets/predata/Home_Scene2` 两个目录。*

### 第3步：计算类别权重 

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

## 高级配置

### 数据集划分策略

本项目支持两种不同的训练集/验证集划分策略，您可以在 `config.yml` 中通过 `data_split.strategy` 字段进行配置。

#### 1. `strategy: 'preprocess'` (默认, 推荐)

这是默认的策略，也是更科学的划分方法。

- **工作方式**: 在运行 `preprocess.sh` 脚本时，程序会以**文件组**（即一次完整的实验录制）为最小单位，将一个场景下的所有文件组按80/20的比例划分，并分别存入 `train/` 和 `val/` 两个子目录。
- **优点**: 这种方法可以有效**防止数据泄露**。它能确保来自同一次实验的所有数据片段（它们具有高度相似的背景噪声和环境特征）要么全部进入训练集，要么全部进入验证集。这使得验证集的评估结果能更真实地反映模型在面对**全新**数据时的泛化能力。
- **适用场景**: 所有正式的、以评估模型性能为目的的实验。

#### 2. `strategy: 'on_the_fly'`

此策略提供了更大的灵活性。

- **工作方式**: 预处理脚本 `preprocess.sh` 仍然会创建 `train/` 和 `val/` 目录。但是，当运行 `start_training.sh` 启动训练时，程序会**忽略**这个预划分，而是将所有指定数据源的 `train/` 和 `val/` 目录下的全部 `.pt` 文件合并成一个大的数据集，然后**按单个样本**进行完全随机的80/20切分。
- **优点**: 可以在不重新进行耗时预处理的情况下，尝试不同的随机划分。
- **缺点**: 存在数据泄露的风险。来自同一个文件组的数据片段可能同时出现在训练集和验证集中，导致验证集准确率**过于乐观**，无法完全代表模型的泛化性能。
- **适用场景**: 用于快速迭代、代码调试或初步探索，不作为最终的模型性能评估依据。

*注意：无论使用哪种策略，`calculate_pos_weight.py` 脚本都能智能地识别并只在正确的训练数据子集上计算权重。*

### 自定义模型文件名

您可以自定义训练过程中保存的最佳模型的文件名。在 `config.yml` 中，修改 `training.model_save_name` 字段。

该字段支持以下占位符：
- `{project_name}`: 项目名称 (在 `config.yml` 中配置)
- `{model_name}`: 使用的模型名称 (例如 `simple_transformer` 或 `vit`)
- `{timestamp}`: 训练开始时的时间戳 (格式: `YYYYMMDD-HHMMSS`)。**注意：此值为程序运行时自动生成，无需在config中配置。**

**示例**:
```yaml
# config.yml
training:
  model_save_name: "{project_name}_{model_name}_{timestamp}_best.pth"
```
这将会生成类似 `sdp-wifi-sensing_vit_20231027-143000_best.pth` 的文件名。 