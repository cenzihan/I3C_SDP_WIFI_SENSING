# 基于WiFi CSI的人- 体存在检测

本项目使用WiFi信道状态信息（CSI），通过深度学习模型在多个房间中检测人体存在。

## 功能

- **模型架构**:
  - **双流Transformer (Dual-Stream Transformer)**: 分别处理来自两个独立WiFi流的数据，然后进行融合。
  - **多任务Transformer (Multi-Task Transformer)**: 使用单个模型同时预测三个区域（例如，房间A、房间B和客厅）的存在情况，包含共享的特征提取层和任务特定的分类头。
  - **自适应权重 (Adaptive Weighting)**: 在多任务模型中，此机制可让网络自行学习各输入流对每个预测任务的贡献度。
- **数据处理**:
  - **并行化预处理**: 利用多进程处理原始CSI文本文件，并将其转换为`.pt`张量文件。
  - **数据划分策略**: 提供多种训练集与验证集的划分策略，其中包括`group_level_random_split`，该策略旨在防止同一实验记录中的数据泄露。
- **配置与执行**:
  - **YAML配置**: `config.yml`文件用于配置模型架构、超参数、数据源和训练参数。
  - **命令行覆盖**: 运行时可通过命令行参数覆盖`config.yml`中的设置。
- **训练与监控**:
  - **日志系统**: 将训练过程中的进度、指标和配置输出到日志文件（`training.log`）和控制台。
  - **TensorBoard集成**: 将损失、准确率、学习率等关键指标写入TensorBoard事件文件，用于训练过程的可视化。
  - **评估指标**: 计算并记录总体准确率、完全匹配率和各子任务的准确率。

## 项目结构

```
.
├── datasets/
│   ├── Home_Scene1/          # 场景1的原始数据
│   ├── Home_Scene2/          # 场景2的原始数据
│   └── predata/              # 按场景组织的预处理后的.pt文件
├── models/                   # 保存的模型检查点 (.pth)
├── results/                  # 保存的产物，如混淆矩阵
├── scripts/                  # 辅助性Shell脚本
├── src/                      # Python源代码
├── training/                 # TensorBoard日志和训练日志文件
├── config.yml                # 项目主配置文件
├── env/                      # 环境设置文件
└── README.md                 # 本文档
```

## 工作流程

### 第1步：环境设置

本项目推荐使用Conda进行环境设置。

```bash
# 为设置脚本授予执行权限
chmod +x env/*.sh

# 运行设置脚本
./env/setup_env.sh
```

### 第2步：解压原始数据集

在预处理之前，需解压原始CSI数据。

```bash
# 导航到datasets目录
cd datasets

# 授予执行权限并运行解压脚本
chmod +x start_extract.sh
./start_extract.sh

# 返回项目根目录
cd ..
```

### 第3步：数据预处理

此步骤将原始的`.txt`数据转换为`.pt`张量文件。脚本会依据`config.yml`中的`scenes_to_process`列表进行处理，并将输出保存到`datasets/predata/`目录。

```bash
# 授予执行权限并运行预处理脚本
chmod +x scripts/preprocess.sh
./scripts/preprocess.sh
```

### 第4步：（可选）计算类别权重

如果数据集中存在类别不平衡，可以为损失函数计算类别权重。

```bash
# 运行权重计算脚本
python -m scripts.calculate_pos_weight
```

脚本会输出`pos_weight`值，例如 `pos_weight: [4.44, 3.97, 3.76]`。可将此列表更新到`config.yml`的`training.pos_weight`字段。

### 第5步：开始训练

数据准备就绪后，开始模型训练。

```bash
# 授予执行权限并运行训练脚本
chmod +x scripts/start_training.sh
./scripts/start_training.sh
```

如需覆盖配置，可使用命令行参数。例如，更改学习率：

```bash
./scripts/start_training.sh --learning_rate 0.0005
```

### 第6步：使用TensorBoard监控

训练指标记录在`training/`目录中，可使用TensorBoard进行可视化。

```bash
# 启动TensorBoard
tensorboard --logdir=training/
```

## 配置详情

### 数据划分策略 (`data_split.strategy`)

在`config.yml`中，可定义数据集的划分方式：

- **`group_level_random_split`**: 在文件组（对应一次完整的实验记录）级别上进行随机划分。这可以确保来自同一记录的数据不会同时出现在训练集和验证集中。
- **`preprocess`**: `preprocess.sh`脚本在执行时会将文件组划分为`train`和`val`子目录，此策略直接使用这个预划分。
- **`on_the_fly`**: 此策略将合并所有指定数据源的`.pt`文件，并在单个样本级别上执行新的随机划分。注意，此方法可能导致来自同一记录的样本同时出现在训练集和验证集中。

### 模型保存名称 (`training.model_save_name`)

在`config.yml`中，可使用占位符自定义保存模型的名称。

- `{project_name}`: 项目名称。
- `{model_name}`: 正在训练的模型的名称。
- `{timestamp}`: 训练开始时生成的时间戳。

**`config.yml`中的示例:**
```yaml
training:
  model_save_name: "{project_name}_{model_name}_{timestamp}_best.pth"
```
这将生成一个类似`sdp-wifi-sensing_multi_task_transformer_20231115-103000_best.pth`的文件名。 