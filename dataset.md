# 原始数据集结构与命名规则

本文档旨在阐明本项目所使用的原始 `.txt` 数据集的目录结构与文件命名规则。这套规则是数据预处理脚本 (`src/preprocess.py`) 能够正确配对5个关联文件（2个CSI数据文件 + 3个GroundTruth文件）的基础。

## 核心理念：文件配对

一个有效的训练样本，必须由来自同一时间、同一实验条件下的5个文件组成：
- `RoomA` 的CSI数据
- `RoomB` 的CSI数据
- `RoomA` 的在场状态 (Ground Truth)
- `RoomB` 的在场状态 (Ground Truth)
- `LivingRoom` 的在场状态 (Ground Truth)

`find_file_groups` 函数通过`os.walk`遍历文件夹，以 `RoomAData` 目录中的文件为基准，根据文件名中编码的数字逻辑，推导出其余4个文件的路径，从而将它们组合成一个“文件组”。

## 文件名编码规则

文件名（如 `1100_01.txt` 或 `1201_03.txt`）是理解配对逻辑的关键。文件名由一个4位数的**逻辑码**和一个2位数的**重复码**组成。

### 4位数逻辑码 (例如 `1100`, `1201`)

这个4位数的代码蕴含了实验的所有条件。

- **前两位 (`11` 或 `12`)**: 用于区分不同的场景和带宽。
  - `11xx`: 对应 `Home_Scene1` 和 `Home_Scene2` 中的 `5G_80M` 带宽数据。
  - `12xx`: 对应 `Home_Scene2` 中的 `5G_160M` 带宽数据。

- **第三位 (`0`, `1`, `2`)**: 标记房间。
  - `x10x` 或 `x20x`: RoomA
  - `x11x` 或 `x21x`: RoomB
  - `x12x` 或 `x22x`: LivingRoom

- **第四位 (`0`, `1`, `2`)**: 标记干扰类型。
  - `xx00`: NoInterference (无干扰)
  - `xx01`: GreenPlantFan (绿植风扇干扰)
  - `xx02`: RobotVacuum (扫地机器人干扰)

### 文件配对逻辑示例

假设脚本在 `RoomAData` 目录下找到了文件 `1201_03.txt`。
- **解析**:
  - 场景/带宽码: `12` (Scene2, 160M)
  - 房间码: `0` (RoomA)
  - 干扰码: `1` (GreenPlantFan)
  - 重复码: `03`
- **推导**:
  - `RoomB` 对应的数据文件，房间码应为 `1`，所以是 `1211_03.txt`。
  - `LivingRoom` 对应的GroundTruth文件，房间码应为 `2`，所以是 `groundtruth_1221_03.txt`。
  - 其余GroundTruth文件只是在原文件名上加前缀。

这个逻辑被硬编码在 `src/dataset.py` 的 `find_file_groups` 函数中，使其能够为不同场景、不同带宽下的数据文件正确配对。