# 数据集结构与文件命名说明

本文档用于说明本项目所使用的原始 `.txt` 数据集的目录结构与文件命名规则。这套规则是数据预处理脚本 (`src/preprocess.py`) 正确配对5个关联文件（2个CSI数据文件 + 3个GroundTruth文件）的基础。

## 文件配对机制

为了构成一个有效的训练样本，系统需要将来自同一时间、同一实验条件下的5个文件进行组合：
- `RoomA` 的CSI数据 (`RoomAData`)
- `RoomB` 的CSI数据 (`RoomBData`)
- `RoomA` 的在场状态标签 (`RoomAGroundTruth`)
- `RoomB` 的在场状态标签 (`RoomBGroundTruth`)
- `LivingRoom` 的在场状态标签 (`LivingRoomGroundTruth`)

`src/dataset.py` 中的 `find_file_groups` 函数通过 `os.walk` 遍历文件夹，以 `RoomAData` 目录下的文件名为基准，依据文件名中编码的数字逻辑，推导出其余4个关联文件的路径，并将它们组合成一个文件组。

## 文件名编码规则

文件名（例如 `1100_01.txt` 或 `1201_03.txt`）包含了文件配对所需的信息。文件名由一个4位数的**逻辑码**和一个2位数的**重复码**组成，中间以下划线 `_` 分隔。

### 4位数逻辑码 (例如 `1100`, `1201`)

该代码定义了实验的特定条件。

- **前两位 (`11` 或 `12`)**: 用于区分场景和带宽。
  - `11xx`: 对应 `Home_Scene1` 和 `Home_Scene2` 中的 `5G_80M` 带宽数据。
  - `12xx`: 对应 `Home_Scene2` 中的 `5G_160M` 带宽数据。

- **第三位 (`0`, `1`, `2`)**: 标记房间。
  - `x10x` 或 `x20x`: RoomA
  - `x11x` 或 `x21x`: RoomB
  - `x12x` 或 `x22x`: LivingRoom

- **第四位 (`0`, `1` 或 `2`)**: 标记干扰类型。
  - `xx00`: NoInterference (无干扰)
  - `xx01`: GreenPlantFan (绿植风扇干扰)
  - `xx02`: RobotVacuum (扫地机器人干扰)

### 文件配对逻辑示例

假设脚本在 `RoomAData` 目录中找到文件 `1201_03.txt`。
- **解析**:
  - 场景/带宽码: `12` (Scene2, 160M)
  - 房间码: `0` (RoomA)
  - 干扰码: `1` (GreenPlantFan)
  - 重复码: `03`
- **推导**:
  - `RoomB` 对应的数据文件，其房间码应为 `1`，因此文件名为 `1211_03.txt`。
  - `LivingRoom` 对应的GroundTruth文件，其房间码应为 `2`，因此文件名为 `groundtruth_1221_03.txt`。
  - 其余GroundTruth文件则通过在原始文件名上添加前缀来定位。

该逻辑在 `src/dataset.py` 的 `find_file_groups` 函数中实现，以确保在不同场景和带宽条件下，数据文件都能被正确地组合。