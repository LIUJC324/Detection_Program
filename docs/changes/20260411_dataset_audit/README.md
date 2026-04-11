# 2026-04-11 Dataset Audit

## 本次动作

1. 已停止训练会话 `rgbt_reliability_refine_20260411`
2. 审计本地 `datasets/dronevehicle_like_refined`
3. 复核本地生成脚本与原始数据来源
4. 检索可作为下一阶段补充的数据集

## 结论先说

当前问题不能简单归因成“模型不行”或“数据不行”中的单一一项。

更准确的判断是：

1. 早期主问题是模型框架问题
2. 当前继续提分困难时，数据已经成为主要瓶颈之一
3. 当前本地数据集最关键的问题不是“脏”，而是：
   - 规模太小
   - 类别严重不均衡
   - train / val 切分口径不合理
   - 数据来源只覆盖了官方 `validation` 部分

## 本地数据审计结果

### 1. 数据完整性

`dronevehicle_like_refined` 当前结果：

- train:
  - RGB: `1175`
  - Thermal: `1175`
  - Annotation: `1175`
- val:
  - RGB: `294`
  - Thermal: `294`
  - Annotation: `294`

配对结论：

- `rgb_only = 0`
- `thermal_only = 0`
- `missing_annotations_for_pairs = 0`

说明：

- 当前数据不是“文件配对错了”或“标签缺失一大片”

### 2. 图像尺寸与白边处理

本地原始 `DroneVehicle` 图像尺寸：

- `840 x 712`

与官方说明一致。

本地处理后尺寸分布：

- 宽度范围：`640 ~ 645`
- 高度范围：`512 ~ 518`
- 主峰尺寸：`640 x 512`

处理摘要：

- `cropped_samples = 1469`
- 平均裁掉边距约：
  - left `99.43`
  - top `99.39`
  - right `99.80`
  - bottom `99.75`

边缘白边抽样结果：

- train 抽样 `sampled_white_border_like_ratio = 0.0`
- val 抽样 `sampled_white_border_like_ratio = 0.0`

说明：

- 白边裁剪基本是对的
- 这一点不是当前主要问题

### 3. 标注质量

标注框审计：

- `invalid_boxes = 0`
- `out_of_bounds_boxes = 0`

说明：

- 当前 JSON 标注不存在大规模退化框或越界框
- 标注格式本身基本健康

### 4. 目标分布

当前训练集：

- 总框数：`17098`
- `car = 14440`
- `truck = 1056`
- `bus = 587`
- `van = 505`
- `freight_car = 510`

说明：

- `car` 占比约 `84%`
- 长尾类别非常弱
- 这会直接推高：
  - 主类偏置
  - 小类漏检
  - 复杂背景误检

### 5. 目标尺度与边界分布

当前本地统计：

- train:
  - `bbox_area_median = 1914`
  - `bbox_area_p10 = 880`
  - `bbox_area_p90 = 4644`
- val:
  - `bbox_area_median = 1922`
  - `bbox_area_p10 = 940`
  - `bbox_area_p90 = 5110`

边界框贴边比例：

- train: `edge_touch_box_ratio = 0.1455`
- val: `edge_touch_box_ratio = 0.1446`

说明：

- 这类“靠边目标”在 UAV 视角里是常见现象
- 不一定是错误，但会增加检测难度

## 最关键的问题：本地数据口径不合理

### 1. 当前本地数据只来自官方 validation 原始包

本地原始目录只有：

- `datasets/raw/dronevehicle/val_unpack/val/...`

原始图像 / 标注数量：

- `valimg = 1469`
- `vallabel = 1469`
- `vallabelr = 1469`

生成脚本默认参数也明确写成：

- `--source-root .../datasets/raw/dronevehicle/val_unpack/val`

而且生成脚本后续做的是：

1. 先把全部样本收集到一起
2. 再 `shuffle`
3. 再按 `train_ratio=0.8` 随机切成：
   - train `1175`
   - val `294`

### 2. 这意味着什么

这意味着当前本地训练 / 验证集其实是：

- **从官方 validation 子集随机再切出来的**

而不是：

- 官方 train / validation 的标准口径

这会带来几个问题：

1. 数据规模被锁死在 `1469` 对
2. 没有利用官方训练集
3. 当前验证结果和官方口径不可直接对齐
4. 训练与验证的场景分布本身更接近，评估代表性有限

这是目前最需要先修的地方。

## 次关键问题：标注融合策略可能引入噪声

当前数据生成采用：

- `annotation_source = merged_union`

处理摘要显示：

- `rgb_objects = 21198`
- `thermal_objects = 23035`
- `final_objects = 21198`
- `merged_objects = 20902`
- `unused_secondary_objects = 2133`

说明：

- 当前最终监督目标本质上仍以 RGB 标注数为主
- thermal 里一部分目标不会进入最终监督
- 跨模态 union 也可能把框扩得更大，带来定位噪声

这不是当前第一优先问题，但后续应该做对照实验：

1. `annotation_source = rgb`
2. `annotation_source = thermal`
3. `annotation_source = merged_union`

比较哪一种对 `FP` 更敏感。

## 对“是不是数据集的问题”的最终判断

是，但不是“纯数据质量差”这么简单。

当前更准确的结论是：

1. **数据规模与切分口径有问题**
2. **类别分布有问题**
3. **低照 / 弱模态 / hard negative 的覆盖仍不够**
4. **模型结构也仍有限制**

所以不能只做一件事。

最优先顺序应是：

1. 先修数据口径
2. 再扩充同类数据
3. 再做辅助预训练

## 推荐补充的数据集

下面按优先级排序。

### A. 第一优先级：同源主数据

1. `DroneVehicle` 官方全集

为什么第一优先：

- 任务、类别、模态与当前项目完全同源
- 官方页面明确提供：
  - `Train`
  - `Validation`
  - `Test`

应该优先做的不是“换数据集”，而是：

1. 下载并接入官方 `Train`
2. 保留官方 `Validation`
3. 不再从 `Validation` 里随机切 train/val

### B. 第二优先级：UAV 视角车辆 / 小目标数据

2. `VisDrone`

价值：

- 大规模 UAV 视角数据
- 适合给 RGB backbone 学 aerial small-object 先验

3. `UAVDT`

价值：

- 也是 UAV 车辆场景
- 对“复杂背景 + 小目标 + 车辆类别”更贴近

4. `DOTA`

价值：

- 大型航空遥感检测基准
- 适合给 aerial detection backbone 学泛化能力

说明：

- `VisDrone / UAVDT / DOTA` 不是 RGB-T
- 它们更适合做：
  - RGB branch warmup
  - detection head warmup
- 不适合直接替代最终主任务评估

### C. 第三优先级：RGB-T / low-light 辅助数据

5. `M3FD`

价值：

- 配准 RGB-T
- 带检测标签
- 适合给 thermal / fusion 分支做预训练

6. `FLIR ADAS Dataset`

价值：

- 可见光 + thermal
- 类别里含 car / bus / truck
- 适合补 thermal objectness 和复杂低照条件

7. `LLVIP`

价值：

- 极低照
- 时空严格对齐
- 适合补弱模态 / 低照鲁棒性

说明：

- `M3FD / FLIR / LLVIP` 都不是 UAV 主任务口径
- 更适合做：
  - fusion warmup
  - thermal warmup
  - low-light robustness warmup

### D. 低优先级辅助集

8. `KAIST Multispectral Pedestrian`
9. `MSRS`
10. `RoadScene`
11. `TNO`

说明：

- 这些更适合做：
  - RGB-T 对齐
  - 弱光
  - 融合方法验证
- 不适合作为当前车辆检测主训练集

## 最务实的下一步

### 先做

1. 下载官方 `DroneVehicle Train + Validation`
2. 重建本地 `dronevehicle_like_refined_v2`
3. 保持官方 split，不再随机切官方 validation
4. 跑一条最朴素的 baseline 对照

### 再做

5. 给 RGB 分支加 `VisDrone / UAVDT` warmup
6. 给 fusion / thermal 分支加 `M3FD / FLIR / LLVIP` warmup

### 暂时不要做

1. 继续只在当前 1469 对数据上长时间调小结构
2. 一次性混入太多异构数据集
3. 不做类别重映射就直接混训

## 参考来源

- 官方 / 高质量公开源：
  - `https://github.com/VisDrone/DroneVehicle`
  - `https://github.com/VisDrone/VisDrone-Dataset`
  - `https://sites.google.com/view/daweidu/projects/uavdt`
  - `https://captain-whu.github.io/DOTA/dataset.html`
  - `https://github.com/JinyuanLiu-CV/TarDAL`
  - `https://oem.flir.com/solutions/automotive/adas-dataset-form/`
  - `https://bupt-ai-cz.github.io/LLVIP/`
  - `https://soonminhwang.github.io/rgbt-ped-detection/`
  - `https://github.com/Linfeng-Tang/MSRS`
  - `https://github.com/jiayi-ma/RoadScene`
  - `https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029`
