# 2026-04-11 Optimization Roadmap

## 关机前状态

截至 `2026-04-11 14:57 +0800`：

1. `reliability_aware` 新结构训练已停止
2. 当前不建议继续恢复该训练
3. 当前优先级已经从“继续改小结构”切换到“修数据口径 + 扩充数据 + 再做对照实验”

当前关键参考：

- 主训练跟踪文档：
  [docs/training/模型端问题排查、框架修复与续训方案_20260409.md](/home/liujuncheng/rgbt_uav_detection/docs/training/模型端问题排查、框架修复与续训方案_20260409.md)
- 本轮 reliability refine 留档：
  [docs/changes/20260411_reliability_refine/README.md](/home/liujuncheng/rgbt_uav_detection/docs/changes/20260411_reliability_refine/README.md)
- 本地数据审计：
  [docs/changes/20260411_dataset_audit/README.md](/home/liujuncheng/rgbt_uav_detection/docs/changes/20260411_dataset_audit/README.md)

## 当前判断

当前最重要的结论有三条：

1. 早期主问题是模型框架问题，这部分已经修过
2. 现在继续提分困难，数据已经是主要瓶颈之一
3. 当前本地数据集最大问题不是“标签坏了”，而是：
   - 只用了官方 `validation` 原始包
   - 再从里面随机切出 train / val
   - 规模太小
   - 类别严重不均衡

所以后续不要再按“继续调一轮训练参数”作为主路线。

## 下次开机后直接执行的顺序

### 第 0 步：先看这三份文档

1. [docs/changes/20260411_optimization_roadmap/README.md](/home/liujuncheng/rgbt_uav_detection/docs/changes/20260411_optimization_roadmap/README.md)
2. [docs/changes/20260411_dataset_audit/README.md](/home/liujuncheng/rgbt_uav_detection/docs/changes/20260411_dataset_audit/README.md)
3. [docs/training/模型端问题排查、框架修复与续训方案_20260409.md](/home/liujuncheng/rgbt_uav_detection/docs/training/模型端问题排查、框架修复与续训方案_20260409.md)

### 第 1 步：修正主数据口径

目标：

- 不再用“官方 validation 再随机切 train/val”的口径
- 改成官方 `Train + Validation` 的标准使用方式

要做的事：

1. 下载 / 补齐官方 `DroneVehicle` 的 `Train` 和 `Validation`
2. 保留原始数据目录，不覆盖现有 `val_unpack/val`
3. 新建目标数据集目录，例如：
   - `datasets/dronevehicle_like_official_v2`

完成标准：

1. 原始数据目录中同时存在官方 train 和 val
2. 新数据集生成脚本支持显式指定官方 split
3. 不再做随机切分 validation 充当 train

### 第 2 步：重建本地数据集 v2

目标：

- 用官方 split 重新生成训练可用数据集

建议保留的处理：

1. 白边裁剪
2. RGB/Thermal 成对对齐
3. 统一输出本地 JSON 标注格式

建议新增的对照实验开关：

1. `annotation_source = rgb`
2. `annotation_source = thermal`
3. `annotation_source = merged_union`

建议不要继续默认只保留一种策略而不做对照。

完成标准：

1. 新数据集完整生成
2. 重新输出一份数据审计摘要
3. 确认 train / val 数量、类别分布、图像尺寸与官方口径一致

### 第 3 步：在 v2 数据集上跑最朴素 baseline

目标：

- 不叠加太多新技巧，先验证“数据口径修正”本身是否带来改善

建议配置原则：

1. 先使用当前已验证过的稳定框架
2. 不先引入复杂新 fusion
3. 不先混入外部异构数据集

优先观察：

1. `TP`
2. `FP`
3. `recall50`
4. `small_recall50`

完成标准：

如果只修数据口径就能明显优于旧基线，则说明数据问题是当前主因之一。

旧基线可参考：

- [outputs/train_runs/run_20260411_121944/run.log](/home/liujuncheng/rgbt_uav_detection/outputs/train_runs/run_20260411_121944/run.log)

### 第 4 步：做 annotation_source 消融

目标：

- 判断当前 `merged_union` 是否真的有利

建议最少跑三条短实验：

1. `rgb`
2. `thermal`
3. `merged_union`

重点判断：

1. 谁的 `FP` 更低
2. 谁的 `TP` 更稳
3. 谁的 `small_recall50` 更好

如果 `merged_union` 明显拉高 `FP`，后续应考虑放弃它。

### 第 5 步：再决定是否加外部数据集

只有在下面情况之一成立时，再引入外部数据：

1. 官方 `DroneVehicle` 标准口径跑完后，`TP / recall50 / small_recall50` 仍不理想
2. `FP` 仍长期卡在高位
3. 低照 / 弱模态场景明显不稳

引入顺序建议：

1. UAV 视角可见光预训练：
   - `VisDrone`
   - `UAVDT`
   - `DOTA`
2. RGB-T / low-light 辅助预训练：
   - `M3FD`
   - `FLIR`
   - `LLVIP`

注意：

- 先预训练 / warmup
- 再回到 `DroneVehicle-like` 主任务精调
- 不要一上来把所有异构数据直接混训

## 每一步的停 / 继续标准

### A. 如果修正官方数据口径后

出现下面任意两条，就说明这条线值得继续：

1. `TP` 明显高于旧基线
2. `FP` 明显低于旧基线
3. `recall50` 提升
4. `small_recall50` 提升

### B. 如果修正官方数据口径后仍然没有明显改善

则继续做：

1. `annotation_source` 消融
2. 外部数据 warmup

### C. 如果连官方口径 + 消融之后仍然没明显改善

才考虑：

1. 更强 backbone
2. 更系统的 fusion 重构
3. 更大规模迁移学习路线

## 不要做的事

下次开机后，尽量不要直接做下面这些：

1. 继续恢复当前已停掉的 `reliability_aware` run
2. 再在当前 `1469` 对随机切分数据上长时间炼
3. 一次性混入太多异构数据集
4. 不做对照实验就同时改：
   - 数据口径
   - backbone
   - fusion
   - loss
   - augment

## 下次第一轮实际执行建议

最务实的第一轮任务就是：

1. 补齐官方 `DroneVehicle Train + Validation`
2. 新建 `dronevehicle_like_official_v2`
3. 重新审计
4. 跑 baseline

这四步做完，再决定要不要加 `VisDrone / UAVDT / M3FD / FLIR / LLVIP`。

## 这份文档的用途

下次回来时，不需要重新回忆讨论过程，直接按本文件从上往下执行即可。
