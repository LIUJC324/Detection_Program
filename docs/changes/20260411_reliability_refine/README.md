# 2026-04-11 Reliability Refine

## 背景

截至 `run_20260411_121944`：

- `epoch 40`
- `recall50 = 0.084878`
- `small_recall50 = 0.122330`
- `TP = 348`
- `FP = 29052`

结论：

- 修复后主线已经从“不会检”走到“能检但不稳”
- 但 `FP` 仍然偏高
- `epoch 30 ~ 40` 已经出现平台期

因此本轮不再继续只靠同一版轻量主线硬炼，而是开始落第一步借鉴优化。

## 本轮改动

### 1. 借鉴优化实现

已改文件：

- [model/network/fusion_module.py](/home/liujuncheng/rgbt_uav_detection/model/network/fusion_module.py)
- [model/network/backbone.py](/home/liujuncheng/rgbt_uav_detection/model/network/backbone.py)
- [model/detector.py](/home/liujuncheng/rgbt_uav_detection/model/detector.py)
- [data/transforms.py](/home/liujuncheng/rgbt_uav_detection/data/transforms.py)
- [scripts/train.py](/home/liujuncheng/rgbt_uav_detection/scripts/train.py)
- [configs/experiment_reliability_refine.yaml](/home/liujuncheng/rgbt_uav_detection/configs/experiment_reliability_refine.yaml)

本轮引入两类借鉴思路：

1. `EAEF / UA-CMDet` 风格的弱模态建模
   - 训练增强里新增弱模态模拟
   - 随机让 RGB 或 thermal 其中一支退化
2. `CALNet / EAEF` 风格的融合改进
   - 新增 `reliability_aware` fusion
   - 融合时显式考虑模态可靠性和冲突区域

### 2. Warm Start 兼容性

为避免新模块一加就无法继续利用旧权重，本轮同时放开了：

- `--init-checkpoint` 的非严格加载

这意味着：

- 旧的 `best.pt / last.pt` 仍可作为新结构的 warm start
- 但 `--resume` 仍保持严格恢复，用于同结构断点续训

## 文档整理

本轮把游离文档收回到了 `docs/` 体系内：

- 接口契约文档移动到：
  [docs/integration/interface.md](/home/liujuncheng/rgbt_uav_detection/docs/integration/interface.md)
- 后端架构图移动到：
  [docs/architecture/后端架构图.png](/home/liujuncheng/rgbt_uav_detection/docs/architecture/后端架构图.png)

同时新增：

- [docs/changes/README.md](/home/liujuncheng/rgbt_uav_detection/docs/changes/README.md)

后续每轮非平凡改动都应新增一个 `docs/changes/YYYYMMDD_topic/README.md`。

## 下一步

1. 先完成本轮借鉴优化代码验证
2. 再按 [experiment_reliability_refine.yaml](/home/liujuncheng/rgbt_uav_detection/configs/experiment_reliability_refine.yaml) 启动新实验
3. 训练结果仍不理想时，再切到更强 backbone 或外部辅助预训练路线

## 已启动实验

本轮新实验已启动：

- `tmux` 会话：`rgbt_reliability_refine_20260411`
- stdout 日志：
  [outputs/reliability_refine_20260411_140333.log](/home/liujuncheng/rgbt_uav_detection/outputs/reliability_refine_20260411_140333.log)
- run 目录：
  [outputs/train_runs/run_20260411_140543](/home/liujuncheng/rgbt_uav_detection/outputs/train_runs/run_20260411_140543)
- 解析后的训练配置：
  [outputs/train_runs/run_20260411_140543/config_resolved.yaml](/home/liujuncheng/rgbt_uav_detection/outputs/train_runs/run_20260411_140543/config_resolved.yaml)

为避免公共权重在新实验里被覆盖后找不到上轮基线，已额外保留快照：

- [weights/best_before_reliability_refine_20260411.pt](/home/liujuncheng/rgbt_uav_detection/weights/best_before_reliability_refine_20260411.pt)
