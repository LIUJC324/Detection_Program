# 2026-04-11 Hardcase Refine Light

## 背景

上一版 hard-case 增强过重，训练结果明显回落：

- 暗光增强过强
- 弱模态扰动过强
- 运动模糊增强过强

导致：

- `TP / recall50 / small_recall50` 都比当前较优线上版本更差

因此本次切到：

- **更轻量的 hard-case 精修**

## 借鉴思路

这次主要借鉴的是开源项目里的处理方向，而不是整套结构迁移：

1. `UA-CMDet`
   - illumination-aware
   - uncertainty-aware
   - 启发：低照处理要有，但不能把正常样本大面积扰坏

2. `CALNet-Dronevehicle`
   - 跨模态冲突处理
   - 启发：弱模态模拟要更温和，避免把融合训练直接打散

3. `EAEFNet`
   - 弱模态显式建模
   - 启发：继续保留弱模态增强，但幅度要轻

## 本次配置

- 配置文件：
  [configs/experiment_hardcase_refine_light.yaml](/home/liujuncheng/rgbt_uav_detection/configs/experiment_hardcase_refine_light.yaml)

核心变化：

1. 从当前较优快照继续：
   - [weights/deploy_best_calibrated_20260411.pt](/home/liujuncheng/rgbt_uav_detection/weights/deploy_best_calibrated_20260411.pt)
2. 降低 hard-case 扰动强度
3. 保留但减弱：
   - lowlight augmentation
   - weak modality augmentation
   - motion blur augmentation

## 目的

目标不是再强行追求“更重的 hard-case”，而是：

1. 保住当前较好的基础命中率
2. 小幅补暗处与高速场景
3. 避免重增强把几何框学习再次打坏
