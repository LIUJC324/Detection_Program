# 2026-04-11 Hardcase Refine

## 背景

当前线上观感已经明显改善，但剩余问题集中在：

1. 暗处车辆
2. 车辆密集区域
3. 高速运动导致的模糊目标

这说明当前瓶颈已经从几何框偏差，进一步转向：

- hard-case robustness

## 本次处理

已改文件：

- [data/transforms.py](/home/liujuncheng/rgbt_uav_detection/data/transforms.py)
- [scripts/train.py](/home/liujuncheng/rgbt_uav_detection/scripts/train.py)
- [configs/experiment_hardcase_refine.yaml](/home/liujuncheng/rgbt_uav_detection/configs/experiment_hardcase_refine.yaml)

新增增强：

1. `RandomMotionBlurPair`
   - 同时对 RGB / Thermal 应用同一模糊核
   - 目标是补高速运动和轻微拖影

当前 hard-case 配置方向：

1. 降低随机裁剪强度
2. 增强低照模拟
3. 保留弱模态扰动
4. 增加运动模糊增强

## 训练目标

下一阶段不再主要追求：

- 普通场景继续堆总体召回

而是重点追求：

1. 暗处命中率
2. 密集区漏检率
3. 高速目标可检出性

## 下一步

当前低学习率训练跑完后，优先启动：

- [configs/experiment_hardcase_refine.yaml](/home/liujuncheng/rgbt_uav_detection/configs/experiment_hardcase_refine.yaml)
