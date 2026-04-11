# 2026-04-11 Quick Preintegration Refine

## 背景

`2026-04-11 17:00` 左右将启动新一轮联调。

为避免临时联调前没有任何新尝试，本次启动一条：

- 结构不变
- 从稳定基线 warm start
- 只跑 `6 epoch`
- checkpoint 独立输出

的短训快跑实验。

## 联调前兜底处理

由于先前 `reliability_aware` 实验曾改写全局：

- `weights/best.pt`

为避免 `deploy_stable.yaml` 或 `deploy.yaml` 误加载到新结构权重，本次先执行了：

- 用 `weights/best_before_reliability_refine_20260411.pt` 恢复 `weights/best.pt`

## 本次短训配置

- 配置文件：
  [configs/experiment_quick_preintegration_refine.yaml](/home/liujuncheng/rgbt_uav_detection/configs/experiment_quick_preintegration_refine.yaml)
- 初始化权重：
  [weights/best_before_reliability_refine_20260411.pt](/home/liujuncheng/rgbt_uav_detection/weights/best_before_reliability_refine_20260411.pt)
- 运行目录：
  [outputs/train_runs/run_20260411_163753](/home/liujuncheng/rgbt_uav_detection/outputs/train_runs/run_20260411_163753)
- stdout 日志：
  [outputs/quick_preintegration_20260411_163800.log](/home/liujuncheng/rgbt_uav_detection/outputs/quick_preintegration_20260411_163800.log)
- 独立 checkpoint 目录：
  [weights/quick_preintegration_20260411](/home/liujuncheng/rgbt_uav_detection/weights/quick_preintegration_20260411)

## 本次参数

1. `epochs = 6`
2. `lr = 8e-5`
3. `warmup_epochs = 1`
4. `num_workers = 0`
5. 不改 backbone，不改 fusion
6. 不覆盖联调使用的独立稳定快照

## 目的

这轮不是为了做结构性突破，而是：

1. 在联调前给出一个可能略优于当前稳定基线的候选快照
2. 同时不破坏现有联调入口
