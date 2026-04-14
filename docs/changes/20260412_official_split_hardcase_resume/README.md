# 2026-04-12 Official Split Hardcase Resume Note

## 当前状态

截至 `2026-04-12` 当前这次官方 split 数据准备已经完成。

生成结果：

1. 数据集目录：
   - [datasets/dronevehicle_like_official_rgb_expand_v1](/home/liujuncheng/rgbt_uav_detection/datasets/dronevehicle_like_official_rgb_expand_v1)
2. 数据集摘要：
   - [dataset_summary.json](/home/liujuncheng/rgbt_uav_detection/datasets/dronevehicle_like_official_rgb_expand_v1/dataset_summary.json)

关键统计：

1. `num_train = 17962`
2. `num_val = 1469`
3. `split_strategy = official_split`
4. `annotation_source = rgb`
5. `bbox_expand_ratio = 0.12`
6. `bbox_expand_min_pixels = 2.0`
7. `skipped_corrupt_samples = 9`

说明：

1. 官方 `train.zip` 仍存在坏样本与少量异常标注
2. 当前脚本已经兼容：
   - 坏图跳过
   - 未知原始标签跳过
3. 因此这版数据集是“可用优先”的官方 split 版本

## 训练配置

当前用于这条线的训练配置：

- [configs/experiment_hardcase_official_rgb_expand_light.yaml](/home/liujuncheng/rgbt_uav_detection/configs/experiment_hardcase_official_rgb_expand_light.yaml)

核心口径：

1. 数据集根目录：
   - `./datasets/dronevehicle_like_official_rgb_expand_v1`
2. 初始化权重：
   - `./weights/deploy_best_calibrated_20260411.pt`
3. 目标是轻量 hardcase 精修
4. 当前 `sampleFps` 与模型端联调无关，这份配置只对应训练

## 当前训练状态

当前 **没有正在运行的训练进程**。

本轮曾经自动拉起过两次训练 run：

1. [run_20260412_113915](/home/liujuncheng/rgbt_uav_detection/outputs/train_runs/run_20260412_113915)
2. [run_20260412_113926](/home/liujuncheng/rgbt_uav_detection/outputs/train_runs/run_20260412_113926)

但这两次都不是最终有效启动，原因是：

1. 自动等待脚本在数据集生成过程中误触发过早
2. `run.log` 里显示当时落在了 `device=cpu`
3. 当前并没有后续活跃训练进程

因此：

1. 这两个 run 只作为留档参考
2. 不作为本轮 hardcase 正式训练结果

## 模型端状态

当前模型服务已经手动下线：

1. `rgbt-model.service` 为 `inactive (dead)`

这是为了在数据集处理阶段不占 GPU / CPU 资源。

## 重启电脑后建议执行顺序

### 第 1 步：确认模型端是否需要继续保持下线

如果优先跑训练，不要先启动模型端服务。

### 第 2 步：确认没有残留训练进程

建议先检查：

```bash
ps -ef | rg 'scripts/train.py|prepare_dronevehicle_like.py|wait_and_start_hardcase_official_20260412'
```

如果没有输出，再继续下面步骤。

### 第 3 步：正式启动 hardcase 训练

推荐命令：

```bash
cd /home/liujuncheng/rgbt_uav_detection
/home/liujuncheng/miniconda3/bin/python scripts/train.py \
  --config configs/experiment_hardcase_official_rgb_expand_light.yaml \
  --num-workers 0
```

说明：

1. 当前建议继续用 `--num-workers 0`
2. 这样最稳，便于先确认首轮能稳定启动和写日志

### 第 4 步：确认训练是否真的跑在 GPU

启动后第一时间看：

```bash
tail -n 40 outputs/train_runs/$(find outputs/train_runs -maxdepth 1 -type d -name 'run_*' | sort | tail -n 1 | xargs basename)/run.log
```

重点确认：

1. `device=cuda`
2. `train_samples=17953` 或接近当前数据集统计
3. 没有新的 dataset / label / file corruption 报错

注意：

当前训练代码里的 `train_samples` 会来自真正成功读到的数据集样本数，和 `dataset_summary.json` 中的统计口径可能有极少量差别时，优先以实际训练日志为准。

## 如果训练启动失败，优先检查什么

1. 是否落在 `device=cpu`
2. 是否出现新的异常原始标签
3. 是否仍有坏图在训练阶段被读到
4. 是否有历史自动等待脚本残留，重复拉起训练

## 当前最重要的结论

这条“官方 split + rgb 标注 + bbox expand + 轻量 hardcase 精修”的训练前置条件已经具备。

重启电脑后，不需要重新下载数据集，也不需要重新解包。

最直接的下一步就是：

1. 保持模型端离线
2. 手动启动
   - [configs/experiment_hardcase_official_rgb_expand_light.yaml](/home/liujuncheng/rgbt_uav_detection/configs/experiment_hardcase_official_rgb_expand_light.yaml)
3. 确认第一轮正式训练确实跑在 `cuda`
