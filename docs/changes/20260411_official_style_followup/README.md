# 2026-04-11 Official Style Follow-up

## 当前状态

截至 `2026-04-11` 晚间：

1. 线上服务继续使用：
   - 高召回阈值
   - 重合框抑制
   - 非对称框校正
2. 后台正在跑一条更小学习率的精修：
   - [configs/experiment_rootcause_rgb_expand_finetune.yaml](/home/liujuncheng/rgbt_uav_detection/configs/experiment_rootcause_rgb_expand_finetune.yaml)
3. 当前本地数据仍然只来自官方 `validation` 原始包
4. 训练脚本与数据准备脚本已经支持下一阶段切到官方标准 split

## 为什么当前这条低学习率训练不应无限继续

原因：

1. `lr` 已经很低，后续边际收益会明显下降
2. 当前主要瓶颈已经从“纯参数还能不能再抠一点”转成：
   - 数据口径
   - 监督框尺度
   - 官方 split

因此：

- 当前这条低学习率训练跑完后，可以保留最佳 checkpoint
- 但后续主线不应继续只在同一数据口径上硬炼

## 借鉴开源数据集处理模式：当前已落地的部分

已经借鉴并落地的点：

1. 官方白边裁剪
   - 当前脚本会先裁掉 `DroneVehicle` 原始图像外围白边
   - 本地统计也验证了处理后图像主峰接近 `640 x 512`

2. 官方 split 支持
   - [scripts/prepare_dronevehicle_like.py](/home/liujuncheng/rgbt_uav_detection/scripts/prepare_dronevehicle_like.py)
   - 已支持：
     - `--train-source-root`
     - `--val-source-root`
   - 也就是一旦本地拿到官方 `Train + Validation`，可直接保留官方 split

3. 更干净的监督口径
   - 已支持：
     - `annotation_source = rgb`
     - `annotation_source = thermal`
     - `annotation_source = merged_union`
   - 当前根因训练已经切到 `rgb`

4. 针对框偏小的监督修正
   - 已支持：
     - `--bbox-expand-ratio`
     - `--bbox-expand-min-pixels`
   - 当前扩张标注版训练已经在用

## 训练跑完后的下一阶段建议顺序

### 第 1 步：固定当前最佳 checkpoint

不要直接拿最后一轮。

要做的事：

1. 查看当前精修 run 的最佳 epoch
2. 记录：
   - `TP`
   - `FP`
   - `recall50`
   - `small_recall50`
3. 保留一份独立快照

### 第 2 步：优先切官方 split

条件：

- 本地拿到官方 `DroneVehicle Train + Validation`

要做的事：

1. 新建：
   - `datasets/dronevehicle_like_official_v2`
2. 用官方 split 重建数据集
3. 不再从单一 `validation` 里随机切 train/val

推荐命令形态：

```bash
cd /home/liujuncheng/rgbt_uav_detection
/home/liujuncheng/miniconda3/bin/python scripts/prepare_dronevehicle_like.py \
  --train-source-root /path/to/official/train_root \
  --val-source-root /path/to/official/val_root \
  --target-root /home/liujuncheng/rgbt_uav_detection/datasets/dronevehicle_like_official_v2 \
  --annotation-source rgb \
  --bbox-expand-ratio 0.12 \
  --bbox-expand-min-pixels 2 \
  --copy-mode symlink \
  --clear-target
```

### 第 3 步：只做小范围参数搜索

下一阶段参数优化，建议只在下面这几个维度里做小范围网格：

1. `lr`
   - `1e-5`
   - `2e-5`
   - `3e-5`
2. `random_crop_prob`
   - `0.04`
   - `0.06`
   - `0.08`
3. `bbox_expand_ratio`
   - `0.08`
   - `0.10`
   - `0.12`
   - `0.14`
4. `bbox_expand_min_pixels`
   - `1`
   - `2`
   - `3`

不要再同时大改：

1. backbone
2. fusion
3. loss
4. 数据口径
5. 线上后处理

### 第 4 步：指标判断标准

下一阶段继续 / 停止，重点看：

1. `TP`
2. `FP`
3. `recall50`
4. `small_recall50`
5. 预览视频里的：
   - 左上偏差是否减轻
   - 小框是否变少
   - 一大一小重合框是否变少

如果只涨 `TP` 但框观感没有改善，不算真正完成。

## 下一阶段重点不是继续加大训练轮数

真正的重点是：

1. 更合理的数据口径
2. 更合理的监督框尺度
3. 更小范围、更有针对性的参数搜索

如果只保留一句最务实的话：

- 当前这条低学习率训练跑完后，优先切官方 split 和官方式数据准备，不要继续长期停留在“validation 随机切分 + 小幅继续炼”的阶段。
