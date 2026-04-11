# 2026-04-11 Hardcase Official Dataset Plan

## 目标

当前模型已经把：

1. 框大小
2. 重合框
3. 左上偏移

这几类问题初步压下去了。

后续主要瓶颈已经收敛到：

1. 暗处车辆
2. 车辆密集区
3. 高速运动目标

因此下一阶段主线不应再停留在：

- 基于官方 `validation` 的随机切分数据

而应切到：

- **官方 `Train + Validation` split**
- 再在这个标准口径上做 hard-case 精修

## 本地现状

当前本地已经具备：

1. 官方压缩包：
   - `datasets/raw/dronevehicle/train.zip`
   - `datasets/raw/dronevehicle/val.zip`
2. 现有解包数据：
   - `datasets/raw/dronevehicle/val_unpack/val`
3. 数据准备脚本已经支持：
   - 官方 split
   - `annotation_source`
   - `bbox_expand_ratio`
   - `bbox_expand_min_pixels`

脚本：

- [scripts/prepare_dronevehicle_like.py](/home/liujuncheng/rgbt_uav_detection/scripts/prepare_dronevehicle_like.py)

## 推荐执行步骤

### 第 1 步：解包官方 train / val

建议目录：

1. `datasets/raw/dronevehicle/train_unpack`
2. `datasets/raw/dronevehicle/val_unpack`

示例命令：

```bash
cd /home/liujuncheng/rgbt_uav_detection/datasets/raw/dronevehicle
mkdir -p train_unpack val_unpack
unzip -q train.zip -d train_unpack
unzip -q val.zip -d val_unpack
```

解包后应看到类似结构：

- `train_unpack/train/trainimg`
- `train_unpack/train/trainimgr`
- `train_unpack/train/trainlabel`
- `train_unpack/train/trainlabelr`
- `val_unpack/val/valimg`
- `val_unpack/val/valimgr`
- `val_unpack/val/vallabel`
- `val_unpack/val/vallabelr`

### 第 2 步：生成官方 split 数据集

推荐先做一版：

- `annotation_source = rgb`
- `bbox_expand_ratio = 0.12`
- `bbox_expand_min_pixels = 2`

目标目录建议：

- `datasets/dronevehicle_like_official_rgb_expand_v1`

示例命令：

```bash
cd /home/liujuncheng/rgbt_uav_detection
/home/liujuncheng/miniconda3/bin/python scripts/prepare_dronevehicle_like.py \
  --train-source-root /home/liujuncheng/rgbt_uav_detection/datasets/raw/dronevehicle/train_unpack/train \
  --val-source-root /home/liujuncheng/rgbt_uav_detection/datasets/raw/dronevehicle/val_unpack/val \
  --target-root /home/liujuncheng/rgbt_uav_detection/datasets/dronevehicle_like_official_rgb_expand_v1 \
  --annotation-source rgb \
  --bbox-expand-ratio 0.12 \
  --bbox-expand-min-pixels 2 \
  --copy-mode symlink \
  --clear-target
```

### 第 3 步：先审计再训练

生成后优先检查：

1. `dataset_summary.json`
2. train / val 数量
3. 主峰尺寸是否仍接近 `640x512`
4. `split_strategy` 是否为：
   - `official_split`

### 第 4 步：基于官方 split 跑 hard-case 轻量精修

建议沿用当前已验证过更稳的思路：

1. 从当前认可快照 warm start
   - [weights/deploy_best_calibrated_20260411.pt](/home/liujuncheng/rgbt_uav_detection/weights/deploy_best_calibrated_20260411.pt)
2. 保留轻量 hard-case 增强
3. 不再用过重 lowlight / motion blur

建议搜索范围：

1. `lowlight_aug_prob`
   - `0.12`
   - `0.15`
   - `0.18`
2. `weak_modality_prob`
   - `0.05`
   - `0.08`
   - `0.10`
3. `motion_blur_prob`
   - `0.05`
   - `0.08`
   - `0.10`
4. `lr`
   - `1e-5`
   - `1.2e-5`
   - `1.5e-5`

### 第 5 步：只做小步实验，不要大混改

下一阶段不要同时改：

1. backbone
2. fusion
3. 数据口径
4. 线上后处理
5. hard-case 增强强度

优先保持：

- 结构不变
- 监督更正
- 数据 split 正确
- hard-case 增强轻量

## 当前推荐优先级

1. 先用官方 split 重建数据集
2. 再用当前较优快照做轻量 hard-case 精修
3. 再看是否需要接入额外 hard-case 数据集

## 为什么要先这样做

因为当前最务实的提升空间依然在：

1. 数据 split 正确化
2. 监督框尺度更稳
3. 轻量 hard-case 处理

而不是立刻上新 backbone 或大改融合结构。
