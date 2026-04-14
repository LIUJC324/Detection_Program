# 2026-04-12 YOLO-OBB 真 OBB 标注补齐方案

## 目标

在继续 `YOLO-OBB` 主训练之前，先把“真 OBB 标注”方案补齐，回答三个问题：

1. 最少需要补多少图，主训练才值得跑。
2. 应该优先补哪些图。
3. 主训练和后续精修应该用什么参数。

## 当前前提

当前已经具备：

1. `YOLO-OBB` 训练链路已跑通。
2. `RGB-only` OBB smoke test 已经证明模型能收敛。
3. 官方 split 主数据已经整理完成：
   - [dataset_summary.json](/home/liujuncheng/rgbt_uav_detection/datasets/dronevehicle_like_official_rgb_expand_v1/dataset_summary.json)

当前还不具备的是：

1. 真正的 `OBB` / `angle` 监督标签。
2. 面向正式主训练的稳定参数文件。

## 真 OBB 标注的最小可行规模

### 最小可行版本

如果目标是“让主训练里的 angle 有实际意义”，建议最低补齐：

1. `val`：
   - `300-500` 张图
   - 尽量覆盖明显斜车、密集车、长车、边缘车
2. `train`：
   - `800-1200` 张图
   - 以 hardcase 为主

这是一条“能开始跑主训练”的最低线。

### 推荐首版版本

如果目标是“让第一版主训练结果更可信”，建议补齐：

1. `val`：
   - `500-800` 张图
2. `train`：
   - `1500-2500` 张图

这是更推荐的首版规模。

### 更完整版本

如果目标是“后续前端真的要吃 angle，而且希望上线后稳定”，建议逐步推进到：

1. `val`：
   - 全量 `1469` 图
2. `train`：
   - `3000+` hardcase 图

## 为什么不建议一上来全量重标

原因很现实：

1. 当前主风险不在“量不够大”，而在“路线还没通过真 OBB 验证”。
2. 先做一个中等规模真 OBB 子集，更容易快速确认：
   - angle 是否显著改善
   - OBB 是否真比 HBB 更适合当前业务视频
3. 一次性全量重标成本高，回报未验证前不划算。

## 真 OBB 图像筛选优先级

优先补这些图：

1. 车辆明显倾斜：
   - 横穿视角
   - 斜停
   - 机位俯视导致车身斜向分布
2. 长目标：
   - `truck`
   - `bus`
   - `freight_car`
3. 密集场景：
   - 目标间间隔小
   - 水平框互相覆盖严重
4. 画面边缘目标：
   - 左右边缘
   - 上下边缘
5. 当前误检/漏检高发样本：
   - 优先来自 hardcase 精修池

## 标注规则

建议统一按 `四点顺时针` 标注。

### 文件格式

Ultralytics OBB 标签行格式：

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

### 坐标规则

1. 归一化到 `[0, 1]`
2. 以 `RGB` 图像为准
3. 点序固定：
   - 从“最靠近左上”的顶点开始
   - 之后顺时针

### 业务规则

1. 车辆长边尽量沿真实车身方向。
2. 不要为了贴边而画成任意四边形，优先画旋转矩形。
3. 极小目标如果方向完全不可辨，可以保留近似水平矩形。
4. 如果目标被严重遮挡，按可见主体长轴方向估计，不要随意拉大。

## 数据组织建议

建议新增这一版正式真 OBB 数据：

- `datasets/yolo_obb_official_rgb_trueobb_v1`

内部结构：

```text
datasets/yolo_obb_official_rgb_trueobb_v1/
  images/
    train/
    val/
  labels/
    train/
    val/
  dataset.yaml
  dataset_summary.json
  annotation_manifest.json
```

说明：

1. `images/` 可以继续软链接。
2. `labels/` 放真 OBB 标签。
3. `annotation_manifest.json` 记录：
   - 哪些图是真 OBB
   - 哪些图仍是矩形四点 pseudo OBB

## 主训练参数方案

### 第一阶段：稳定主训练

建议新增配置：

- [yolo_obb_official_rgb_stage1_stable.yaml](/home/liujuncheng/rgbt_uav_detection/configs/yolo_obb_official_rgb_stage1_stable.yaml)

建议参数：

1. `epochs = 30`
2. `batch = 4`
3. `workers = 0`
4. `amp = false`
5. `plots = false`
6. `model = yolo11s-obb.pt`

这组参数的意义是：

1. 先求稳
2. 先验证官方 split 在本机完整可跑
3. 不在第一轮就把资源打满

### 第二阶段：真 OBB 精修

建议新增配置：

- [yolo_obb_official_rgb_trueobb_finetune.yaml](/home/liujuncheng/rgbt_uav_detection/configs/yolo_obb_official_rgb_trueobb_finetune.yaml)

建议参数：

1. `epochs = 20`
2. `batch = 4`
3. `workers = 0`
4. `amp = false`
5. `plots = false`
6. `lr0 = 0.001`
7. `close_mosaic = 5`

这组参数的意义是：

1. 在主训练权重基础上做真 OBB hardcase 精修
2. 更强调方向监督和细框拟合

## 推荐执行顺序

### 路线 A：最小代价验证

1. 补：
   - `val 300-500`
   - `train 800-1200`
2. 跑：
   - `stage1 stable`
3. 看：
   - OBB 可视化质量
   - `truck / bus / freight_car` 是否改善明显

### 路线 B：更稳的首版主线

1. 补：
   - `val 500-800`
   - `train 1500-2500`
2. 跑：
   - `stage1 stable`
   - `trueobb finetune`
3. 再决定是否值得全量继续标

## 留档要求

每次补标后，必须同步留档：

1. `dataset_summary.json`
2. `annotation_manifest.json`
3. 标注图数
4. 标注目标数
5. 各类占比
6. 斜车样本占比

训练后同步留档：

1. `results.csv`
2. `best.pt`
3. `best.onnx`
4. `best_fp16.onnx`
5. `frontend_model_config.json`

## 一句话结论

如果现在就问“主训练前最少补多少真 OBB 才值得跑”，我的建议是：

1. `val` 至少补 `300-500` 图
2. `train` 至少补 `800-1200` 图

如果你要的是更可信的第一版主训练结果，则建议直接按：

1. `val 500-800`
2. `train 1500-2500`

再启动正式训练。
