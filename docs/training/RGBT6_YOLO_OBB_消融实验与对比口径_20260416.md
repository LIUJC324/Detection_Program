# RGBT6 YOLO-OBB 消融实验与对比口径（2026-04-16）

本文档用于把当前 `RGB-T 6ch YOLO-OBB` 后续增强线需要做的：

- 消融实验口径
- 公开模型对比口径
- 结果记录格式

统一成一套可直接执行、可直接写材料的规范。

---

## 1. 当前对比基线

当前建议固定三条基线：

1. `RGB-only YOLO-OBB` 展示线
   - 参考留档：
     - [20260413_yolo_obb_stage2_archive/README.md](/home/liujuncheng/rgbt_uav_detection/docs/changes/20260413_yolo_obb_stage2_archive/README.md)

2. `RGB-T 6ch YOLO-OBB baseline`
   - 参考权重：
     - [official_rgbt6_full_official_speedup_v3/weights/best.pt](/home/liujuncheng/rgbt_uav_detection/outputs/yolo_obb_runs/official_rgbt6_full_official_speedup_v3/weights/best.pt)

3. `RGB-T FCOS`
   - 作为“多模态融合与小目标增强”主线说明对照

说明：

- `RGB-only YOLO-OBB` 用来回答“RGB-T 是否比单 RGB 更有效”
- `RGB-T FCOS` 用来回答“当前项目并非只做 YOLO，还有更早的多模态主线积累”

---

## 2. 本轮建议的消融矩阵

建议至少做下面四组：

### 2.1 对比 1：RGB-only vs RGB-T 6ch

目的：

- 证明双光输入有效

建议配置：

- `RGB-only`
  - 参考现有 `official_rgb_stage2`
- `RGB-T baseline`
  - [yolo_obb_rgbt6_stage8_resume_from_v3_lowmem.yaml](/home/liujuncheng/rgbt_uav_detection/configs/yolo_obb_rgbt6_stage8_resume_from_v3_lowmem.yaml)

### 2.2 对比 2：有无模态质量门控

目的：

- 证明我们的融合不是简单 6 通道拼接

建议配置：

- `no gate`
  - baseline
- `gate only`
  - [yolo_obb_rgbt6_ablation_gate_only_v1.yaml](/home/liujuncheng/rgbt_uav_detection/configs/yolo_obb_rgbt6_ablation_gate_only_v1.yaml)

### 2.3 对比 3：有无弱模态模拟

目的：

- 证明全天候/弱模态场景鲁棒性增强有效

建议配置：

- `no weak modality simulation`
  - baseline
- `weak only`
  - [yolo_obb_rgbt6_ablation_weak_only_v1.yaml](/home/liujuncheng/rgbt_uav_detection/configs/yolo_obb_rgbt6_ablation_weak_only_v1.yaml)

### 2.4 对比 4：有无小目标加强模块

目的：

- 证明小目标增强对无人机高空视角是有效的

建议配置：

- `full innovation without small-target block`
  - [yolo_obb_rgbt6_innovation_v1.yaml](/home/liujuncheng/rgbt_uav_detection/configs/yolo_obb_rgbt6_innovation_v1.yaml)
- `full innovation + small-target block`
  - [yolo_obb_rgbt6_smalltarget_v1.yaml](/home/liujuncheng/rgbt_uav_detection/configs/yolo_obb_rgbt6_smalltarget_v1.yaml)

---

## 3. 本轮已落地的创新配置

### 3.1 模态质量门控 + 弱模态模拟

创新模块：

- [rgbt6_yolo_modules.py](/home/liujuncheng/rgbt_uav_detection/model/network/rgbt6_yolo_modules.py)

包含：

- `ReliabilityAwareStemGate`
- `WeakModalityDropout`
- `SmallTargetStemBlock`

创新版结构：

- [yolo11_obb_rgbt6_innov_v1.yaml](/home/liujuncheng/rgbt_uav_detection/configs/yolo11_obb_rgbt6_innov_v1.yaml)

创新版训练配置：

- [yolo_obb_rgbt6_innovation_v1.yaml](/home/liujuncheng/rgbt_uav_detection/configs/yolo_obb_rgbt6_innovation_v1.yaml)

### 3.2 小目标增强

当前采用的落地方式：

- 在输入 stem 前增加 `SmallTargetStemBlock`
- 不改主干大结构，尽量降低 warm start 风险

小目标增强结构：

- [yolo11_obb_rgbt6_smalltarget_v1.yaml](/home/liujuncheng/rgbt_uav_detection/configs/yolo11_obb_rgbt6_smalltarget_v1.yaml)

小目标增强训练配置：

- [yolo_obb_rgbt6_smalltarget_v1.yaml](/home/liujuncheng/rgbt_uav_detection/configs/yolo_obb_rgbt6_smalltarget_v1.yaml)

---

## 4. 建议记录的指标

所有消融都建议统一记录：

- `precision`
- `recall`
- `mAP50`
- `mAP50-95`
- `best epoch`
- `final epoch`
- `params`
- `GFLOPs`
- `avg_model_latency_ms`
- `empty_detection_ratio`

其中：

- `precision/recall/mAP50/mAP50-95` 来自训练验证结果
- `avg_model_latency_ms` 和 `empty_detection_ratio` 来自本地预览 JSON

---

## 5. 建议记录的预览视频口径

建议所有对比都统一：

- 同一输入视频
- 同一 `conf=0.25`
- 同一 `iou=0.5`
- 同一 `imgsz=640`
- 同一 `sample_every`

当前推荐输入：

- [dronevehicle_rgb_thermal_side_by_side.mp4](/home/liujuncheng/rgbt_uav_detection/outputs/demo_video/dronevehicle_rgb_thermal_side_by_side.mp4)

当前 `RGBT6` 专用预览脚本：

- [render_local_paired_video_yolo_obb_rgbt6.py](/home/liujuncheng/rgbt_uav_detection/scripts/render_local_paired_video_yolo_obb_rgbt6.py)

---

## 6. 公开模型对比口径

### 6.1 可直接说的内容

可以直接说：

- 当前结果已经达到公开强方法附近
- 明显强于老一代 R-CNN/S²A-Net/UA-CMDet 一类基线
- 与 TarDAL、C²Former 这类近年 RGB-IR 方法处于同一档

### 6.2 不能直接说的内容

不建议直接说：

- “严格超过某篇论文”

除非满足：

1. 同一数据划分
2. 同一评估协议
3. 同一指标口径
4. 同一输入模态设定

### 6.3 对外更稳妥的表述

推荐表述：

> 在 DroneVehicle 场景下，我们当前的 `RGB-T 6ch YOLO-OBB` 结果已经达到公开强基线附近，明显超过经典 R-CNN 系和早期 RGB-IR 检测方法，并在 `mAP50-95` 这一更严格指标上表现出较强的竞争力。同时，当前模型参数量更小，具备较好的精度-复杂度比和工程落地优势。

---

## 7. 结果记录模板

建议后续每次实验都补一条：

| 实验名 | 配置 | precision | recall | mAP50 | mAP50-95 | latency(ms) | empty ratio | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| RGB baseline | ... |  |  |  |  |  |  |  |
| RGBT baseline | ... |  |  |  |  |  |  |  |
| gate only | ... |  |  |  |  |  |  |  |
| weak only | ... |  |  |  |  |  |  |  |
| full innovation | ... |  |  |  |  |  |  |  |
| full + small-target | ... |  |  |  |  |  |  |  |

---

## 8. 当前建议顺序

建议后续执行顺序：

1. 先跑 `gate only`
2. 再跑 `weak only`
3. 再跑 `full innovation`
4. 最后跑 `full + small-target`

原因：

- 先把“融合增强”单独拆开看
- 再看“小目标增强”能否继续抬高最终结果

这样最符合赛题答辩时的逻辑：

- 先证明双光融合不是简单拼接
- 再证明小目标增强是针对无人机视角额外有效
