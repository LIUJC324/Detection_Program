# 2026-04-12 True OBB Pack And Stage1 Training Prep

## 本次完成内容

### 1. 补齐真 OBB 标注数据包框架

已生成：

- [dataset.yaml](/home/liujuncheng/rgbt_uav_detection/datasets/yolo_obb_official_rgb_trueobb_v1/dataset.yaml)
- [dataset_summary.json](/home/liujuncheng/rgbt_uav_detection/datasets/yolo_obb_official_rgb_trueobb_v1/dataset_summary.json)
- [annotation_manifest.json](/home/liujuncheng/rgbt_uav_detection/datasets/yolo_obb_official_rgb_trueobb_v1/annotation_manifest.json)
- [priority_candidates_train.json](/home/liujuncheng/rgbt_uav_detection/datasets/yolo_obb_official_rgb_trueobb_v1/priority_candidates_train.json)
- [priority_candidates_val.json](/home/liujuncheng/rgbt_uav_detection/datasets/yolo_obb_official_rgb_trueobb_v1/priority_candidates_val.json)

说明：

1. `images/` 来自官方 pseudo OBB 数据集软链接
2. `labels/` 为可被后续人工真 OBB 修正覆盖的复制版本
3. `priority_candidates_*` 按 hardcase 优先级排序，便于先补最值钱的图

### 2. 主训练稳定参数已对齐

主训练配置：

- [yolo_obb_official_rgb_stage1_stable.yaml](/home/liujuncheng/rgbt_uav_detection/configs/yolo_obb_official_rgb_stage1_stable.yaml)

当前关键参数：

1. `data = yolo_obb_official_rgb_trueobb_v1/dataset.yaml`
2. `model = smoke_rgb_expand_refined_v1/weights/best.pt`
3. `epochs = 30`
4. `batch = 4`
5. `workers = 0`
6. `amp = false`
7. `plots = false`

意义：

1. 先用 smoke test 最优权重 warm start
2. 先在更稳的官方 split 上跑第一轮 stage1
3. 等真 OBB 人工修正继续补齐后，再做下一轮 finetune

## 当前建议执行顺序

1. 人工按 `priority_candidates_val.json` 先补 `val`
2. 再按 `priority_candidates_train.json` 补 `train`
3. 同时可先启动 stage1 stable 训练，作为“正式大盘 warm start”
4. 补标完成后再跑：
   - [yolo_obb_official_rgb_trueobb_finetune.yaml](/home/liujuncheng/rgbt_uav_detection/configs/yolo_obb_official_rgb_trueobb_finetune.yaml)

## 一句话结论

本次已经把“真 OBB 数据补齐方案”从纸面方案推进到：

1. 有固定数据目录
2. 有固定优先补标清单
3. 有固定主训练参数
4. 可以直接接着开启官方 split 主训练
