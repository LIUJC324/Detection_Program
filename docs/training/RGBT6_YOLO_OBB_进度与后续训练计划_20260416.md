# RGBT6 YOLO-OBB 进度与后续训练计划（2026-04-16）

本文档用于固化当前 `RGB-T 6通道 YOLO-OBB` 路线的真实进度、与赛题要求的对应关系，以及后续训练步骤计划，方便后续继续复用。

适用场景：

- 队内训练交接
- 答辩材料撰写
- 后续继续训练时快速对齐现状

---

## 1. 当前路线划分

当前仓库里实际存在三条模型路线：

1. `RGB-T FCOS` 主线  
   对应赛题中的“多模态融合 + 小目标增强”主要求。

2. `RGB-only YOLO-OBB` 展示线  
   当前在单 RGB 条件下已经达到较好的展示效果和几何框效果。

3. `RGB-T 6ch YOLO-OBB` 在建线  
   目标是把前端在线演示和赛题中的“RGB-T 融合”统一到一条线上。

当前最重要的研发重点是第 3 条。

---

## 2. 已完成进度

### 2.1 单 RGB YOLO-OBB 已达到可展示状态

当前 `RGB-only YOLO-OBB stage2` 已达到：

- `precision = 0.71000`
- `recall = 0.68155`
- `mAP50 = 0.70302`
- `mAP50-95 = 0.61061`

本地预览：

- `empty_detection_ratio = 0.016667`
- `avg_model_latency_ms = 17.434`

对应留档：

- [20260413_yolo_obb_stage2_archive/README.md](/home/liujuncheng/rgbt_uav_detection/docs/changes/20260413_yolo_obb_stage2_archive/README.md)

说明：

- 这条线已经能支撑前端演示
- 但它本身是 `3` 通道，不是真正的 RGB-T 融合

### 2.2 RGB-T FCOS 主线已完成多模态融合与小目标增强

当前 `RGB-T FCOS` 主线已经具备：

- 双分支 `RGB / Thermal` 特征提取
- 跨模态注意力融合
- 可靠性感知融合
- `BiFPN` 多尺度增强
- 小目标精炼头
- `FCOS` 尺度对齐

曾达到的阶段指标：

- `recall50 = 0.084878`
- `small_recall50 = 0.122330`
- `TP = 348`

对应留档：

- [20260411_reliability_refine/README.md](/home/liujuncheng/rgbt_uav_detection/docs/changes/20260411_reliability_refine/README.md)

说明：

- 这条线证明了你们确实已经做了 RGB-T 融合
- 但当前更适合赛题主线说明，不如 `YOLO-OBB` 直观好展示

### 2.3 RGB-T 6通道 YOLO-OBB 链路已打通

当前已完成：

1. `YOLO-OBB` 第一层扩成 `6ch`
2. 导出可被前端直接使用的 `6ch ONNX`
3. 生成 `6ch` 前端 config
4. 构造 `RGB3 + Thermal3` 的输入张量口径
5. 小规模和中等规模的 `6ch` 训练链路验证成功

当前导出的前端包：

- [yolo11_obb_rgbt6_fastdemo.onnx](/home/liujuncheng/rgbt_uav_detection/weights/yolo11_obb_rgbt6_fastdemo.onnx)
- [frontend_model_config_yolo_obb_rgbt6_fastdemo.json](/home/liujuncheng/rgbt_uav_detection/weights/frontend_model_config_yolo_obb_rgbt6_fastdemo.json)

说明：

- `6ch` 口径已经不是纸面方案
- 数据、训练、导出、前端参数都已经能跑起来

---

## 3. 当前真实问题

### 3.1 6通道版本还不能稳定出框

尽管 `6ch` 链路已经打通，但当前几个阶段性模型在本地预览上仍表现为：

- 大量空帧
- 预览视频依然无框

例如：

- [annotated_preview_yolo_obb_rgbt6_continue_from_best_v1_20260416.json](/home/liujuncheng/rgbt_uav_detection/outputs/local_preview/annotated_preview_yolo_obb_rgbt6_continue_from_best_v1_20260416.json)

结果为：

- `frames = 60`
- `empty_detection_frames = 60`
- `empty_detection_ratio = 1.0`

这说明：

- 问题已经不在导出链路
- 也不在前端输入口径
- 而在模型训练效果本身

### 3.2 小子集训练不足以稳定拉起效果

此前已配对的中等规模子集：

- `train = 1175`
- `val = 294`

虽然能让指标偶尔出现非零值，但很不稳定，容易重新掉回全 `0`。

说明：

- 小规模配对子集只适合验证链路
- 不足以支撑稳定的 `RGB-T 6ch OBB` 检测效果

---

## 4. 最新已扩充的数据进度

当前已经基于官方原始热红外目录，构造出一套更大的全量 `RGB-T 6ch` 数据集：

- `train = 17953`
- `val = 1469`

对应留档：

- [build_summary.json](/home/liujuncheng/rgbt_uav_detection/datasets/yolo_obb_official_rgbt6_full_v1/build_summary.json)

当前数据集位置：

- [dataset.yaml](/home/liujuncheng/rgbt_uav_detection/datasets/yolo_obb_official_rgbt6_full_v1/dataset.yaml)

说明：

- 这套数据规模已经不再是“小试验”
- 后续是否能出框，主要取决于训练策略是否合适

---

## 5. 当前训练进度

### 5.1 `continue_from_best_v1`

当前这一轮从前一轮 `best.pt` 继续训练，最好点达到：

- `precision = 0.36`
- `recall = 0.00591`
- `mAP50 = 0.006`
- `mAP50-95 = 0.00399`

对应结果：

- [results.csv](/home/liujuncheng/rgbt_uav_detection/outputs/yolo_obb_runs/official_rgbt6_continue_from_best_v1/results.csv)

说明：

- 比更早阶段有进步
- 但仍然不稳定

### 5.2 `full_official_v1`

当前已经切到全量官方 `RGB-T 6ch` 数据训练：

- [official_rgbt6_full_official_v1](/home/liujuncheng/rgbt_uav_detection/outputs/yolo_obb_runs/official_rgbt6_full_official_v1)

目前首轮日志说明：

- 训练已正常启动
- 数据缓存建立正常
- 当前还在前几轮阶段，后续需要继续观察首轮完整验证结果

---

## 6. 与赛题要求的对应情况

### 6.1 极小目标检测

已做：

- `BiFPN`
- 小目标精炼头
- 小目标损失强化
- `FCOS` 尺度对齐

当前完成度：

- 设计和实现已经有
- 效果仍在继续训练验证

### 6.2 多模态融合

已做：

- `RGB-T FCOS` 主线已完整实现双分支融合
- `YOLO-OBB 6ch` 线已完成输入层改造与训练打通

当前完成度：

- 主线融合已实现
- `6ch OBB` 效果仍未稳定

### 6.3 多类别与密集遮挡

已做：

- 5 类目标检测
- 分数筛选、过滤、去重、结果整理

当前完成度：

- 基本能力有
- 遮挡下稳定性还要继续优化

### 6.4 全天候适应

已做：

- `RGB-T` 双模态
- 弱模态建模
- 暗光增强

当前完成度：

- 思路和实现都在
- 仍需靠更大规模训练把效果真正拉起来

---

## 7. 后续详细训练步骤计划

### 第一步：继续观察当前全量训练

目标：

- 先看全量官方 `RGB-T 6ch` 训练是否能自然拉起非零指标

关注：

- `precision`
- `recall`
- `mAP50`
- `mAP50-95`

判断标准：

- 如果连续多个 epoch 出现非零结果，且有上升趋势，则继续训
- 如果继续长期为 `0`，说明还要进一步调整训练策略

### 第二步：每隔若干 epoch 做一次视频验收

目标：

- 不只看 `csv`
- 必须看预览视频是否从“全空”变成“开始有框”

执行方式：

1. 取当前 `best.pt`
2. 生成一版 `RGB-T 6ch` 本地预览视频
3. 统计：
   - `empty_detection_ratio`
   - `avg_detections_per_frame`
   - `max_detections_per_frame`

### 第三步：如果全量训练仍然不稳，做低增强稳定化续训

目标：

- 避免模型在验证集上“偶尔有框、随后又掉回 0”

建议策略：

- `mosaic -> 0`
- `scale -> 0.2 ~ 0.3`
- `degrees -> 0`
- 更低学习率
- 从当前最好 `best.pt` 继续训

### 第四步：继续借鉴已有主线经验增强训练

建议继续引入：

- 暗光增强
- 弱模态模拟
- 更稳的输入对齐口径

这些思路已经在 `RGB-T FCOS` 路线上证明有价值，不应丢掉。

### 第五步：达标后再统一导出前端包

达标前不要频繁导参数和视频。

建议达到下面条件后，再导：

1. `recall` 不再长期 `0`
2. `mAP50` 连续几个 epoch 稳定为正
3. 本地预览视频不再全空

再统一输出：

- `best.pt`
- `6ch onnx`
- `frontend config`
- `preview video`

---

## 8. 当前最务实的执行顺序

后续按下面顺序推进：

1. 跑完当前全量官方 `RGB-T 6ch` 训练
2. 查看首轮完整验证指标
3. 如果指标开始起来，继续全量训
4. 如果还是不稳，切到低增强稳定化续训
5. 每隔若干 epoch 做一次视频验收
6. 只有出现稳定出框后，再统一导出前端交付包

---

## 9. 一句话总结

当前 `RGB-T 6ch YOLO-OBB` 方案已经从“只能设想”推进到“数据、训练、导出、前端参数全部打通”，但还没有达到稳定出框阶段。后续工作的重点不是继续改前端，而是依靠更大规模的 `RGB-T` 配对数据和更稳的训练策略，把这条线真正训练成熟。
