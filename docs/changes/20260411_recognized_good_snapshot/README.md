# 2026-04-11 Recognized Good Snapshot

## 认可快照

当前已确认保留的较优快照为：

- [weights/deploy_best_calibrated_20260411.pt](/home/liujuncheng/rgbt_uav_detection/weights/deploy_best_calibrated_20260411.pt)

这是从当时线上使用的较优 `best.pt` 额外复制出来的稳定快照，用于：

1. 保留当前较满意的线上效果
2. 避免后续训练继续改写 `weights/best.pt` 后找不到这版
3. 后续可作为回退点或 warm start 基线

## 对应预览

这份快照对应的较优预览为：

- [annotated_preview_online_calibrated_20260411.mp4](/home/liujuncheng/rgbt_uav_detection/outputs/local_preview/annotated_preview_online_calibrated_20260411.mp4)
- [annotated_preview_online_calibrated_20260411.json](/home/liujuncheng/rgbt_uav_detection/outputs/local_preview/annotated_preview_online_calibrated_20260411.json)

摘要：

1. `rendered_frames = 150`
2. `empty_detection_frames = 11`
3. `empty_detection_ratio = 0.073333`
4. `avg_model_latency_ms = 38.514`

## 对应线上参数

对应的是 [configs/deploy_stable.yaml](/home/liujuncheng/rgbt_uav_detection/configs/deploy_stable.yaml) 中这一版参数：

1. `score_thresh = 0.2`
2. `nms_thresh = 0.4`
3. `detections_per_img = 100`
4. `result_min_confidence = 0.4`
5. `callback_min_confidence = 0.4`
6. `callback_fallback_min_confidence = 0.3`
7. `result_duplicate_iou = 0.18`
8. `result_containment_ratio = 0.8`
9. `result_duplicate_center_distance = 24`
10. `result_box_pad_left = -2`
11. `result_box_pad_top = -1`
12. `result_box_pad_right = 16`
13. `result_box_pad_bottom = 22`

## 使用建议

如果后续新训练效果变差，优先回退到：

- [weights/deploy_best_calibrated_20260411.pt](/home/liujuncheng/rgbt_uav_detection/weights/deploy_best_calibrated_20260411.pt)

而不是重新从多个历史 `best_before_*` 文件里倒推。

## 本次 Git 上传范围

本次上传到 Git 的内容以参数、路线图和最佳预览视频为主：

1. 上传当前部署参数与实验配置
2. 上传后续优化路线文档
3. 上传最佳预览视频与对应摘要 JSON

当前这份权重快照仍保留在本地 `weights/` 目录，未纳入本次 Git 提交。
