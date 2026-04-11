# 2026-04-11 Git Upload Bundle

## 本次上传目的

把当前已经验证过的参数更新、后续优化方案，以及一版认可的预览视频整理到同一次 Git 提交里，便于后续追踪和对外同步。

## 当前上传内容

### 1. 参数更新

本次一并上传的参数文件主要包括：

- `configs/deploy_stable.yaml`
- `configs/deploy_demo_high_recall.yaml`
- `configs/deploy.yaml`
- `configs/deploy_integration_stable.yaml`
- `configs/default.yaml`
- `configs/experiment_*.yaml`

其中当前线上认可版本以 `configs/deploy_stable.yaml` 为准，关键参数为：

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
14. `enable_rgb_lowlight_enhance = true`
15. `rgb_lowlight_luma_threshold = 72`
16. `rgb_lowlight_gamma = 1.15`
17. `rgb_lowlight_clahe_clip_limit = 2.0`
18. `rgb_lowlight_confidence_scale = 0.75`

### 2. 最佳预览视频

本次选择上传的预览视频为：

- `outputs/local_preview/annotated_preview_online_calibrated_20260411.mp4`
- `outputs/local_preview/annotated_preview_online_calibrated_20260411.json`

选择依据：

1. `rendered_frames = 150`
2. `empty_detection_frames = 11`
3. `empty_detection_ratio = 0.073333`
4. `avg_model_latency_ms = 38.514`

对应说明文档见：

- `docs/changes/20260411_recognized_good_snapshot/README.md`

### 3. 后续优化方案

后续优化路线以这份文档为准：

- `docs/changes/20260411_optimization_roadmap/README.md`

当前主结论是：

1. 当前继续单纯调小结构或继续恢复旧 run 的收益不高
2. 下一阶段优先修正官方数据口径
3. 先重建标准 `Train + Validation` 数据集，再做 baseline 和 annotation source 消融
4. 只有在标准口径仍不理想时，再考虑引入外部数据 warmup

## 说明

本次 Git 提交不包含 `weights/deploy_best_calibrated_20260411.pt`，权重快照仍保留在本地，避免仓库继续累积模型文件。
