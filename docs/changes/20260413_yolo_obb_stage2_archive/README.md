# 2026-04-13 YOLO-OBB Stage2 Archive

## 本次结论

本轮 `YOLO-OBB` 路线已经完成：

1. `stage2` 模型优化训练
2. 本地 angle 预览视频验证
3. 最终 `ONNX` 导出

可以认为这版结果已经具备“下线留档、交给前端继续接入”的条件。

## 训练结果

本轮主要看：

- [results.csv](/home/liujuncheng/rgbt_uav_detection/outputs/yolo_obb_runs/official_rgb_model_refined_stage2_v1/results.csv)

截至已落盘的最后一轮：

- `epoch 18`
  - `precision = 0.71000`
  - `recall = 0.68155`
  - `mAP50 = 0.70302`
  - `mAP50-95 = 0.61061`

从曲线看：

1. `mAP50` 已经稳定在 `0.70+`
2. `mAP50-95` 已经稳定在 `0.61` 左右
3. `angle_loss` 已下降到 `0.0017` 量级

这说明：

1. `YOLO-OBB` 主线已经稳定收敛
2. 相比初始 smoke test，stage2 进一步提升了几何拟合与总体检测质量
3. 这版权重可作为当前对前端交付的主模型

## 最终权重

本轮留档的主权重：

- [best.pt](/home/liujuncheng/rgbt_uav_detection/outputs/yolo_obb_runs/official_rgb_model_refined_stage2_v1/weights/best.pt)
- [last.pt](/home/liujuncheng/rgbt_uav_detection/outputs/yolo_obb_runs/official_rgb_model_refined_stage2_v1/weights/last.pt)

## ONNX 导出

已导出并校验：

- [best.onnx](/home/liujuncheng/rgbt_uav_detection/outputs/yolo_obb_runs/official_rgb_model_refined_stage2_v1/weights/best.onnx)

校验结论：

1. `onnx.checker` 已通过
2. 输入名：`images`
3. 输出名：`output0`
4. 文件大小约 `10.3MB`

说明：

1. 这是标准 `YOLO-OBB` 单图 `3` 通道输入
2. 与先前 `RGB-T FCOS` 的 `6` 通道 ONNX 不同
3. 前端若使用 `onnxruntime-web`，应按 `YOLO-OBB` 口径做前处理与后处理

## 本地预览视频

本轮 angle 预览视频：

- [annotated_preview_yolo_obb_angle_stage2_latest.mp4](/home/liujuncheng/rgbt_uav_detection/outputs/local_preview/annotated_preview_yolo_obb_angle_stage2_latest.mp4)
- [annotated_preview_yolo_obb_angle_stage2_latest.json](/home/liujuncheng/rgbt_uav_detection/outputs/local_preview/annotated_preview_yolo_obb_angle_stage2_latest.json)

摘要：

1. `rendered_frames = 600`
2. `inferenced_frames = 600`
3. `empty_detection_ratio = 0.016667`
4. `avg_model_latency_ms = 17.434`

说明：

1. 本地预览脚本已经走 `polygon + angle` 可视化链路
2. 视频中看到的斜框来自 `YOLO-OBB` 预测结果，不再是旧的矩形框展示

## 数据相关留档

本轮相关数据准备与修正留档：

1. 真 OBB 数据包方案：
   - [README.md](/home/liujuncheng/rgbt_uav_detection/docs/changes/20260412_yolo_obb_true_obb_data_plan/README.md)
2. 真 OBB 数据包与 stage1 准备：
   - [README.md](/home/liujuncheng/rgbt_uav_detection/docs/changes/20260412_trueobb_pack_and_stage1_prep/README.md)
3. 模型辅助修正摘要：
   - [model_refine_summary.json](/home/liujuncheng/rgbt_uav_detection/datasets/yolo_obb_official_rgb_trueobb_v1/model_refine_summary.json)

## 建议的交付口径

对前端：

1. 直接使用：
   - [best.onnx](/home/liujuncheng/rgbt_uav_detection/outputs/yolo_obb_runs/official_rgb_model_refined_stage2_v1/weights/best.onnx)
2. 如需参考可视化效果：
   - [annotated_preview_yolo_obb_angle_stage2_latest.mp4](/home/liujuncheng/rgbt_uav_detection/outputs/local_preview/annotated_preview_yolo_obb_angle_stage2_latest.mp4)

对后续研发：

1. 如需继续提升 angle 真实性，下一步还是补更多真 OBB 标注
2. 但当前这版已经足够作为可交付版本留档

## 一句话结论

`YOLO-OBB stage2` 当前已经达到：

1. 结果可接受
2. 预览可验证
3. ONNX 可导出
4. 可以留档并下线
