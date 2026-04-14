# 2026-04-12 YOLO-OBB Smoke Test Follow-up

## 本次执行内容

本轮已按 `RGB-only YOLO-OBB` 迁移方案完成以下动作：

1. 生成了小规模 `YOLO-OBB` smoke test 数据集：
   - [dataset.yaml](/home/liujuncheng/rgbt_uav_detection/datasets/yolo_obb_rgb_expand_refined/dataset.yaml)
   - [dataset_summary.json](/home/liujuncheng/rgbt_uav_detection/datasets/yolo_obb_rgb_expand_refined/dataset_summary.json)
2. 安装了 `ultralytics`
3. 启动了 `YOLO11n-obb` smoke test 训练
4. 产出了第一版 `best.pt / last.pt`

本轮训练配置：

- [yolo_obb_smoke_rgb.yaml](/home/liujuncheng/rgbt_uav_detection/configs/yolo_obb_smoke_rgb.yaml)

本轮输出目录：

- [smoke_rgb_expand_refined_v1](/home/liujuncheng/rgbt_uav_detection/outputs/yolo_obb_runs/smoke_rgb_expand_refined_v1)

## 数据口径

这轮不是“真 OBB 标注训练”，而是：

1. 把现有水平框 JSON
2. 转换成 Ultralytics OBB 的四点标签
3. 用“轴对齐矩形四点”先跑通 OBB 训练链路

因此本轮结果用于回答两个问题：

1. `YOLO-OBB` 在本机和当前数据上能不能跑
2. 即便没有真 angle 标注，OBB 框架本身是否有继续投入价值

不能直接把本轮结果解读为“已经学会真实 angle”。

## 训练结果

### 1. 训练是否跑通

跑通了。

证据：

1. 已生成：
   - [best.pt](/home/liujuncheng/rgbt_uav_detection/outputs/yolo_obb_runs/smoke_rgb_expand_refined_v1/weights/best.pt)
   - [last.pt](/home/liujuncheng/rgbt_uav_detection/outputs/yolo_obb_runs/smoke_rgb_expand_refined_v1/weights/last.pt)
2. 已生成：
   - [results.csv](/home/liujuncheng/rgbt_uav_detection/outputs/yolo_obb_runs/smoke_rgb_expand_refined_v1/results.csv)

### 2. 实际停在第几轮

这轮实际只记录到了 `epoch 9`。

也就是说：

1. 没有按计划完整跑满 `20 epoch`
2. 但已经足够说明训练链路是可用的
3. 当前最优结果也已经能拿来评估

### 3. 当前最优指标

按 [results.csv](/home/liujuncheng/rgbt_uav_detection/outputs/yolo_obb_runs/smoke_rgb_expand_refined_v1/results.csv)：

- `epoch 9`
  - `precision = 0.42384`
  - `recall = 0.52327`
  - `mAP50 = 0.46136`
  - `mAP50-95 = 0.35178`

从曲线看：

1. `mAP50` 从 `0.2774 -> 0.46136`
2. `mAP50-95` 从 `0.18902 -> 0.35178`
3. 整体仍在上升，没有出现明显崩塌

这说明：

1. `YOLO-OBB` 在当前数据上是能学到有效几何与检测表征的
2. 即使先用“矩形四点伪 OBB”，也已经比完全不做 OBB 验证更有价值

## 训练程度评估

### 结论

当前属于：

- `链路验证成功`
- `模型已初步收敛`
- `但仍远未完成正式训练`

### 为什么说“已初步收敛”

因为：

1. 前几轮指标明显上升
2. 到 `epoch 8-9` 仍有提升
3. 没出现完全学不动的情况

### 为什么说“远未完成正式训练”

因为：

1. 本轮只到 `epoch 9`
2. 训练计划原本是 `20 epoch`
3. 当前还只是小规模 smoke test 数据
4. 当前标签并不是真 OBB 标注

所以本轮只能回答：

- `值得继续`

还不能回答：

- `已经可作为最终前端部署 OBB 模型`

## 当前最重要判断

### 1. 这条路线值得继续

原因：

1. 环境打通了
2. CUDA 训练正常
3. 模型能收敛
4. 指标有持续提升

### 2. 下一次不应该继续在这版小数据上死磕太久

原因：

1. `1175/294` 的数据规模太小
2. 这轮只是 smoke test
3. 当前最重要的是把正式训练切到今天的官方 split 数据

### 3. 如果业务真的要 angle，必须尽快补真 OBB 标注

这是当前路线最大的限制：

1. 本轮只是矩形四点
2. 真 angle 监督仍然缺失
3. 不补真 OBB，就无法真正证明“模型学会了车的倾斜角”

## 下一次优化训练方案

### 第 1 步：把 smoke test 跑满一次完整 20 epoch

建议：

1. 保留当前 `batch=4`
2. 保留 `workers=0`
3. 保留 `amp=false`
4. 保留 `plots=false`

原因：

1. 当前这组参数已经证明能进训练循环
2. 先用最稳口径拿到一条完整曲线

目标：

1. 先确认 `epoch 20` 的最终指标
2. 看 `best epoch` 是否明显晚于 `epoch 9`

### 第 2 步：切到官方 split 主训练

下一轮正式训练建议改为：

- 数据：
  - [dataset.yaml](/home/liujuncheng/rgbt_uav_detection/datasets/yolo_obb_official_rgb_expand_v1/dataset.yaml)
- 配置：
  - [yolo_obb_official_rgb.yaml](/home/liujuncheng/rgbt_uav_detection/configs/yolo_obb_official_rgb.yaml)

建议先把配置调成更稳的首轮版本：

1. `batch: 4`
2. `workers: 0`
3. `amp: false`
4. `plots: false`
5. `epochs: 20-30` 先做首轮验证

不要一上来就直接冲 `80 epoch`，原因是：

1. 当前小数据 smoke test 还有“未跑满 20 epoch”的现象
2. 大数据集更应该先确保稳定性

### 第 3 步：正式补真 OBB hardcase 标注

优先级建议：

1. `val` 先补真 OBB
2. `train` 的 hardcase 样本补真 OBB
3. 其余样本仍可暂保留矩形四点

这样做的收益最大：

1. 先让验证集有真实 angle 评估
2. 再让训练集逐渐获得真实 angle 监督

### 第 4 步：等主训练稳定后，再做前端部署包

届时需要产出：

1. `best.pt`
2. `best.onnx`
3. `best_fp16.onnx`
4. `frontend_model_config.json`

当前这轮 smoke test 不建议立刻交给前端做最终接入，只适合内部验证。

## 推荐的下一轮具体配置

### smoke 完整版

在当前配置基础上继续：

- `epochs: 20`
- `batch: 4`
- `workers: 0`
- `amp: false`
- `plots: false`

### official 首轮稳定版

建议把 [yolo_obb_official_rgb.yaml](/home/liujuncheng/rgbt_uav_detection/configs/yolo_obb_official_rgb.yaml) 暂时改成：

1. `epochs: 30`
2. `batch: 4`
3. `workers: 0`
4. `amp: false`
5. `plots: false`

等这轮稳定跑通后，再逐步恢复：

1. `batch: 8/16`
2. `amp: true`
3. `epochs: 80`

## 一句话结论

这轮 `YOLO-OBB smoke test` 已经证明：

1. 路线能跑
2. 指标能涨
3. 值得继续

下一步最合理的动作不是继续纠结这轮小数据，而是：

1. 先补真 OBB hardcase 标注
2. 再切到今天的官方 split 数据做正式训练
