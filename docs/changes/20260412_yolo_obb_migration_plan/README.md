# 2026-04-12 YOLO-OBB 改造与训练方案

## 目标

把当前项目从 `RGB-T FCOS + 水平框` 演进到 `YOLO-OBB + angle` 路线，满足两个目标：

1. 支持斜框 / angle 训练与推理。
2. 为前端本地 ONNX 推理准备更标准的模型结构与输出语义。

## 当前事实

### 1. 当前模型不是 YOLO

当前线上与导出模型是：

- 双流 `RGB-T` 特征提取
- `cross_attention` 融合
- `LightweightBiFPN`
- `SmallObjectRefineHead`
- `torchvision FCOS`

对应代码：

- [model/detector.py](/home/liujuncheng/rgbt_uav_detection/model/detector.py)
- [model/network/backbone.py](/home/liujuncheng/rgbt_uav_detection/model/network/backbone.py)
- [model/network/fusion_module.py](/home/liujuncheng/rgbt_uav_detection/model/network/fusion_module.py)

当前模型摘要：

- 参数量：约 `2.97M`
- 输入：`[B, 6, H, W]`
- 输出：`boxes / scores / labels`
- 几何语义：`xyxy` 水平框
- 不支持：`angle / xywhr / polygon`

### 2. 当前数据只有水平框

当前本地训练集与评估集标注格式都是：

- `bbox = [x1, y1, x2, y2]`

对应代码：

- [data/dataset.py](/home/liujuncheng/rgbt_uav_detection/data/dataset.py)
- [data/preprocess.py](/home/liujuncheng/rgbt_uav_detection/data/preprocess.py)

这意味着：

- 现有数据可以转换成“矩形四点”
- 但不能仅靠转换就产生真实 `angle`
- 如果目标真的大量倾斜，想训练出有意义的 `angle`，必须补 `OBB` 标注

### 3. Ultralytics OBB 的现实约束

按官方 OBB 文档，Ultralytics OBB 训练使用四点标注，内部采用 `xywhr` 表示旋转框。

官方参考：

- https://docs.ultralytics.com/tasks/obb/
- https://docs.ultralytics.com/datasets/obb/

同时，Ultralytics 官方 OBB 主线默认是单图像单输入，不是当前项目这种 `6` 通道 `RGB+Thermal` 双流结构。

因此改造路线需要先做决策：

1. `推荐 v1`：先做 `RGB-only YOLO-OBB`
2. `扩展 v2`：再做 `RGB-T YOLO-OBB`

## 推荐路线

### v1：先上 RGB-only YOLO-OBB

这是推荐路线，原因很直接：

1. 前端本地推理最容易接入标准 YOLO-OBB ONNX
2. `angle` 能直接来自官方 OBB 头，而不是自定义输出
3. 数据标注当前本来就以 `RGB` 口径更稳定
4. 改造量显著小于“直接做双流 RGBT OBB”

这条路的目标不是替代当前 `RGB-T FCOS` 所有能力，而是先快速拿到：

1. 真正的 `angle`
2. 标准 OBB ONNX
3. 前端本地部署友好的模型结构

### v2：后续再做 RGBT YOLO-OBB

如果 v1 跑通并且业务确认必须保留热红外加成，再进入 v2：

1. 修改 Ultralytics 模型首层输入为 `6` 通道
2. 保留 `RGB / thermal` 双分支或做早期融合
3. 重写 dataloader 与 augment，适配成对图像
4. 重写导出与前端输入说明

这一步是正式模型研发，不是快速迁移。

## 这次要用的数据

### A. 先前数据

#### 1. 小规模精修 RGB 扩展集

- 路径：[datasets/dronevehicle_like_rgb_expand_refined](/home/liujuncheng/rgbt_uav_detection/datasets/dronevehicle_like_rgb_expand_refined)
- 摘要：[dataset_summary.json](/home/liujuncheng/rgbt_uav_detection/datasets/dronevehicle_like_rgb_expand_refined/dataset_summary.json)
- 统计：
  - `num_train = 1175`
  - `num_val = 294`
  - `annotation_source = rgb`
  - `bbox_expand_ratio = 0.12`

用途：

1. 先验证 OBB 数据转换脚本
2. 做 YOLO-OBB 首轮 smoke test
3. 快速检查 angle 流程是否跑通

#### 2. 小规模 merged_union 集

- 路径：[datasets/dronevehicle_like_refined](/home/liujuncheng/rgbt_uav_detection/datasets/dronevehicle_like_refined)
- 摘要：[dataset_summary.json](/home/liujuncheng/rgbt_uav_detection/datasets/dronevehicle_like_refined/dataset_summary.json)

用途：

1. 可作为误检对比集或补充验证集
2. 不建议作为 v1 主训练集

原因：

- 它的标注口径是 `merged_union`
- 更适合当前 RGB-T 框扩展路线
- 不适合作为标准 RGB-only OBB 主线

### B. 今天做的优质 hardcase 官方 split 数据

- 路径：[datasets/dronevehicle_like_official_rgb_expand_v1](/home/liujuncheng/rgbt_uav_detection/datasets/dronevehicle_like_official_rgb_expand_v1)
- 摘要：[dataset_summary.json](/home/liujuncheng/rgbt_uav_detection/datasets/dronevehicle_like_official_rgb_expand_v1/dataset_summary.json)
- 统计：
  - `num_train = 17962`
  - `num_val = 1469`
  - `split_strategy = official_split`
  - `annotation_source = rgb`
  - `bbox_expand_ratio = 0.12`
  - `skipped_corrupt_samples = 9`

用途：

1. 作为 YOLO-OBB 主训练集
2. 作为后续 hardcase angle 精修主战场
3. 作为最终前端部署模型的主数据来源

## 核心难点：angle 不能从水平框里“变出来”

这是本方案最重要的一条现实判断：

1. 把 `[x1,y1,x2,y2]` 转成四点矩形，只是把水平框换一种存储方式
2. 这样得到的四点仍然是轴对齐矩形
3. 模型可以按 OBB 框架训练，但学不到真正的倾斜角分布

所以要“支持 angle 的训练”，必须把今天这批 hardcase 优质数据里真正斜着的车补成真 OBB。

## 数据改造方案

### 第 1 阶段：把当前 JSON 水平框批量转成 YOLO-OBB 基础标签

新增脚本建议：

- `scripts/convert_hbb_json_to_yolo_obb.py`

输入：

- 当前 JSON 标注：`bbox=[x1,y1,x2,y2]`

输出：

- Ultralytics OBB `labels/*.txt`
- 每行格式：

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

初始转换规则：

1. 点序固定从左上开始
2. 顺时针输出四点
3. 坐标归一化到 `[0,1]`

注意：

- 这一步只用于“打通 OBB 训练管线”
- 它产生的是“矩形四点伪 OBB”
- 还不是最终能学出真实 angle 的标签

### 第 2 阶段：在官方 hardcase 数据上补真 OBB 标注

建议新增数据版本：

- `datasets/dronevehicle_like_official_rgb_expand_obb_v1`

建议标注范围：

1. 先只标 hardcase 子集
2. 优先标注明显斜车、密集遮挡、长车、边缘车
3. val 集优先全部补真 OBB
4. train 集先补一批高价值 hardcase，再逐步扩展

建议优先级：

1. `val` 全量真 OBB
2. `train` 中 hardcase 图片真 OBB
3. 其余 train 样本先保留矩形四点

这样做的好处：

1. 可以先让评估集有真实 angle 指标
2. 训练集不需要一次性全量重标
3. 可以先跑出第一版有 angle 能力的模型

### 第 3 阶段：形成混合训练集

训练集分两部分：

1. 大盘基础样本：
   - 来源：`dronevehicle_like_official_rgb_expand_v1`
   - 标签：矩形四点转换 OBB
2. 真 angle hardcase 样本：
   - 来源：人工补的 OBB 标注
   - 标签：真实四点 OBB

推荐混合方式：

1. 直接合并成一个 OBB 数据集
2. 对真 OBB hardcase 样本做过采样
3. 在后期 finetune 阶段进一步提高 hardcase 权重

## 训练实施方案

### 环境方案

建议新建独立环境，不在当前 FCOS 训练环境里硬塞：

```bash
/home/liujuncheng/miniconda3/bin/pip install ultralytics
```

说明：

1. 当前仓库里还没有 `ultralytics`
2. OBB 训练建议和当前 FCOS 代码分开
3. 避免把现有在线服务依赖链搅乱

### 模型选择

建议采用两阶段模型选择：

1. 首轮 smoke test：
   - 官方 OBB `n` 级预训练权重
2. 主训练：
   - 官方 OBB `s` 级预训练权重

模型命名说明：

1. Ultralytics 官方文档主页已经切到 `YOLO26`
2. 但历史版本与很多命令仍使用 `YOLO11`
3. 实操时以当前安装版本里可用的官方 `OBB n/s` 权重名为准

建议口径：

1. 能用 `yolo26n-obb.pt / yolo26s-obb.pt` 就优先用它
2. 如果安装版本仍沿用旧命名，则换成对应的 `yolo11n-obb.pt / yolo11s-obb.pt`

### 阶段 A：small smoke test

数据：

- `dronevehicle_like_rgb_expand_refined`

目标：

1. 先验证标签转换
2. 先验证训练命令
3. 先验证导出 ONNX

建议参数：

```bash
yolo obb train \
  model=<official_n_obb_checkpoint> \
  data=configs/yolo_obb_rgb_smoke_data.yaml \
  epochs=20 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  project=outputs/yolo_obb_runs \
  name=smoke_rgb_v1
```

验收标准：

1. 训练正常收敛
2. val 能正常出 OBB 结果
3. 能导出 ONNX
4. 前端能吃 `xywhr` 或四点 polygon

### 阶段 B：官方 split 主训练

数据：

- `dronevehicle_like_official_rgb_expand_v1`
- 加上真 OBB hardcase 标注子集

目标：

1. 形成第一版可部署的 angle 模型
2. 保留官方 split 的规模优势
3. 让 hardcase 真 OBB 对模型产生角度监督

建议参数：

```bash
yolo obb train \
  model=<official_s_obb_checkpoint_or_smoke_best> \
  data=configs/yolo_obb_rgb_official_data.yaml \
  epochs=80 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  project=outputs/yolo_obb_runs \
  name=official_rgb_hardcase_v1
```

建议附加策略：

1. 前 `50-60` epoch 跑全量混合集
2. 后 `20-30` epoch 降低学习率
3. 最后 `10-20` epoch 关闭强 mosaic，稳定框角度

### 阶段 C：hardcase 精修

数据：

- 真 OBB hardcase 子集

目标：

1. 继续强化斜车与密集小目标
2. 提升前端实际视频里的视觉观感

建议参数：

```bash
yolo obb train \
  model=outputs/yolo_obb_runs/official_rgb_hardcase_v1/weights/best.pt \
  data=configs/yolo_obb_rgb_hardcase_only_data.yaml \
  epochs=15 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  lr0=0.001 \
  project=outputs/yolo_obb_runs \
  name=official_rgb_hardcase_finetune_v1
```

## 训练时长估算

以下估算基于：

1. 本机 GPU：
   - `RTX 4060 Laptop GPU 8GB`
2. 当前官方 split 数据规模：
   - `17962 train`
   - `1469 val`
3. 当前 FCOS 官方 split 配置在 batch=4 时每 epoch 约 `4489` step
4. YOLO-OBB 如果切到 `RGB-only`，理论上会比当前双流 FCOS 更轻

### 数据准备

1. 水平框转 OBB 矩形四点脚本：
   - `0.5` 天以内
2. 标注抽检与可视化核对：
   - `0.5` 天
3. hardcase 真 OBB 补标：
   - 小批量 `300-500` 图：约 `0.5-1` 天
   - 中批量 `1000-2000` 图：约 `1-3` 天

### 模型训练

#### smoke test

1. `n` 级 OBB，`20` epoch：
   - 约 `0.5-1.5` 小时

#### 主训练

1. `n` 级 OBB，`80` epoch：
   - 约 `4-8` 小时
2. `s` 级 OBB，`80` epoch：
   - 约 `8-16` 小时

#### hardcase finetune

1. `10-20` epoch：
   - 约 `0.5-2` 小时

### 一句话估算

如果不算人工补真 OBB 标注，只算机器训练：

1. 首版能跑通：当天内
2. 首版能拿到可用 OBB 模型：`1` 天左右

如果把“真 angle 标注”也算进去：

1. 更现实的完整周期：`2-4` 天

## 输出留档要求

每一轮都必须留下面这些内容：

1. 数据留档：
   - `dataset_summary.json`
   - `labels_obb_audit.json`
   - `train/val` 样本数
   - 类别分布
2. 训练留档：
   - `train_args.yaml`
   - `results.csv`
   - `confusion_matrix.png`
   - `PR_curve.png`
   - `val_batch*.jpg`
3. 模型留档：
   - `best.pt`
   - `last.pt`
   - `best.onnx`
   - `best_fp16.onnx`
   - `frontend_model_config.json`
4. 说明文档：
   - 本轮数据来源
   - 是否包含真 OBB hardcase
   - 主要指标
   - 前端接入说明

## 推荐新增文件与目录

建议按下面这组路径推进：

### 脚本

- `scripts/convert_hbb_json_to_yolo_obb.py`
- `scripts/build_yolo_obb_rgb_dataset.py`
- `scripts/export_yolo_obb_frontend_package.py`

### 配置

- `configs/yolo_obb_rgb_smoke_data.yaml`
- `configs/yolo_obb_rgb_official_data.yaml`
- `configs/yolo_obb_rgb_hardcase_only_data.yaml`

### 数据

- `datasets/dronevehicle_like_rgb_expand_obb_v1`
- `datasets/dronevehicle_like_official_rgb_expand_obb_v1`

### 输出

- `outputs/yolo_obb_runs/`
- `weights/yolo_obb_rgb_official_v1/`

## 给前端的直接结论

如果你们要的是：

1. 前端本地推理
2. 真 angle
3. 标准 OBB ONNX

那么推荐路线就是：

1. 先放弃当前 `RGB-T FCOS` 直接改 angle 的想法
2. 先做 `RGB-only YOLO-OBB`
3. 在今天的官方 split hardcase 优质数据上补真 OBB
4. 首版模型先交前端本地跑

## 最终建议

按优先级，建议这样执行：

1. 先做 `RGB-only YOLO-OBB`
2. 先把现有 JSON 水平框转成 OBB 四点标签，跑通全链路
3. 同步补今天这批 hardcase 优质数据的真 OBB 标注
4. 先出一版 `n/s` 级 OBB ONNX 给前端本地推理
5. 等前端把 angle 跑顺，再决定是否值得做 `RGB-T YOLO-OBB`
