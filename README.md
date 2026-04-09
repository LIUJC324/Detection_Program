# RGBT 无人机检测项目

面向全天候交通场景的 RGB-T 无人机车辆检测项目。  
仓库当前覆盖：

- 数据预处理与数据转换
- 模型训练与评估
- 导出与单帧推理
- FastAPI 模型服务
- 视频 / 实时流会话式推理
- 前后端联调支撑

当前实现主要基于：

- `PyTorch`
- `torchvision FCOS`
- 自定义 RGB-T 双分支骨干与融合模块
- `FastAPI + ffmpeg/ffprobe`

## 当前状态

这个仓库已经不只是一个“模型骨架”，目前已经具备：

- 训练 / 评估 / 导出 / 服务化完整链路
- 视频与流式会话推理
- 固定公网联调入口 `FRP`
- 评估逻辑修复，指标已能输出可信的 `TP / FP / FN`
- FCOS 特征尺度错位问题已修复
- 一轮修复后续训已完整跑到 `epoch 60`
- 小规模迁移学习入口已预留（`ResNet18`）

当前模型真实状态：

- 已经不再卡在 `TP = 0`
- 修复后框架可以打出正召回
- 误检仍然偏高
- 可以联调，但还不是可直接交付的高质量模型

当前最重要的跟踪文档：

- [docs/training/模型端问题排查、框架修复与续训方案_20260409.md](/home/liujuncheng/rgbt_uav_detection/docs/training/模型端问题排查、框架修复与续训方案_20260409.md)

## 仓库结构

```text
rgbt_uav_detection/
├── configs/          # 训练 / 部署 / 实验配置
├── data/             # 数据加载与预处理
├── docs/             # 训练、联调、部署、测试文档
├── model/            # backbone、fusion、neck、head、detector
├── scripts/          # 训练、评估、导出与工具脚本
├── service/          # FastAPI 服务与流会话管理
├── weights/          # best / last / 部署快照权重
├── outputs/          # 训练产物、日志、评估结果
├── class_mapping.json
├── requirements.txt
└── README.md
```

## 数据格式

默认数据组织方式：

```text
datasets/dronevehicle_like_refined/
├── rgb/
│   ├── train/
│   └── val/
├── thermal/
│   ├── train/
│   └── val/
└── annotations/
    ├── train/
    └── val/
```

标注 JSON 示例：

```json
{
  "image_id": "000001",
  "objects": [
    {
      "bbox": [120, 80, 164, 112],
      "class_id": 0,
      "class_name": "car"
    }
  ]
}
```

当前类别映射：

```json
{
  "0": "car",
  "1": "truck",
  "2": "bus",
  "3": "van",
  "4": "freight_car"
}
```

## 快速开始

### 1. 训练

```bash
cd /home/liujuncheng/rgbt_uav_detection
/home/liujuncheng/miniconda3/bin/python scripts/train.py \
  --config configs/default.yaml \
  --num-workers 0
```

恢复训练：

```bash
/home/liujuncheng/miniconda3/bin/python scripts/train.py \
  --config configs/default.yaml \
  --resume weights/last.pt \
  --num-workers 0
```

### 2. 评估

```bash
/home/liujuncheng/miniconda3/bin/python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint weights/best.pt \
  --output outputs/evaluation.json \
  --num-workers 0
```

### 3. 单次图像对推理

```bash
/home/liujuncheng/miniconda3/bin/python scripts/infer_demo.py \
  --rgb demo/rgb.jpg \
  --thermal demo/thermal.jpg
```

### 4. 导出

```bash
/home/liujuncheng/miniconda3/bin/python scripts/export.py \
  --config configs/default.yaml \
  --checkpoint weights/best.pt \
  --format both
```

## 模型服务

### 本地启动

```bash
cd /home/liujuncheng/rgbt_uav_detection
DEPLOY_CONFIG=/home/liujuncheng/rgbt_uav_detection/configs/deploy_stable.yaml \
/home/liujuncheng/miniconda3/bin/python -m uvicorn service.api.app:app \
  --host 127.0.0.1 \
  --port 18000
```

健康检查：

```bash
curl -fsS http://127.0.0.1:18000/v1/health
curl -fsS http://127.0.0.1:18000/v1/model/info
```

### 已实现接口

- `GET /v1/health`
- `GET /v1/model/info`
- `POST /v1/detect/stream`
- `POST /v1/inference/video/start`
- `POST /v1/inference/video/stop`
- `POST /v1/inference/stream/start`
- `POST /v1/inference/stream/stop`
- `GET /v1/inference/session/{sessionId}`

说明：

- 接口路径对前后端联调保持稳定
- 当前回调模式是 `rectangle`
- 视频 / 流推理通过 `sampleFps` 控制抽帧频率

## 当前工程关键决策

### 1. 评估口径已修复

项目之前存在 AP 计算问题。现在已经修正，当前评估应结合下面这些指标一起看：

- `mAP50`
- `recall50`
- `small_recall50`
- `TP / FP / FN`
- per-class 统计

### 2. FCOS 特征尺度错位已修复

当前最关键的结构性问题，是自定义 RGB-T 特征金字塔与 `torchvision FCOS` 的尺度假设不匹配。  
这一点已经在 detector 装配层修复，模型不再停留在“高分点框 / 空框泛滥”的旧故障模式。

### 3. 空盒问题已做模型端缓解

当前联调回调链路已经加入：

- 退化框过滤
- 回调 fallback 置信度路径

目标是降低：

- 高频 `boxes: []`
- 零面积框 / 空盒 / 线盒

同时不改现有接口字段。

## 小规模迁移学习入口

目前已经预留了一条工程量较小的迁移学习路线：

- 可切换为双分支 `ResNet18`
- 实验配置文件：
  - [configs/experiment_resnet18_transfer.yaml](/home/liujuncheng/rgbt_uav_detection/configs/experiment_resnet18_transfer.yaml)

当前限制：

- 本地还没有可直接加载的 `ResNet18` 预训练权重文件
- 也没有本地缓存的 `torchvision` 官方权重

所以目前是：

- 工程入口和配置已准备好
- 真正的迁移实验还需要补一份本地预训练权重

## 联调与展示建议

推荐的实时演示链路：

1. 前端负责播放视频 / 流
2. 后端创建推理会话并调用模型服务
3. 模型端按 `sampleFps` 抽帧推理
4. 模型端回调检测结果给后端
5. 前端在播放器上叠加检测框

如果更强调展示稳定性，也可以补一条离线方案：

- 模型端离线渲染带框视频
- 前端直接播放成品视频作为演示素材

相关脚本：

- [scripts/export_dronevehicle_annotated_demo.py](/home/liujuncheng/rgbt_uav_detection/scripts/export_dronevehicle_annotated_demo.py)
- [scripts/export_dronevehicle_showcase.py](/home/liujuncheng/rgbt_uav_detection/scripts/export_dronevehicle_showcase.py)

## 文档导航

建议从这里开始看：

1. [docs/training/模型端问题排查、框架修复与续训方案_20260409.md](/home/liujuncheng/rgbt_uav_detection/docs/training/模型端问题排查、框架修复与续训方案_20260409.md)
2. [docs/training/项目需求确认与借鉴优化方案_20260409.md](/home/liujuncheng/rgbt_uav_detection/docs/training/项目需求确认与借鉴优化方案_20260409.md)
3. [docs/integration/发给前后端同学的固定公网联调清单_20260407.md](/home/liujuncheng/rgbt_uav_detection/docs/integration/发给前后端同学的固定公网联调清单_20260407.md)
4. [docs/integration/实时视频演示与前后端衔接说明.md](/home/liujuncheng/rgbt_uav_detection/docs/integration/实时视频演示与前后端衔接说明.md)
5. [docs/ops/稳定部署方案_20260407.md](/home/liujuncheng/rgbt_uav_detection/docs/ops/稳定部署方案_20260407.md)
6. [docs/README.md](/home/liujuncheng/rgbt_uav_detection/docs/README.md)

## 后续路线

近期优先级：

- 继续降低误检
- 持续观察联调日志，验证空盒是否下降
- 启动小规模迁移学习实验
- 保持训练进展、联调状态与主文档同步

中长期优先级：

- 升级 RGB-T 融合模块
- 优化部署与服务切换流程
- 增加离线带框视频导出链路作为演示兜底
