# RGBT UAV Detection

RGB-T UAV vehicle detection project for all-weather traffic scenes.  
The repository covers:

- dataset conversion and preprocessing
- model training and evaluation
- export and inference
- FastAPI model service
- video/stream session inference for frontend/backend integration

The current implementation is based on:

- `PyTorch`
- `torchvision FCOS`
- custom RGB-T dual-branch backbone and fusion modules
- `FastAPI + ffmpeg/ffprobe`

## Current Status

This repository is no longer just a model skeleton. The following items are already in place:

- training / evaluation / export / service pipeline
- stream and video session inference
- fixed public integration entry via `FRP`
- evaluation bug fix and trusted `TP / FP / FN` metrics
- FCOS feature-scale alignment fix
- resumed training run completed to `epoch 60`
- small-scope transfer-learning entry prepared for `ResNet18`

Current model status:

- the model is no longer stuck at `TP = 0`
- the repaired framework can produce positive recall
- false positives are still very high
- integration is usable, but model quality is not yet production-grade

Most important tracking document:

- [docs/training/模型端问题排查、框架修复与续训方案_20260409.md](/home/liujuncheng/rgbt_uav_detection/docs/training/模型端问题排查、框架修复与续训方案_20260409.md)

## Repository Layout

```text
rgbt_uav_detection/
├── configs/          # training / deployment / experiment configs
├── data/             # dataset loading and preprocessing
├── docs/             # training, integration, deployment, testing notes
├── model/            # backbone, fusion, neck, head, detector assembly
├── scripts/          # train / eval / export / utilities
├── service/          # FastAPI service and stream session manager
├── weights/          # best / last / deployment snapshots
├── outputs/          # train runs, logs, evaluation outputs
├── class_mapping.json
├── requirements.txt
└── README.md
```

## Data Format

Default dataset layout:

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

Annotation example:

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

Current class mapping:

```json
{
  "0": "car",
  "1": "truck",
  "2": "bus",
  "3": "van",
  "4": "freight_car"
}
```

## Quick Start

### 1. Train

```bash
cd /home/liujuncheng/rgbt_uav_detection
/home/liujuncheng/miniconda3/bin/python scripts/train.py \
  --config configs/default.yaml \
  --num-workers 0
```

Resume training:

```bash
/home/liujuncheng/miniconda3/bin/python scripts/train.py \
  --config configs/default.yaml \
  --resume weights/last.pt \
  --num-workers 0
```

### 2. Evaluate

```bash
/home/liujuncheng/miniconda3/bin/python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint weights/best.pt \
  --output outputs/evaluation.json \
  --num-workers 0
```

### 3. Single-Pair Inference

```bash
/home/liujuncheng/miniconda3/bin/python scripts/infer_demo.py \
  --rgb demo/rgb.jpg \
  --thermal demo/thermal.jpg
```

### 4. Export

```bash
/home/liujuncheng/miniconda3/bin/python scripts/export.py \
  --config configs/default.yaml \
  --checkpoint weights/best.pt \
  --format both
```

## Service

### Local Service

```bash
cd /home/liujuncheng/rgbt_uav_detection
DEPLOY_CONFIG=/home/liujuncheng/rgbt_uav_detection/configs/deploy_stable.yaml \
/home/liujuncheng/miniconda3/bin/python -m uvicorn service.api.app:app \
  --host 127.0.0.1 \
  --port 18000
```

Health check:

```bash
curl -fsS http://127.0.0.1:18000/v1/health
curl -fsS http://127.0.0.1:18000/v1/model/info
```

### Integration Endpoints

Implemented service endpoints:

- `GET /v1/health`
- `GET /v1/model/info`
- `POST /v1/detect/stream`
- `POST /v1/inference/video/start`
- `POST /v1/inference/video/stop`
- `POST /v1/inference/stream/start`
- `POST /v1/inference/stream/stop`
- `GET /v1/inference/session/{sessionId}`

Notes:

- API paths are kept stable for frontend/backend integration.
- current callback mode is `rectangle`
- stream/video inference uses `sampleFps` for frame sampling

## Current Engineering Decisions

### Trusted Evaluation

The project previously had an AP computation issue.  
This has been fixed, and the current evaluation outputs should be interpreted using:

- `mAP50`
- `recall50`
- `small_recall50`
- `TP / FP / FN`
- per-class stats

### FCOS Scale Fix

The most important structural issue was feature-scale mismatch between the custom RGB-T feature pyramid and `torchvision FCOS`.  
This has been fixed in the detector assembly so the model is no longer trapped in the old “high-score point box” failure mode.

### Empty-Box Handling

Integration-side callback logic now includes:

- degenerate-box filtering
- callback fallback confidence path

This is meant to reduce:

- frequent `boxes: []`
- zero-area / empty rectangle artifacts

without changing API fields.

## Small-Scope Transfer Learning

A low-engineering-cost transfer-learning entry has been prepared:

- configurable dual-branch `ResNet18`
- current experiment config:
  - [configs/experiment_resnet18_transfer.yaml](/home/liujuncheng/rgbt_uav_detection/configs/experiment_resnet18_transfer.yaml)

Current limitation:

- there is no local cached pretrained `ResNet18` weight file in the environment
- the repository is ready for transfer-learning experiments, but the actual pretrained checkpoint still needs to be provided locally

## Integration and Demo Strategy

Recommended real-time demo route:

1. frontend plays video/stream
2. backend creates session and calls model service
3. model service samples frames and performs detection
4. model service callbacks results to backend
5. frontend overlays boxes on the player

For a more stable presentation effect, a second route is also available:

- render annotated demo videos offline and play the result as a showcase asset

Useful scripts:

- [scripts/export_dronevehicle_annotated_demo.py](/home/liujuncheng/rgbt_uav_detection/scripts/export_dronevehicle_annotated_demo.py)
- [scripts/export_dronevehicle_showcase.py](/home/liujuncheng/rgbt_uav_detection/scripts/export_dronevehicle_showcase.py)

## Documentation Map

Start here:

1. [docs/training/模型端问题排查、框架修复与续训方案_20260409.md](/home/liujuncheng/rgbt_uav_detection/docs/training/模型端问题排查、框架修复与续训方案_20260409.md)
2. [docs/training/项目需求确认与借鉴优化方案_20260409.md](/home/liujuncheng/rgbt_uav_detection/docs/training/项目需求确认与借鉴优化方案_20260409.md)
3. [docs/integration/发给前后端同学的固定公网联调清单_20260407.md](/home/liujuncheng/rgbt_uav_detection/docs/integration/发给前后端同学的固定公网联调清单_20260407.md)
4. [docs/integration/实时视频演示与前后端衔接说明.md](/home/liujuncheng/rgbt_uav_detection/docs/integration/实时视频演示与前后端衔接说明.md)
5. [docs/ops/稳定部署方案_20260407.md](/home/liujuncheng/rgbt_uav_detection/docs/ops/稳定部署方案_20260407.md)
6. [docs/README.md](/home/liujuncheng/rgbt_uav_detection/docs/README.md)

## Roadmap

Near-term priorities:

- continue reducing false positives
- verify whether empty callback frames have materially decreased in integration
- run the prepared small-scope transfer-learning experiment
- keep integration logs and training status synchronized into the main tracking document

Longer-term priorities:

- stronger RGB-T fusion module migration
- cleaner deployment process
- optional offline annotated-video generation path for demos
