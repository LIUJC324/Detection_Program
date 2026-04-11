# 2026-04-11 Dual Stream Start API

## 变更背景

按 `2026-04-11 16:51` 的新联调口径，模型端实时流启动参数改为双路输入：

```json
{
  "sessionId": "sid_xxx",
  "rgbPullUrl": "http://8.137.107.232:8088/{rgbKey}.flv",
  "irPullUrl": "http://8.137.107.232:8088/{irKey}.flv",
  "sampleFps": 4,
  "callbackToken": "aerialeye"
}
```

`stop` 接口保持不变。

## 本次改动

已改文件：

- [service/core/schemas.py](/home/liujuncheng/rgbt_uav_detection/service/core/schemas.py)
- [service/api/app.py](/home/liujuncheng/rgbt_uav_detection/service/api/app.py)
- [service/streaming/session_manager.py](/home/liujuncheng/rgbt_uav_detection/service/streaming/session_manager.py)
- [docs/integration/interface.md](/home/liujuncheng/rgbt_uav_detection/docs/integration/interface.md)
- [docs/integration/发给前后端同学的固定公网联调清单_20260407.md](/home/liujuncheng/rgbt_uav_detection/docs/integration/发给前后端同学的固定公网联调清单_20260407.md)

## 行为变化

1. `/v1/inference/stream/start` 现在支持：
   - `rgbPullUrl`
   - `irPullUrl`
2. 双路 URL 会由模型端内部用 ffmpeg 拼成现有的双模态帧流，再沿用原有推理链。
3. 历史 `sourceUrl` 单路 mixed-stream 写法仍保留兼容。
4. 历史 `rgbPushUrl / irPushUrl` 拼写也保留兼容。
5. `app / streamKey / resultCallbackUrl / pairLayout / rgbPosition / callbackToken / sampleFps` 等其他字段继续保留。
6. `stop` 接口不变。

## sampleFps

`sampleFps` 仍然保留。

当前建议：

1. 前后端显式传 `sampleFps`
2. 当前联调口径可直接传 `4`

当前稳定部署默认通常是：

- `2.0`

## 后续补充

在联调阶段又补了一层展示侧修正：

1. 针对“框恒定偏小”，在推理后处理里增加了可配置的框扩张
2. 针对“一大一小重合框”，增加了同类框重合/包含抑制

这两步都属于：

- 联调展示侧的临时止血

不替代后续基于新数据口径的根因训练。
