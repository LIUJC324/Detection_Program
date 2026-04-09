，y# 接口变更详解（2026-04-03）

## 1. 变更背景

本次为同一轮联调中的两组改造：

1. 实时流会话改造：前端返回双路 WebRTC，模型端仅消费单路 HTTP-FLV 合流地址。
2. 视频会话改造：支持 RGB/IR 双文件物理拼接后再推理，并补充能力探测接口。
3. 模型视频启动命令新增固定字段：`sample_fps=3`、`pair_layout=side_by_side_h`、`rgb_position=left`。

## 2. 受影响接口

### 2.1 前端调用后端

1. `POST /api/v1/stream/sessions/start`
2. `GET /api/v1/stream/sessions/{sessionId}`
3. `POST /api/v1/stream/sessions/{sessionId}/stop`
4. `POST /api/v1/detect/video/sessions/start`
5. `GET /api/v1/detect/video/sessions/capability`

### 2.2 后端调用模型端

1. `aerialeye.inference.stream-start-endpoint`
2. `aerialeye.inference.video-start-endpoint`

## 3. 契约变更

### 3.1 实时流会话（`/api/v1/stream/sessions`）

#### 3.1.1 `POST /start`

变更后请求体：

```json
{
  "app": "live",
  "streamKey": "abc123"
}
```

变更后响应体（移除 `sourceUrl`，新增双路 WebRTC）：

```json
{
  "sessionId": "sid_xxx",
  "app": "live",
  "streamKey": "rgb_key",
  "rgbWebrtcUrl": "webrtc://127.0.0.1:1985/live/rgb_key",
  "irWebrtcUrl": "webrtc://127.0.0.1:1985/live/ir_key",
  "status": "RUNNING",
  "startedAt": "2026-04-03T10:10:00"
}
```

#### 3.1.2 `GET /{sessionId}`

与 `POST /start` 的响应结构对齐：

1. 返回 `rgbWebrtcUrl`、`irWebrtcUrl`
2. 不再返回 `sourceUrl`

#### 3.1.3 `POST /{sessionId}/stop`

除停止模型会话外，新增：

1. 释放会话绑定合流资源
2. 引用计数归零时销毁 FFmpeg 合流进程

### 3.2 视频会话（`/api/v1/detect/video/sessions`）

#### 3.2.1 `POST /start`

支持双模式入参：

1. 单文件模式：`objectKey`
2. 双文件拼接模式：`rgbObjectKey + irObjectKey`

示例：

```json
{
  "bucket": "aerialeye-video",
  "objectKey": "video/origin/2026-04-01/uuid.mp4",
  "rgbObjectKey": "video/origin/2026-04-01/rgb.mp4",
  "irObjectKey": "video/origin/2026-04-01/ir.mp4"
}
```

规则：

1. 传入 `rgbObjectKey` 或 `irObjectKey` 任一字段时，按双文件模式处理。
2. 双文件模式要求二者都非空。
3. 双文件模式由后端执行物理拼接：RGB 在左、IR 在右（`hstack`）。

#### 3.2.2 `GET /capability`

用于前端探测是否可用双文件拼接：

```json
{
  "pairComposeEnabled": true,
  "legacySingleEnabled": true,
  "ffmpegReady": true,
  "ffmpegPath": "ffmpeg",
  "pairLayout": "side_by_side_h",
  "rgbPosition": "left",
  "pairRequiredFields": ["rgbObjectKey", "irObjectKey"],
  "message": "Pair compose is ready."
}
```

### 3.3 模型端命令

#### 3.3.1 `stream-start-endpoint`

命令结构不新增字段，但 `sourceUrl` 语义变更为：

1. 不再是原始推流地址
2. 固定为后端合流后的单路 HTTP-FLV 地址
   `http://{httpFlvHost}/{app}/mix_{rgbKey}_{irKey}.flv`

#### 3.3.2 `video-start-endpoint`

命令新增固定字段：

1. `sample_fps`: `3`
2. `pair_layout`: `"side_by_side_h"`
3. `rgb_position`: `"left"`

示例：

```json
{
  "sessionId": "sid_xxx",
  "sourceUrl": "https://...",
  "bucket": "aerialeye-video",
  "objectKey": "video/result/2026-04-03/mix_rgb_ir.mp4",
  "sample_fps": 3,
  "pair_layout": "side_by_side_h",
  "rgb_position": "left",
  "resultCallbackUrl": "http://host:8080/api/v1/detect/model/result",
  "callbackToken": "change-me"
}
```

## 4. 机制变更（详细）

### 4.1 实时流启动机制

1. 以后端 `app + streamKey` 反查同设备 RGB/IR 路由。
2. 按规则命名合流 key：`mix_{rgbKey}_{irKey}`。
3. 启动 FFmpeg 合流（`hstack`，RGB 左 IR 右）。
4. 生成模型端拉流地址：`http://{httpFlvHost}/{app}/mix_{rgbKey}_{irKey}.flv`。
5. 启动模型流会话并绑定 `sessionId -> mixKey`。
6. 向前端返回 `rgbWebrtcUrl`、`irWebrtcUrl`。

### 4.2 视频双文件机制

1. 校验 `rgbObjectKey` 与 `irObjectKey` 都存在。
2. 使用 FFmpeg 进行物理拼接（RGB 左 IR 右）。
3. 上传拼接文件到 MinIO（`video/result/...`）。
4. 启动模型视频会话，`sourceUrl` 指向拼接产物。
5. 同步下发固定三元组：`sample_fps/pair_layout/rgb_position`。

## 5. 错误码与失败场景

1. 路由反查失败：`4000`
2. 同一路由已有运行会话：`3001`
3. 缺失 RGB/IR 配对通道：`2101`
4. 视频文件不存在：`5000`
5. FFmpeg 合流/拼接失败：`500`
6. 模型启动失败：`3003`

## 6. 前置条件

1. 模型端可拉取 HTTP-FLV 与 MinIO 预签名视频地址。
3. 模型端已兼容新增字段：`sample_fps`、`pair_layout`、`rgb_position`。

## 7. 对接改造清单

### 7.1 前端

1. 实时流页面改用 `rgbWebrtcUrl/irWebrtcUrl`。
2. 移除对流会话 `sourceUrl` 的依赖。
3. 视频会话启动前可调用 `/capability` 决定是否显示双文件拼接入口。

### 7.2 模型端

1. 流会话继续读取 `sourceUrl`，但按“单路 HTTP-FLV 合流地址”处理。
2. 视频会话读取新增固定字段并按固定语义处理。
3. 不再假设输入一定是原始单路 RTMP。

## 8. 兼容性说明

1. 流会话响应字段为不兼容变更：`sourceUrl` 已移除。
2. 视频会话启动请求为向后兼容扩展：保留 `objectKey` 单文件模式。
3. 视频模型命令 JSON 为新增字段扩展，模型端需同步升级解析逻辑。
