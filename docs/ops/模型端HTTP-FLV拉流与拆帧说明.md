# 模型端 HTTP-FLV 拉流与拆帧说明

这份说明只讲一个问题：

`模型端怎样主动拉 HTTP-FLV，并把视频拆成 RGB-T 图片对再喂给当前检测模型。`

## 1. 当前实现做了什么

我已经在模型服务里补了最小可用的“主动拉流会话”能力。

新增接口：

- `POST /v1/inference/video/start`
- `POST /v1/inference/stream/start`
- `POST /v1/inference/session/stop`
- `GET /v1/inference/session/{session_id}`

对应代码位置：

- `service/api/app.py`
- `service/streaming/session_manager.py`
- `service/core/predictor.py`
- `service/core/schemas.py`

## 2. 当前实现的核心假设

这是最重要的一点。

当前实现默认假设：

- 输入给模型端的是一个单路视频流
- 这个视频流的每一帧里已经同时包含 RGB 和 Thermal 两部分
- 两部分的排布方式默认是“左右拼接”

也就是一帧长这样：

`[ RGB | Thermal ]`

默认配置在：

- `configs/deploy.yaml`

对应字段：

- `default_pair_layout: side_by_side_h`
- `default_rgb_position: left`

如果未来你们的视频是上下拼接，也支持：

- `pair_layout = stacked_v`

## 3. 为什么这样设计

因为你前面说的约束是：

- 最终拿到的不是两路独立图片
- 而是视频数据
- 但当前模型推理入口需要 RGB/Thermal 图像对

所以必须在模型端补一层：

1. 拉视频流
2. 解码成帧
3. 从每帧中拆出 RGB 和 Thermal 两块
4. 转成当前 `Predictor` 可用的输入

这层本质上就是：

`video frame -> paired images -> detector`

## 4. HTTP-FLV 是怎么被支持的

当前实现没有自己手写 FLV 解封装器，而是直接复用：

- `ffmpeg`
- `ffprobe`

做法是：

1. `ffprobe` 先拿视频宽高
2. `ffmpeg` 从 `source_url` 持续读取视频
3. 用 `fps=...` 做抽帧
4. 把帧以 `rgb24 rawvideo` 形式输出到管道
5. Python 进程按固定字节数一帧一帧读取

这样做的好处是：

- 实现简单
- 兼容性强
- HTTP-FLV、普通 mp4、其他 ffmpeg 可识别的 URL 都能走同一条链

所以严格讲：

- 当前实现支持的是“ffmpeg 可读取的视频 URL”
- HTTP-FLV 只是其中一种典型输入

## 4.1 当前已经验证到什么程度

目前已经验证通过的链路包括：

- 本地 side-by-side `mp4` 文件
- 本地 side-by-side 视频会话
- 本地 HTTP 静态服务暴露的视频 URL，再由模型端主动读取

为了方便本地验证，我还补了两个脚本：

- `scripts/make_dronevehicle_demo_videos.sh`
- `scripts/test_http_paired_video_session.py`

其中前者会生成：

- `dronevehicle_rgb_thermal_side_by_side.mp4`
- `dronevehicle_rgb_thermal_side_by_side.flv`

后者会：

1. 在本机起一个 HTTP 静态服务
2. 暴露上面的 side-by-side 视频文件
3. 让模型端按 HTTP URL 主动去拉
4. 验证“HTTP 视频 -> 拆帧 -> 图片对 -> 检测”整条链

这说明当前实现不只是“理论上支持”，而是已经能跑通本地主动拉取的验证链路。

并且当前已经在本机完成过下面这条验证：

- 本地静态 HTTP 服务暴露 `.flv`
- 模型端按 `http://127.0.0.1:8765/...flv` 主动拉取
- 成功拆帧并完成检测会话

## 5. 一帧是怎样被拆成图片对的

在 `service/streaming/session_manager.py` 里，核心逻辑是：

1. 读到一帧完整 RGB 图像
2. 按布局规则切一刀
3. 左半边/上半边作为 RGB
4. 右半边/下半边作为 Thermal
5. 再调用 `predictor.predict_arrays(rgb_frame, thermal_frame)`

也就是说，当前模型端补的是：

- 从“视频帧”
- 到“图像对”

的桥接层。

## 6. Thermal 为什么还能直接复用当前模型

当前模型输入要求是：

- RGB 三通道
- Thermal 也喂成三通道

所以实现里做了这件事：

- Thermal 半帧如果是彩色图，先转灰度
- 再复制成三通道

这样就和现有训练/推理的 Thermal 处理口径一致。

## 7. 当前接口怎么用

### 7.1 启动视频文件会话

```json
POST /v1/inference/video/start
{
  "session_id": "video_demo_001",
  "source_url": "http://host/demo_side_by_side.mp4",
  "result_callback_url": "http://backend/api/v1/detect/model/result",
  "callback_token": "change-me",
  "sample_fps": 2,
  "pair_layout": "side_by_side_h",
  "rgb_position": "left"
}
```

### 7.2 启动实时流会话

```json
POST /v1/inference/stream/start
{
  "session_id": "stream_demo_001",
  "app": "live",
  "stream_key": "abc123",
  "source_url": "http://host/live/abc123.flv",
  "result_callback_url": "http://backend/api/v1/detect/model/result",
  "callback_token": "change-me",
  "sample_fps": 2,
  "pair_layout": "side_by_side_h",
  "rgb_position": "left"
}
```

这里的 `source_url` 就可以是 HTTP-FLV。

如果只是本地联调测试，也可以直接用：

- `http://127.0.0.1:8765/dronevehicle_rgb_thermal_side_by_side.flv`
- `http://127.0.0.1:8765/dronevehicle_rgb_thermal_side_by_side.mp4`

### 7.3 停止会话

```json
POST /v1/inference/session/stop
{
  "session_id": "stream_demo_001"
}
```

### 7.4 查询会话

```text
GET /v1/inference/session/stream_demo_001
```

它会返回：

- 会话状态
- 已处理帧数
- 最近错误
- 最近一次检测结果

## 8. 结果是怎样回给后端的

每处理完一帧，模型端会把结果回调给后端，主要字段包括：

- `session_id`
- `source_type`
- `frame_index`
- `frame_timestamp`
- `stream_key`
- `app`
- `detections`
- `inference_time`
- `image_size`
- `model_version`

也就是说，模型端现在已经能满足：

- 连续处理视频帧
- 连续输出时序检测结果

## 9. 当前实现的边界

这个版本是“最小可用实现”，不是完整生产版。

当前边界包括：

1. 默认假设单路视频里已经拼好了 RGB-T
2. 还不支持“两路独立流自动按时间戳配对”
3. 还没有做复杂重试、断线重连、缓冲队列治理
4. 还没有做高并发优化
5. 依赖本机存在 `ffmpeg` 和 `ffprobe`

补充：

- 当前回调后端失败时，会记录 `last_error`，但不会让整条视频会话立刻崩掉
- 这样更适合联调和演示阶段排查问题

## 10. 如果现场流不是拼接视频怎么办

那就不是当前这版代码的假设了。

如果现场拿到的是：

- 一路 RGB 流
- 一路 Thermal 流

那么下一步需要补的是：

- 双流拉流器
- 两路时间同步
- 两路帧配对器

也就是：

`RGB stream + Thermal stream -> timestamp align -> paired images -> detector`

这比当前“单路拼接视频拆帧”复杂一层。

## 11. 你现在可以怎么跟前后端说

可以直接这样说：

`模型端现在已经支持主动读取视频流，并可以用 HTTP-FLV 作为 source_url 输入。前提是这路视频每帧里已经拼好了 RGB 和 Thermal。模型端会自动按布局拆成图像对，再复用现有检测模型，并把逐帧结果回调后端。`
