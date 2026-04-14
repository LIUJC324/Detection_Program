# 2026-04-12 Dual Stream Latency And Blocking Archive

## 背景

本次归档记录的是 `2026-04-11` 晚到 `2026-04-12` 凌晨这轮双路 `RGB + IR HTTP-FLV` 联调中暴露出的启动延迟、阻塞、积压、误判和上游流质量问题，以及已经落地的修复方式。

当前联调输入形态是：

1. 后端调用模型端 `POST /v1/inference/stream/start`
2. 模型端接收：
   - `rgbPullUrl`
   - `irPullUrl`
   - `sampleFps`
   - `pairLayout`
   - `rgbPosition`
3. 模型端内部使用 `ffmpeg` 同时拉两路，再在内部拼成一张大帧后送入检测流程

本轮联调中，前后端侧感受到的主要症状是：

1. 会话进入 `RUNNING` 很慢
2. 前端迟迟看不到结果
3. 有时半分钟内只零星收到几个框
4. 有时 stop 之后链路还在继续卡着
5. 有时出现 `Invalid NAL unit size`、`Packet mismatch` 之类的流媒体错误

## 问题归类

### 1. 启动慢

实际日志显示，最近几条真实双流会话经常是：

1. `session_created`
2. 过很久才 `session_probe_ok`
3. 再过一段时间才 `session_running`

典型现象：

1. `start -> probe_ok` 经常要二十几秒到四十几秒
2. `start -> running` 可能要四十秒以上

这不是模型前向本身慢，而是启动阶段卡在：

1. 双路 `ffprobe`
2. 上游流首个可解码关键帧等待
3. `ffmpeg` 双流 filter graph 成帧

### 2. 旧帧积压导致“看起来越来越慢”

旧实现是：

1. 主线程阻塞读完整一帧 rawvideo
2. 做一次推理
3. 再继续读下一帧

这意味着：

1. 如果推理慢一点，上游帧就会堆积
2. 如果流抖动或回调有波动，延迟会继续向后累加
3. 用户看到的不是“当前最新帧”，而是“已经排队的旧帧”

### 3. STARTING 阶段 stop 仍然继续跑

旧逻辑在 `STARTING` 阶段如果收到 stop：

1. 只是设置 `stop_event`
2. 但阻塞中的 `ffprobe` 不会立刻退出
3. 后面的 retry / decoder 启动仍可能继续发生

所以会出现：

1. 前后端已经 stop
2. 模型端还在 probe
3. 甚至后面还会拉起新的 `ffmpeg`

### 4. 双流 filter graph 之前存在实现错误

本轮联调中曾出现：

1. `Cannot find a matching stream for unlabeled input pad 1 on filter Parsed_scale2ref_*`

这不是前后端问题，也不是流本身问题，而是模型端 `scale2ref` 输入 pad 连接写错了。

### 5. 上游 bind 流存在真实质量问题

本轮也多次出现：

1. `Invalid NAL unit size`
2. `missing picture in access unit`
3. `Packet mismatch`

这说明并不只是模型端工程实现慢，上游 `bind_rgb` / `bind_ir` 流本身也存在：

1. 封装不完整
2. 新订阅者拿到关键帧慢
3. H.264 NAL 包不稳定

## 已落地的修复

### 1. 修正双流 `scale2ref + hstack/vstack` filter graph

已在：

- [service/streaming/session_manager.py](/home/liujuncheng/rgbt_uav_detection/service/streaming/session_manager.py)

修正双流 `filter_complex` 构造，避免再出现 `scale2ref` 输入 pad 不匹配错误。

### 2. 切换在线服务为低延迟 ffmpeg 参数

已在：

- [configs/deploy_stable.yaml](/home/liujuncheng/rgbt_uav_detection/configs/deploy_stable.yaml)
- [configs/deploy_integration_stable.yaml](/home/liujuncheng/rgbt_uav_detection/configs/deploy_integration_stable.yaml)

启用或收紧：

1. `ffmpeg_low_latency = true`
2. `ffmpeg_startup_analyzeduration = 1000000`
3. `ffmpeg_startup_probesize = 1048576`

目的是降低直播启动等待时间和内部缓冲深度。

### 3. 双路 probe 改为并行

已在：

- [service/streaming/session_manager.py](/home/liujuncheng/rgbt_uav_detection/service/streaming/session_manager.py)

新增 `_probe_dual_video_size()`，把 RGB / IR 两路 `ffprobe` 从串行改为并行，减少双路会话启动等待。

### 4. 启动阶段 stop 可中断

已在：

- [service/streaming/session_manager.py](/home/liujuncheng/rgbt_uav_detection/service/streaming/session_manager.py)

让 `ffprobe` 支持感知 `stop_event`，在 stop 发生时尽量及时退出，不再让会话停了还继续 probe / retry。

### 5. 主处理链改为“最新帧优先”

已在：

- [service/streaming/session_manager.py](/home/liujuncheng/rgbt_uav_detection/service/streaming/session_manager.py)

核心改动：

1. 新增读帧线程持续读取 rawvideo
2. 只保留最新一帧到共享缓冲
3. 如果旧帧还没消费就被新帧覆盖，记入 `frames_dropped`

这意味着：

1. 系统更偏向“当前最新结果”
2. 不再因为旧帧排队而不断累加显示延迟
3. 可用性优先于逐帧不丢帧

### 6. 当前联调固定尺寸默认内置到模型端

为了避免立刻改前后端接口，当前模型端已经默认把双流会话尺寸固定为：

1. `1700 x 720`

配置位置：

- [configs/deploy_stable.yaml](/home/liujuncheng/rgbt_uav_detection/configs/deploy_stable.yaml)
- [configs/deploy_integration_stable.yaml](/home/liujuncheng/rgbt_uav_detection/configs/deploy_integration_stable.yaml)

字段：

1. `default_dual_stream_frame_width = 1700`
2. `default_dual_stream_frame_height = 720`

模型端启动逻辑会在：

- [service/streaming/session_manager.py](/home/liujuncheng/rgbt_uav_detection/service/streaming/session_manager.py)

中优先吃这组默认值，从而跳过 `ffprobe`。

### 7. 保留可选接口扩展，但不是当前联调必需

另外也已经加了可选字段：

1. `frameWidth`
2. `frameHeight`

位置：

- [service/core/schemas.py](/home/liujuncheng/rgbt_uav_detection/service/core/schemas.py)
- [service/api/app.py](/home/liujuncheng/rgbt_uav_detection/service/api/app.py)

但当前联调阶段不要求前后端立刻改接口，因为模型端已经默认按固定 `1700x720` 工作。

## 当前建议口径

### 1. 给后端的口径

当前联调先保持原有接口不动：

1. 继续传 `rgbPullUrl`
2. 继续传 `irPullUrl`
3. 继续传 `sampleFps`
4. 继续传 `pairLayout`
5. 继续传 `rgbPosition`

本轮额外建议：

1. 默认把 `sampleFps` 调到 `2`
2. 不要在 `STARTING` 阶段过早 stop
3. 当前双流尺寸先按固定 `1700x720` 处理

### 2. 给前端的口径

前端不要用“本次给了几个 box”判断快慢。

原因：

1. `box` 数量是这一帧的目标数量
2. 不是速度指标
3. 真正速度要看：
   - `start -> running` 时延
   - `framesProcessed`
   - 回调次数

前端仍应区分：

1. `STARTING`
2. `RUNNING`
3. `STOPPED / FAILED / COMPLETED`

## 当前剩余隐患

### 1. 上游 bind 流质量问题还没有根治

模型端已经尽量降低工程阻塞，但下面这些问题仍然可能继续出现：

1. `Invalid NAL unit size`
2. `Packet mismatch`
3. 新订阅者长时间拿不到关键帧

这类问题根因更偏向：

1. SRS / bind 转发链路
2. 编码器 GOP / SPS/PPS 策略
3. 上游流本身质量

### 2. 固定尺寸方案依赖当前联调流长期不变

当前默认固定：

1. `1700x720`

如果后续联调尺寸改变，这个默认值也必须同步更新，否则会出现：

1. `frame_bytes` 计算错误
2. rawvideo 读帧错位
3. 画面切分错误

### 3. “最新帧优先”意味着允许丢帧

当前策略是为了降低展示延迟而主动允许：

1. 旧帧被覆盖
2. 结果更接近当前时刻

这对演示链路更合理，但并不适合拿来做严格逐帧统计。

### 4. 当前还缺一次“真实线上双流重新回归”

本轮已经完成：

1. 本地语法检查
2. 本地单路 dry-run
3. 本地双路 dry-run
4. 带固定尺寸默认值的本地验证

但在本次归档写入时，还缺少一轮基于真实 `bind_rgb` / `bind_ir` 的完整回归对照：

1. `start -> probe_ok` 是否明显缩短
2. `start -> running` 是否明显缩短
3. `framesProcessed` 是否更接近目标 `sampleFps`
4. 前端观感是否改善

## 验证结论

当前可以明确确认：

1. 模型端关于双流 filter graph、启动阻塞、stop 可中断、最新帧优先这些工程问题已经做了第一轮修复
2. 模型服务已重启并在线
3. 本地验证未回归
4. 当前联调阶段前后端可以先不改接口

但同时也要明确：

1. 真正决定“是否还能继续慢”的，仍然有一部分取决于上游双流质量
2. 本轮修复的是模型端链路阻塞，不是把流媒体源质量问题一并消灭

## 后续建议

下一步最值得优先做的是：

1. 用真实双流再跑一轮回归
2. 对比：
   - 首次回调时间
   - `framesProcessed`
   - 前端实际叠框观感
3. 如果仍然慢，继续追：
   - 上游 GOP
   - 新订阅关键帧到达时间
   - bind 流封装质量

如果这一步完成后仍不稳定，再考虑第二轮工程改造：

1. 进一步缩短 probe 超时
2. 对真实双流加入更激进的 decoder startup 超时
3. 针对 bind 流单独增加更严格的 retry / kill 策略
