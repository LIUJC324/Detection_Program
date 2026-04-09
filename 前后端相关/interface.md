# 接口文档

代码基线：`main` 工作区（`aerialeye` 多模块后端）
文档目标：给前端、模型端、联调与运维提供可直接落地的接口契约与策略说明。

## 1. 全局约定

### 1.1 API 前缀与鉴权

- HTTP API 前缀：`/api/v1`
- Access Token Header：`Authorization: Bearer <accessToken>`
- Sa-Token 关键配置（默认）：

1. `timeout=1800s`
2. `active-timeout=1800s`
3. `is-concurrent=false`（单用户单会话）
4. `is-share=false`

### 1.2 免登录白名单

以下路径不经过登录拦截：

1. `/api/v1/auth/login`
2. `/api/v1/auth/refresh`
3. `/api/v1/auth/register`
4. `/api/v1/auth/email-code/send`
5. `/api/v1/auth/password/reset-code/send`
6. `/api/v1/auth/password/reset`
7. `/api/v1/stream/srs/callback`
8. `/api/v1/detect/model/result`
9. `/ws/**`
10. `/actuator/health`、`/actuator/info`、`/actuator/prometheus` 等观测接口

说明：`file`、`device`、`detect/video/sessions`、`stream/sessions`、`stats` 都要求登录（即便方法上没有显式 `@SaCheckLogin`）。

### 1.3 统一响应结构

所有接口都返回 `Result<T>`：

```json
{
  "code": 200,
  "message": "success",
  "data": {},
  "timestamp": 1711800000000
}
```

字段语义：

1. `code`：业务码（200 成功；其他失败）
2. `message`：错误或提示信息
3. `data`：业务数据，失败时通常为 `null`
4. `timestamp`：服务端毫秒时间戳

分页统一使用 `PageResult<T>`：

```json
{
  "total": 100,
  "pages": 5,
  "current": 1,
  "size": 20,
  "records": []
}
```

### 1.4 错误处理语义（非常重要）

1. 系统通过 `Result.code` 表达业务成功/失败。
2. 许多失败场景返回体里的 `code=4xx/5xx`，但 HTTP 状态本身可能仍是 200。
3. 前端必须以 `body.code` 作为主判断条件，不要只看 HTTP status。

### 1.5 通用时间与序列化

1. `LocalDate`：`yyyy-MM-dd`
2. `LocalDateTime`：ISO-8601（如 `2026-04-01T09:30:00`）
3. `timestamp`（Result / WebSocket 事件）：Unix 毫秒

## 2. 场景：认证与会话

路径前缀：`/api/v1/auth`

### 2.1 注册验证码发送

接口：`POST /email-code/send`
鉴权：否

请求体：

```json
{
  "email": "user@example.com"
}
```

参数约束：

1. `email` 必填，且必须为邮箱格式

成功响应：`Result<Void>`

策略与说明：

1. 发送频控：同邮箱冷却默认 60 秒。
2. IP 限流：默认每 IP 每小时最多 20 次。
3. 验证码默认有效期 300 秒。
4. 验证码只存哈希，不存明文。

前端建议：

1. 倒计时按钮基于 60 秒冷却。
2. 对 `1102`（发送过频）单独提示“请稍后再试”。

### 2.2 注册

接口：`POST /register`
鉴权：否

请求体：

```json
{
  "username": "demo_user",
  "email": "user@example.com",
  "password": "Abcd1234!",
  "emailCode": "123456"
}
```

参数约束：

1. `username`：`^[a-zA-Z0-9_]{3,32}$`
2. `email`：邮箱格式
3. `password`：8-64
4. `emailCode`：6位数字
5. Service 额外密码策略：必须包含大写、小写、数字、特殊字符

成功响应：`Result<Void>`

常见失败码：

1. `1103` 邮箱已注册
2. `1104` 用户名已存在
3. `1100` 验证码错误
4. `1101` 验证码过期
5. `1109` 密码强度不满足
6. `1112` 验证码尝试次数过多锁定

策略与说明：

1. 注册成功后自动赋予默认角色（默认 `USER`）。
2. 邮箱统一会被转小写后处理。

### 2.3 密码重置验证码发送

接口：`POST /password/reset-code/send`
鉴权：否

请求体：

```json
{
  "email": "user@example.com"
}
```

成功响应：`Result<Void>`

策略与说明：

1. 为防枚举：邮箱不存在时也返回成功，不暴露用户是否存在。
2. 频控与验证码策略同注册验证码。

### 2.4 密码重置

接口：`POST /password/reset`
鉴权：否

请求体：

```json
{
  "email": "user@example.com",
  "emailCode": "123456",
  "newPassword": "Abcd1234!"
}
```

成功响应：`Result<Void>`

策略与说明：

1. 重置成功后会撤销该用户 refresh token 并强制下线。
2. 密码策略与注册一致。

### 2.5 登录

接口：`POST /login`
鉴权：否

请求体：

```json
{
  "account": "user@example.com",
  "password": "Abcd1234!",
  "rememberMe": false
}
```

说明：

1. `account` 支持用户名或邮箱。
2. `rememberMe` 默认为 `false`。

成功 `data`（`TokenResponse`）：

```json
{
  "accessToken": "xxx",
  "accessExpiresIn": 1800,
  "refreshToken": "yyy",
  "refreshExpiresIn": 86400,
  "tokenType": "Bearer"
}
```

常见失败码：

1. `1001` 用户名或密码错误
2. `1004` 用户被禁用
3. `1107` 账号被锁定
4. `1111` IP 被锁定

策略与说明：

1. 登录成功会先踢掉同用户旧会话，再签发新 token（单会话）。
2. `rememberMe=true` 时 refresh token TTL 默认 259200 秒；否则 86400 秒。
3. 登录失败会累计账号/IP 失败次数，达到阈值触发锁定。

前端建议：

1. 收到 `1107/1111` 时提示剩余等待时间（文案可固定“稍后重试”）。
2. 登录后立即拉取 `/me` 缓存权限。

### 2.6 刷新 Token

接口：`POST /refresh`
鉴权：否

请求体：

```json
{
  "refreshToken": "yyy"
}
```

成功响应：同登录 `TokenResponse`

常见失败码：

1. `1106` refresh token 过期
2. `1105` refresh token 无效
3. `1000` 用户不存在
4. `1004` 用户禁用

策略与说明：

1. 刷新采用轮换：旧 refresh token 会失效。
2. 当前 access token 会被拉黑（黑名单）。

### 2.7 登出

接口：`POST /logout`
鉴权：是

成功响应：`Result<Void>`

策略与说明：

1. 当前 access token 拉黑。
2. 用户 refresh token 撤销。
3. 用户会话退出。

### 2.8 获取当前登录用户

接口：`GET /me`
鉴权：是

成功 `data`（`MeResponse`）：

```json
{
  "id": 1,
  "username": "admin",
  "email": "admin@example.com",
  "roles": ["ADMIN"],
  "permissions": ["auth:role:manage"]
}
```

### 2.9 申请 WebSocket Ticket

接口：`POST /ws-ticket`
鉴权：是

成功 `data`（`WsTicketResponse`）：

```json
{
  "ticket": "uuid_like_string",
  "expiresIn": 60
}
```

策略与说明：

1. ticket 一次性：WebSocket 握手成功后即删除（`getAndDelete`）。
2. TTL 最多 300 秒，默认 60 秒。

### 2.10 管理员重置用户角色

接口：`PUT /admin/users/{userId}/roles`
鉴权：需 `auth:role:manage`

请求体：

```json
{
  "roleCode": "ADMIN"
}
```

成功响应：`Result<Void>`

策略与说明：

1. 重置角色后会清空该用户旧角色关系。
2. 目标用户会被强制下线并撤销 refresh token。

### 2.11 管理员查询用户 RBAC

接口：`GET /admin/users/{userId}/rbac`
鉴权：需 `auth:permission:manage`

成功 `data`（`UserRbacResponse`）：

```json
{
  "userId": 2,
  "username": "demo",
  "roles": ["USER"],
  "permissions": []
}
```

## 3. 场景：设备接入、绑定与协作

路径前缀：`/api/v1/device`
鉴权：全部需要登录

角色规则：

1. `OWNER`：可读可写，可分享
2. `EDITOR`：可读可写，不可分享
3. `VIEWER`：只读

### 3.0 推荐新增流程：临时地址绑定后再入库

说明：相比直接 `POST /api/v1/device` 预创建设备，推荐前端走“先绑定成功，再落库”的流程。

#### 3.0.1 开始绑定（发放双路临时推流地址）

接口：`POST /api/v1/device/bindings/start`

请求体：

```json
{
  "deviceCode": "UAV_001",
  "name": "一号机",
  "deviceType": "UAV",
  "vendor": "AerialEye",
  "model": "AE-X1",
  "remark": "测试设备"
}
```

成功 `data`（`DeviceBindingStartResponse`）：

```json
{
  "bindId": "bind_xxx",
  "status": "PENDING",
  "expiresIn": 600,
  "streamApp": "bind",
  "rgbStreamKey": "bind_rgb_xxx",
  "irStreamKey": "bind_ir_xxx",
  "rgbPushUrl": "rtmp://127.0.0.1:1935/bind/bind_rgb_xxx",
  "irPushUrl": "rtmp://127.0.0.1:1935/bind/bind_ir_xxx"
}
```

策略与说明：

1. 临时绑定默认 10 分钟过期（可配置）。
2. 仅当双路流都被 SRS 回调确认后，后端才会创建正式 `device + channels + member` 记录。
3. 绑定成功后自动创建两路通道：`RGB_MAIN`、`IR_MAIN`。

#### 3.0.2 查询绑定状态

接口：`GET /api/v1/device/bindings/{bindId}`

成功 `data`（`DeviceBindingStatusResponse`）：

```json
{
  "bindId": "bind_xxx",
  "status": "BINDING",
  "expiresIn": 420,
  "deviceCode": "UAV_001",
  "deviceName": "一号机",
  "deviceId": null,
  "rgbBound": true,
  "irBound": false,
  "message": null
}
```

状态枚举：

1. `PENDING`：未收到任一路回调
2. `BINDING`：已收到至少一路回调
3. `COMPLETED`：双路确认并已入库（`deviceId` 可用）
4. `FAILED`：入库失败
5. `EXPIRED`：超时过期

### 3.1 兼容接口：直接创建设备

接口：`POST /api/v1/device`

请求体：

```json
{
  "deviceCode": "UAV_001",
  "name": "一号机",
  "deviceType": "UAV",
  "vendor": "AerialEye",
  "model": "AE-X1",
  "remark": "测试设备"
}
```

参数约束：

1. `deviceCode`：`^[A-Za-z0-9_-]{3,64}$`
2. `name`：2-128
3. `deviceType`：`^[A-Za-z0-9_-]{2,32}$`（默认 UAV）

成功响应：`Result<Long>`（`data` 为 `deviceId`）

常见失败码：

1. `2100` 设备编码已存在

策略与说明：

1. 创建后自动把当前用户写入 `device_member` 为 `OWNER`。

### 3.2 更新设备

接口：`PUT /api/v1/device/{deviceId}`

请求体（按需字段）：

```json
{
  "name": "一号机-改",
  "status": 1,
  "remark": "已换电池"
}
```

字段约束：

1. `status` 仅 `0/1`
2. 仅 `OWNER/EDITOR` 可操作

### 3.3 分享设备给其他账号

接口：`POST /api/v1/device/{deviceId}/share`

请求体：

```json
{
  "account": "other_user_or_email"
}
```

成功 `data`（`DeviceShareResponse`）：

```json
{
  "deviceId": 1,
  "userId": 9,
  "account": "other_user_or_email",
  "roleCode": "VIEWER",
  "created": true
}
```

策略与说明：

1. 仅 OWNER 可分享。
2. 分享目标会被设置为 `VIEWER`（新建或覆盖非 OWNER 角色）。
3. 不允许分享给自己。

常见失败码：

1. `403` 非 OWNER
2. `1000` 目标用户不存在
3. `1004` 目标用户禁用

### 3.4 设备详情

接口：`GET /api/v1/device/{deviceId}`

成功 `data`（`DeviceDetailResponse`）核心字段：

1. 设备信息：`id/deviceCode/name/deviceType/vendor/model/status/onlineStatus/lastSeenAt`
2. 权限信息：`myRole`
3. 通道列表：`channels[]`

### 3.5 我的设备分页

接口：`GET /api/v1/device/my?current=1&size=20&keyword=uav&status=1`

参数：

1. `current` 最小 1，默认 1
2. `size` 范围 1-100，默认 20
3. `status` 可选，`0/1`

成功：`Result<PageResult<DeviceListItem>>`

### 3.6 查询设备通道列表

接口：`GET /api/v1/device/{deviceId}/channels`

成功：`Result<List<DeviceChannelResponse>>`

### 3.7 新建通道

接口：`POST /api/v1/device/{deviceId}/channels`

请求体：

```json
{
  "channelCode": "RGB_MAIN",
  "channelType": "RGB",
  "streamApp": "live",
  "streamKey": "optional",
  "ingestProtocol": "RTMP",
  "status": 1,
  "sortOrder": 0
}
```

参数约束：

1. `channelCode`：`^[A-Za-z0-9_-]{2,32}$`
2. `channelType`：`RGB|IR`
3. `ingestProtocol`：`RTMP|RTSP|GB28181`

策略与说明：

1. 未传 `streamApp` 默认 `live`。
2. 未传 `streamKey` 自动生成 UUID 紧凑串。
3. `streamApp + streamKey` 冲突会报 `2103`。

### 3.8 更新通道

接口：`PUT /api/v1/device/channels/{channelId}`

请求体（按需字段）：

```json
{
  "streamApp": "live",
  "streamKey": "new_key",
  "ingestProtocol": "RTSP",
  "status": 1,
  "sortOrder": 2
}
```

策略与说明：

1. 仅 OWNER/EDITOR 可更新。
2. 更新后若路由冲突，返回 `2103`。

### 3.9 轮换 Stream Key

接口：`POST /api/v1/device/channels/{channelId}/stream-key/rotate`

请求体：

```json
{
  "streamApp": "live"
}
```

成功 `data`：

```json
{
  "channelId": 11,
  "streamApp": "live",
  "streamKey": "new_generated_key"
}
```

策略与说明：

1. 每次都会生成新 key，旧 key 立即失效。
2. 建议前端在执行前二次确认。

### 3.10 按流路由反查设备

接口：`GET /api/v1/device/stream/resolve?app=live&streamKey=xxx`

成功 `data`（`DeviceStreamResolveResponse`）包含：

1. 设备维度：`deviceId/deviceCode/deviceName/deviceStatus/onlineStatus`
2. 通道维度：`channelId/channelCode/channelType/streamApp/streamKey/channelStatus`

常见失败码：

1. `400` 参数缺失
2. `4000` 路由不存在
3. `2105` 设备禁用
4. `2104` 通道禁用

## 4. 场景：文件上传与回放

路径前缀：`/api/v1/file/video`
鉴权：需要登录

### 4.1 视频上传

接口：`POST /upload`（`multipart/form-data`）

表单字段：

1. `file`（必填）

校验：

1. 扩展名：`mp4/mov/avi/mkv/flv/webm`
2. 大小：最大 1024MB

成功 `data`（`VideoUploadResponse`）：

```json
{
  "bucket": "aerialeye-video",
  "objectKey": "video/origin/2026-04-01/uuid.mp4",
  "originalFilename": "demo.mp4",
  "contentType": "video/mp4",
  "size": 12345678,
  "playbackUrl": "https://..."
}
```

策略与说明：

1. bucket 不存在且允许自动建桶时会自动创建。
2. 上传失败返回 `5001`。

### 4.2 生成回放地址

接口：`GET /presign?objectKey=...&bucket=...`

说明：

1. `bucket` 可不传，默认视频桶。
2. 返回 `playbackUrl`（MinIO 预签名 GET）。

成功 `data`：

```json
{
  "bucket": "aerialeye-video",
  "objectKey": "video/origin/2026-04-01/uuid.mp4",
  "playbackUrl": "https://..."
}
```

### 4.3 检查对象是否存在

接口：`GET /exists?objectKey=...&bucket=...`

成功 `data`：

```json
{
  "bucket": "aerialeye-video",
  "objectKey": "video/origin/2026-04-01/uuid.mp4",
  "exists": true
}
```

## 5. 场景：视频推理会话

路径前缀：`/api/v1/detect/video/sessions`
鉴权：需要登录

状态枚举：`STARTING | RUNNING | STOPPING | STOPPED`

### 5.1 启动视频推理

接口：`POST /start`

请求体：

```json
{
  "bucket": "aerialeye-video",
  "objectKey": "video/origin/2026-04-01/uuid.mp4"
}
```

参数约束：

1. `objectKey` 必填
2. `bucket` 可选，不传默认视频桶

成功 `data`（`VideoSessionResponse`）：

```json
{
  "sessionId": "sid_xxx",
  "bucket": "aerialeye-video",
  "objectKey": "video/origin/2026-04-01/uuid.mp4",
  "sourceUrl": "https://...",
  "status": "RUNNING",
  "startedAt": "2026-04-01T10:00:00"
}
```

策略与说明：

1. 启动前会校验对象存在，不存在返回 `5000`。
2. 启动失败返回 `3003`（模型错误）。
3. 会向模型端下发 `startVideoSession` 控制命令（见第 10 章）。

### 5.2 停止视频推理

接口：`POST /{sessionId}/stop`

成功 `data`：

```json
{
  "sessionId": "sid_xxx",
  "stopped": true
}
```

常见失败码：

1. `3000` session 不存在
2. `3001` session 状态不允许（非 RUNNING）
3. `3003` 模型停止失败

### 5.3 查询视频推理会话

接口：`GET /{sessionId}`

成功：`Result<VideoSessionResponse>`

## 6. 场景：实时流推理会话

路径前缀：`/api/v1/stream/sessions`
鉴权：需要登录

### 6.1 启动实时流推理

接口：`POST /start`

请求体：

```json
{
  "app": "live",
  "streamKey": "abc123",
  "sourceUrl": "optional"
}
```

参数约束：

1. `app` 必填
2. `streamKey` 必填
3. `sourceUrl` 可选

成功 `data`（`StreamSessionResponse`）：

```json
{
  "sessionId": "sid_xxx",
  "app": "live",
  "streamKey": "abc123",
  "sourceUrl": "rtmp://127.0.0.1:1935/live/abc123",
  "status": "RUNNING",
  "startedAt": "2026-04-01T10:10:00"
}
```

策略与说明：

1. 启动前会调用设备路由反查验证该流是否有效绑定。
2. 若没传 `sourceUrl`，后端按 `aerialeye.stream.forward` 自动拼接。
3. 同一路由（`app+streamKey`）只允许一个活动会话，带分布式锁保护。
4. 若路由已有 RUNNING 会话，返回 `3001`。

### 6.2 停止实时流推理

接口：`POST /{sessionId}/stop`

成功 `data`：

```json
{
  "sessionId": "sid_xxx",
  "stopped": true
}
```

失败语义同视频会话停止。

### 6.3 查询实时流会话

接口：`GET /{sessionId}`

成功：`Result<StreamSessionResponse>`

## 7. 场景：模型回调（模型端 -> 后端）

路径：`POST /api/v1/detect/model/result`
鉴权：白名单接口（不需要用户登录）
安全头：`X-Model-Token`（与配置 `aerialeye.inference.callback-token` 比对）

请求体（`InferenceResult`）：

```json
{
  "taskId": "task_xxx",
  "sessionId": "sid_xxx",
  "streamKey": "abc123",
  "annotationMode": "rectangle",
  "boxes": [
    {
      "tag": "person",
      "score": 0.98,
      "x1": 0.10,
      "y1": 0.20,
      "x2": 0.40,
      "y2": 0.50
    }
  ],
  "modelLatencyMs": 42,
  "error": ""
}
```

返回语义：

1. token 无效：`Result.fail(403, "invalid model callback token")`
2. `sessionId` 缺失：`Result.fail(400, "sessionId is required")`
3. session 非活动：`code=200` 且 `data.accepted=false`
4. 正常：`code=200` 且 `data.accepted=true`

成功示例：

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "accepted": true,
    "sessionId": "sid_xxx"
  },
  "timestamp": 1711800000000
}
```

策略与说明：

1. 回调被接受后会发布到实时推送通道，并异步进入统计落库链路。
2. `error` 非空会被作为模型错误事件计入指标。
3. 若配置中的 callback token 为空字符串，则默认放行（不建议生产这么做）。
4. 当前联调阶段 `annotationMode=rectangle`，`x1/y1/x2/y2` 分别表示矩形框左上角与右下角。
5. 字段仍沿用 `x1/y1/x2/y2`，便于前后端直接按矩形框渲染。

模型端建议：

1. 回调失败（网络/5xx）要做有限重试与指数退避。
2. `error` 建议使用 `CODE:message` 格式，便于后端拆出 `errorCode`。

## 8. 场景：SRS 回调分发（可选模式）

路径：`POST /api/v1/stream/srs/callback`
鉴权：白名单接口
用途：把帧级回调转成推理队列任务（默认配置 `dispatch-enabled=false`）

请求体（`SrsCallbackRequest`）：

```json
{
  "action": "on_publish",
  "app": "live",
  "stream": "abc123",
  "frameUrl": "http://srs/frame.jpg",
  "timestamp": 1711800000,
  "nonce": "random",
  "sign": "hmac_sha256_hex"
}
```

签名规则：

1. 参与字段固定顺序：`action/app/stream/timestamp/nonce`
2. canonical payload：每个字段 `trim` 后按换行拼接
3. 算法：`HMAC-SHA256(secret, payload)`，hex 小写
4. 时间窗：`abs(now - timestamp) <= maxSkewSeconds`（默认 60 秒）

返回语义：

1. 签名失败：`code=403`
2. 分发关闭：`queued=false` + reason
3. 会话非激活：`queued=false` + reason
4. frameUrl 为空：`queued=false` + reason
5. 队列满：`code=503`
6. 入队成功：返回 `taskId/sessionId/queued/queueSize`

策略与说明：

1. 推荐主路径是“显式 start/stop 会话 + 模型端拉流”，SRS 回调仅作兼容。
2. 当队列满时要监控 `aerialeye_inference_queue_rejected_total`。

## 9. 场景：统计分析

路径前缀：`/api/v1/stats`
鉴权：需要登录

范围规则（三接口一致）：

1. 默认最近 7 天（含当天）
2. 仅传 `startDate` 或 `endDate` 时自动补齐 7 天窗
3. 最长 90 天
4. 非 ADMIN 用户只看自己有权限的设备数据

`metric` 可选值：

1. `frame_count`
2. `detection_count`（默认）
3. `error_count`
4. `avg_latency_ms`

### 9.1 总览

接口：`GET /overview?startDate=2026-03-25&endDate=2026-03-31`

成功 `data`（`StatsOverviewResponse`）：

```json
{
  "startDate": "2026-03-25",
  "endDate": "2026-03-31",
  "loginSuccessCount": 10,
  "loginFailureCount": 2,
  "deviceCount": 3,
  "onlineDeviceCount": 2,
  "channelCount": 6,
  "frameCount": 1000,
  "detectionCount": 2300,
  "errorCount": 12,
  "avgLatencyMs": 45.67,
  "callbackErrorRate": 1.2
}
```

### 9.2 趋势

接口：`GET /trend?startDate=2026-03-25&endDate=2026-03-31&metric=detection_count`

成功 `data`（`StatsTrendResponse`）：

```json
{
  "metric": "detection_count",
  "startDate": "2026-03-25",
  "endDate": "2026-03-31",
  "points": [
    { "date": "2026-03-25", "value": 120.0 }
  ]
}
```

说明：区间内无数据的日期会补 0 点。

### 9.3 设备榜单

接口：`GET /top/devices?startDate=2026-03-25&endDate=2026-03-31&metric=avg_latency_ms&limit=10`

参数：

1. `limit`：1-50，默认 10

成功 `data`（`StatsTopDeviceResponse`）：

```json
{
  "metric": "avg_latency_ms",
  "startDate": "2026-03-25",
  "endDate": "2026-03-31",
  "items": [
    {
      "deviceId": 1,
      "deviceName": "一号机",
      "frameCount": 100,
      "detectionCount": 210,
      "errorCount": 1,
      "avgLatencyMs": 36.22,
      "metricValue": 36.22
    }
  ]
}
```

## 10. 场景：WebSocket 实时订阅

端点：`/ws/detect`（可配置）
认证方式：query 参数一次性票据 `ticket`（通过 `/api/v1/auth/ws-ticket` 获取）

### 10.1 握手

连接示例：

```text
ws://host:8080/ws/detect?ticket=<ticket>
```

握手策略：

1. 缺少 ticket -> 401 拒绝握手
2. ticket 无效或已使用 -> 401
3. ticket 成功后立即删除（不可重放）

### 10.2 客户端命令

订阅：

```json
{ "action": "subscribe", "sessionId": "sid_xxx" }
```

取消订阅：

```json
{ "action": "unsubscribe" }
```

心跳：

```json
{ "action": "ping" }
```

### 10.3 服务端消息

连接确认：

```json
{ "type": "connected", "sessionId": null, "timestamp": 1711800000000 }
```

订阅确认：

```json
{ "type": "subscribed", "sessionId": "sid_xxx", "timestamp": 1711800000000 }
```

检测结果推送：

```json
{
  "type": "DETECT_RESULT",
  "sessionId": "sid_xxx",
  "timestamp": 1711800000000,
  "data": {
    "taskId": "task_xxx",
    "sessionId": "sid_xxx",
    "streamKey": "abc123",
    "boxes": [],
    "modelLatencyMs": 40,
    "error": ""
  }
}
```

错误消息：

```json
{ "type": "error", "message": "forbidden" }
```

### 10.4 订阅权限策略

1. `sessionId` 必须是活动会话。
2. 如果是流会话，会按 `app+streamKey -> device` 做读权限校验。
3. 如果是视频文件会话（无设备路由），默认允许订阅。

前端建议：

1. 会话切换先 `unsubscribe` 再 `subscribe`。
2. 断线重连要重新申请 ticket（旧 ticket 已失效）。
3. 收到 `session is not active` 应停止自动重订阅。

## 11. 模型端对接契约（后端主动调用）

这部分是后端服务调用模型服务的 HTTP 契约，供模型端实现参考。

### 11.1 视频会话启动命令

目标地址：`aerialeye.inference.video-start-endpoint`

请求体：

```json
{
  "sessionId": "sid_xxx",
  "sourceUrl": "https://...",
  "bucket": "aerialeye-video",
  "objectKey": "video/origin/2026-04-01/uuid.mp4",
  "resultCallbackUrl": "http://host:8080/api/v1/detect/model/result",
  "callbackToken": "change-me"
}
```

### 11.2 实时流会话启动命令

目标地址：`aerialeye.inference.stream-start-endpoint`

请求体：

```json
{
  "sessionId": "sid_xxx",
  "app": "live",
  "streamKey": "abc123",
  "sourceUrl": "rtmp://127.0.0.1:1935/live/abc123",
  "resultCallbackUrl": "http://host:8080/api/v1/detect/model/result",
  "callbackToken": "change-me"
}
```

### 11.3 停止命令

目标地址：

1. `aerialeye.inference.video-stop-endpoint`
2. `aerialeye.inference.stream-stop-endpoint`

请求体：

```json
{
  "sessionId": "sid_xxx"
}
```

### 11.4 SRS 帧任务模式下的推理调用

当启用 `dispatch-enabled=true`，后端会向 `aerialeye.inference.endpoint` 发请求：

```json
{
  "taskId": "task_xxx",
  "sessionId": "sid_xxx",
  "app": "live",
  "streamKey": "abc123",
  "frameUrl": "http://srs/frame.jpg",
  "timestamp": 1711800000
}
```

期望模型返回：

```json
{
  "taskId": "task_xxx",
  "sessionId": "sid_xxx",
  "streamKey": "abc123",
  "boxes": [],
  "modelLatencyMs": 42,
  "error": ""
}
```

## 12. 错误码附录（常用）

### 12.1 通用 ResultCode

1. `200` success
2. `400` bad request
3. `401` unauthorized
4. `403` forbidden
5. `404` not found
6. `500` internal error
7. `1000` user not found
8. `1001` invalid credentials
9. `1002` token expired
10. `1003` token invalid
11. `1004` user disabled
12. `2000` device not found
13. `3000` task not found
14. `3001` task status error
15. `3003` model error
16. `4000` stream not found
17. `5000` file not found
18. `5001` file upload failed
19. `5002` file too large

### 12.2 认证扩展 AuthResultCode

1. `1100` 邮箱验证码错误
2. `1101` 邮箱验证码过期
3. `1102` 验证码发送过频
4. `1103` 邮箱已注册
5. `1104` 用户名已存在
6. `1105` refresh token 无效
7. `1106` refresh token 过期
8. `1107` 账号锁定
9. `1108` 邮件发送失败
10. `1109` 密码策略不符合
11. `1110` 角色不存在
12. `1111` IP 登录锁定
13. `1112` 验证码校验锁定

### 12.3 设备扩展 DeviceResultCode

1. `2100` 设备编码已存在
2. `2101` 通道不存在
3. `2102` 通道编码已存在
4. `2103` 流路由冲突
5. `2104` 通道禁用
6. `2105` 设备禁用

## 13. 联调建议（前端 + 模型端）

### 13.1 前端联调最小闭环

1. 登录拿 `accessToken`
2. 创建或查询设备与通道，拿到 `app+streamKey`
3. 启动流会话 `/api/v1/stream/sessions/start`
4. 申请 `/api/v1/auth/ws-ticket` 并连接 WebSocket
5. 发送 `subscribe(sessionId)`，观察 `DETECT_RESULT`
6. 结束时调用 `/stop` 并 `unsubscribe`

### 13.2 模型端联调最小闭环

1. 实现 start/stop 控制接口（视频+流）
2. 处理 `resultCallbackUrl + callbackToken`
3. 向 `/api/v1/detect/model/result` 回调标准 `InferenceResult`
4. 回调失败做有限重试

### 13.3 生产化建议

1. 所有 `change-me` 类密钥必须改为高强度随机值。
2. WebSocket 建议限制允许域名，不使用 `*`。
3. 给 `Result.code != 200` 建立统一告警埋点。
