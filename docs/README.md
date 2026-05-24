# Docs Index

`docs/` is the repository knowledge base.  
Long-lived knowledge stays in its topic folder.  
Each non-trivial change set also gets its own folder under `docs/changes/`.
当前仓库主线已经进入模型端完成态，训练、推理、服务、联调和部署文档都按这条主线维护。

## Start Here

1. [reference/ML_基本原理与代码对应说明.md](/home/liujuncheng/rgbt_uav_detection/docs/reference/ML_基本原理与代码对应说明.md)
2. [architecture/模型端架构图讲解_20260416.md](/home/liujuncheng/rgbt_uav_detection/docs/architecture/模型端架构图讲解_20260416.md)
3. [integration/实时视频演示与前后端衔接说明.md](/home/liujuncheng/rgbt_uav_detection/docs/integration/实时视频演示与前后端衔接说明.md)
4. [integration/发给前后端同学的固定公网联调清单_20260407.md](/home/liujuncheng/rgbt_uav_detection/docs/integration/发给前后端同学的固定公网联调清单_20260407.md)
5. [ops/稳定部署方案_20260407.md](/home/liujuncheng/rgbt_uav_detection/docs/ops/稳定部署方案_20260407.md)
6. [training/模型端问题排查、框架修复与续训方案_20260409.md](/home/liujuncheng/rgbt_uav_detection/docs/training/模型端问题排查、框架修复与续训方案_20260409.md)
7. [changes/README.md](/home/liujuncheng/rgbt_uav_detection/docs/changes/README.md)

## By Topic

- `training/`
  - 训练状态
  - 框架修复
  - 续训留档
  - 后续优化方案
- `integration/`
  - 前后端联调说明
  - 接口契约
  - 会话与回调口径
  - 演示链路说明
- `ops/`
  - 部署方案
  - 服务日志
  - 流媒体 / FFmpeg 说明
  - 运维命令留档
- `testing/`
  - 测试记录
  - 演示验证
  - 测试数据建议
- `architecture/`
  - 架构图
  - 后端架构图
  - 分层图与原理图
  - PlantUML 架构源码
- `reference/`
  - 长文档原理说明
  - 代码与方法映射
  - 赛题创新点与关键模块说明
- `changes/`
  - 每一轮非平凡改动的归档目录
  - 目录名统一为 `YYYYMMDD_topic`
  - 每个目录至少放一份 `README.md`

## Current Reading Order

If you are joining the project now, read in this order:

1. ML 主说明文档
2. 模型端架构说明
3. 实时视频演示与前后端衔接说明
4. 固定公网联调清单
5. 稳定部署方案
6. 模型端历史跟踪与问题修复记录
7. 变更归档入口

## Placement Rules

1. 新文档必须放到对应主题目录里，不再放到仓库根目录或独立散落文件夹。
2. 变更类文档统一放到 `docs/changes/YYYYMMDD_topic/README.md`。
3. 接口协议、联调口径、会话说明统一放到 `docs/integration/`。
4. 架构图和配套说明统一放到 `docs/architecture/`。
5. 模型训练事实、问题排查、续训建议优先更新 `docs/training/` 下的主跟踪文档。

## Canonical Paths

- 接口契约主文档：
  [integration/interface.md](/home/liujuncheng/rgbt_uav_detection/docs/integration/interface.md)
- 架构图资源目录：
  [architecture](/home/liujuncheng/rgbt_uav_detection/docs/architecture)
- 变更归档入口：
  [changes/README.md](/home/liujuncheng/rgbt_uav_detection/docs/changes/README.md)
