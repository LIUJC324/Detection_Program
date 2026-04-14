# Docs Index

`docs/` is the repository knowledge base.  
Long-lived knowledge stays in its topic folder.  
Each non-trivial change set also gets its own folder under `docs/changes/`.

## Start Here

1. [training/模型端问题排查、框架修复与续训方案_20260409.md](/home/liujuncheng/rgbt_uav_detection/docs/training/模型端问题排查、框架修复与续训方案_20260409.md)
2. [training/项目需求确认与借鉴优化方案_20260409.md](/home/liujuncheng/rgbt_uav_detection/docs/training/项目需求确认与借鉴优化方案_20260409.md)
3. [changes/20260411_optimization_roadmap/README.md](/home/liujuncheng/rgbt_uav_detection/docs/changes/20260411_optimization_roadmap/README.md)
4. [changes/20260411_dataset_audit/README.md](/home/liujuncheng/rgbt_uav_detection/docs/changes/20260411_dataset_audit/README.md)
5. [changes/20260411_reliability_refine/README.md](/home/liujuncheng/rgbt_uav_detection/docs/changes/20260411_reliability_refine/README.md)
6. [integration/发给前后端同学的固定公网联调清单_20260407.md](/home/liujuncheng/rgbt_uav_detection/docs/integration/发给前后端同学的固定公网联调清单_20260407.md)
7. [ops/稳定部署方案_20260407.md](/home/liujuncheng/rgbt_uav_detection/docs/ops/稳定部署方案_20260407.md)

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
- `reference/`
  - 长文档原理说明
  - 代码与方法映射
- `changes/`
  - 每一轮非平凡改动的归档目录
  - 目录名统一为 `YYYYMMDD_topic`
  - 每个目录至少放一份 `README.md`

## Current Reading Order

If you are joining the project now, read in this order:

1. 模型端跟踪与当前状态
2. 项目需求确认与借鉴方向
3. 下次直接可执行的优化路线
4. 数据审计结论
5. 最近一次改动归档
6. 固定公网联调清单
7. 部署与服务日志说明

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
