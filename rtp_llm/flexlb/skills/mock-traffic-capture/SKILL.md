---
name: mock-traffic-capture
description: 用 RTP Frontend 和 Mock ViT 采集匿名请求，分段归档并构建全复用的 MCKINDEX 数据集。用于 Frontend 叠加镜像出包、Whale 采集部署、触发、监控、下载、恢复和离线验收。
---

# Frontend 流量采集

交付 `replay.mckidx`、`report.json` 和校验过的 SHA-256。新数据存入 `<shared-root>/data/traffic/`，Case 用 `--shared-root` 读取。

- [架构与字段](references/architecture.md) 定义入口、归档、跨段前缀复用和输出契约。
- [出包与镜像适配](references/build.md) 从现有 RTP 镜像构建 Frontend 适配层，从当前源码构建 Mock ViT。
- [部署与角色配置](references/deploy.md) 固定目标版本、配置签名与发现、确认入口覆盖。
- [触发、取数与恢复](references/operations.md) 从同一台协调 Pod 跟进一个批次。
- [生产来源标记](references/production-source.md) 定义简版统计、同窗取证和历史回填。
- [离线验收](references/validation.md) 用同一批封存分片测入库时间、工作盘峰值与理论复用率。

程序在 [traffic-capture](../../mock-framework/scripts/traffic-capture/)。按目标角色的实际镜像、架构和 Python 解释器选择构建参数。Whale 发布使用 `whale-biz-operation`，Pod 命令和传输使用 `asish`。

只在采集入口和目标 Pod 已核对后开启窗口。验收以实际提交段、每个入口的请求覆盖及成品校验为准。部分数据交付必须标出实际覆盖、缺段及缺失原因。
