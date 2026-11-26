# 生产来源标记

## 交付口径

使用 `<模型>-<时长>.mckidx` 及同名 `.stats.json`。简版统计的 `source` 只列采集窗内有实际请求证据的生产源应用，按 `service_id` 去重。没有证实任何生产源时填 `null`。以下是已核实单个来源时的字段示例。

```json
{ "source": [{ "application": "glm-5.3-mint", "service_id": "v-71b85312" }] }
```

来源名单表示同窗已证实出现的应用。完整采集报告记录采集窗、目标 Mock 部署、目标 Chat、Broadcaster 链路、Trace 证据和各来源计数的覆盖范围。未验证的 ID 及来源字段缺失数也留在完整报告中。MCKIDX 仅包含回放字段。

## 采集与取证

1. Frontend 收到 KServe 首帧时，解析请求体 `parameters["ds_header_attributes"]`。从 `x-dashscope-service-id` 取来源候选 ID，按采集窗、分段和 ID 聚合，每条请求计一次。字段缺省、空值或解析失败计入来源缺失数，另报具备首帧的请求总数。该口径独立于回放入库数与缺失率。
2. 从同一采集窗少量抽取 `x-dashscope-requestid` 查 Spectrum Trace。优先覆盖各候选 ID，确认 Trace 同时包含目标采集分支及该 ID 对应的原始生产分支。只将通过验证的生产应用写入简版 `source`。完整报告注明抽查范围、未核实 ID、来源缺失数，以及未取得完整来源计数的区间。
3. 同一 Trace 可能经过多条 Broadcaster 路径。记录实际链路和重叠证据，保留按 Frontend 实际到达请求计数的口径。重复交付或生产独立请求数需要独立证据。保留必要的 Trace 证据，避免持久化整份请求头。

Frontend 叠加层把聚合计数放进压缩分片封尾。协调器在完整 `report.json` 的 `validation.production_source` 输出整窗汇总，每个 `validation.segments[].production_source` 输出该段汇总。字段包括 `first_frames`、`by_service_id`、`missing_service_id` 和 `shards_with_source_stats`。应满足 `first_frames = sum(by_service_id.values()) + missing_service_id`。对照同级的 `frontend_shards` 判断分片覆盖。旧分片无来源计数，读取时视为来源统计不可用。KServe 首帧钩子之前关闭的 RPC 也不在首帧分母内。

采集前核对目标 Frontend 镜像、启动钩子配置及协调器均包含该功能。旧部署可继续做同窗 Trace 抽查，在完整报告注明来源计数不可用。来源 ID 聚合不发起 Trace 查询，简版 `source` 由同窗证据确认后填写。

## 证据与历史回填

- 2026-11-24 19:27–19:29（北京时间），Whale KVCM Mock 部署 `6ae8a3522f3ed52b2d2a697d` 的两个 Frontend Pod 共抽样 35 条。Chat 转发后两字段均可读。Trace 核实 `v-4890ef96` 对应 `deepseek-v4.1-flash-1017-probe-1018-mint-bj`，`v-a2a65be6` 对应 `deepseek-v4.1-flash-1017-probe-1018-mint-sg`。两条 Trace 均出现 Mint 和 Frontend 两个 Broadcaster，证实这两个样本的复制链路有重叠。
- GLM-5.3 历史 5、30、60 分钟各有同窗 Trace 证实 `glm-5.3-mint`（`v-71b85312`）实际出现。证据粒度为出现，完整来源占比未知。
- 历史产物只能用各自采集窗的证据回填。仅有当前路由或当前样本时，历史 DeepSeek 的来源仍按未知处理。

实现挂在 `rtp_llm/dash_sc/inference/servicer.py` 的 `extract_span_external_request_id` 调用前。它用 gRPC 到达钩子记录的时间选段，在解析生成参数前仅统计首帧。请求关联 ID 仍沿用现有日志和 Trace 取证。以部署实际版本及同窗 Trace 为准。
