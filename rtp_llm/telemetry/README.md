# RTP-LLM Trace 运维配置

## 启用 Trace

在需要采集的每个 RTP-LLM 进程注入 `RTP_LLM_TRACE_CONFIG`。HTTP frontend、Dash、Prefill、Decode、Fusion 和 FlexLB 需要分别配置。

```bash
export RTP_LLM_TRACE_CONFIG='{"enabled":true,"endpoint":"https://collector.example/v1/traces","headers":{"x-arms-license-key":"<key>","x-arms-project":"<proj>","x-cms-workspace":"<ws>"}}'
```

`endpoint` 填写完整的 HTTP/HTTPS 地址。上例使用 ARMS/CMS 接收端要求的三个 header：`x-arms-license-key`、`x-arms-project`、`x-cms-workspace`。不同接收端按其要求替换 header 名和值；凭证由发布系统注入，不要写入镜像或日志。

## 可选参数

```json
{
  "enabled": true,
  "endpoint": "https://collector.example/v1/traces",
  "headers": {
    "x-arms-license-key": "<key>",
    "x-arms-project": "<proj>",
    "x-cms-workspace": "<ws>"
  },
  "sampler_ratio": 0.1,
  "certificate": "/path/to/ca.pem",
  "max_queue_size": 2048,
  "max_export_batch_size": 512,
  "schedule_delay_ms": 5000,
  "http_timeout_ms": 3000
}
```

参数说明：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `enabled` | 是否启用当前进程 Trace | `false` |
| `endpoint` | OTLP/HTTP 接收端完整地址；启用时必填 | 无 |
| `headers` | 接收端鉴权 header；启用时必须为非空对象 | `{}` |
| `sampler_ratio` | 没有已采样父 Trace 时的新请求采样比例，范围 `0` 到 `1` | `1.0` |
| `certificate` | HTTPS 接收端的 CA 文件路径 | 系统 CA |
| `max_queue_size` | SDK 待发送 Span 队列上限 | `2048` |
| `max_export_batch_size` | 单次发送的最大 Span 数，不能超过队列上限 | `512` |
| `schedule_delay_ms` | 批量发送的最长等待时间（毫秒） | `5000` |
| `http_timeout_ms` | HTTP 发送超时时间（毫秒） | `3000` |

未填写的参数使用默认值。`sampler_ratio=0` 仍会保留已采样的上游请求；关闭 Trace 请使用 `enabled=false`。

`enabled` 支持布尔值、0/1 及对应字符串（大小写不敏感）；数值参数支持数字字符串。整数参数的小数部分会被截断。

## 关闭 Trace

不设置 `RTP_LLM_TRACE_CONFIG`，或设置：

```bash
export RTP_LLM_TRACE_CONFIG='{"enabled":false}'
```

旧的 `RTP_LLM_OTEL_*`、`OTEL_*` 和 FlexLB `-Dotel.*` 配置不再控制 RTP-LLM Trace。

## 实例标识

发布系统为各进程单独注入 `POD_IP`，用于 Trace 的 `rtp_llm.pod_ip`：

```bash
export POD_IP=10.0.0.12
```

服务名、进程标识、角色和后端 rank 自动生成，无需在 Trace JSON 中配置。

## 配置错误

配置缺失 endpoint、headers，或包含非法 JSON、header、证书和地址时，该进程关闭 Trace，但不影响推理或调度启动。配置修改后需重启进程生效。

## 验证

发送一次受控请求，在 Unitrace 中按 trace ID 检查 Trace 是否出现。若没有 Trace，确认所有相关进程都注入了 JSON，并检查启动日志和接收端鉴权状态。

## ViT 单媒体阶段

ViT 复用上述配置、采样和 OTLP/HTTP exporter，不增加独立开关。媒体 span
`rtp_llm.vit.media` 挂在处理该输入的 RPC 下，带有 `request_id`、
`rtp_llm.vit.media.index`（从 0 开始）、`rtp_llm.vit.media.type` 和
`rtp_llm.vit.media.url`。URL **原样上报**，保留 query、fragment 和 data URL，
不脱敏、不生成 `url_hash`；URL 只作属性，不作 span 名或 metric 标签。

每个阶段是独立的子 span，开始/结束时间表示实际进入/退出；阶段名称均以
`rtp_llm.vit.` 为前缀：

| 阶段 | 计时边界 |
|---|---|
| `compute.queue` | 异步计算任务提交到计算线程开始 |
| `preprocess.queue` | CPU 任务提交到 worker 开始，包括 dispatch/IPC；本地执行通常接近零 |
| `download` / `download.cache` | 已有媒体 loader 的调用；网络、文件、data URL 沿用原逻辑，下载缓存命中单独标记 |
| `preprocess` | CPU 解码和预处理，下载期间暂停；一次输入可能有多个不重叠区间 |
| `embedding.queue` | scheduler 入队到执行，包括凑批、等待和可选预取 |
| `embedding` / `embedding.batch` | scheduler 的 host 调用，包括 batched embedding、结果校验和缓存回填；不是纯 GPU kernel 时间 |
| `greennet.total` | 绿网分支开始到最终 verdict；仅启用时出现 |
| `greennet.prepare` / `greennet` | 旧接口准备/提交与等待判定，或 source 接口提交到判定 |
| `greennet.dispatch` / `greennet.queue` | CPU worker 绿网协程分发与检查容量排队 |
| `cache_wait` | 等待已有/本次提交的缓存 entry 就绪；不代表重新计算 |

同一请求的多图用原始序号区分；hash 查询筛选未命中的输入时仍保留原始序号。
全量 hash 命中不会进入计算链路，因此不会虚构上述计算 span。相同 cache key 的
实际生产任务共享一次计算，消费者通过 span link 关联生产者，并记录 `media.cache_state`。
异步生产者的媒体 span 在任务实际完成/失败/取消时结束，不随提交 RPC 提前结束。

跨请求合批只产生一个 batch span，link 到各媒体；每个媒体的 embedding span
用 `rtp_llm.batch_id` 和 link 关联这次共同执行，不应把各媒体耗时相加当作 GPU 工作量。
不可拆分的多输入处理标记 `rtp_llm.vit.shared_execution=true`，不伪造逐图独立耗时。

CPU 子进程只通过已有结果通道回传时间戳，由 ViT 主进程创建并导出 span，
不在 CPU worker 初始化 exporter。每个任务最多回传 256 条阶段记录，超出时
记录 `rtp_llm.vit.timing_records_dropped`。进程崩溃/超时而未返回的阶段不会伪造完成时间。
所有 span 使用现有 trace ID；埋点不增加 CUDA synchronize，也不改变调度/缓存/公共结果协议。
