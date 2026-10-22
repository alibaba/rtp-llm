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
