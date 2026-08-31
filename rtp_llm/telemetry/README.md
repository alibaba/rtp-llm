# RTP-LLM Trace 部署指南

RTP-LLM 内置 OpenTelemetry trace（Python frontend + C++ engine 双侧自产 span，OTLP/HTTP 直连导出）。

## 1. 开启方式

```bash
export RTP_LLM_OTEL_TRACE_ENABLE=1        # 总开关，默认关闭
# 二选一：
export RTP_LLM_OTEL_REGION=cn-hangzhou    # region 映射自动解析 endpoint/headers/CA
# 或显式指定 endpoint：
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://<collector>/v1/traces
```

行为要点（代码依据：`tracing.py` / `cpp/telemetry/TelemetryRuntime.cc`）：

- **fail-open**：telemetry 任何初始化/导出失败只降级关闭，不影响推理。
- **仅 tp_rank 0 产 span**，其余 rank 自动禁用；DP 部署下每个 DP 组的 tp_rank0 均产 span（请求只路由到一组，trace 不重复）。C++ 侧 Resource 带 `rtp_llm.dp_rank` / `rtp_llm.world_rank` 用于区分副本；Python frontend 侧 Resource 只有 `service.name` / `service.instance.id` / `process.pid` / `rtp_llm.role`，副本靠 `service.instance.id`（`hostname-pid`）区分。
- 开关打开但无 endpoint 时 telemetry 静默禁用（error 日志可查）。

## 2. endpoint 解析优先级

1. `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`（原样使用）
2. `OTEL_EXPORTER_OTLP_ENDPOINT`（自动拼接 `/v1/traces`）
3. `RTP_LLM_OTEL_REGION` + region 配置文件

生产环境优先由部署平台显式注入 `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` 和
`OTEL_EXPORTER_OTLP_TRACES_HEADERS`。region 配置文件包含接入凭证，不应内置于镜像或发布包；
若使用 region 单变量模式，应通过 Secret 在运行时挂载配置，并用
`RTP_LLM_OTEL_REGION_CONFIG_FILE` 指向挂载路径（也兼容挂载到
`/etc/rtp_llm/trace_regions.json`）。region 解析结果不会覆盖用户已显式设置的 env。
region 解析在 launcher 进程（`start_server.py`）中执行后随环境继承给 C++ backend 子进程。

## 3. POD_IP 与平台指标面板（重要）

Python/C++ 两侧均在 **`POD_IP` 环境变量非空**时向 Resource 写入 `host.ip`。
观测平台的请求数/错误数/耗时面板依赖该属性做实例维度的过滤统计，
**span 缺少它时这些面板恒为"暂无数据"**——trace 本身仍然完整，只是指标面板统计不到。

- **k8s 部署**：`POD_IP` 通常已由 downward API 注入，无需额外配置。
- **非 k8s 部署（物理机 / docker 直跑）**：必须显式设置，例如：

  ```bash
  export POD_IP=$(hostname -i | awk '{print $1}')
  ```

注：时间窗内无错误请求时错误数显示"暂无数据"属正常现象（口径为 OTel status=ERROR 的 span 数）。

## 4. 环境变量一览

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RTP_LLM_OTEL_TRACE_ENABLE` | `0` | trace 总开关 |
| `RTP_LLM_OTEL_REGION` | 空 | region 映射（自动解析 endpoint/headers/CA） |
| `RTP_LLM_OTEL_REGION_CONFIG_FILE` | 空 | region 配置文件路径覆盖 |
| `RTP_LLM_OTEL_SERVICE_NAME` | 空 | 整体覆盖 service.name；默认按角色拆分为 `rtp_llm_frontend/prefill/decode/pdfusion` |
| `RTP_LLM_OTEL_TRACE_SAMPLER_RATIO` | `1.0` | ParentBased(TraceIdRatio) 采样率 |
| `RTP_LLM_OTEL_BSP_MAX_QUEUE_SIZE` | `2048` | BatchSpanProcessor 队列上限（满则静默丢弃） |
| `RTP_LLM_OTEL_BSP_SCHEDULE_DELAY_MS` | `5000` | BSP 导出周期 |
| `RTP_LLM_OTEL_BSP_MAX_EXPORT_BATCH_SIZE` | `512` | 单批导出条数（自动 clamp 到不超过队列上限） |
| `RTP_LLM_OTEL_HTTP_TIMEOUT_MS` | `3000` | OTLP HTTP 导出超时 |
| `POD_IP` | 空 | 非空时写 Resource `host.ip`（指标面板依赖，见第 3 节） |

以上默认值 Python 与 C++ 两侧一致；两侧读取同一组环境变量，无需分别配置。

## 5. 部署后验证

1. 发一条 chat completions 请求（trace 仅覆盖 `/v1/chat/completions` 入口）。
2. 从 access log 取 trace_id：`grep <prompt关键词> logs/access_r*_s*.log`，取 `trace_id` 字段。
3. 确认导出无失败：`grep 'failed to export' logs/*.log` 应无命中。
4. 在观测平台用 trace_id 检索，确认 span 树完整（PD 分离 11 span，含 decode 侧 `load_cache` 子 span；Fusion 6 span）、
   流式请求的 POST span 附加信息 Events(1) 为 `first_response_chunk`；非流式请求无该 event、
   平台上能看到该实例 IP（`<POD_IP>`）、且请求数/耗时指标有数据。

新 service.name 首批 trace 的平台索引可能有分钟级延迟，属正常现象。
