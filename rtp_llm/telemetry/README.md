# RTP-LLM Trace 部署指南

RTP-LLM 内置 OpenTelemetry trace（Python frontend + C++ engine 双侧自产 span，OTLP/HTTP 直连导出）。

## 1. 开启方式

```bash
export RTP_LLM_OTEL_TRACE_ENABLE=1        # 总开关，默认关闭
# 二选一：
export RTP_LLM_OTEL_REGION=cn-hangzhou    # region 映射自动解析 endpoint/headers/CA
# 或显式指定（此时需自行把 endpoint 和鉴权 headers 一起给齐）：
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://<collector>/v1/traces
# collector 需鉴权时必须同时给 headers，否则导出被拒（401/403）；
# 引擎不把 headers 传给 exporter 构造参数，而是依赖 OTel SDK 从该 env 读取。
export OTEL_EXPORTER_OTLP_TRACES_HEADERS='x-arms-license-key=<key>,x-arms-project=<proj>,x-cms-workspace=<ws>'
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

## 5. FlexLB（Java master）的 trace

FlexLB 是独立的 Java 进程，与 Python/C++ 引擎共用
`RTP_LLM_OTEL_TRACE_ENABLE` 总开关，导出配置使用标准 `OTEL_*` 环境变量。
开关为真且配置了 OTLP endpoint 时，`Application` 会在 Spring 启动前自动启用
OpenTelemetry provider；无需额外添加 JVM `-D` 参数或 Java agent。

FlexLB 内的埋点分两部分：
- **手工埋点**（`GrpcTraceInterceptor` + `FlexlbTrace`）：负责 Schedule 的 SERVER span、
  业务属性（`flexlb.schedule.*`、`rtp_llm.*`）与业务拒绝的 ERROR 状态。默认只启用
  trace exporter；metrics/log exporter 默认关闭。
- **自动埋点**（OpenTelemetry Java agent，可选）：额外补 gRPC 客户端/服务端 span、
  `rpc.*` / `network.*` 等属性。当前生产启动脚本不加载 `-javaagent`。

### 5.1 手工埋点的最小开启方式

```bash
export RTP_LLM_OTEL_TRACE_ENABLE=1
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://<collector>/apm/trace/opentelemetry/otlp/v1/traces
export OTEL_EXPORTER_OTLP_TRACES_HEADERS='x-arms-license-key=<key>,x-arms-project=<proj>,x-cms-workspace=<ws>'

java -jar flexlb.jar ...
```

应用默认设置 `OTEL_SERVICE_NAME=rtp_llm_flexlb`，并关闭 metrics/log exporter；部署侧显式配置
对应 `OTEL_*` 变量时以用户配置为准。若总开关关闭或 endpoint 缺失，provider 保持 no-op，
不影响调度或远端 traceparent 透传。`OTEL_EXPORTER_OTLP_TRACES_HEADERS` 承载接入凭证，
应通过 Secret 在运行时注入，不应硬编码进镜像或发布包（与第 2 节口径一致）。

### 5.2 启用自动埋点（Java agent，可选）

若还想要 gRPC/HTTP 层的自动 span，再额外挂 agent：

```bash
java -javaagent:/path/to/opentelemetry-javaagent.jar -jar flexlb.jar ...
```

启用 agent 前需要知道的两点（均为实测结论）：

1. **成功请求的 Schedule span 状态是 `UNSET`，不是 `OK`。** agent 拥有 SERVER span 时
   由 agent 结束，而 OTel 规范要求 instrumentation 成功时不设 OK。按 `status == OK`
   过滤成功请求的看板会漏掉这些 span —— 应改用「非 ERROR」或依赖 `flexlb.schedule.code`。
   （业务拒绝仍会被手工埋点标为 `ERROR` + `error.type=FLEXLB_BUSINESS_REJECTED`，不受影响。）
2. **agent 会显著放大 span 量。** 实测约 88% 是定时健康检查等后台任务产生的 span，
   接入前需评估 collector 侧的采样与配额。

手工 SERVER span 与 agent SERVER span **不会重复**：`GrpcTraceInterceptor` 检测到已有
current span 时复用它（`ownsSpan=false`），不会再建第二个。

## 6. 部署后验证

1. 发一条 chat completions 请求（trace 仅覆盖 `/v1/chat/completions` 入口）。
2. 从 access log 取 trace_id：`grep <prompt关键词> logs/access_r*_s*.log`，取 `trace_id` 字段。
3. 确认导出无失败：`grep 'failed to export' logs/*.log` 应无命中。
4. 在观测平台用 trace_id 检索，确认 span 树完整、
   流式请求的 POST span 附加信息 Events(1) 为 `first_response_chunk`；非流式请求无该 event、
   平台上能看到该实例 IP（`<POD_IP>`）、且请求数/耗时指标有数据。

   按拓扑对照 span 数（实测值）：

   | 拓扑 | span 数 | 特征 span |
   |---|---|---|
   | Fusion | 6 | — |
   | PD 分离（frontend 直连 prefill） | 11 | `rtp_llm.prefill_generate_stream_call`；decode 侧 `load_cache` |
   | PD 分离 + master 凑批 | 14 | `rtp_llm.prefill_batch_request`、`rtp_llm.fetch_response`；FlexLB 的 `rtp_llm.flexlb.schedule` |

新 service.name 首批 trace 的平台索引可能有分钟级延迟，属正常现象。
