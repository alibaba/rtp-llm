# Frontend 指标与 engine 统计口径

本文描述基于 main `1812ac310f` 移植的 frontend 监控功能。指标适用于经过 Python frontend 请求观测链路的 HTTP 推理、OpenAI 兼容接口和 DashScope 请求。直接调用模型或 native op 不会自动产生这组 frontend 请求指标。

## 开启方式

新增的 15 个核心 frontend 指标及 `py_rtp_frontend_recent_cache_key_*` 探针默认关闭。在启动 frontend 进程前设置：

```bash
export RTP_LLM_FRONTEND_METRICS_ENABLE=1
```

开关接受 `1`、`true`、`yes`、`on`，忽略大小写与首尾空格；未设置、空值、`0`、`false`、`no`、`off` 均关闭。其他值记录警告并保持关闭。开关在 `FrontendRequestMetrics` 初始化时读取，修改环境变量后应重启相应 frontend 进程。

还需按部署环境初始化 KMonitor，并完成下文的服务就绪流程。此开关控制新增 frontend 观测链路，不关闭既有 `py_rtp_*`、engine 指标或服务状态管理。启用后内部请求会携带私有指标传输配置；用户请求 JSON 不需要、也不能用 `frontend_metric_streaming` 开启该内部通道。

## 核心 15 个指标

下表中的 `I` 为 engine 输入长度，包含内部物化的 prefix；`O` 为所有返回序列的生成长度之和；`R` 为复用 token 数。请求汇总优先采用私有后端快照，公开的 usage、AuxInfo 和返回 token 内容不因监控改变。

对推测解码，`V` 为 verify 轮数，`A` 为接受 token 总数，**包含每轮必有的一个 target token**，`P` 为提出的 draft token 总数，不包含 target token。`T_ctx`、`T_cache` 为对应 context 执行时间累计值，单位 µs；`Δw` 为 frontend 实际上报窗口，单位秒。

| 指标 | 单位 | 统计定义与 engine 对应关系 |
| --- | --- | --- |
| `py_rtp_frontend_input_length` | token | 请求的 `I`；对应 engine `inputLength()`，可与公开 usage 的 prompt 长度不同。 |
| `py_rtp_frontend_output_length` | token | 请求的 `O`；batch 汇总各请求，多个返回序列逐行相加。独立序列提前结束时保留其实际终态长度。 |
| `py_rtp_frontend_cached_token_length` | token | 请求的 `R`；来自 engine 复用元数据。PD 使用合并后的 prefill/decode 元数据，避免把 handoff 传来的 prompt 再计作一次命中。 |
| `py_rtp_frontend_cache_hit_ratio` | **百分数，0–100** | `min(R / I, 1) × 100`；`I = 0` 时不报。 |
| `py_rtp_frontend_concurrency` | 请求数 | 当前 frontend 进程已开始、尚未结束的受观测请求数量。一个 batch 请求占一个请求槽位，不等同于 engine decode batch size。 |
| `py_rtp_frontend_ttft_ms` | ms | frontend 请求起点至首个有效输出的时间；流式输出与非流式估计的边界见下文。 |
| `py_rtp_frontend_tpot_ms` | ms | **engine 推测解码 TPOT 样本**：`rtp_llm_sp_estimate_tpot_us / 1000`。不是按请求 RT、输出长度重算的普通 TPOT。 |
| `py_rtp_frontend_request_rt_ms` | ms | frontend 请求观测开始到成功完成、错误或取消收尾的时长。 |
| `py_rtp_frontend_speculative_avg_accept_length` | token/verify 轮 | `A / V`，`V > 0` 时上报；与 engine `rtp_llm_sp_avg_accept_token_num` 采用相同的接受 token 定义。 |
| `py_rtp_frontend_speculative_accept_rate` | 比率，0–1 | `max(A − V, 0) / P`，限制在 `[0, 1]`，`P > 0` 时上报；对应 engine **`rtp_llm_sp_avg_fix_accept_rate`** 的 draft-only 语义。 |
| `py_rtp_frontend_stream_first_output_token_length` | token | 流式请求第一个实际对外输出帧增加的 token 数；空帧、工具调用缓冲帧及内部 metric-only 帧不算首帧。 |
| `py_rtp_frontend_input_token_tps` | token/s | `Σ context_tokens_with_cache × 10^6 / Σ T_cache`；对应 engine `rtp_llm_context_tps_with_cache`。 |
| `py_rtp_frontend_noncache_input_token_tps` | token/s | `Σ context_tokens × 10^6 / Σ T_ctx`；对应 engine `rtp_llm_context_tps`。 |
| `py_rtp_frontend_output_token_tps` | token/s | `Σ engine_generate_tokens / Δw`；使用 engine 的生成计数，窗口差异见下文。 |
| `py_rtp_frontend_noncache_output_token_tps` | token/s | 与上一项相同；decode 生成 token 不使用 prompt cache 命中率扣减。 |

字段不可用时不编造对应请求统计。没有有效输出的失败请求仍有 RT 和并发收尾，但可能没有 TTFT、长度或推测解码指标。没有 SP 样本的请求不补报 `tpot_ms=0`。

请求汇总标签包含 `rank_id`、`server_id`、`source`、`streaming`、`priority`。并发、TPS 和共享 SP TPOT 使用 frontend 容器维度的 `rank_id`、`server_id`，避免把整个 engine batch 的贡献误归因到某一个承载请求的 priority。服务状态层统一追加 `is_serving`。C++ stream 指标同时保留 `priority` 与超时的 `timeout=true` 标签。

## 执行计数与 TPOT 的来源

### Context 与生成吞吐

NormalExecutor 的 context 分子来自 `StreamGroups.contextExecuteTokenSize()` 和 `contextExecuteTokenSizeWithCache()`，decode 分子来自 `totalDecodeBatchSize()`；执行时间使用传给 engine `reportMetrics()` 的同一个 `tps_execute_time_us`。同步路径在 dispatch 完成后取样，异步路径沿用 engine 的 dispatch launch 前取样边界。

MTP prefill 使用 `collectPrefillMetrics()` 中相同的 context 分子和时间。MTP decode 使用 `sp_engine_collector.total_accepted_token_num`，保留 main 对接受长度的**一轮延迟采样**，不拿当前对外输出长度代替。私有 `frontend_generate_execute_time_us` 携带 engine 调用所用的 `decode_time`/`tps_execute_time_us`，但不作为 frontend output TPS 的分母。

Context 只有在 token 数和执行时间均大于零时才累计；generate 分子只要求 token 数大于零，与 engine `RtpLLMTokenPSMetricsCollector` 相同。每个 engine step 的计数和时间由一个有效 RPC stream 承载，frontend 用累计计数的正向差分去重，不能按 batch 请求数重复相加同一个 step。

### SP TPOT 与接受率

共享 TPOT 的唯一公式是：

```text
engine_sp_tpot_us = step_latency_us /
                   (total_accepted_token_num / total_stream_num)
frontend_tpot_ms  = engine_sp_tpot_us / 1000
```

C++ reporter 和私有 RPC 通道调用同一个 `estimateTpotUs()`。只有 engine collector 满足正 proposed、stream、accepted 计数条件时才发送样本。普通 NormalExecutor 不生成 SP TPOT。是否产生样本由实际执行路径和 collector 决定，不根据请求中的 `force_disable_sp_run` 标志推测或补造一个值。

SP 请求累计 `V` 复用 main 的 `sp_iter_count_`，`A = V + sum(speculative_accepted_tokens_per_pos_)`；`P` 随相同的 `specUpdate` 按 `speculative_propose_step` 累计。若每轮提出 `s` 个 draft token，frontend 接受率是 `(A/V − 1)/s`；engine 的另一项 `rtp_llm_sp_avg_accept_rate` 是 `(A/V)/(s+1)`，两者含义不同。Frontend 对请求全部轮次先累计再求比率，engine 按 step 上报 gauge；在变动 batch 或跨请求聚合时，直接平均 gauge 不保证与请求加权结果相等。

SP 样本序号在每个后端 request attempt 内递增。序列化只读取已完成的样本，不等待未来轮次。当前 step 的终态业务帧在本轮样本完成后发布；旧的非终态 token 帧和错误/取消不会等待未来 step。失败的候选 stream 会让出样本承载权，本轮刚结束但终态尚未发送的 stream 仍可承载本轮样本。Batch RPC 收集过程保留中间 metric-only 帧中的样本，PD 交接也会先发送真正的 prefill 输出。

如果请求已取消或传输已断开，未送达 frontend 的样本不能由 frontend 凭空恢复；因此跨进程指标不承诺在网络中断时与 engine 完全逐点一致。

### Output TPS 的窗口差异

Engine 现有 `rtp_llm_generate_tps` 由约一秒的 reporter loop 上报窗口内**原始 token 计数**；`generateTPS()` 不除以每步执行时间，也不按 reporter 的实际调度延迟重新归一化。

Frontend 同样收集 engine token 分子，但除以自己的真实 heartbeat 间隔 `Δw`，默认目标间隔为一秒。例如 120 个 token 在一个实际 1.2 秒窗口到达，frontend 报 100 token/s；engine 原有计数窗口若收集到同样 120 个 token，会报原始值 120。RPC 传输延迟、两个进程的窗口起点及 heartbeat 调度会改变单个点的分布。比较时应先对齐时间范围与聚合方法，不能把同一时刻的 gauge 差异直接解释为吞吐丢失。

Context TPS 的分母是 engine 执行时间，因此不按 frontend heartbeat 的实际时长计算。Frontend 在窗口内先合并计数/时间再求比值，不平均各请求的 TPS。

## 请求边界、重试与公开输出

流式 TTFT 从 frontend 请求观测起点计到 renderer 首次产生有效 token 增量；空 SSE 帧、角色帧、工具调用缓冲以及 metric-only 帧都不触发首 token。它包含进入观测之后的路由、RPC 与渲染等待，不是 engine `rtp_llm_first_token_latency_us` 的单位换算，也不是客户端网络侧 TTFT。

非流式请求没有逐 token 的公开帧，使用后端 side-channel 估计首 token 时刻：以帧到达 frontend 的时间为锚点，减去 `backend_cost − backend_first_token_cost`，并限制在请求开始与帧到达之间。缺少有效后端时间时使用首个有效后端输出的到达时间。该估计仍带有传输和接收调度影响。

原生 `/batch_infer` 和 `/v1/batch/chat/completions` 保持一次 unary batch RPC；整个 batch 只占一个请求和并发槽位，各 item 的私有长度与执行计数按独立单元聚合。不为指标开启公开 AuxInfo 或物化 terminal beam token 行。只有最终观测且没有有效 AuxInfo 时间时，TTFT 包含整个 batch 的等待时间；没有逐 token 观测，不代表其中最早 item 的真实首 token 时刻。

RT 覆盖 frontend 的请求观测生命周期，成功、错误、取消均执行一次收尾；不是 engine `rtp_llm_latency_us` 的同边界指标。并发在 begin/finish 时增减，异常与关闭生成器不会重复递减。

对每个逻辑 batch 单元，请求长度、缓存和接受率汇总采用**已经观察到的最新 backend attempt**，避免把失败尝试的输出长度叠进最终响应。吞吐保留**所有已观察到的 attempts** 的 engine 累计正向差分；失败重试已经消耗的计算不能从吞吐中删除。各 attempt 的 SP 样本序号也独立去重。

非流式请求可以通过私有 metric-only 帧报告进度。这些帧在公开响应链路被过滤，不改变 HTTP、OpenAI、DashScope 的响应帧数、usage、AuxInfo、output IDs 或 beam 输出优化路径，也不推进业务帧的首输出状态。

## 服务状态与 warmup/drain

服务指标状态独立于 `RTP_LLM_FRONTEND_METRICS_ENABLE`。Python 和 C++ 采用以下生命周期：

| 状态 | `is_serving` | 上报行为 |
| --- | --- | --- |
| STARTING / WAITING_FOR_WARMUP | 尚未就绪 | 不接受普通指标，不上报服务状态心跳；warmup 数据不混入首个生产窗口。 |
| SERVING | `true` | 普通指标与 `rtp_llm_service_status=1` 正常上报。 |
| DRAINING | `false` | 继续上报在途请求收尾与 `rtp_llm_service_status=0`，保留排空阶段可观测性。 |
| STOPPED | 已停止 | 停止上报；同一次生命周期不能由晚到的 ready 回调重新进入 SERVING。 |

标准服务器由 BackendManager 在模型加载后申请 serving；若配置了 `RTP_LLM_STARTUP_WARMUP_HEALTH_GATE_FILE`，需要已有启动 warmup 流程完成并创建 gate 文件后才正式开放指标。不要为了开启监控提前创建该文件。NormalEngine 的 warmup/preRun 还使用 scoped suppression，避免后续预热动作污染生产统计。指标状态不替代原有 HTTP/RPC 健康检查与就绪判断。

关闭时通过共享 drain event 把状态传播至 frontend、backend 和相应辅助进程；backend 状态监听器仅更新指标状态，实际排空与停止仍由既有进程管理流程执行。晚到的 warmup/ready 回调不能覆盖 drain。

**Native/offline 使用者需显式开启 serving。** 直接构造 `RtpLLMOp`、`RtpEmbeddingOp` 或 Python engine 并调用 `init` 不会自动开放指标。在初始化和自己的预热完成后，对对应对象调用：

```python
engine.set_serving(True)
# 执行生产请求或需要测量的离线推理。
engine.set_serving(False)
# 然后按既有接口停止并释放 engine。
```

`RtpLLMOp`/`RtpEmbeddingOp` 的 Python 接口同样名为 `set_serving`。若离线代码还独立使用 Python KMonitor reporter，应先完成 `kmonitor.init()`，就绪后调用 `kmonitor.set_serving(True)`，退出时调用 `kmonitor.set_serving(False)`。纯 C++ 集成使用 `startKmonServiceStatus(reporter)`、`setKmonServiceServing(true/false)`、`stopKmonServiceStatus()`；若使用已有 KMonitor factory 管理启动/停止，则复用其生命周期。调用 `set_serving(True)` 也遵守已配置的 warmup gate。

## 私有 RPC 契约

字段只用于后端与 frontend 内部通信，不添加到公开 AuxInfo 或用户请求 JSON。保留 main 原有字段编号：GenerateConfig 的 `prompt_logits_end=66`、`enable_disk_cache=74`、`aux_info=75`、`prefill_only=76`，以及 GenerateOutputs 的 `supports_prefill_completion=10`。

| Message | 字段号 | 新增字段 |
| --- | --- | --- |
| `GenerateConfigPB` | 77 | `frontend_metric_streaming` |
| `GenerateOutputsPB` | 11 | `frontend_metric_only` |
| 同上 | 12 | `frontend_generate_token_num` |
| 同上 | 13 / 14 | `frontend_context_token_num` / `frontend_context_token_num_with_cache` |
| 同上 | 15 / 16 | `frontend_context_execute_time_us` / `frontend_context_execute_time_with_cache_us` |
| 同上 | 17 | `frontend_generate_execute_time_us` |
| 同上 | 18 / 19 | `frontend_input_len` / `frontend_output_len` |
| 同上 | 20 / 21 / 22 | `frontend_speculative_verify_rounds` / `frontend_speculative_accepted_token_num` / `frontend_speculative_proposed_draft_tokens` |
| 同上 | 23 | repeated `frontend_sp_tpot_samples` |
| `FrontendSpTpotSamplePB` | 1 / 2 | `sequence_id`（uint64）/ `tpot_us`（double） |

私有累计计数字段使用 `Int64Value`，区分缺失和零。Python 样本格式是 `[(sequence_id, tpot_us), ...]`。新后端没有分配给某请求 context 贡献时，不应拿它的完整输入长度再次补算同一个 batch 的贡献。旧后端缺字段时只能保留已有元数据所能支持的统计；不能声称仍具备精确 engine TPS/SP TPOT。

除核心 15 项外，新增 frontend recent-cache-key 探针包含 hit count、total count、request count、empty-request count 与 hit ratio。它们统计 frontend 最近窗口内是否见过 prompt block key；窗口由已有 `CACHE_HIT_TIME_WINDOW_MS` 控制，默认 30 分钟。这里的 `py_rtp_frontend_recent_cache_key_hit_ratio` 是 **0–1 比率**，不同于上表请求 KV cache 命中率的 0–100 百分数，也不能代替 engine 的真实 KV cache 命中情况。

## 来源与 main 能力保留

以下 commit 均可通过 `git show <sha>` 复核；覆盖按功能核对，不按作者筛选。此次是基于新 main 的功能移植，未整体复制源分支的模型推理、CUDA kernel 或动态 MTP 改动。

| 来源 commit | 作者 | 覆盖方式 |
| --- | --- | --- |
| `74d31e227b`、`b8e97961ec` | yangchengjun.ycj | frontend 请求、缓存、吞吐与聚合指标；`180db1dd8b` 为旧 main 独立移植参考，重新解决与当前 main 的差异。 |
| `5f026a0383` | yangchengjun.ycj | engine SP TPOT 私有样本传输；本次同时修复源实现已记录的未来轮次阻塞和结束/error stream 丢样问题。 |
| `c38484ce75` | yangchengjun.ycj | DashScope 请求范围的 speculative acceptance 汇总日志。 |
| `ec4c91fa43` | zw193905 | priority/error/timeout 标签与相关监控修正；main 已有部分不重复移植。 |
| `bf87bf6023` | luoli | 补齐返回失败的 route QPS 观测缺口；不为此整体引入 auto-TPM 业务改动。 |
| `19d694b693`、`3b74d8bb0a`、`4f1e39cacd` | zw193905 | `is_serving`、warmup 抑制与 drain 状态传播，结合当前 main 生命周期适配。 |

Main 已有的 auto-TPM/priority 行为（源历史 `19d06d7f67`）、DashScope business-error 统计（`7cda7e68bd`）及日志同意策略（`afc69f3553`）保留等效行为。`69aba52b60e` 整合的启动 warmup health gate 已存在，本次把指标生命周期接入该 gate；`95ce33f438` 已加入 engine context-token count 指标，本次复用。当前 main 的 stage 时延、缓存层级、prompt logits、prefill-only、异步执行与返回序列行为继续保留。

## 代码与回归入口

- [Frontend 聚合和生命周期](../../rtp_llm/frontend/frontend_request_metrics.py)、[对应测试](../../rtp_llm/frontend/test/frontend_request_metrics_test.py)。
- [内部协议](../../rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto)、[Python RPC 转换测试](../../rtp_llm/cpp/model_rpc/test/model_rpc_client_test.py)。
- [当前 step 的指标交接](../../rtp_llm/cpp/engine_base/stream/FrontendMetricsStep.h)、[stream 边界测试](../../rtp_llm/cpp/engine_base/stream/test/GenerateStreamTest.cc)、[batch/PD RPC 回归](../../rtp_llm/cpp/model_rpc/test/LocalRpcServerTest.cc)。
- [Engine 指标公式](../../rtp_llm/cpp/metrics/RtpLLMMetrics.h)、[C++ 服务状态](../../rtp_llm/cpp/metrics/ServiceStatus.cc)、[Python 服务状态](../../rtp_llm/metrics/kmonitor_metric_reporter.py)。

这些回归覆盖累计值去重、batch/multi-sequence、重试、终态/错误/取消、metric-only 过滤、SP 样本归属及服务状态。单元测试不能替代真实 GPU、PD 拆分部署下的端到端窗口对齐验证。
