# Mock 指标合同

指标名称、单位、P/D 角色、HTTP snapshot 字段和汇总方式统一定义于 `MockMetricContract`。`MockControlServer` 根据该定义生成 HELP/TYPE 与两种 HTTP exposition；Whale 上报按同一份定义检查指标与角色，未知指标显式失败。KMonitor 的注册类型与 Prometheus 类型按各自适配器解释，不能直接互换。测试采集与报告声明位于 [online_eval 指标配置](../tools/online_eval/config/monitoring/README.md)；不在此复制完整 metric registry。

## 来源与身份

HTTP `/metrics` 默认按 role 汇总；`?per_engine=true` 带 `engine_name`、`role`、`grpc_port`、`engine_ip` 等身份标签。重启后的计数器属于新一代引擎，不能直接跨代作差。Whale KMonitor 与 HTTP 是独立观察者，不应混用窗口或把多观察者速率相加。

`rtp_llm_` 表示有对应真实引擎语义的观测，不保证模拟时间或数值等于真实 GPU。`mock_` 保留模拟器专属状态和口径，不能只换前缀来合并。

## 执行 TPS 与墙钟 TPS

Prefill 批次 i 的 C 为实际计算 context token，I 为含复用的输入 token，E 为实测执行微秒：

- `rtp_llm_context_tps` = `1e6 × ΣC / ΣE`，只纳入 C、E 都为正的批次。
- `rtp_llm_context_tps_with_cache` 独立按 I、E 都为正的批次计算。
- `rtp_llm_context_wall_tps` 和 `rtp_llm_context_wall_tps_with_cache` 分别以实际 report 墙钟窗口为分母；`rtp_llm_wall_tps_report_interval_us` 记录该窗口。

完全命中的批次可能 C=0、I>0，因此两个执行 TPS 的分母可能不同，不能相减推导命中率。Wall pair 可用于窗口复用率，但不同引擎 report interval 不同时，直接汇总速率的比值不等于精确汇总 token 的比值。

Token 成员在执行开始时冻结，完成派发后与实际执行时间原子发布。执行前取消不计工作；执行后取消不抹去已执行工作。成功请求累计 token 是另一种业务口径。
长步骤尚未完成时没有执行样本，报告保持缺失；完全空闲的窗口报告 0。独立观察者保留各自 cursor 和 wall origin，崩溃后重新建立 generation。

参考为 `RtpLLMMetrics.h` 的 `RtpLLMTokenPSMetricsCollector`、`RtpLLMMetrics.cc` 及执行器时间边界。Mock 实测时间包含模拟等待和运行时开销，不证明绝对 GPU 吞吐；不模拟完整 chunked prefill / beam 执行或 per-priority TPS。比较时固定输入、完整 Fetch、模型与采集版本，并对齐 DP / 引擎聚合口径。

Decode 的 `rtp_llm_sp_estimate_tpot_us` 使用模拟 step 微秒除以实际推进 stream 新产生 token 的平均数；不包含 Prefill 首 token、中途加入或 KV 增长失败的 stream。它是执行估计，不能替代客户端 TTFT / TPOT，也不伪造 draft proposal 和接受率。

Decode 的 `rtp_llm_generate_tps` 对齐真实 `RtpLLMTokenPSMetricsCollector::generateTPS()`：它是观察窗口内的执行 token 数，单位为 token，名字中的 TPS 不代表按秒归一化。`mock_decode_wall_tps` 才是该增量除以观察窗口秒数，性能门禁与速率曲线使用后者。HTTP 与 Whale 共用 `CounterRateMetrics` 累计账本，保留独立游标；首次采样包含引擎启动以来的执行，重启按新 generation 重建窗口。`mock_decode_step_tokens_total` 是对应执行累计量，`mock_generate_tokens_total` 是成功完成请求的输出 token 累计量，不能混用。

两个 sink 的完成数、device cache 驱逐数和准入 cache-key 数统一使用 `mock_engine_completed_total`、`mock_engine_cache_evictions_total`、`mock_engine_cache_key_hits_total`、`mock_engine_cache_keys_requested_total`。平台 dashboard 别名及微秒到毫秒转换保留在 Whale 适配器中。HTTP 的汇总模式和单引擎模式共用导出定义；report interval 仅按单引擎导出，平均执行时间按样本数加权。

## 队列、批次与累计量

| 指标 | 口径 |
|---|---|
| `rtp_llm_running_stream_size` | 正在执行的 stream；不等于 snapshot 的全部生命周期条目 |
| `rtp_llm_wait_stream_size` | 调度等待；不把 Decode ALLOCATE 预留算作等待执行 |
| `mock_prefill_batch_size` | 每个实际执行批次的 histogram，包含 scrape 之间执行的批次 |
| `mock_engine_prefill_ms_avg`、`mock_engine_decode_ms_avg` | 有界近期模拟执行样本的 ms 均值，不能换单位后冒充单次 `rtp_llm_model_forward_us` |
| `mock_engine_cache_key_hits_total`、`mock_engine_cache_keys_requested_total` | 准入时 prefix-match 的累计 key 数；真实 recent-cache-key 观测是 per-request token gauge，两者不可合并 |
| `mock_engine_accepted_total`、`mock_engine_completed_total` | 模拟生命周期 counter，不是 QPS gauge |

`prefill_batches`、`prefill_batch_requests` 和 `max_prefill_batch_size` 描述引擎执行组批，不能用它们反推 Master FIXED_WINDOW 批形。Counter 需要根据声明窗口作差或 rate，并显式处理重启；采样缺失不是 0。

## KV 与 Memory cache

`rtp_llm_kv_cache_pool_total_blocks` 表示模拟 device pool 容量，
`rtp_llm_kv_cache_pool_used_ratio` 为 `(total - free) / total * 100`，包括可回收缓存，不能用 `total - available` 代替。
`rtp_llm_kv_cache_pool_available_blocks` 包括 free 与可驱逐 cache block，排除 held 和被请求引用的 block。
`mock_engine_held_blocks` 是无 key 分配，`mock_engine_referenced_blocks` 是在用的 cache-key block，二者之和才是本模型的请求持有量，不能把任一项改名为 free 或完整 request-ref。

Device 与 Memory 复用 token 分别由
`rtp_llm_kv_cache_device_reuse_length` 和 `rtp_llm_kv_cache_host_reuse_length` 表示，不重叠。复用事件在资源准备完成、执行开始时记录，执行后的取消不抹去匹配观测。
`rtp_llm_kv_cache_hit_rate` 按一分钟窗口的总复用 token / 总输入 token 计算百分比，窗口不足或没有输入不补零，不能平均逐请求百分比。
Memory `mock_memory_cache_available_blocks` 包括可回收条目，`mock_memory_cache_unavailable_ratio` 描述 pinned 与 pending 容量的百分比；常驻前缀的占用需看 `mock_memory_cache_occupancy_ratio`。

`mock_cache_evicted_entry_age_ms` 必须按 `scope=gpu,backing=device` 或 `scope=memory,backing=memory` 分开分析。没有驱逐表示没有年龄观测，不能补 0。容量比较还需固定 block size、CP 与拓扑；这些是元数据模型，不是实际内存页观测。

## 指标命名与保留依据

`rtp_llm_` 名称必须存在于当前 C++ 指标注册表，并对齐生产者的单位、窗口和采样边界；注册表测试检查名称，语义测试检查窗口与分母。真实名称存在不意味着任意 Mock 值都可以使用它。

`mock_` 用于模拟器独有的准入/排空状态、执行累计量和明确区分的辅助观测。理论历史 key 命中与物理缓存复用不同；累计事件与真实窗口 gauge 不同；Memory pending/pinned 元数据不等同真实 transfer 或统一 block-pool 观测。保留这些指标时必须说明边界，不能只因名字相近就合并。

`mock_prefill_admitted_requests` 包含已准入、预留执行批次中的请求，`rtp_llm_running_stream_size` 只数已开始执行的请求；两者不共享 running 名称。等待与 Decode 执行计数使用对应的 scheduler 指标，不再重复发布同值指标。

`mock_backend_ttft_us`、`mock_backend_latency_us` 和 `mock_decode_success_qps` 分别记录后端延迟与成功终态速率。它们的边界不等于 frontend ingress 到客户端响应，不发布 `py_rtp_` 前端别名；转换单位或更换标签不能使两者等价。

模拟缓存条目的移除数量、创建到移除的年龄和 host 元数据容量使用 `mock_` 名称；它们不能代替真实的驱逐计划数量、candidate idle/age、tier residence 或 host pool 指标。没有实际传输计时就不产生传输延迟样本。

## 引擎移除准入

`mock_engine_admission_open` 表示新工作 RPC 闸门；
`mock_engine_admitted_rpcs_total` 和 `mock_engine_rejected_rpcs_total` 分别统计通过和被移除闸门拒绝的 RPC。
一次 batch 算一个 RPC，不能当请求数。关闭入口不暂停已有任务推进；受理计数在关闭后不得增长。

来源类型、PromQL、窗口和缺失规则必须随 run 归档。未知指标、单位冲突和必需样本缺失应显式失败；日志、文件或 debug API 证据须独立声明，不建立降级回退链。
