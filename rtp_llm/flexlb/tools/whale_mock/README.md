# Whale 寄生 mock 模式

显式设置 `RTP_LLM_MOCK_BUNDLE=1`、`FETCH_OUTPUT_STREAM=0` 后启用。
一个 CPU Pod 运行 master 与 mock 两个独立 JVM；mock 内每个逻辑引擎拥有独立端口、KV 池、队列和生命周期。
默认的本地 case 模式和 Whale 单引擎 Pod 模式不变。

`bundle.yaml` 定义 P/D 数量、每引擎容量、线程数、堆大小与观测参数。
默认 48P/192D，P=11042 块、D=27686 块，每块 1024 token，均为单池。
性能文件不修改输入或输出长度；现有 `FLEXLB_CONFIG` 优先传给 master 与 mock，避免两份估算配置漂移。

### GLM-5.3 标定配置（显式启用）

`glm53-calibration.json` 保存 WLCB `l20d_wlcb_zhipu` 的近似标定，不能作为所有 GLM 硬件的通用公式。
将 `bundle_overrides` 写入 `MOCK_BUNDLE_OVERRIDES_YAML`，`performance` 写入
`MOCK_PERFORMANCE_CONFIG_JSON`，`prefill_estimator` 写入现有 `FLEXLB_CONFIG` 的
`router.roles.prefill.executionTimeEstimator`。这些环境变量重启后生效。
`MOCK_EOS_CONFIG_JSON` 仍可单独覆盖 EOS；未设置时保留 performance 中的 EOS 设置。
物理 `block_size` 必须与性能配置一致，否则启动失败。未配置新变量时保留原行为。

P 耗时（毫秒）为 `127.473174679 + 77.754525464 × batchSize +
12.534367685 × sum(computeTokens/1024) + 0.601112784 × sum(hitCacheTokens/1024)`，下限 1ms。
D 每步耗时为 `26.079977409 + 0.968553807 × running` 毫秒，每步每请求平均推进 2.741241181 token。
系数来自 2026-09-13 的主机级 20 秒窗口，不是请求级性能回归：P 9908 个配对样本，
D 7240 个；按时间前 80% 拟合、后 20% 验证，平均相对误差约 6.2% / 2.4%。
D 样本仅覆盖 batch 3.09–7.70，超出范围是外推，不能据此宣称大 batch 已对齐。
此显式 GLM 配置启用均值 1900 的几何 EOS 作为初始标定，来自生产输出长度窗口均值约 1927；
窗口均值不是按请求加权的分布，仍须用实际完成请求复核，不能将其解释为真实 EOS 模型。

55P/40D 对应当时该部署的逻辑规模。P/D 单池容量分别为 31218/46157 块，
每块 64 token；依据同主机同窗口 `available_blocks / (1-used_ratio/100)` 的中位数估计。
生产 `staticCacheBlockSize=500` 是旧路由缓存索引参数，不替代物理 KV 块大小。
旧生产 `CACHE_AFFINITY_FIRST / WEIGHTED_CACHE` 与当前 schema v3 路由器不能直接视为等价；
这份文件标定引擎执行和容量，不声明调度器策略完全相同。

master 通过本地 discovery 文件发现各引擎，不依赖 P/D VIP。
控制端口使用 Pod IP；引擎 RPC 使用独立 loopback IP 和端口，保证 master 的 engineIp 指标不互相覆盖。同 Pod P→D 仍使用现有 RPC 协议。
框架自动接续，不等待客户端 Fetch；启动任何对端失败时 supervisor 会关闭另一 JVM。

指标保留真实 Pod 的 `hippo_role` 和 `container_ip`，用 `engine`、`engine_port`、`dp_rank` 区分逻辑引擎。
每个引擎的累计计数、时间基准和执行轮采样单独维护；不能把共享 Pod 的总数当作单引擎值。
现有 Grafana 查询应选择 master 角色，并按 engine 或 dp_rank 查看逻辑实例，不能要求出现不存在的 P/D Pod。
不修改 Grafana 面板。

## 验收边界

分别核对调度接受 QPS、decode 完成 QPS、失败数、输出 token 数、Fetch RPC=0、队列与 KV 归零。
调度确认不代表推理完成。现有复制流量的 gRPC 入口通过显式 `RTP_LLM_MOCK_SCHEDULE_ONLY=1` 开关调用测试侧的 schedule_only.py；返回带 schedule_accepted=true、inference_completed=false 的确认帧，不输出 token 或推理完成标志。未开启时走原推理路径。
默认未开启 EOS 时，max_new_tokens 被模拟器视为实际输出长度；超大上限请求可能长期占用资源。可选模型见下文。
未完成真实复制流量验证之前，不宣称成功率或性能已对齐。

Bundle jars use the explicit Maven profile `opensource,!internal,whale-bundle`: KMonitor is included, while engine discovery remains local to the Pod. VipServer is intentionally absent from these test jars; the default internal profile is unchanged.


### 可选的提前 EOS 模型（默认关闭）

真实引擎把 `max_new_tokens` 当上限，在 `min_new_tokens` 后允许 EOS；
`ignore_eos=true` 禁止 EOS。测试 mock 可在 performance JSON 中配置：

```json
{"decode":{"eos":{"enabled":true,"distribution":"geometric","mean_tokens":1024,"seed":20260912}}}
```

Whale bundle 也可通过 `MOCK_EOS_CONFIG_JSON` 传入上述 `eos` 对象。
不设置或 `enabled=false` 完整保留旧行为；仓库默认配置不启用。
启动时生成运行目录下的 performance.json，不修改原始流量及 master 可见参数。

这是常量 EOS 概率模型：允许 EOS 后，每 token 的结束概率为 `1/mean_tokens`。
默认最小长度为 1 时，未被请求上限截断的平均长度是 `mean_tokens`；
较大的 min_new_tokens 会推迟分布起点。请求 ID 和 seed 决定采样，P/D 与重试一致，
不会随并发或执行步数重新抽样。显式 replay output_len 优先，启用模型时仍遵守请求上下限。
完成沿用现有 decode 正常收尾路径，释放 KV/执行位，无需 Fetch。

示例 1024 是暂定模拟参数，**没有当前生产输出分布校准，不代表生产均值**。
此模型对齐结束约束，不模拟语言内容、真实 token EOS、stop words 或 beam search。
真实分布不是几何分布时，应使用真实 replay output_len 或重新校准模型，不能靠调短长度证明生产吞吐。

验证使用实际 D 完成计数及 `mock_generate_tokens_total` 增量，输入侧用
`mock_context_tokens_total` / `mock_context_compute_tokens_total`；这些累计计数不受抓取窗口重置影响。
前端 schedule acknowledgement 仍不是推理完成，不能当端到端成功率。


### Cache and request metrics

Whale mock requests without explicit `unique_key.block_cache_keys` derive rolling
block hashes from token IDs using the production `HashUtil.h` rule. Only full
blocks are indexed for reuse; a partial final block still consumes physical KV.
Explicit test metadata (including an empty key list) retains its original meaning.
This models token-prefix reuse for the current single-model deployment; it does
not model multimodal embedding identity or multiple LoRA cache namespaces.

KMonitor reports request input length at P completion and input/actual output
length at D completion, plus reuse/effective context length. KV eviction lifetime
is sampled per removed block, and direct eviction block count per chain event;
there is no simulated memory-tier writeback. Cumulative eviction and cache-key
hit/request counters are also exposed by the Whale metric adapter.

In bundled mode `hippo_role` identifies the physical master Pod. Split logical
engines by `role=ROLE_TYPE_PREFILL|ROLE_TYPE_DECODE` and `engine`/`dp_rank`; do not
interpret a role-merged running-stream series as the P batch size. The existing
running/context/generate batch metrics are emitted for both logical roles.

### 按负载扩缩逻辑引擎

Whale master zone 可设置 `MOCK_BUNDLE_OVERRIDES_YAML`，重启 bundle 后生效：

```yaml
prefill: 48
decode: 1024
mock_heap: 32g
```

只允许覆盖 P/D 数量与两 JVM 堆大小，不设置时保留 bundle.yaml 默认值。
这是同 Pod 内的逻辑引擎扩容；每引擎容量、执行上限、流量内容与 EOS 分布不变。
需同时检查 Pod 内存、CPU、运行数、排队与完成率。1024 不是默认生产规模，
只是针对当前复制流量的起始容量实验，不能通过降低 running 指标值伪造空闲。

### Prefill 匹配率与 TPS 口径

`rtp_llm_prefill_worker_recent_cache_key_hit_ratio` 与真实引擎一样，按请求统计
最近历史窗口内出现过的完整块占输入 token 的比例（0–1）。窗口默认 30 分钟，
可用 `PREFILL_CACHE_HIT_TIME_WINDOW_MS` 或后备 `CACHE_HIT_TIME_WINDOW_MS` 配置。
同请求重复 key 不会自命中；`theory_cache_all_hit_ratio` 按累计 token 加权。
这些是历史理论匹配率，缓存驱逐后历史仍然存在；实际池复用率单独上报
`mock_prefill_kv_match_ratio`，不能拿理论值解释实际计算量。

`rtp_llm_context_tps` / `_with_cache` 使用每批执行时间的累计增量，
`rtp_llm_context_wall_tps` / `_with_cache` 使用实际采样墙钟时间。
`rtp_llm_generate_tps` 使用逐步生成 token 的增量，包括尚未结束的请求；
`mock_generate_tokens_total` 仍专门统计已完成请求输出，供完成性验收使用。

### Production timing calibration

The bundle uses the DSv4 decode step fit `19.5 + 0.175 * running` ms with
2.6 accepted tokens per step. It must not use the old fixed 20 ms / one-token
smoke configuration for production traffic comparisons.

Prefill execution follows `FLEXLB_CONFIG.router.roles.prefill.executionTimeEstimator`.
An explicit constant expression such as `30` overrides the built-in DSv4 fit;
production comparison deployments must supply the full calibrated expression.
Changing this expression changes both routing estimates and mock execution time.

`MOCK_EOS_CONFIG_JSON.mean_tokens` describes the geometric stopping model, not
an observed mean: request limits, minimum length and ignore-EOS still apply.
Calibrate it against completed output lengths, then size the cluster using the
resulting lifetime. Do not tune reported TPS counters or truncate observations.
