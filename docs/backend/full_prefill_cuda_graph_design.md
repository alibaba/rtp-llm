# No-Prefix Full Prefill CUDA Graph 设计

| 项目 | 内容 |
| --- | --- |
| 状态 | Review 已通过，首版实现完成 |
| 实现状态 | 核心逻辑、算子门槛、真实 checkpoint 端到端验证与 exact-bucket A/B 基线已完成；待长期稳定性、qps=10 和 timeline 验证 |
| 设计基线 | `feature/beam-search-perf-bugfix-2026-08`（调研时 HEAD `de1b66746905`） |
| 目标后端 | FlashInferTRTLLMFMHAv2PrefillImpl，非 paged、no-prefix 路径 |
| 初始运行范围 | BF16、dense MHA/GQA、动态多请求纯 prefill batch，`1 <= B <= Bmax` |
| Graph 范围 | Python model forward 的完整 prefill 主干 |
| 明确不支持 | Prefix cache、Piecewise/Breakable Graph、Paged FMHA、MoE、MLA、prefill/decode mixed batch |

> 本文档是当前首版实现的设计基线。代码默认关闭，只有显式开启独立开关并满足 allowlist 时才创建 generative prefill runner。

## 1. 背景

RTP-LLM 已经有 Decode CUDA Graph、Embedding Prefill Graph、MTP Draft Prefill Graph 的基础设施，也已经在 TRT-LLM FMHA v2 算子测试中验证了 CUDA Graph replay 时更新 sequence metadata 的基本能力。

但是，普通生成模型的 prefill 当前不会进入 CUDA Graph：

- 普通 PyWrappedModel 只持有一个面向 decode 的 graph runner。
- CudaGraphRunner 的 prefill capture 只接受 embedding-style 或 MTP draft layout。
- 普通 prefill 在 canRun 阶段直接回退 eager。
- 现有 support_cuda_graph 判定只检查实现是否存在 prepare_cuda_graph，无法表达“仅支持 no-prefix Full Prefill Graph”。

本设计只处理一个明确 graph-safe 的 backend：

**FlashInferTRTLLMFMHAv2PrefillImpl 的 non-paged no-prefix 路径。**

下列观测到的 SM120 kernel 属于该路径：

~~~
fmha_v2_flash_attention_bf16_64_64_S_q_kv_64_causal_alibi_softmax_output_bf16_sm120_kernel_nl_tiled
~~~

它是 TRT-LLM FMHA v2 的 CONTIGUOUS_Q_KV GQA prefill kernel，不是 paged KV attention。它支持在固定 capture shape 下，通过稳定地址上的 cu_seqlens、seq_lens 和 KV 写入映射更新每次 replay 的请求信息。

## 2. 设计结论

本方案采用以下约束，不实现通用 Prefill CUDA Graph：

1. 只 allowlist FlashInferTRTLLMFMHAv2PrefillImpl 的 non-paged no-prefix 路径。
2. 第一阶段支持动态多请求纯 prefill batching；真实请求数可以在 `1..Bmax` 内变化，超过配置上限时回退 eager。
3. 每个 token bucket capture 一张完整的 prefill graph，不支持 Breakable/Piecewise Graph。
4. 每张 graph 使用固定的 request-slot capacity：`Bmax` 个真实 request slots 加一个 padding sentinel slot。未使用的真实 slots 以零长度 sequence 填充。
5. sentinel 的 KV 写入只能落入 graph runner 专用 scratch KV blocks，不能触碰调度器分配给真实请求的 KV blocks。
6. 普通模型同时保留 decode graph runner 和新增的 prefill graph runner，两者不能共用 graph state。
7. 任何非零 prefix、backend 不匹配、shape 不匹配或 scratch 资源不足，都在 capture/replay 前回退 eager。
8. 该功能使用独立开关，默认关闭；不能因为配置了 PREFILL_CAPTURE_CONFIG 就隐式打开。

### 2.1 动态多请求 batching 的可行性结论

**可以实现，并且不需要为每个实际 batch size 分别 capture 一张 graph。** 当前分支、SGLang 和 vLLM 的代码给出了三层证据：

1. 当前分支的 `NormalModelInputGatherer::processContextStreams` 已经能把多个 context requests flatten 成 packed tokens、逐请求 `input_lengths/prefix_lengths`、`cu_seqlens` 和二维 KV block table。调度器与 gatherer 不需要为 CUDA Graph 改写多请求 batch 语义。
2. 当前 `CudaGraphRunner::prepareAttentionInputs` 已有固定 capacity buffer、按本次真实 batch size 刷新 metadata、清零 request tail，以及把较小 live block table 拷入 captured table 的能力。缺少的主要是 generative prefill 专用的 slot 归一化、sentinel KV mapping 和独立 runner/state。
3. TRT-LLM FMHA v2 CUDA Graph UT 已扩展为固定 `Bmax+1` slots 的生产 contract：capture `[0,0,0,0,64]`，连续 replay `[24,32,0,0,8]`、`[64,0,0,0,0]`、`[32,32,0,0,0]` 和 `[16,16,16,16,0]`，并校验真实 RoPE、真实 BF16 KV 写入和 sentinel scratch isolation。

本地参考实现也采用固定 request axis：

- SGLang Full Prefill CUDA Graph 按 aggregate `num_tokens` 选图，以固定 `full_prefill_max_req` 作为 request-slot capacity，实际请求数不足时将剩余 slots 置为零长度。因此 graph 数量不随实际 batch size 线性增长。
- vLLM Full CUDAGraph 同时 pad token capacity 和 request rows，并为 padded rows/tokens 提供 null/invalid KV mapping。它验证了动态逻辑 batch 可以映射到固定 capture shape；RTP-LLM 的 packed TRT FMHA contract 不使用 invalid token slots，因此本设计改用一个显式 sentinel sequence 和 scratch KV。

SGLang 是 Full Prefill 固定 request-axis 的直接参考；vLLM 在这里是 full-graph shape/KV padding 的工程参考，不代表 vLLM 的所有 prefill attention backends 都支持 Full CUDAGraph。RTP-LLM 仍必须以自身目标 backend 的 operator gate 为准。

因此首版的边界不是“只支持一个真实请求”，而是“支持配置上限内的动态多请求纯 prefill batch；固定 capture slot 数，动态更新各 slot 长度和 KV mapping”。

本文中的 `B` 精确定义为 `input_lengths.size(0)`，也就是 `stream_groups.totalContextBatchSize()` 展开后的 backend context sequence rows 数，不等同于 frontend RPC 数。一个 stream 的 `currentBatchSize()` 大于 1（例如 beam 展开）时会占用多个真实 slots，也必须计入 `Bmax`。

本次调研快照与主要证据位置如下：

| 代码库 | 调研版本 | 主要证据 |
| --- | --- | --- |
| RTP-LLM | `feature/beam-search-perf-bugfix-2026-08@de1b66746905` | `NormalModelInputGatherer::processContextStreams`、`CudaGraphRunner::prepareAttentionInputs`、`test_trtllm_fmha_v2_prefill.py::test_cuda_graph` |
| SGLang | `43226af8121f20de0e217bc5c344fe04e746397f` | `cuda_graph_config.py::full_prefill_max_req`、`PrefillCudaGraphRunner::_capture_req_slots/can_replay_locally/load_batch` |
| vLLM | `a02cfccbc6187344325e364f09f6d8c33c4b253b` | `CudagraphDispatcher` 的 padded `BatchDescriptor`、`gpu_model_runner.py` 的 padded request rows、null block 和 `slot_mapping=-1` |

### 2.2 可实现性判断与验证门槛

结论是**工程上可实现，且 operator gate 已通过；但在真实 checkpoint 的端到端正确性和性能验证完成前仍不能默认开启**。算子门槛包括：

1. 固定 `Bcap` capture，跨 replay 改变真实 B、每个 `Li` 和 block-table rows。
2. 打开目标模型一致的 RoPE，使用真实 BF16 KV cache，并让 padding sentinel 写入独立 scratch blocks。
3. 比较每个真实请求的 attention output 与 KV 内容，覆盖 `B=1`、`B=Bmax`、相同 T 不同 layout、非零/零长度 sentinel，以及连续交替 replay。
4. graph replay 后执行一次真实 decode，确认每个请求只消费自身 prefill KV；这一项与完整模型输出一致性一起放在 checkpoint 端到端验证阶段完成。

如果该 gate 发现 fused RoPE/KV writer 无法在固定 slots 下安全刷新 mapping，则首版不能绕过检查强开 Full Graph，应将该 backend capability 保持为 `NEVER`。如果 gate 通过，后续主要工作属于 runner/state、scratch ownership 和模型路由改造，不存在需要修改 scheduler batching 语义的结构性阻塞。

## 3. 目标与非目标

### 3.1 目标

- 对明确 graph-safe 的 no-prefix TRT-LLM FMHA v2 backend capture 完整 prefill model forward。
- 首版支持动态多请求纯 prefill batching，实际请求数和每个请求的 sequence length 可以在 profile capacity 内变化。
- 保证每次 replay 可以更新 input tokens、真实 sequence length、RoPE metadata 和 KV cache block mapping。
- 支持真实 token 数落在 capture bucket 以内，而不是只支持与 bucket 完全相等的 prompt。
- 保证 CUDA Graph replay 后生成结果、logits/hidden states 和 KV cache 内容与 eager 路径一致。
- 不影响已有 Decode CUDA Graph、Embedding Prefill Graph 和 MTP Draft Prefill Graph。
- 所有不满足条件的请求可观测地回退 eager，不进行运行中 recapture。

### 3.2 非目标

- Prefix cache、prefix reuse 或任何 prefix_lengths 大于 0 的请求。
- Breakable CUDA Graph、Piecewise CUDA Graph 或 attention 前后分段 capture。
- Paged TRT-LLM FMHA v2、FlashInfer paged prefill、TRTLLMGen、MLA、Sparse、HeadWise、Context Parallel。
- prefill 与 decode request 混合在同一次 model forward 的 mixed batch。
- `enable_layer_micro_batch=1` 的 `forward_micro_batch` 路径；首版支持 scheduler 形成的动态 pure-prefill batch，但不再做 layer micro-batch 拆分。
- MoE、multimodal/MRoPE、LoRA、PD disaggregation、cache-store 等尚未完成 graph-safety 审计的模型路径。
- TP 大于 1 或 capture NCCL collective；首版限定 TP=1。
- FP8 model/KV cache；首版使用 BF16 model 和 BF16 KV cache。
- 将 post layers、采样、scheduler、tokenizer 或 RPC 纳入 CUDA Graph。

## 4. “Full Prefill Graph”的边界

本设计中的 Full Prefill Graph 指一次 py_model.forward 的完整生成模型 prefill 主干，包括：

~~~
input embedding
    -> all transformer layers
    -> prefill attention and KV cache write
    -> model forward 返回的 hidden states
~~~

以下工作仍在 graph 外：

- scheduler 组 batch；
- host metadata 构造；
- H2D metadata 更新；
- graph eligibility 判断；
- graph bucket 选择；
- prepare_cuda_graph；
- C++ post-layer/postprocess；
- logits/sampling；
- tokenizer 和 RPC。

因此，本文的 Full 表示“不拆分 transformer forward”，不表示整个端到端请求都被 capture。

## 5. Backend 能力模型

### 5.1 问题

当前 Python attention base class 的 support_cuda_graph 只判断 prepare_cuda_graph 是否可调用。这个条件对 Full Prefill Graph 过宽：

- 某个 backend 可以支持 decode graph，但不一定支持 prefill graph。
- 某个 prefill backend 可以支持 no-prefix，但不一定支持 prefix。
- “能够更新 metadata”不等于“完整 model forward 中所有 shape、分配和 KV 写入都安全”。

### 5.2 新能力接口

引入显式 capability，而不是根据类名或是否存在方法进行推断：

~~~
PrefillCudaGraphCapability:
    NEVER
    FULL_NO_PREFIX
~~~

Attention implementation 默认返回 NEVER。首版只有 FlashInferTRTLLMFMHAv2PrefillImpl 在满足自身 support 条件时返回 FULL_NO_PREFIX。

attention factory 接收明确的 graph selection mode：

~~~
EAGER
DECODE_GRAPH
FULL_PREFILL_NO_PREFIX_GRAPH
~~~

在 FULL_PREFILL_NO_PREFIX_GRAPH 模式下，factory 必须同时满足：

- capability 等于 FULL_NO_PREFIX；
- has_prefix 为 false；
- backend 的常规 support 检查通过；
- non-paged 路径；
- 当前硬件、dtype 和 RoPE 配置在首版 allowlist 内，head dimension 只需通过 backend 的常规 support 检查。

不得回退选择另一个“也有 prepare_cuda_graph 方法”的 attention implementation。找不到完全匹配的实现时，禁用该 prefill graph profile 并报告原因。

### 5.3 首版 allowlist

| 维度 | 首版条件 |
| --- | --- |
| Python implementation | FlashInferTRTLLMFMHAv2PrefillImpl |
| Attention layout | non-paged，MHA 或 GQA |
| Prefix | 所有 prefix_lengths 必须为 0 |
| Dtype | BF16 input/output，BF16 KV cache |
| CUDA | 不低于 backend 现有最低要求 |
| GPU | 首先验证 SM120；SM90 通过同等测试后再加入 enable matrix |
| Model | dense causal decoder-only |
| Parallelism | TP=1、无 CP |
| Layer micro batch | 关闭；当前 `forwardMicroBatched` 路径不经过 CUDA Graph runner |
| Real prefill batch | 动态 `1..FULL_PREFILL_CUDA_GRAPH_MAX_REQUESTS`，只允许 pure prefill batch |

allowlist 按“完成测试的平台配置”逐项扩展，不能仅依据 TRT-LLM FMHA v2 的理论 support 范围自动开放。

Head dim 不作为 Full Prefill Graph 的额外 allowlist 维度。FMHA、RoPE 和 KV writer 对 head dim 的约束统一由目标 backend 的常规 `support()` 检查负责；CUDA Graph 不增加额外限制。

## 6. 为什么可以按 total token 选图，但 request axis 必须固定

TRT-LLM FMHA v2 capture 时会固化以下 launch 相关信息：

- batch size；
- max_q_len/max_kv_len 上界；
- Q/K/V tensor capacity；
- kernel grid；
- workspace 和 tensor 地址。

相同 total token 可能有完全不同的 batch layout，例如：

~~~
[512]
[256, 256]
[128, 128, 128, 128]
~~~

这些 layout 不能以不同的 runtime tensor shape 直接命中同一张 graph，但可以先归一化为同一个固定 capture contract：

~~~
total token capacity = Tg
captured request slots = Bcap = Bmax + 1
~~~

其中 `Bmax` 个 slots 是真实请求容量，额外一个 slot 是 token padding sentinel。实际 batch size `B` 小于 `Bmax` 时，将未使用的真实 slots 写成零长度 sequence；实际 token 数 `T` 小于 `Tg` 时，由 sentinel 承担 `Tg-T` 个 padding tokens。

当前 TRT-LLM FMHA v2 实现的 `prepare()` 会在 capture 时固化 Python 侧 `batch_size` 和 `max_q_len/max_kv_len`，而 `prepare_cuda_graph()` 只原地刷新 `seq_lens`、RoPE/KV offset 等内容。因此 replay 时不能真的改变 backend batch shape，但可以改变固定 slots 中哪些是有效请求，以及各 slot 的长度。

首版 capture 固定：

~~~
Bmax 个真实 request slots + 1 个 padding sentinel slot
max_q_len = max_kv_len = Tg
~~~

这样 `[512]`、`[256,256]` 和 `[128,128,128,128]` 在 `Bmax >= 4` 时都可以命中同一个 `Tg=512` profile，graph key 仍只使用 token bucket。动态 batch 不会把 graph 数量从 `|token buckets|` 放大成 `|token buckets| x |batch sizes|`。

该方案的代价是 backend 始终以 `Bcap` 个 slots capture，且 `max_q_len` 采用保守的 `Tg` 上界。首版先通过 `FULL_PREFILL_CUDA_GRAPH_MAX_REQUESTS` 控制该开销；如果 benchmark 证明固定大 request capacity 明显降低 kernel 效率，再引入少量 `(Tg, Bg)` sparse profiles，而不是为所有组合 capture 笛卡尔积。

## 7. 固定 bucket 与 padding sentinel

### 7.1 Graph profile

继续复用 PREFILL_CAPTURE_CONFIG 表达 token buckets。例如：

~~~
PREFILL_CAPTURE_CONFIG=128,256,384,512,768,1024
~~~

每个 bucket Tg capture 一张 graph。首版 profile 的固定 layout 是：

| 项目 | Capture 值 |
| --- | --- |
| 总 token capacity | Tg |
| backend batch size | `Bcap = Bmax + 1` |
| real request slots | `Bmax` |
| sentinel slots | 1 |
| max_q_len | Tg |
| max_kv_len | Tg |
| prefix lengths | 长度为 `Bcap`，全部为 0 |

### 7.2 Replay layout

对于由 `B` 个真实请求组成的 pure prefill batch：

~~~
real_lengths = [L0, L1, ..., L(B-1)]
T = sum(real_lengths)
1 <= B <= Bmax
~~~

选择满足 `Tg >= T` 的最小 bucket。固定 `Bcap` slots 的 replay metadata 构造为：

~~~
input_lengths = [L0, ..., L(B-1), 0, ..., 0, Tg-T]
                 |------ Bmax real slots -----| |sentinel|
prefix_lengths = zeros(Bcap)
cu_seqlens = prefix_sum_with_leading_zero(input_lengths)，最后一个值恒为 Tg
~~~

sentinel 固定使用 slot `Bmax`，而不是随 `B` 改变位置；真实 slots `B..Bmax-1` 为零长度。token buffer 的前 `T` 个位置保持 gatherer 的真实 packed input_ids 顺序，后 `Tg-T` 个位置填固定合法 pad token。由于每个真实请求和 sentinel 都是独立 causal sequence，sentinel token 不会参与真实 prompt 的 attention，也不会改变前 `T` 个真实 token 的输出。

例如 `Bmax=4`、真实长度为 `[24,32]`、`Tg=64` 时：

~~~
input_lengths = [24, 32, 0, 0, 8]
cu_seqlens = [0, 24, 56, 56, 56, 64]
~~~

当 `T == Tg` 时，sentinel 是合法的零长度 sequence。SM120 FMHA v2 算子测试已经覆盖固定总 token、固定 slot capacity、动态 `B=1/2/4`、动态 sequence layout、零长度 slots、真实 RoPE、真实 KV cache 写入和 sentinel 为零/非零的两种情况。

### 7.3 为什么不把 padding 追加到真实请求

如果把 padding token 直接当作真实请求的尾部 token：

- attention 虽然是 causal，前部输出可能保持正确；
- 但 KV writer 会把 padding KV 写入真实请求后续位置；
- scheduler 的真实 KV block 分配不一定覆盖 bucket 上界；
- decode 阶段可能观察到被污染的 KV bookkeeping。

因此 padding 必须是独立 sentinel sequence，并使用独立 scratch KV mapping。

## 8. Scratch KV 资源设计

### 8.1 资源所有权

`PyWrappedModel` 在创建 prefill graph runner 前，从 KVCacheManager 为每个 KV cache group/tag 预留专用 scratch blocks，并将稳定的 kernel block IDs 交给 runner：

~~~
scratch_blocks[tag] =
    ceil(max(PREFILL_CAPTURE_CONFIG) / tokens_per_block[tag])
~~~

这些 blocks：

- 生命周期由 `PyWrappedModel` 持有，晚于 prefill graph runner 释放；
- 不进入 scheduler 的可分配 free list；
- 不计入任何真实请求的 block table；
- 在对应 KV group 的 block table 中只提供给 sentinel slot；
- runner 析构时归还；
- 预留失败时整体禁用 Full Prefill Graph，不允许借用真实请求 blocks。

同一组最大容量 scratch blocks 可以被所有 token buckets 复用，因为首版 graph replay 维持串行执行，不允许多个 prefill graph 同时写入 sentinel 区域。

### 8.2 Replay KV mapping

每次 replay 前更新稳定地址上的 block table：

- slots `0..B-1` 分别指向 scheduler 为各真实请求分配的 KV blocks；
- slots `B..Bmax-1` 是零长度 inactive slots，其 table rows 清零或填明确的 invalid block id；
- slot `Bmax` 固定指向 runner-owned scratch blocks；
- 未使用的 table tail 清零或填入明确的 invalid block id；
- graph 内不得读取 host pointer，也不得动态分配 block table。

### 8.3 并发约束

首版要求同一个 model instance 的 prefill graph replay 串行化，并沿用现有 forward event/stream 同步契约。

如果以后允许多 stream 并发 replay，则必须改为：

- 每个 in-flight replay 独占 scratch block set；或
- 通过 scratch pool 获取资源并在 graph 完成事件后归还。

在完成该改造前，不能通过放松 event/mutex 来并行 replay。

## 9. 静态与动态数据契约

CUDA Graph 内所有 tensor 地址和影响 launch topology 的 shape 必须稳定。内容可以在 replay 前原地更新。

| 数据 | Capture 后是否固定 | Replay 前动作 |
| --- | --- | --- |
| input_ids buffer 地址与 shape `[Tg]` | 固定 | 前 T 个 copy，tail 填 pad token |
| hidden/QKV/output capacity | 固定 | graph 内覆盖 |
| backend batch size=`Bcap` | 固定 | 不变；实际真实请求数单独记录为 B |
| max_q_len/max_kv_len=`Tg` | 固定 | 不变 |
| input_lengths buffer 地址 `[Bcap]` | 固定 | 写入 B 个真实长度、inactive zeros 和 sentinel padding length |
| prefix_lengths buffer 地址 `[Bcap]` | 固定 | 始终全部写 0 |
| cu_seqlens buffer 地址 `[Bcap+1]` | 固定 | 对固定长度 input_lengths 做 cumulative sum，末值为 Tg |
| seq_lens buffer 地址 `[Bcap]` | 固定 | `prepare_cuda_graph` 原地刷新 |
| position/rope metadata 地址 | 固定 | 按 B 个真实 sequence、inactive slots 和 sentinel layout 原地刷新 |
| KV block table 地址与 capacity | 固定 | 写入 real + scratch mapping |
| KV cache base pointer | graph 生命周期固定 | 不变 |
| workspace 地址 | 固定 | 不变 |
| output capacity [Tg,...] | 固定 | graph 后只消费前 T 个真实输出 |

`lm_output_indexes`、request ids、post-layer batch size 等 scheduler-visible metadata 继续保持真实 B，不扩成 `Bcap`。sentinel 只存在于 graph runner 的 attention metadata 和 token/KV scratch 空间，不能进入 post layer、采样、监控或响应构造。

禁止事项：

- replay 热路径创建新 CUDA tensor；
- 替换上述 tensor 的 storage；
- 在 graph 内根据 T 进行 host 分支；
- replay 时改变 batch size、max_q_len 或 tensor rank；
- 用 .item() 等 device-to-host 同步判断 eligibility。

prefix eligibility 使用 scheduler 已在 host 侧持有的 metadata 判断，必须在任何 graph state 修改前完成。

实现已拆分 live batch 与 captured capacity 语义，并显式记录：

~~~
real_request_count = B
real_token_count = T
graph_token_capacity = Tg
graph_request_capacity = Bmax
captured_backend_batch_size = Bcap
~~~

不能把 `Bcap` 伪装成 scheduler 的真实 batch size，否则 output index、KV accounting 和 post layer 都可能错误地把 sentinel 当成业务请求。

## 10. Runner 架构

### 10.1 显式 Graph Role

当前 runner 通过 is_prefill_cuda_graph_mode 和 num_tokens_per_bs 推断 graph 用途，容易把 embedding、MTP draft 和普通 prefill 混淆。

改为显式 role：

~~~
CudaGraphRole:
    DECODE
    TARGET_VERIFY
    EMBEDDING_PREFILL
    MTP_DRAFT_PREFILL
    GENERATIVE_PREFILL_NO_PREFIX
~~~

capture、canRun、profile key 和 metadata preparation 都基于 role 分派，不再用 num_tokens_per_bs 猜测模式。

### 10.2 普通模型使用双 runner

普通生成模型同时持有：

~~~
decode_graph_runner
decode_graph_state

prefill_graph_runner
prefill_graph_state
~~~

路由规则：

~~~
attention_inputs.is_prefill == false
    -> decode_graph_runner

attention_inputs.is_prefill == true
    -> prefill_graph_runner
       -> eligibility 失败则 eager
~~~

两个 runner 必须有独立的：

- graph map；
- input buffers；
- graph state；
- capture/replay event；
- CUDA graph memory pool（首版）。

首版不共享 activation graph pool，优先保证 decode 和 prefill graph 的地址及并发安全。确认模型执行严格串行并完成显存收益评估后，才考虑共享 pool。

### 10.3 Graph state 不能共享

decode 和 prefill 的 input layout、batch semantics、output slicing、attention metadata 完全不同。共享 CudaGraphState 会导致以下风险：

- prefill replay 覆盖 decode 的 KV block table；
- canRun 使用错误的 current_batch_size；
- output tensor 指向另一种 shape 的 captured storage；
- event 只保护其中一个 graph。

因此不能通过“临时把现有单 runner 切换成 prefill mode”实现。

## 11. Capture 流程

启动阶段按 bucket 从大到小 capture，以复用现有 capture 过程并控制内存峰值：

1. 解析并校验 PREFILL_CAPTURE_CONFIG。
2. 检查 Full Prefill Graph 独立开关。
3. 检查模型级静态条件：dense、BF16、TP=1、无 CP/LoRA/multimodal 等。
4. 预留最大 bucket 所需的 sentinel scratch KV blocks。
5. 为 bucket Tg 构造固定 `Bcap=Bmax+1` slots 的 capture input；使用 `[0,...,0,Tg]`，即固定最后一个 sentinel slot 作为保守的 `max_q_len=Tg` capture layout，并将非零 dummy sequence 映射到 scratch KV。
6. 让 attention factory 以 FULL_PREFILL_NO_PREFIX_GRAPH mode 选择实现。
7. 验证最终选择的每个 attention implementation capability 均为 FULL_NO_PREFIX。
8. warmup，确保 lazy workspace、JIT kernel 和临时 buffer 已完成初始化。
9. capture py_model.forward。
10. 保存 graph、稳定输入输出 buffers、backend instance 和 profile metadata。
11. capture 结束后执行一次 eager/captured 对照自检；失败则删除该 profile。

如果模型包含多个 attention tag/group，首版要求它们全部解析为同一个 allowlisted backend；否则禁用 prefill graph，避免部分 layer graph-safe、部分 layer 静默使用其他实现。

capture 失败只能影响 prefill graph。已有 decode graph 仍应正常初始化和运行。

## 12. Replay 流程

一次 pure prefill batch 进入模型后按以下顺序执行：

1. 判断请求是否为 prefill。
2. 在 host 侧执行 eligibility 检查。
3. 读取真实请求数 B 和每个请求长度，要求 `1 <= B <= Bmax`，并计算 `T=sum(Li)`。
4. 要求所有 prefix lengths 等于 0。
5. 选择最小的 Tg 大于等于 T 的 bucket。
6. 检查 padding ratio 是否在阈值内。
7. 等待该 runner 上一次 replay 的 completion event。
8. 将 B 个请求的 packed input_ids copy 到固定 buffer，tail 填 pad token。
9. 写入固定 `Bcap` 容量的 input lengths、全零 prefix lengths 和 cumulative `cu_seqlens`；inactive slots 为零长度，sentinel 长度为 `Tg-T`。
10. 更新 B 个真实 request rows、inactive rows 与 sentinel scratch row 的 KV block mapping。
11. 在 graph 外调用 attention prepare_cuda_graph，原地刷新 seq_lens、RoPE 和 KV offset。
12. launch graph。
13. 记录 completion event。
14. 仅把前 T 个位置作为真实 hidden-state output 交给后续 post layer。

任何一步验证失败都在 launch 前回退 eager。已经开始 graph replay 后不允许中途切换路径。

## 13. Graph 选择与回退

### 13.1 Bucket 选择

候选 bucket 为：

~~~
Tg = lower_bound(PREFILL_CAPTURE_CONFIG, T)
~~~

以下情况回退 eager：

- 没有大于等于 T 的 bucket；
- 真实请求数 B 超过 `FULL_PREFILL_CUDA_GRAPH_MAX_REQUESTS`；
- profile capture 失败或未完成；
- padding ratio 超过阈值；
- scratch KV capacity 不足。

建议增加：

~~~
FULL_PREFILL_CUDA_GRAPH_MAX_PADDING_RATIO=0.25
~~~

计算方式：

~~~
(Tg - T) / Tg
~~~

该阈值避免短 prompt 命中过大的 graph，导致 padding compute 抵消 kernel launch 收益。首版默认值已确认为 0.25，后续仍应通过 benchmark 校准。

### 13.2 Fallback reason

回退必须带结构化 reason，至少包括：

| Reason | 含义 |
| --- | --- |
| feature_disabled | 独立开关未开启 |
| unsupported_model | 模型结构不在 allowlist |
| unsupported_backend | attention implementation capability 不匹配 |
| prefix_nonzero | 任一 prefix length 非零 |
| mixed_prefill_decode_batch | 同一次 model forward 同时包含 prefill 和 decode requests |
| layer_micro_batch_enabled | 当前执行走 `forward_micro_batch`，不进入首版 Full Prefill Graph |
| request_capacity_exceeded | 真实 prefill request 数超过 capture capacity |
| bucket_miss | 无可用 token bucket |
| padding_ratio_exceeded | padding 比例过高 |
| scratch_kv_unavailable | scratch blocks 不足 |
| profile_not_ready | capture 失败或 profile 不完整 |
| metadata_mismatch | replay metadata 超出 capture contract |

不允许把异常吞掉后只打印通用“cannot run graph”日志。

## 14. 配置接口

新增独立开关，默认关闭：

~~~
ENABLE_FULL_PREFILL_CUDA_GRAPH=0
~~~

依赖关系：

~~~
ENABLE_CUDA_GRAPH=1
ENABLE_FULL_PREFILL_CUDA_GRAPH=1
PREFILL_CAPTURE_CONFIG=<非空 bucket 列表>
FULL_PREFILL_CUDA_GRAPH_MAX_REQUESTS=<正整数>
~~~

推荐首轮验证配置：

~~~
ENABLE_CUDA_GRAPH=1
ENABLE_FULL_PREFILL_CUDA_GRAPH=1
PREFILL_CAPTURE_CONFIG=128,256,384,512,768,1024
FULL_PREFILL_CUDA_GRAPH_MAX_REQUESTS=8
FULL_PREFILL_CUDA_GRAPH_MAX_PADDING_RATIO=0.25
~~~

约束：

- ENABLE_FULL_PREFILL_CUDA_GRAPH 不影响 decode graph 开关。
- PREFILL_CAPTURE_CONFIG 为空时，不创建 generative prefill runner。
- `FULL_PREFILL_CUDA_GRAPH_MAX_REQUESTS` 固定每个 profile 的真实 request-slot capacity；首版所有 token buckets 共用同一个 Bmax，因此 graph 数量仍等于 token bucket 数量。
- 配置错误时禁用 prefill graph 并给出明确日志；不影响服务 eager 启动。
- 首版不能提供“允许 prefix”的隐藏开关。

`Bmax=8` 只是首轮验证建议，不是最终默认值。它需要按线上 prompt/batch 分布和 FMHA kernel 效率校准。若动态 batch 大于 Bmax，首版直接 eager fallback，不在运行期 recapture。

## 15. 已实现的代码边界

首版已按以下边界实现：

| 文件/模块 | 首版实现 |
| --- | --- |
| fmha_impl_base.py | 增加 PrefillCudaGraphCapability |
| attn_factory.py | 增加显式 graph selection mode 和 capability filter |
| cuda_impl/trt.py | 为 non-paged TRT-LLM FMHA v2 声明 FULL_NO_PREFIX，并加强运行时断言 |
| cuda_graph_base.h | 增加显式 CudaGraphRole |
| cuda_graph_runner.h/.cc | 增加 generative no-prefix prefill role、动态真实 batch state、eligibility、profile 和 fallback reason |
| cuda_graph_prefill.cc | 增加固定 request capacity、inactive slots 和 sentinel layout 的 capture/replay metadata 构造 |
| PyWrappedModel.h/.cc | 普通模型改为 decode/prefill 双 runner 和双 state 路由 |
| PyWrappedModel + KVCacheManager 接口 | 按 KV cache group/tag 预留、持有和释放 sentinel scratch blocks；不修改 allocator 本身 |
| server args | 增加独立 enable、最大真实请求数和 padding ratio 配置 |
| attention/cuda_graph tests | 增加动态真实请求数、真实 RoPE、真实 KV 写入和跨 replay metadata 测试 |
| cpp/cuda_graph tests | 增加多请求 replay metadata/sentinel contract 测试；完整 runner 通过模型集成目标编译链接 |

实现与验证按以下逻辑层次组织；提交前可据此整理 commits：

1. operator contract gate：动态 B/layout、RoPE、真实 KV 和 sentinel scratch UT，不改变生产路径；
2. capability 与配置，不改变默认行为；
3. scratch KV resource；
4. explicit role 与双 runner；
5. Full Prefill capture/replay；
6. correctness、timeline 和性能验证。

## 16. 正确性测试

### 16.1 Attention 算子级

扩展 TRT-LLM FMHA v2 prefill CUDA Graph 测试：

- BF16、SM120；
- MHA 和 GQA；
- 使用与目标模型一致的 RoPE，而不是 RoPE disabled；
- 使用真实 KV cache buffer 和 block table；
- capture total token 与 request-slot capacity 固定，replay 时 B、T 和各 `Li` 在同一 profile 内变化；
- 覆盖 capture `[32,8,8,8]` 后 replay `[24,32,0,0]` 类 layout，并补齐 RoPE/KV writer 验证；
- 覆盖相同 T、不同 B 与不同 length distribution，例如 `[64]`、`[32,32]`、`[16,16,16,16]`；
- 覆盖 T=Tg、T=Tg-1、bucket 下界附近和较高 padding；
- inactive slots 始终为零长度，sentinel slot 覆盖非零长度和零长度；
- 比较 eager/captured attention output；
- 比较真实请求前 T 个 KV cache；
- 验证真实请求 T 以后未发生越界写；
- 验证 sentinel 写入只落入 scratch blocks；
- 验证每个真实 request 的 KV 只写入自身 block-table row，不发生跨 request 污染。

### 16.2 Full model

使用 dense Qwen-like BF16 模型覆盖：

- eager 与 graph hidden states 数值一致；
- eager 与 graph logits 在既定 tolerance 内一致；
- greedy generated tokens 完全一致；
- 连续 replay 不同 prompt 长度，验证无 stale cu_seqlens/seq_lens；
- 连续 replay 在 `B=1..Bmax` 间变化，验证 inactive tail 清理完整；
- 连续 replay 相同 total token、不同请求数和长度分布；
- 连续 replay 不同真实 block ids，验证无 stale KV mapping；
- prefill graph 后立即 decode，验证 KV cache 可被正确消费；
- decode graph 和 prefill graph 交替运行；
- prefix 大于 0 时明确回退 eager；
- context batch 在 `1..Bmax` 内命中 graph，`B>Bmax` 时明确回退 eager；
- prefill/decode mixed batch 明确回退 eager；
- `enable_layer_micro_batch=1` 时保持现有 eager `forward_micro_batch` 路径；
- unsupported backend、bucket miss、padding 超限时明确回退；
- prefill capture 失败不影响 decode graph。

### 16.3 稳定性

- 至少进行 1000 次不同长度 replay；
- 在 compute-sanitizer 或等价检查下无非法访问；
- scratch blocks 不出现在 scheduler free/allocated accounting 中；
- runner 析构后 scratch blocks 正确归还；
- graph 生命周期内 KV cache base pointer 不变化；
- 不发生运行时 recapture 或 graph 数量持续增长。

### 16.4 当前验证记录（2026-08-18）

已完成：

- `//rtp_llm/cpp/cuda_graph/tests:cuda_graph_replay_contracts_test` 通过，覆盖固定 request capacity、inactive slots 和 sentinel metadata contract；
- `//rtp_llm/models_py/modules/factory/attention/cuda_impl/test:test_trtllm_fmha_v2_prefill_sm12x` 的 Full Prefill Graph case 通过，覆盖 BF16、SM120、真实 RoPE、真实 KV 写入、scratch isolation，以及 `[24,32]`、`[64]`、`[32,32]`、`[16,16,16,16]` 的跨 B/layout replay；
- `//rtp_llm/cpp/models/test:pywrapped_model_cache_store_integration_test` 通过，验证双 runner/scratch 资源相关生产代码可以完整编译链接；
- 通过 `bazelisk test //rtp_llm/test:server_test` 启动 `/home/silu.zsl/ckpt/algr_bs`（BF16、dense GQA、SM120），`seq_len=64` Full Prefill Graph capture、首次 replay 和数值自检成功；
- 同一条 `max_new_tokens=4`、greedy 请求分别在 eager-prefill 与 Full Prefill Graph 下返回完全一致的 4-token 结果；将 scheduler 配置为 `MAX_CONTEXT_BATCH_SIZE=4` 后，4 个并发请求被动态调度为 `B=1` 与 `B=3` 两次 Full Prefill Graph replay，4 个请求均成功返回，证明真实服务路径不限制为单请求。

服务测试 target 是常驻进程，完成请求验证后由测试端主动中断，因此该次 Bazel invocation 的最终状态为 interrupted，而不是断言失败。exact-bucket 短测基线见 17.2；长期 replay、compute-sanitizer、qps=10、padding/fallback 分布和 timeline 门槛仍按 16.3、17 节继续执行。

## 17. 性能与 Timeline 验证

功能正确后再执行性能验证，不能用性能结果替代正确性门槛。

### 17.1 Benchmark

对相同模型、相同输入分布分别测：

- eager prefill；
- decode graph only；
- decode graph + Full Prefill Graph。

分别记录：

- model forward latency；
- TTFT；
- 端到端 RT；
- graph metadata preparation；
- H2D copy；
- graph replay；
- padding token 数和 padding ratio；
- capture 时间；
- graph/scratch 显存增量。

至少覆盖：

- bucket exact hit；
- 小幅 padding；
- 接近 padding ratio 阈值；
- 超过阈值的 eager fallback；
- 相同 total token 下的 `B=1`、中等 B、`B=Bmax`；
- 线上代表性的请求长度分布与 request-slot occupancy；
- qps=10 的稳定压测。

### 17.2 当前 A/B 实测（2026-08-18）

实测环境为 `/home/silu.zsl/ckpt/algr_bs`（BF16 dense GQA、TP=1）和 NVIDIA RTX PRO 5000 72GB Blackwell（SM120）。A/B 两组都开启 decode CUDA Graph、关闭 KV reuse，仅切换 `ENABLE_FULL_PREFILL_CUDA_GRAPH`。`PREFILL_CAPTURE_CONFIG=64,128,256,512,1024`、`Bmax=4`、padding ratio 阈值为 0.25。请求使用 `max_new_tokens=1`，每个点预热 20 次后采集 100 次，下表均为 exact bucket hit，不包启动 capture/self-check 开销。

`B=4` 通过 `/batch_infer` 原子入队，避免 4 个独立 HTTP 请求被 scheduler 拆成 `B=1+B=3`；TTFT 取该 batch 中最后完成请求的 p50，E2E 为整个 HTTP batch 的 wall-time p50。

| 形态 | Total tokens | Eager TTFT p50 | Graph TTFT p50 | TTFT 收益 | Eager E2E p50 | Graph E2E p50 | E2E 收益 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B=1` | 64 | 3.468 ms | 2.353 ms | 32.15% | 4.775 ms | 3.668 ms | 23.18% |
| `B=1` | 256 | 4.089 ms | 3.493 ms | 14.58% | 5.589 ms | 5.077 ms | 9.16% |
| `B=1` | 1024 | 12.654 ms | 12.047 ms | 4.80% | 14.966 ms | 14.435 ms | 3.55% |
| `B=4`, each=16 | 64 | 3.474 ms | 2.393 ms | 31.12% | 4.762 ms | 3.652 ms | 23.31% |
| `B=4`, each=64 | 256 | 3.871 ms | 3.286 ms | 15.11% | 5.341 ms | 4.761 ms | 10.86% |
| `B=4`, each=256 | 1024 | 11.489 ms | 10.855 ms | 5.52% | 13.793 ms | 13.143 ms | 4.71% |

结论：CUDA Graph 节省的主要是 Python/逐层 kernel launch 固定开销，因此 total token 越小，相对收益越大；到 1024 token 时 GPU 计算占比上升，TTFT 收益收敛到约 5%。所有点的 TTFT p99 仍为正收益；`B=1,total=1024` 的 HTTP E2E p99 受单个客户端/前端抖动影响出现回退，因此还不能以这轮短测代替 qps=10 长稳压测和 timeline 门槛。

4 个独立 HTTP 并发请求在当前 scheduler 下会常被拆成多个 prefill batch。此时每个子 batch 独立选 bucket 并执行 padding-ratio fallback，其结果不能用来作为固定 `B=4` 的 graph 微基准，但应在后续线上请求分布压测中保留，用于评估真实 graph hit/fallback 比例。

### 17.3 Timeline 预期

目标 timeline 中：

- py_model.forward 主干由一次 CUDA Graph launch 提交；
- FMHA v2、GEMM、RoPE/KV write 等 kernel 位于该 graph 内；
- graph 外只保留 metadata preparation、copy、post layer 等明确工作；
- 不应在 replay 中看到 Python 逐层 kernel launch；
- 不应每个请求重新 capture、JIT 或分配 workspace。

采集 timeline 时只保留足够分析的短窗口，避免 profiler 本身显著干扰 RT。

### 17.4 Enable 门槛

建议首版达到以下条件后才允许默认业务配置开启：

- 所有正确性和 KV safety case 通过；
- eligible 请求的 model-forward latency 有稳定正收益；
- TTFT p50 目标提升不低于 5%，否则继续保持实验开关；
- TTFT/RT p99 相比 eager 不回退超过 2%；
- eager fallback 路径开销回退不超过 1%；
- capture 和 scratch 显存成本有明确上限并写入启动日志。

具体性能阈值可在 Review 时调整，但 correctness、KV isolation 和无动态 recapture 是硬门槛。

## 18. 可观测性

当前首版核心逻辑已提供 capture/scratch/enable 日志、限频的 replay 命中日志，以及按 reason 限频的 fallback 日志。下列完整 KMonitor metrics 在默认业务开启前继续补齐：

新增或复用 metrics：

- full_prefill_graph_capture_total；
- full_prefill_graph_capture_failure_total；
- full_prefill_graph_replay_total；
- full_prefill_graph_fallback_total，按 reason 分类；
- full_prefill_graph_selected_bucket；
- full_prefill_graph_padding_tokens；
- full_prefill_graph_padding_ratio；
- full_prefill_graph_real_request_count；
- full_prefill_graph_request_capacity；
- full_prefill_graph_request_slot_occupancy；
- full_prefill_graph_prepare_latency_us；
- full_prefill_graph_replay_latency_us；
- full_prefill_graph_capture_memory_bytes；
- full_prefill_graph_scratch_kv_blocks。

启动日志必须打印：

- 功能是否开启；
- backend capability；
- 实际 enable matrix；
- 成功/失败的 buckets；
- 固定的 request-slot capacity；
- 每个失败 profile 的具体原因；
- scratch KV 预留数量；
- graph 显存开销。

热路径日志需要限频，不能因 fallback 每请求刷日志。

## 19. 风险与缓解

| 风险 | 缓解措施 |
| --- | --- |
| padding KV 污染真实请求 | 独立 sentinel slot 和 runner-owned scratch blocks |
| total token 相同但 batch layout 不同 | 固定 `Bcap`，每次 replay 原地刷新全部 lengths/cu_seqlens/block-table rows；跨 B/layout UT |
| inactive slot 残留上次 KV mapping | replay 前清零全部 inactive rows，零长度 slot UT 与交替 batch replay |
| 固定 Bmax/max_q_len 使 kernel 过度保守 | 首版配置较小 Bmax 并 benchmark；有明确数据后才增加 sparse `(Tg,Bg)` profiles |
| prefix 请求误入 graph | host eligibility 前置检查，capability 只允许 FULL_NO_PREFIX |
| backend factory 选到另一个实现 | 显式 graph selection mode，所有 layer capability 必须匹配 |
| decode graph 被 prefill state 覆盖 | 双 runner、双 state、独立 event 和 pool |
| padding 计算抵消收益 | 最小 bucket + padding ratio 阈值 + metrics |
| capture 显存过高 | buckets 显式配置、逐 profile 记录内存、默认关闭 |
| stale metadata | 所有 metadata 原地刷新，交替长度/block id 重放测试 |
| sentinel scratch 并发写冲突 | 首版 replay 串行；并发前引入 scratch pool |
| operator UT 与真实模型存在差距 | 补齐 RoPE、真实 KV writer、prefill-to-decode 端到端测试 |

## 20. 后续扩展

以下能力不属于首版，也不能在首版实现中预留未经验证的隐式路径：

1. Request-capacity/max-q-len sparse profiles：只有固定 Bmax 或保守 `max_q_len=Tg` 被实测为瓶颈时，才增加少量显式 profile，避免笛卡尔积爆炸。
2. Prefix backend：需要 paged attention、动态 block table、prefix length 和 KV reuse 的独立 graph-safety 设计。
3. SM90：按硬件配置补齐测试后扩展 allowlist。
4. TP/NCCL capture：需要 collective graph capture 和多 rank 一致性验证。
5. MoE：需要 routing、capacity、通信和动态 shape 审计。
6. graph pool 共享：只有证明 decode/prefill 不并发且地址安全后才实施。

## 21. Review 清单

首版实现采用以下已确认决策：

- [x] 首版只支持 FlashInferTRTLLMFMHAv2PrefillImpl non-paged no-prefix。
- [x] 首版支持 `1..Bmax` 个动态真实 prefill 请求，固定 `Bmax` 个真实 slots 并使用一个 sentinel slot。
- [x] 为 sentinel 预留最大 bucket 对应的 scratch KV blocks。
- [x] 普通模型使用 decode/prefill 双 runner，而不是改造现有单 runner 复用 state。
- [x] Full Graph 边界为 py_model.forward，不包含 post layer/sampling。
- [x] 使用独立 ENABLE_FULL_PREFILL_CUDA_GRAPH 开关且默认关闭。
- [x] PREFILL_CAPTURE_CONFIG 继续表达一维 token buckets。
- [x] 首版所有 buckets 共用 `FULL_PREFILL_CUDA_GRAPH_MAX_REQUESTS`，graph 数量不乘以 batch size。
- [x] padding ratio 阈值默认值为 0.25。
- [x] 首个 enable matrix 限定 BF16、TP=1、dense、SM120；head dim 沿用 backend 的常规 support matrix。
- [x] Prefix、prefill/decode mixed batch、Paged backend 和 Breakable Graph 均不进入首版。

真实 checkpoint 的 capture、动态多请求 replay、decode-after-prefill 与 exact-bucket A/B 基线已完成；长期稳定性、qps=10、padding/fallback 分布和 timeline 门槛仍按第 16、17 节执行，未通过前保持实验开关默认关闭。
