# GLM-5 Runtime 显存预留分析

> 生产部署配置: Prefill 8卡 8CP 8EP / Decode 8卡 8DP 8EP / max_seq_len=256K / 总并发=128 / Decode 开启 CUDA Graph

---

## 1. 模型关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| hidden_size | 6144 | |
| num_hidden_layers | 78 + 1 MTP | layer 0-77 主干, layer 78 MTP |
| num_attention_heads | 64 | |
| vocab_size | 154880 | |
| kv_lora_rank | 512 | MLA compressed KV |
| q_lora_rank | 2048 | MLA compressed Q |
| qk_nope_head_dim / qk_rope_head_dim | 192 / 64 | |
| v_head_dim | 256 | |
| n_routed_experts | 256 | |
| num_experts_per_tok | 8 | |
| moe_intermediate_size | 2048 | |
| n_shared_experts | 1 | |
| intermediate_size (dense FFN) | 12288 | 前 3 层 dense 用 |
| shared expert intermediate_size | 2048 | 同 moe_intermediate_size |
| first_k_dense_replace | 3 | 前 3 层 dense, 75 层 MoE |

## 2. 部署配置推导

| | Prefill | Decode |
|---|---------|--------|
| 卡数 | 8 | 8 |
| 并行方式 | CP=8 (tp=8), EP=8 | DP=8, EP=8 |
| 每 rank local experts | 256 / 8 = 32 | 256 / 8 = 32 |
| 总并发 | - | 128 |
| 每 rank max_generate_batch_size | 1 (CP 强制 single prefill) | 128 / 8 = **16** |
| 每 rank 处理 token 数 (prefill) | 256K / 8 = **32K** | **16** (每 batch 1 token/request) |
| max_seq_len | 262144 (256K) | 262144 (256K) |

## 3. Runtime Memory 计算框架

Runtime memory 是指模型权重和 KV cache 之外，forward pass 所需的临时显存。引擎通过以下公式决定：

```
runtime_required = max(
    default_floor,       // max(2048 MiB, 5% × total_gpu_mem)
    warm_up_peak,        // warm-up forward pass 实测峰值
    sampler_mem          // max_generate_batch_size × vocab_size × 4 × 8
)

kv_cache_budget = gpu_available_after_weights - runtime_required
```

**代码位置**: `rtp_llm/cpp/cache/MemoryEvaluationHelper.cc`

### 3.1 Default Floor (最低保底)

```cpp
// MemoryEvaluationHelper.cc:63
minimal_runtime_bytes = max(2048L * 1024 * 1024, (long)(total_gpu_bytes * 0.05));
```

| GPU 型号 | 显存 | 5% | Floor |
|----------|------|-----|-------|
| H800 | 80 GiB | 4096 MiB | **4096 MiB (4 GiB)** |
| H20 | 96 GiB | 4915 MiB | **4915 MiB (~4.8 GiB)** |

### 3.2 Sampler Memory

```cpp
// MemoryEvaluationHelper.cc:128-129
sample_need_mem = (size_t)max_generate_batch_size * model_config.vocab_size * 4 * 8;
```

| 场景 | max_generate_batch_size | 计算 | 结果 |
|------|------------------------|------|------|
| Prefill | 1 | 1 × 154880 × 32 | **~5 MiB** (可忽略) |
| Decode (dp=8, 并发=128) | 16 | 16 × 154880 × 32 | **~76 MiB** |

### 3.3 Warm-up Peak Memory (核心主导项)

引擎初始化时通过实际 forward pass 测量峰值显存 (`NormalEngine::warmUp`)。

- **Prefill warm-up**: 构造 `max_seq_len - 1 = 262143` tokens 的 fake input，CP=8 后每 rank 处理 ~32K tokens
- **Decode warm-up**: 构造 fake input，执行 decode forward + **CUDA Graph capture**

## 4. Prefill 侧显存分析 (8CP, 8EP)

CP=8 时每 rank 处理 32K tokens (262144/8 = 32768)。warm-up 会以 ~32K tokens 跑一次 prefill forward。

### 4.1 MegaMoE max_tokens_per_rank 计算

```python
# fused_moe_wrapper.py 计算链:
max_tokens_per_rank = max(8192, max_seq_len) = 262144

# resolve_moe_max_tokens_per_rank (prefill):
cp_bound = cp_padded_tokens_per_rank_bound(262144, 8) = 32768
budget = min(262144, 32768) = 32768

# chunked_moe_enabled() == True (默认):
#   DEFAULT_MOE_CHUNK_TOKENS = 16384
# → 最终: min(32768, 16384) = 16384
# DeepGEMM 内部对齐后 buf.num_max_tokens_per_rank ≈ 16512
```

> **注意**: 默认 chunked MoE 会把 symm buffer 限制到 16384 tokens。如果 CP ZigZag 分片不均导致某 rank 收到 >16512 tokens，会触发 `GLM5 MegaMoE input tokens=X exceeds num_max_tokens_per_rank` 报错。需设置 `DSV4_MOE_CHUNK_PREFILL=0` 或增大 `DSV4_MOE_CHUNK_TOKENS=33000` 来解决。

### 4.2 每层激活显存 (峰值, BF16)

| 组件 | 计算 | 估算 |
|------|------|------|
| Hidden states (residual) | 32K × 6144 × 2B | **384 MiB** |
| MLA Q projection 中间量 | 32K × q_lora_rank(2048) × 2B | **128 MiB** |
| MLA KV projection 中间量 | 32K × kv_lora_rank(512) × 2B | **32 MiB** |
| Attention output (pre-down proj) | 32K × 64 × v_head_dim(256) × 2B | **1024 MiB** |
| Shared expert FFN 中间量 | 32K × 12288 × 2B (SiGLU 需 2x gate) | **1536 MiB** |
| MoE dispatch/combine buffer | MegaMoE symm buffer (32K tokens, 32 local experts) | **~1-3 GiB** |

> 注: 各层串行执行，同一时刻只有一层的激活存在。FlashAttention 采用 chunk 计算，不会产生完整 QK^T 矩阵。

### 4.3 其他固定开销

| 组件 | 估算 |
|------|------|
| cuBLAS workspace | ~200 MiB |
| FlashAttention workspace | ~100-200 MiB |
| CP 通信 buffer (`PrefillCPConfig::comm_buffer_size`) | 512 MiB (默认) |
| PyTorch CUDACachingAllocator 碎片 | ~500 MiB |

### 4.4 Prefill 汇总

| 项目 | 估算 |
|------|------|
| 激活峰值 (32K tokens × 78 层串行) | 3-5 GiB |
| MegaMoE symm buffer | 1-3 GiB |
| cuBLAS / FlashAttn workspace | ~0.3 GiB |
| CP comm buffer | 0.5 GiB |
| PyTorch allocator overhead | ~0.5 GiB |
| **Prefill 估算总计** | **~6-10 GiB** |
| **建议 reserve_runtime_mem_mb** | **8192-10240 (8-10 GiB)** |

> Smoke test 中所有 GLM-5 配置统一使用 `reserver_runtime_mem_mb=8192`。

## 5. Decode 侧显存分析 (8DP, 8EP, CUDA Graph)

### 5.1 CUDA Graph Capture 开销

Decode 开启 CUDA graph 时，引擎会按默认策略 capture 多个 batch size 的 graph instance。

**capture batch sizes** (max_generate_batch_size=16 时):

```
{1, 2, 3, 4, 5, 6, 7, 8, 16}  →  共 9 个 graph instances
```

代码位置: `rtp_llm/cpp/cuda_graph/cuda_graph_decode.cc:11-38`

每个 graph instance 包含:

| 组件 | 说明 |
|------|------|
| 图节点结构 | CUDA driver 内部, 10-50 MiB/instance |
| Captured tensors | graph 私有的中间 tensor 快照, 100-300 MiB/instance |
| Graph pool 碎片 | PyTorch CUDACachingAllocator 分配对齐 |

**CUDA Graph 总开销估算:**

| 因素 | 估算 |
|------|------|
| 9 个 graph instances | 9 × (100~300) MiB = 900~2700 MiB |
| Graph pool 碎片 | ~500 MiB |
| **CUDA Graph 总计** | **~1.5-3 GiB** |

> 启动日志中可精确观测每个 graph 的显存增量:
> ```
> [CudaGraph Memory] captured batch size 16: pool_delta=XXX MiB, total_reserved=XXX MiB
> ```

### 5.2 Decode Forward 激活显存

Decode 每步处理 16 tokens (128 并发 / 8DP = 16 per rank, 每 request 1 token)，激活显存很小:

| 组件 | 计算 | 估算 |
|------|------|------|
| Hidden states | 16 × 6144 × 2B | 0.19 MiB |
| MLA projection | 16 × 2048 × 2B | 0.06 MiB |
| Attention decode workspace | FlashInfer/SparseMLA | ~50-100 MiB |
| MoE (32 local experts, 16 tokens) | dispatch + GEMM | ~100-200 MiB |

### 5.3 MegaMoE Symmetric Buffer (Decode)

```python
# resolve_moe_max_tokens_per_rank for decode:
# is_decode_role=True → max_tokens_per_rank = max_generate_batch_size × 1 = 16
```

Decode 时 16 tokens 的 symm buffer 较小 (~200-500 MiB)。

### 5.4 Decode 汇总

| 项目 | 估算 |
|------|------|
| 激活峰值 (16 tokens decode) | ~0.5 GiB |
| **CUDA Graph (9 instances)** | **1.5-3 GiB** (主导项) |
| MegaMoE symm buffer (decode, 16 tokens) | ~0.3-0.5 GiB |
| cuBLAS workspace | ~0.2 GiB |
| Sampler | ~0.08 GiB |
| PyTorch allocator overhead | ~0.5 GiB |
| **Decode 估算总计** | **~3-5 GiB** |
| **建议 reserve_runtime_mem_mb** | **8192 (8 GiB)** |

> Smoke test 统一使用 `reserver_runtime_mem_mb=8192`，对 decode 侧有充足余量。

## 6. 建议配置值

| 场景 | reserve_runtime_mem_mb | 说明 |
|------|----------------------|------|
| **Prefill (8CP 8EP, 256K)** | **8192-10240** (8-10 GiB) | 32K tokens/rank 激活 + MegaMoE buffer + CP comm; smoke 参考值 8192 |
| **Decode (8DP 8EP, 128并发, CUDA Graph)** | **8192** (8 GiB) | CUDA Graph 9 instances 主导; smoke 参考值 8192 |

## 7. Smoke Test 参考配置

以下配置摘自 `internal_source/rtp_llm/test/smoke/BUILD`，所有配置均使用 `reserver_runtime_mem_mb=8192`。

### 7.1 Full Checkpoint 配置 (8卡)

| smoke test | Prefill | Decode | max_seq_len | 并发 | CUDA Graph | MTP |
|------------|---------|--------|-------------|------|------------|-----|
| `mla_mega_moe_fp8_attn_cp_pd_full_ckpt` | tp=4, ep=4, CP=ALL_GATHER | dp=4, ep=4 | 32768 | 8 | 开 | 无 |
| `mla_mtp_mega_moe_cudagraph_pd_full_ckpt` | tp=4, ep=4, CP=ALL_GATHER | dp=4, ep=4 | 32768 | 8 | 开 | gen=3 |

### 7.2 2卡 PD 配置

| smoke test | Prefill | Decode | CUDA Graph | MTP |
|------------|---------|--------|------------|-----|
| `mla_mega_moe_cp_pd` | tp=2, ep=2 | dp=2, ep=2 | 开 | 无 |
| `mla_mega_moe_fp8_attn_cp_pd` | tp=2, ep=2, FP8_PER_BLOCK_NO_MOE | dp=2, ep=2 | 开 | 无 |
| `mla_mtp_mega_moe_cudagraph_pd` | tp=2, ep=2, 并发=8 | dp=2, ep=2, 并发=8 | 开 | gen=3 |

### 7.3 Smoke 与生产对比

| 维度 | Smoke (full_ckpt) | 生产配置 | 影响 |
|------|-------------------|---------|------|
| max_seq_len | 32768 | **262144 (256K)** | Prefill 激活与 symm buffer 成正比 |
| CP | 4 | **8** | 每 rank tokens: smoke 8K vs 生产 32K |
| EP | 4 | **8** | 每 rank experts: smoke 64 vs 生产 32 |
| 并发 | 8 | **128** | Decode bs/rank: smoke 2 vs 生产 16 |
| DP (decode) | 4 | **8** | 配合并发变化 |
| reserve_runtime_mem_mb | 8192 | **8192-10240** | 生产 prefill 更大，可能需上调 |

> Smoke 的 per-rank tokens (8K) 只有生产 (32K) 的 1/4，因此 smoke 的 8192 MiB 在生产 prefill 侧可能不够。Decode 侧 graph instances 从 2 个增到 9 个，但 8192 MiB 仍有余量。

## 8. 模型权重显存分析 (FP8 vs FP8+MegaMoE FP4)

FP8 checkpoint 开启 MegaMoE FP4 后，MoE 专家权重从 FP8 (1B/param) 转为 FP4 (0.5B/param + scale)，其余权重保持 FP8。

### 8.1 权重 shape 一览 (从 checkpoint 实测)

**MLA Attention 权重 (每层, FP8)**:

| 权重 | Shape | 元素数 | FP8 (MiB) | TP=8 后 (MiB) | 说明 |
|------|-------|--------|-----------|---------------|------|
| q_a_proj | [2048, 6144] | 12.6M | 12.0 | 12.0 | 不分片 (compressed space) |
| q_b_proj | [16384, 2048] | 33.6M | 32.0 | 4.0 | 按 head 分片 ÷8 |
| kv_a_proj_with_mqa | [576, 6144] | 3.5M | 3.4 | 3.4 | 不分片 |
| kv_b_proj | [28672, 512] | 14.7M | 14.0 | 1.8 | 按 head 分片 ÷8 |
| o_proj | [6144, 16384] | 100.7M | 96.0 | 12.0 | 按 head 分片 ÷8 |
| indexer.wk | [128, 6144] | 0.8M | 0.8 | 0.8 | DSA indexer |
| indexer.wq_b | [4096, 2048] | 8.4M | 8.0 | 1.0 | 按 head 分片 ÷8 |
| **合计** | | **174.2M** | **166.6** | **35.3** | + layernorm/scale 忽略 |

**Dense FFN (前 3 层, FP8)**:

| 权重 | Shape | 元素数 | 说明 |
|------|-------|--------|------|
| gate_proj | [12288, 6144] | 75.5M | |
| up_proj | [12288, 6144] | 75.5M | |
| down_proj | [6144, 12288] | 75.5M | |
| **合计** | | **226.5M** | 216 MiB/层 (full), 27 MiB/层 (TP=8) |

**MoE 专家 (每个 expert, 3 个矩阵)**:

| 权重 | Shape | 元素数 |
|------|-------|--------|
| gate_proj | [2048, 6144] | 12.6M |
| up_proj | [2048, 6144] | 12.6M |
| down_proj | [6144, 2048] | 12.6M |
| **合计** | | **37.7M** |

**其他 (每 MoE 层)**:

| 权重 | Shape/dtype | 大小 |
|------|-------------|------|
| Router gate | [256, 6144] BF16 | 3.0 MiB |
| e_score_correction_bias | [256] FP32 | ~0 |
| Shared expert (gate+up+down) | 3 × [2048, 6144] FP8 | 36.0 MiB |

**全局权重**:

| 权重 | Shape/dtype | 大小 |
|------|-------------|------|
| embed_tokens | [154880, 6144] BF16 | 1.77 GiB |
| lm_head | [154880, 6144] BF16 | 1.77 GiB |
| model.norm | [6144] BF16 | ~0 |

### 8.2 FP8 vs FP4 专家权重对比

| 指标 | FP8 | FP4 (MegaMoE) |
|------|-----|---------------|
| 存储精度 | float8_e4m3fn, 1B/param | int8 packed (2 values/byte), 0.5B/param |
| Scale | FP32 per 128×128 block | int32 per FP4_BLOCK=32 elements |
| **每 expert 权重** | **36.0 MiB** | **22.5 MiB** |
| **每 expert scale** | **~0.01 MiB** | **~4.5 MiB** |
| **每 expert 总计** | **~36.0 MiB** | **~22.5 MiB** |
| FP4/FP8 比值 | — | **62.5%** |

FP4 计算细节:
```
w1 (gate+up stacked [4096, 6144]):
  packed: [4096, 3072] int8        = 12.0 MiB
  scale:  [4096, 192]  int32       =  3.0 MiB
w2 (down [6144, 2048]):
  packed: [6144, 1024] int8        =  6.0 MiB
  scale:  [6144, 64]   int32       =  1.5 MiB
────────────────────────────────────
单 expert FP4 总计                    22.5 MiB
```

### 8.3 每 GPU 权重显存汇总

#### Decode (DP=8, EP=8, 无 TP)

每 rank 拥有完整 attention 权重 + 32 local experts:

| 组件 | FP8 (GiB) | FP8 + FP4 MoE (GiB) |
|------|-----------|---------------------|
| MLA Attention (78 层, 全量) | 12.69 | 12.69 |
| Dense FFN (3 层, 全量) | 0.63 | 0.63 |
| **MoE experts (75 层 × 32 experts)** | **84.40** | **52.73** |
| Shared experts (75 层) | 2.64 | 2.64 |
| Router (75 层) | 0.22 | 0.22 |
| Embedding + lm_head | 3.54 | 3.54 |
| **Total per GPU** | **~104.1 GiB** | **~72.5 GiB** |
| H800 剩余 (80 GiB) | **-24.1 (不可部署!)** | **7.5 GiB** |

#### Prefill (CP=8 = TP=8, EP=8)

Attention 和 Dense/Shared FFN 按 TP=8 分片:

| 组件 | FP8 (GiB) | FP8 + FP4 MoE (GiB) |
|------|-----------|---------------------|
| MLA Attention (78 层, TP=8) | 2.69 | 2.69 |
| Dense FFN (3 层, TP=8) | 0.08 | 0.08 |
| **MoE experts (75 层 × 32 experts)** | **84.40** | **52.73** |
| Shared experts (75 层, TP=8) | 0.33 | 0.33 |
| Router (75 层) | 0.22 | 0.22 |
| Embedding + lm_head | 3.54 | 3.54 |
| **Total per GPU** | **~91.3 GiB** | **~59.6 GiB** |
| H800 剩余 (80 GiB) | **-11.3 (不可部署!)** | **20.4 GiB** |

### 8.4 关键结论

1. **FP8-only 无法在 H800 部署**: 仅 MoE 专家权重就占 84.4 GiB/GPU (EP=8)，加上 attention 等远超 80 GiB
2. **FP4 MoE 是 H800 必选项**: 专家从 84.4 → 52.7 GiB，节省 **31.7 GiB/GPU**
3. **Decode 侧显存最紧张**: FP4 后仅剩 ~7.5 GiB，需分配给 KV cache + runtime memory (8 GiB reserve)，实际可用 KV cache 很少
4. **Prefill 侧余量充裕**: TP=8 大幅压缩 attention，FP4 后剩余 ~20.4 GiB 给 KV cache + runtime

### 8.5 MTP 层额外开销

GLM-5 MTP 层 (layer 78) 结构同主干 MoE 层，如果开启 MTP:

| 组件 | Decode FP4 (GiB) | Prefill FP4 (GiB) |
|------|-----------------|-------------------|
| MTP Attention | 0.16 | 0.03 |
| MTP MoE experts (32 local, FP4) | 0.70 | 0.70 |
| MTP Shared + Router | 0.04 | 0.01 |
| MTP 特有 (eh_proj, norms) | ~0.04 | ~0.04 |
| **MTP 总增量** | **~0.94 GiB** | **~0.78 GiB** |

> 注: MTP 与主模型共享 embedding/lm_head，不产生额外嵌入开销。

## 9. 注意事项

### 9.1 框架自动调整机制

`reserve_runtime_mem_mb` 只是一个下限参数。框架会取:

```
actual_runtime = max(reserve_runtime_mem_mb, 5%_floor, warm_up_peak, sampler_mem)
```

- 设太小: 框架自动拉高到 `max(5%_floor, warm_up_peak)`，**不会** OOM
- 设太大: 浪费 KV cache 空间，降低可服务的并发/序列长度
- **推荐开启 warm_up** (默认开启)，让框架实测峰值

### 9.2 关键日志观测点

启动时关注以下日志，可获取精确值:

```
# Runtime memory 决策
RuntimeConfig has reserve_runtime_mem_mb=XXX
tp_size X needs at least XXX MiB memory for runtime by default

# Warm-up 实测
warm up consumed XXX MiB max memory, env runtime memory XXX MiB, final runtime memory XXX MiB

# CUDA Graph 逐 instance 开销
[CudaGraph Memory][before_capture] cudaMemGetInfo: used=XXX MiB, free=XXX MiB
[CudaGraph Memory] captured batch size N: pool_delta=XXX MiB, total_reserved=XXX MiB
[CudaGraph Memory][after_capture] cudaMemGetInfo: used=XXX MiB, free=XXX MiB

# MegaMoE symm buffer
[GLM5 MegaMoE] allocated symm buffer: ... actual=X.XXX GiB

# Sampler 估算
sampler needs XXX MiB memory, model runtime needs XXX MiB memory

# 最终 KV cache 分配
cache config final decided kv cache memory size XXX MiB
```

### 9.3 自定义 CUDA Graph Batch Sizes

如果默认 capture 的 batch sizes 过多 (显存紧张), 可通过 `decode_capture_batch_sizes` 参数指定:

```
# 例: 只 capture 关键 batch sizes, 减少 CUDA graph 显存
decode_capture_batch_sizes = "1,4,8,16"
```

这会将 graph instances 从 9 个减少到 4 个，节省约 500-1500 MiB。

### 9.4 MTP 对 runtime memory 的影响

GLM-5 有 1 层 MTP (layer 78)。如果开启 speculative decoding (`sp_type=MTP`):

- 框架会强制 `reserve_runtime_mem >= 2 GiB` (`MemoryEvaluationHelper.cc:78-81`)
- MTP draft model 有独立的 CUDA graph capture，需额外显存
- Decode 侧 MTP target verify 阶段单 batch tokens = 16 × (gen_num_per_cycle + 1)
- 建议在上述基础上额外增加 **1-2 GiB**

### 9.5 MegaMoE Chunked MoE 注意事项

默认 `DSV4_MOE_CHUNK_PREFILL=1` 会将 MegaMoE symm buffer 限制到 `DEFAULT_MOE_CHUNK_TOKENS=16384` tokens。在 256K + CP=8 场景下，每 rank 最大 32K tokens，CP ZigZag 分片可能不均。如果报错:

```
GLM5 MegaMoE input tokens=X exceeds num_max_tokens_per_rank=16512
```

解决方案:
- `DSV4_MOE_CHUNK_PREFILL=0` — 关闭 chunked MoE，budget 直接取 cp_bound=32768
- 或 `DSV4_MOE_CHUNK_TOKENS=33000` — 增大 chunk 上限以覆盖 CP 分片上界

### 9.6 理论 vs 实测

上述所有数值均为基于代码和模型参数的**理论估算**。实际显存受以下因素影响:

- CUDA driver 版本和 GPU 型号差异
- PyTorch 版本的 allocator 策略
- DeepGEMM 版本和 MegaMoE kernel 实现细节
- 是否启用 chunked MoE (`DSV4_MOE_CHUNK_PREFILL`)
- Triton kernel JIT 编译缓存

**强烈建议**: 生产环境开启 `warm_up=1`，让框架实测峰值后自动调整。
