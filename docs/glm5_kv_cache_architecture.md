# GLM-5 KV Cache 存储结构与使用链路

## 1. 概述

GLM-5 采用 **MLA (Multi-Latent Attention)** 架构，与 DeepSeek V2/V3 共用同一套 KV Cache 基础设施。MLA 的核心思想是将传统 MHA 中按 head 分别存储的 K/V 压缩为一个低秩的 **latent vector**，大幅降低 KV Cache 的显存占用。

GLM-5 注册在 `rtp_llm/models/deepseek_v2.py`，直接复用 `DeepSeekV2` 基类：

```python
register_model("glm_5", DeepSeekV2, ["GlmMoeDsaForCausalLM"])
register_model("glm_5_mtp", Glm5Mtp, ["GlmMoeDsaMtpForCausalLM"])
```

## 2. MLA 关键参数

MLA 参数从 HuggingFace `config.json` 解析（`deepseek_v2.py:616-632`）：

| 参数 | 含义 | GLM-5 典型值 | DeepSeek V3 典型值 |
|------|------|-------------|-------------------|
| `kv_lora_rank` | KV 压缩的 LoRA 秩 | **448** | 512 |
| `qk_nope_head_dim` | Q/K 非位置编码维度 | 128 | 128 |
| `qk_rope_head_dim` | Q/K RoPE 维度 | **64** | 64 |
| `v_head_dim` | V head 维度 | 128 | 128 |
| `q_lora_rank` | Q 压缩的 LoRA 秩 | 1536 | 1536 |

GLM-5 与 DeepSeek 的**关键差异**在于 `kv_lora_rank`：GLM-5 为 448，DeepSeek 为 512。这导致了不同的 FP8 cache 布局（`fp8_model1_mla` vs `fp8_ds_mla`）。

## 3. MLA vs MHA 的 KV Cache 结构对比

### 3.1 MHA (Multi-Head Attention) 传统方式

- K 和 V **按 head 分别存储**
- 每个 token 存储量 = `2 × num_kv_heads × size_per_head`
- Cache shape: `[num_blocks, 2 × local_head_num_kv × size_per_head × seq_size_per_block]`（2D per layer）
- TP 并行时按 head 切分

定义在 `rtp_llm/cpp/cache/MHAKVCacheSpec.h`。

### 3.2 MLA (Multi-Latent Attention) 压缩方式

- K 和 V **合并压缩为一个 latent vector** + 一个 RoPE 向量
- 每个 token 存储量 = `kv_lora_rank + rope_head_dim`（GLM-5: 448 + 64 = 512 个元素）
- Cache shape: `[num_blocks, seq_size_per_block, kv_lora_rank + rope_head_dim]`（3D per layer）
- `local_head_num_kv = 1` — 所有 head 共享同一组压缩 KV，**TP 并行时不按 head 切分 cache**

定义在 `rtp_llm/cpp/cache/MLAKVCacheSpec.h`。

```
MHA 每个 token (e.g., 128 heads, head_dim=128):
┌──────────────────────────┬──────────────────────────┐
│  K: [num_kv_heads, dim]  │  V: [num_kv_heads, dim]  │
└──────────────────────────┴──────────────────────────┘

MLA 每个 token (GLM-5):
┌──────────────────────────────────────────┬────────────┐
│    compressed_kv (kv_lora_rank=448)      │  k_pe (64) │
│    latent vector，同时编码 K 和 V 信息     │  RoPE 部分  │
└──────────────────────────────────────────┴────────────┘
```

## 4. KV Cache Spec 类层次

```
KVCacheSpec (base)                   ← rtp_llm/cpp/cache/KVCacheSpecBase.h
  ├── MHAKVCacheSpec                 ← rtp_llm/cpp/cache/MHAKVCacheSpec.h
  ├── MLAKVCacheSpec                 ← rtp_llm/cpp/cache/MLAKVCacheSpec.h
  └── LinearKVCacheSpec              ← rtp_llm/cpp/cache/LinearKVCacheSpec.h
```

`MLAKVCacheSpec` 核心字段：

```cpp
struct MLAKVCacheSpec : public KVCacheSpec {
    uint32_t kv_lora_rank;   // 448 (GLM-5) / 512 (DeepSeek)
    uint32_t rope_head_dim;  // 64

    // MLA 特有：local_head_num_kv 固定为 1
    // block_size = (kv_lora_rank + rope_head_dim) * seq_size_per_block
};
```

## 5. KV Cache 物理内存布局

### 5.1 BF16 模式

```
全局 buffer reshape 为 4D:
[layer_num, block_num, seq_size_per_block, kv_lora_rank + rope_head_dim]

每层拿到 3D tensor:
kv_cache_base: [block_num, seq_size_per_block, 512]
                                                ↑
                                    448 (compressed_kv) + 64 (k_pe)

每个 token 占用: 512 × 2 bytes = 1024 字节
```

布局由 `MemoryLayoutStrategy::processKVTensor()`（`rtp_llm/cpp/cache/MemoryLayoutStrategy.cc:58-75`）实现。

### 5.2 FP8 模式 — `fp8_model1_mla`（GLM-5 专用）

GLM-5 使用 `fp8_model1_mla` 布局，每个 token 占 **584 字节**：

```
┌─────────────────────────────────────────────────────────────────┐
│                    每个 token 的内存布局 (584B)                   │
├────────────────────────────────┬────────────────────────────────┤
│  NoPE 部分: 448 bytes          │  RoPE 部分: 128 bytes           │
│  (kv_lora_rank=448 个 FP8 值)  │  (64 个 BF16 值，不量化)         │
├────────────────────────────────┴────────────────────────────────┤
│  Scale Factors: 7 × fp8_e8m0 + 1 byte padding = 8 bytes       │
│  (每 64 个 FP8 元素一个 scale，448/64 = 7 个 scale)              │
│  (scales 存放在 block 尾部，按 token 索引)                       │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 FP8 模式 — `fp8_ds_mla`（DeepSeek V3 专用，对比参考）

DeepSeek V3 使用 `fp8_ds_mla` 布局，每个 token 占 **656 字节**：

```
┌────────────────────────┬──────────────────┬────────────────────┐
│ NoPE: 512 bytes (FP8)  │ Scales: 16 bytes │ RoPE: 128 bytes    │
│ (kv_lora_rank=512)     │ (4 × float32)    │ (64 × BF16，不量化) │
└────────────────────────┴──────────────────┴────────────────────┘
```

### 5.4 布局差异总结

| 模式 | 每 token 字节 | NoPE 存储 | Scale 格式 | RoPE 存储 |
|------|-------------|----------|-----------|----------|
| BF16 (GLM-5) | 1024 | 448 × BF16 | 无 | 64 × BF16 |
| `fp8_model1_mla` (GLM-5) | 584 | 448 × FP8 | 7 × fp8_e8m0 | 64 × BF16 |
| BF16 (DeepSeek) | 1152 | 512 × BF16 | 无 | 64 × BF16 |
| `fp8_ds_mla` (DeepSeek) | 656 | 512 × FP8 | 4 × float32 | 64 × BF16 |

## 6. KV Cache 分配链路

```
模型初始化
  │
  ▼
CacheConfigCreator::create()
  ├── 创建 MLAKVCacheSpec (kv_lora_rank, rope_head_dim, dtype)
  ├── 填充 CacheConfig {use_mla=true, cache_specs=[MLAKVCacheSpec]}
  │
  ▼
BlockPoolConfigHelper::createMemoryLayoutConfig()    ← BlockPoolConfigHelper.h:164
  ├── cfg.use_mla = cache_config.use_mla             (true)
  ├── cfg.is_mla  = use_mla || is_sparse             (控制 scale 3D 布局)
  ├── cfg.kv_block_stride_bytes = spec->block_size_bytes()
  ├── 计算 kv_block_pool_size_bytes = layer_num × block_num × kv_block_stride_bytes
  │
  ▼
MemoryLayoutStrategy::init()                          ← MemoryLayoutStrategy.cc:9
  ├── processKVTensor()
  │   └── MLA 路径: reshape 为 [layer_num, block_num, seq_size_per_block, stride_elems]
  │   └── 每层切出 3D: [block_num, seq_size_per_block, stride_elems]
  ├── processScaleTensor() (FP8 时分配 scale buffer)
  │
  ▼
KVCacheManager
  ├── 管理 block 的分配/释放 (paged attention)
  ├── 每个 request 分配 block_table: [max_blocks_per_seq]
  └── slot_mapping 映射 token → (block_idx, block_offset)
```

## 7. KV Cache 写入链路

### 7.1 Attention 前向传播中的写入

```python
# rtp_llm/models_py/modules/hybrid/mla_attention.py:190
MlaAttention.forward(hidden_states, fmha_impl, kv_cache)
    │
    ├── fused_qkv_a_proj(hidden_states)     # 联合投影，输出 q + compressed_kv + k_pe
    ├── torch.split → q, compressed_kv      # 分离 q 和 kv
    ├── torch.split → compressed_kv, k_pe   # 分离 compressed_kv 和 k_pe
    ├── kv_a_layernorm(compressed_kv)       # RMSNorm (可 fused_strided)
    │
    └── fmha_impl.forward(q_view, compressed_kv, k_pe, kv_cache, layer_idx)
```

### 7.2 FMHA 内部的 cache 写入

```python
# rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/flashinfer_mla_wrapper.py:95
MlaFlashInferImplBase.forward()
    │
    ├── rope_impl.forward(q_pe, k_pe, rope_params)   # 对 Q_pe 和 K_pe 做 RoPE
    │
    ├── kv_cache_write_op.forward(compressed_kv, k_pe, kv_cache, rope_params)
    │   │
    │   └── MlaKVCacheWriteOp.forward()   ← mla_kv_cache_write_op.py:29
    │       └── compute_ops.concat_and_cache_mla(
    │             kv_c=compressed_kv,           # [num_tokens, kv_lora_rank]
    │             k_pe=k_pe,                    # [num_tokens, rope_head_dim]
    │             kv_cache=kv_cache.kv_cache_base,  # [num_blocks, block_size, stride]
    │             slot_mapping=slot_mapping,     # [num_tokens]
    │             kv_cache_dtype="auto" | "fp8_model1_mla",
    │             scale=scale)
    │
    └── fmha_impl.forward(q_nope, q_pe, kv_cache, layer_id)   # 注意力计算
```

### 7.3 CUDA Kernel 实现

位于 `rtp_llm/models_py/bindings/cuda/kernels/mla_quant_kernel.cu`。

**BF16 模式** — `concat_and_cache_mla_kernel` (line 614)：
```
每个 CUDA block 处理一个 token:
1. slot_idx = slot_mapping[token_idx]
2. block_idx = slot_idx / block_size
3. block_offset = slot_idx % block_size
4. 拷贝 compressed_kv[token_idx, 0:kv_lora_rank]  → cache[block_idx, block_offset, 0:kv_lora_rank]
5. 拷贝 k_pe[token_idx, 0:rope_head_dim]           → cache[block_idx, block_offset, kv_lora_rank:kv_lora_rank+rope_head_dim]
```

**FP8 模式 (GLM-5)** — `concat_and_cache_ds_model1_kernel` (line 760)：
```
96 threads per block:
- Threads 0-55:  处理 NoPE 部分 (448 个元素)
  - 7 个 tile，每 tile 64 个 FP8 元素
  - 每 tile 8 个线程，计算 tile 内 max_abs → fp8_e8m0 scale
  - 量化 BF16 → FP8_E4M3 并写入 cache
- Threads 56-63: 空闲 (warp 对齐)
- Threads 64-95: 处理 RoPE 部分 (64 个 BF16 元素)
  - 不量化，直接拷贝 BF16 值
- Scale factors 写入 block 尾部 (每 token 8 bytes)
```

## 8. KV Cache 读取链路

### 8.1 Decode 阶段

```python
MlaFlashInferDecodeOp.forward(q_nope, q_pe, kv_cache, layer_id)
    │
    ├── BF16 模式:
    │   └── BatchMLAPagedAttentionWrapper (FlashInfer)
    │       ├── 通过 block_table + seq_lens 索引 cache blocks
    │       ├── 内核内部：从 cache 读取 compressed_kv + k_pe
    │       ├── 用 W_vc (absorbed weight) 将 compressed_kv → V
    │       ├── 用 W_kc (absorbed weight) 将 compressed_kv → K_nope
    │       ├── K = concat(K_nope, K_pe_from_cache)
    │       └── 标准 softmax(Q·K^T) · V
    │
    └── FP8 模式:
        └── flash_mla_with_kvcache()
            ├── 直接读取 fp8_model1_mla 格式 cache
            ├── 内核内部 dequant NoPE (FP8 → BF16，用 scale)
            └── 直接读 RoPE (已是 BF16)
```

### 8.2 Prefill 阶段

```python
MlaFlashInferPrefillImpl.forward()
    │
    ├── 写入 cache (同上述写入链路)
    │
    └── compute_prefill_context()
        ├── 长序列: MlaFlashInferPrefillOp
        │   └── BatchPrefillWithRaggedKVCacheWrapper (FlashInfer)
        │       └── 从 cache 读取做 paged attention
        │
        └── 短序列 + cache reuse (prefix_lengths > 0 && q_len < absorb_opt_len):
            └── absorb_fmha (MlaFlashInferDecodeOp)
                └── 复用 decode 路径的 absorbed attention
```

### 8.3 FP8 Cache Gather + Upconvert

当需要将 FP8 cache 转换为 BF16 workspace 时（如 sparse attention indexer 需要 BF16 输入），使用专用 kernel：

```cpp
// rtp_llm/models_py/bindings/cuda/kernels/mla_quant_kernel.h

// 分离输出版本
cp_gather_and_upconvert_fp8_kv_cache(
    src_cache,           // [NUM_BLOCKS, BLOCK_SIZE, 656] uint8
    dst_compressed_kv,   // [TOT_TOKENS, 512] bfloat16
    dst_k_pe,            // [TOT_TOKENS, 64] bfloat16
    block_table, seq_lens, workspace_starts, batch_size
);

// 融合输出版本
cp_gather_and_upconvert_fp8_kv_cache_v2(
    src_cache,       // [NUM_BLOCKS, BLOCK_SIZE, 656] uint8
    dst_fused,       // [TOT_TOKENS, 576] bfloat16  (512 + 64 合并)
    block_table, seq_lens, workspace_starts, batch_size, total_tokens
);
```

## 9. GLM-5 特殊处理

### 9.1 DSA (DeepSeek Sparse Attention)

GLM-5 主模型支持 DSA 稀疏注意力（`attn_config.is_sparse = True`），通过 `Indexer` 模块实现 top-k 选择性注意力。Indexer 需要额外的 `kv_scale_base` cache 来存储量化后的 K 用于快速相似度计算。

但 **MTP 层关闭了 sparse**（`deepseek_v2.py:960`）：

```python
# MTP propose 模型不分配 indexer 的 kv_scale_base cache，
# 因此禁用 indexer，回退到标准 MLA
config.attn_config.is_sparse = False
config.attn_config.indexer_topk = 0
```

### 9.2 MTP 的 KV Cache

`Glm5Mtp` 继承 `DeepSeekV2`，其 cache 配置：
- 独立的 `CacheConfig.mtp_sub_configs`
- `layer_num = 1`（单层 MTP）
- 与主模型共享相同的 `MLAKVCacheSpec`（相同的 kv_lora_rank、rope_head_dim）
- 独立的 block pool

### 9.3 TP 并行下的 cache 处理

MLA 的 `local_head_num_kv = 1`，cache 不按 head 切分。`MLAKVCacheSpec::splitKVPartitionBytes()` 返回完整的 K/V block，不做 head-based partitioning：

```cpp
// MLAKVCacheSpec.h:62-82
static KVPartitionBytes splitKVPartitionBytes(...) {
    // For MLA: return full blocks without head-based partitioning
    return {0, k_block_bytes, k_block_bytes, v_block_bytes};
}
```

这意味着每个 TP rank 持有**完整的** KV cache 副本。注意力的并行化在 Q 端完成（Q 按 head 切分到不同 TP rank），attention 输出通过 `all_reduce` 聚合。

## 10. 端到端数据流总结

```
                        ┌─────────────────────────┐
                        │   hidden_states [B, D]   │
                        └────────────┬────────────┘
                                     │
                          fused_qkv_a_proj (Linear)
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              q [B, q_lora]   compressed_kv [B, 448]  k_pe [B, 64]
                    │                │                │
              q_a_layernorm    kv_a_layernorm      RoPE
                    │                │                │
              q_b_proj             │                │
                    │                │                │
              q [B, H, d_q]        │                │
                    │                ▼                ▼
                    │         ┌──────────────────────────┐
                    │         │  concat_and_cache_mla    │
                    │         │  写入 paged KV cache       │
                    │         │  [block, slot, 448+64]    │
                    │         └──────────────────────────┘
                    │                       │
                    ▼                       ▼
              ┌─────────────────────────────────┐
              │   FlashInfer MLA Attention       │
              │   从 cache 读取 compressed_kv     │
              │   内核内部: kv → K_nope, V        │
              │   (通过 absorbed weights W_kc,    │
              │    W_vc 在注意力内部完成)          │
              └──────────────┬──────────────────┘
                             │
                        attn_output
                             │
                          o_proj
                             │
                       all_reduce (TP)
                             │
                        output [B, D]
```

## 11. 关键文件索引

| 文件 | 作用 |
|------|------|
| `rtp_llm/models/deepseek_v2.py` | GLM-5 模型定义、MLA 参数解析、模型注册 |
| `rtp_llm/cpp/cache/MLAKVCacheSpec.h` | MLA cache spec：block size 计算、FP8 布局 |
| `rtp_llm/cpp/cache/MHAKVCacheSpec.h` | MHA cache spec（对比参考） |
| `rtp_llm/cpp/cache/KVCacheSpecBase.h` | Cache spec 基类和枚举 |
| `rtp_llm/cpp/cache/CacheConfig.h` | Cache 配置结构体 |
| `rtp_llm/cpp/cache/BlockPoolConfigHelper.h` | Block pool 配置创建（MLA flag 设置） |
| `rtp_llm/cpp/cache/MemoryLayoutStrategy.cc` | KV tensor reshape（MLA 3D vs MHA 2D） |
| `rtp_llm/cpp/cache/MemoryLayoutConfig.h` | 内存布局配置（use_mla、is_mla） |
| `rtp_llm/models_py/modules/hybrid/mla_attention.py` | MLA Attention 模块（forward 入口） |
| `rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/flashinfer_mla_wrapper.py` | FlashInfer MLA 实现（prefill + decode wrapper） |
| `rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/flashinfer_mla.py` | FlashInfer MLA 算子（decode/prefill op） |
| `rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/mla_kv_cache_write_op.py` | MLA cache 写入 op |
| `rtp_llm/models_py/bindings/cuda/kernels/mla_quant_kernel.cu` | CUDA kernel: concat_and_cache_mla、fp8 gather |
| `rtp_llm/models_py/bindings/cuda/kernels/mla_quant_kernel.h` | CUDA kernel 头文件 |
| `rtp_llm/models_py/modules/base/common/kvcache_store.py` | WriteCacheStoreOp（prefix cache store） |
