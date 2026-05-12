# Phase 4 增强：KV Cache 昇腾适配详细计划

> 本文档是对 `ascend_npu_adaptation_plan.md` 中 Phase 4 KV Cache 适配的详细完善。
> 基于对 vllm-ascend、xllm、rtp-llm 三个项目 KV Cache 实现的对比分析。
> 实现路径：**torch_npu Python API 为主**（类似 vllm-ascend 方案）。

---

## 一、三框架 KV Cache 适配方案对比

通过分析 vllm-ascend、xllm、rtp-llm 三个项目的 KV Cache 实现，总结各框架的 NPU 适配策略差异：

| 维度 | vllm-ascend | xllm | rtp-llm (目标) |
|------|------------|------|---------------|
| **API 层级** | `torch_npu` Python API | ATB C++ 算子 | `torch_npu` Python API |
| **KV Cache 写入** | `torch_npu._npu_reshape_and_cache` | `atb::npu_reshape_and_cache` | `torch_npu._npu_reshape_and_cache` |
| **Prefill Attention** | `torch_npu.npu_fused_infer_attention_score` | `atb::npu_flash_attention` | `torch_npu.npu_fused_infer_attention_score` |
| **Decode Attention** | `torch_npu._npu_paged_attention` | `atb::npu_paged_attention` | `torch_npu._npu_paged_attention` |
| **内存格式** | NCHW/ND (标准) | ND / FRACTAL_NZ (DeepSeek) | ND |
| **MLA 支持** | `npu_fused_infer_attention_score_v2` + `npu_kv_rmsnorm_rope_cache` | ATB monolithic decoder op | 待评估 |
| **ACL Graph** | `torch.npu.NPUGraph` | `atb::npu_custom_paged_attention` | `torch.npu.NPUGraph` |
| **Block 大小** | 128 | 128 | 128 (保持不变) |
| **KV Cache Shape** | `[2, blocks, block_size, kv_heads, head_dim]` K/V dim0 分离 | `[blocks, block_size, kv_heads, head_dim]` × 2 独立 | `[blocks, seq_size, kv_heads, head_dim]` × 2 独立(NHD) |
| **K/V 存储方式** | K/V 独立 tensor | K/V 独立 tensor | **需从合并改为独立** |
| **Layout** | NHD | NHD (标准) / FRACTAL_NZ (DeepSeek) | **需从 HND 改为 NHD** |

### 选型理由

选择 torch_npu Python API 路径的原因：
1. rtp-llm 的 C++ KV Cache 管理（`BlockPool`/`KVCacheManager`）已基于 `torch::Tensor`，设备无关性好
2. `torch_npu` 提供了完整的 Paged Attention NPU 算子，可替代 FlashInfer
3. 相比 ATB C++ 路径（xllm），torch_npu 路径侵入性更小，开发周期更短
4. vllm-ascend 已验证该路径的可行性

---

## 二、C++ 层 KV Cache 内存管理适配

rtp-llm C++ 层的 KV Cache 管理**大部分是设备无关的**，以下逐组件分析：

### 2.1 需要修改的组件

#### 2.1.1 BlockPool 内存分配

**文件**: `rtp_llm/cpp/cache/BlockPool.cc:39-50`

**现状**：
```cpp
void BlockPool::initializeCacheBuffer() {
    cache_aligned_buffer_ = torch::empty({total_size_bytes},
        torch::dtype(torch::kUInt8).device(torch::kCUDA));
}
```

**适配方案**：参数化设备类型
```cpp
void BlockPool::initializeCacheBuffer() {
    auto device = (allocation_type_ == AllocationType::HOST)
                  ? torch::kCPU : getTorchDevice();  // 返回 kCUDA 或 kNPU
    cache_aligned_buffer_ = torch::empty({total_size_bytes},
        torch::dtype(torch::kUInt8).device(device));
    // NPU: torch::empty 会自动分配在 NPU 上，无需 npu_format_cast
    //      因为 KV cache 使用 ND 格式，非 5HD/NZ 格式
}
```

#### 2.1.2 GPU 内存查询

**文件**: `rtp_llm/cpp/cache/MemoryEvaluationHelper.cc`

**现状**: 使用 `cudaMemGetInfo`

**适配方案**：
```cpp
#if USING_ASCEND
    size_t free_mem, total_mem;
    aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem);
#endif
```

#### 2.1.3 Block 批量拷贝

**文件**: `rtp_llm/cpp/cache/KVCacheAllocator.cc:124-191`

**现状**: `execBatchCopy()` 使用 `cudaMemcpyAsync`，按 D2H/H2D/D2D 分类

**适配方案**：
- D2D: `torch::Tensor::copy_()` 或 `aclrtMemcpyAsync`
- H2D: `tensor.to(torch::kNPU)` 或 `aclrtMemcpyAsync`
- D2H: `tensor.to(torch::kCPU)` 或 `aclrtMemcpyAsync`
- 优先使用 `torch::Tensor` 的 `.to(device)` 方法，避免直接调用 `aclrt` API

#### 2.1.4 CUDA 同步调用

**文件**: `rtp_llm/cpp/cache/KVCacheManager.cc:212/496`

**现状**: `cudaSyncAndCheck()`

**适配方案**：
```cpp
#if USING_ASCEND
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
    aclrtSynchronizeStream(stream);
#endif
```

#### 2.1.5 BlockPool::where() 设备检测

**文件**: `rtp_llm/cpp/cache/BlockPool.cc:502-504`

**现状**: `cache_aligned_buffer_.is_cuda()`

**适配方案**：使用设备无关的检测方式，如 `!is_cpu()` 或引入设备枚举判断

### 2.2 KV 分离存储适配（K/V Cache 独立 Tensor）

昇腾 NPU 的 `torch_npu` 算子（`_npu_reshape_and_cache`、`_npu_paged_attention`、`npu_fused_infer_attention_score`）要求 `key_cache` 和 `value_cache` 作为独立的 Tensor 传入，而 rtp-llm 当前将 K/V 合并在一个 Tensor 中（dim=1 分离：`[blocks, 2, kv_heads, seq, dim]`）。

#### 2.2.1 CacheConfig 配置扩展

**文件**: `rtp_llm/cpp/cache/CacheConfig.h`

```cpp
struct CacheConfig {
    // 新增：是否分离 K/V 存储（昇腾 NPU 路径需要）
    bool separate_kv_cache = false;

    // 现有字段
    size_t kv_block_stride_bytes = 0;  // 合并存储时使用
    size_t k_block_stride_bytes = 0;   // 新增：分离存储时的 K stride
    size_t v_block_stride_bytes = 0;   // 新增：分离存储时的 V stride
};
```

> **注意**: 代码中已有 `k_block_size()` 和 `v_block_size()` 方法（`MHAKVCacheSpec.h:37-42`），说明设计上已预留 K/V 分离能力，只需打通整个链路。

#### 2.2.2 BufferTypes 数据结构扩展

**文件**: `rtp_llm/cpp/cache/BufferTypes.h`

```cpp
struct CacheLayerLayout {
    // 现有字段（合并存储）
    std::vector<torch::Tensor> layers_to_kv_buffer_ptrs;
    std::vector<torch::Tensor> layers_to_scale_buffer_ptrs;

    // 新增字段（分离存储）
    std::vector<torch::Tensor> layers_to_k_buffer_ptrs;   // K cache per layer
    std::vector<torch::Tensor> layers_to_v_buffer_ptrs;   // V cache per layer
};
```

#### 2.2.3 BlockPool 内存分配支持分离

**文件**: `rtp_llm/cpp/cache/BlockPool.h` / `BlockPool.cc`

在 `initializeCacheBuffer()` 中增加分离分配路径：

```cpp
void BlockPool::initializeCacheBuffer() {
    auto device = (allocation_type_ == AllocationType::HOST)
                  ? torch::kCPU : getTorchDevice();

    if (separate_kv_cache_) {
        // 分离分配 K/V cache
        k_cache_buffer_ = torch::empty(
            {total_block_num, k_block_stride_bytes},
            torch::dtype(torch::kUInt8).device(device));
        v_cache_buffer_ = torch::empty(
            {total_block_num, v_block_stride_bytes},
            torch::dtype(torch::kUInt8).device(device));
    } else {
        // 合并分配（现有逻辑）
        cache_aligned_buffer_ = torch::empty(
            {total_size_bytes},
            torch::dtype(torch::kUInt8).device(device));
    }
}
```

#### 2.2.4 KVCache / LayerKVCache C++ 绑定扩展

**文件**: `rtp_llm/models_py/bindings/OpDefs.h`

```cpp
struct KVCache {
    // 现有字段（合并存储）
    std::vector<torch::Tensor> kv_cache_base_by_layer;
    std::vector<torch::Tensor> kv_scale_base_by_layer;

    // 新增字段（分离存储）
    std::vector<torch::Tensor> k_cache_base_by_layer;
    std::vector<torch::Tensor> v_cache_base_by_layer;

    bool separate_kv_cache = false;
    // ... 其他现有字段
};

struct LayerKVCache {
    torch::Tensor kv_cache_base;     // 合并存储
    torch::Tensor kv_scale_base;     // FP8 scale
    torch::Tensor k_cache_base;      // 新增：分离的 K cache
    torch::Tensor v_cache_base;      // 新增：分离的 V cache
};
```

#### 2.2.5 getLayerCache 方法适配

**文件**: `OpDefs.h:49` `getLayerCache()`

```cpp
LayerKVCache KVCache::getLayerCache(int idx) {
    LayerKVCache layer_cache;

    if (separate_kv_cache) {
        // 分离存储模式：分别 reshape 为 NHD layout
        // NHD: [kernel_block_num, kernel_seq_size_per_block, num_kv_heads, head_dim]
        layer_cache.k_cache_base = k_cache_base_by_layer[idx].reshape(
            {kernel_block_num, (int64_t)kernel_seq_size_per_block,
             (int64_t)num_kv_heads, (int64_t)head_dim});
        layer_cache.v_cache_base = v_cache_base_by_layer[idx].reshape(
            {kernel_block_num, (int64_t)kernel_seq_size_per_block,
             (int64_t)num_kv_heads, (int64_t)head_dim});
    } else {
        // 合并存储模式（现有逻辑）
        layer_cache.kv_cache_base = kv_cache_base_by_layer[idx].reshape(
            {kernel_block_num, 2, (int64_t)num_kv_heads,
             (int64_t)kernel_seq_size_per_block, (int64_t)head_dim});
    }
    return layer_cache;
}
```

> **设计要点**: 分离存储模式下，`getLayerCache()` 直接将 K/V reshape 为昇腾 NPU 需要的 NHD layout `[blocks, seq_per_block, kv_heads, head_dim]`，避免在 Python 端再做 permute。

#### 2.2.6 配置传递链路

```
启动参数 / 配置文件
    │
    └─ KVCacheConfig.separate_kv_cache = true  (昇腾 NPU 路径)
         │
         ├─ CacheConfig.separate_kv_cache
         │
         ├─ BlockPool::initializeCacheBuffer() → 分配 k_cache_buffer_ + v_cache_buffer_
         │
         ├─ KVCacheManager::getMainModelCacheLayerLayout() → 返回分离的 layer layout
         │
         └─ PyWrappedModel → KVCache 结构 → Python LayerKVCache
```

### 2.3 无需修改的组件

| 组件 | 文件 | 原因 |
|------|------|------|
| `FullKVCacheGroup` | `FullKVCacheGroup.cc` | 纯 block 计数 + 引用计数，无设备 API |
| `LinearKVCacheGroup` | `LinearKVCacheGroup.cc` | 稀疏 block 分配逻辑，无设备 API |
| `BlockCache` | `BlockCache.cc` | 纯 LRU 数据结构 |
| `BlockRefCounter` | `BlockRefCounter.h` | 纯引用计数数组 |
| `MemoryLayoutStrategy` | `MemoryLayoutStrategy.cc` | 基于 `torch::Tensor` reshape/narrow，设备无关 |
| `KVCacheSpec` (MHA/MLA/Linear) | `KVCacheSpecBase.h` 等 | 纯 block 大小计算 |

---

## 三、KV Cache 写入适配（替代 FlashInfer）

### 3.1 现状分析

rtp-llm 的 KV Cache 写入依赖 FlashInfer 库的 CUDA kernel：

**MHA 写入**（`kv_cache_write_op.py:35-90`）：
```python
flashinfer.page.append_paged_kv_cache(
    key, value,
    params.batch_indice_d, params.positions_d,
    (k_cache, v_cache),      # [num_pages, kv_heads, page_size, head_dim] (HND)
    params.page_indice_d,
    params.decode_page_indptr_d,
    params.paged_kv_last_page_len_d,
    "HND",                    # FlashInfer HND layout
)
```

**MLA 写入**（`mla_kv_cache_write_op.py:35-56`）：
```python
compute_ops.concat_and_cache_mla(
    append_ckv_t, key_pe,
    kv_cache.kv_cache_base,   # [blocks, block_size, kv_lora_rank + rope_head_dim]
    fmha_params.slot_mapping,
    self.kv_cache_type,        # "fp8_ds_mla" 或 "auto"
    self.scale,
)
```

### 3.2 MHA KV Cache 写入 — NPU 适配方案

**参考**: vllm-ascend 使用 `torch_npu._npu_reshape_and_cache()` 实现相同功能。

**新增文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_kv_cache_write_op.py`

```python
import torch_npu

class AscendKVCacheWriteOp:
    def forward(self, key, value, kv_cache, fmha_params):
        if kv_cache is None:
            return

        # 分离存储 + NHD layout（由 C++ getLayerCache() 保证）
        k_cache = kv_cache.k_cache_base   # [pages, seq_per_block, kv_heads, head_dim] NHD
        v_cache = kv_cache.v_cache_base   # 同上

        slot_mapping = fmha_params.slot_mapping

        torch_npu._npu_reshape_and_cache(
            key=key,
            value=value,
            key_cache=k_cache,
            value_cache=v_cache,
            slot_indices=slot_mapping,
        )
```

### 3.3 Layout 差异与 NHD 配置适配

| 框架 | KV Cache Shape | Layout 说明 |
|------|---------------|------------|
| rtp-llm (MHA 现有) | `[num_pages, 2, kv_heads, page_size, head_dim]` | K/V 在 dim=1 合并，HND |
| rtp-llm (MHA 昇腾) | `[num_pages, page_size, kv_heads, head_dim]` × 2 | K/V 独立存储，NHD |
| vllm-ascend | `[2, num_blocks, block_size, num_kv_heads, head_size]` | K/V 在 dim=0 分离，NHD |
| xllm | `[n_blocks, block_size, n_kv_heads, head_dim]` × 2 | K/V 分开存储，NHD |

#### 3.3.1 问题根源

rtp-llm MHA 默认使用 **HND layout**（`[pages, kv_heads, seq, dim]`），而昇腾 NPU 的 `torch_npu` 算子（`_npu_reshape_and_cache`、`_npu_paged_attention`、`npu_fused_infer_attention_score`）均要求 **NHD layout**（`[pages, seq, kv_heads, dim]`）。

HND vs NHD 的区别：
- **HND**: 同一个 head 的所有 token 在内存中连续存储，对 GPU attention kernel 友好
- **NHD**: 同一个 token 的所有 head 连续存储，对 token 级别操作友好

#### 3.3.2 解决方案：KV 分离 + NHD Layout

通过 KV 分离存储适配（见 2.2 节），在 C++ 端 `getLayerCache()` 中直接将分离的 K/V reshape 为 NHD layout，避免 Python 端额外 permute：

```cpp
// C++ OpDefs.h getLayerCache() - 昇腾路径
layer_cache.k_cache_base = k_cache_base_by_layer[idx].reshape(
    {kernel_block_num, kernel_seq_size_per_block, num_kv_heads, head_dim});
// 结果: [blocks, seq_per_block, kv_heads, head_dim] = NHD layout
```

Python 端直接使用分离后的 NHD tensor，无需 permute：
```python
# ascend_kv_cache_write_op.py
k_cache = kv_cache.k_cache_base   # 已经是 NHD [pages, seq, kv_heads, dim]
v_cache = kv_cache.v_cache_base   # 同上
```

#### 3.3.3 NHD Layout 对 slot_mapping 和 block_table 的影响

**不需要修改** `slot_mapping` 和 `block_table` 的计算逻辑。Layout 只影响 slot 内部的数据排布，不影响逻辑索引。详见 `kv cache管理-rtp-llm.md` 中的分析。

#### 3.3.4 需要修改 Layout 配置的地方

| 修改项 | 文件 | 说明 |
|--------|------|------|
| KV Cache reshape 维度顺序 | `OpDefs.h` `getLayerCache()` | 昇腾路径使用 `[blocks, seq, kv_heads, dim]` |
| Attention wrapper layout 参数 | `ascend_prefill.py` / `ascend_decode.py` | `input_layout="TND"` 传给 `npu_fused_infer_attention_score` |
| `slot_mapping` 计算 | `CudaFlashInfer.cc` / `FlashInferMlaParams.cc` | **不需要修改** |
| `block_table` 计算 | 同上 | **不需要修改** |

### 3.4 MLA KV Cache 写入 — NPU 适配方案

**新增文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_mla_kv_cache_write_op.py`

**vllm-ascend 参考**: 使用 `torch_npu.npu_kv_rmsnorm_rope_cache()` 融合 RMSNorm + RoPE + cache write

**方案** (分步实现):
```python
class AscendMlaKVCacheWriteOp:
    def forward(self, append_ckv_t, key_pe, kv_cache, fmha_params, total_global_ids=None):
        if kv_cache is None:
            return
        slot_mapping = fmha_params.slot_mapping

        # 拼接 compressed KV 和 RoPE key
        concat_kv = torch.cat([append_ckv_t, key_pe], dim=-1)
        # [tokens, kv_lora_rank + rope_head_dim]

        # MLA cache 使用独立的 k_cache_base（与 MHA 分离存储一致）
        cache_base = kv_cache.k_cache_base  # 或使用专门的 mla_cache 字段
        # [blocks, block_size, kv_lora_rank + rope_head_dim]

        # 使用 torch_npu scatter 写入
        # 或使用 reshape_and_cache 的 MLA 模式
        torch_npu._npu_reshape_and_cache(
            key=concat_kv.unsqueeze(1),   # 添加 kv_heads 维度
            value=concat_kv.unsqueeze(1),
            key_cache=cache_base,
            value_cache=cache_base,
            slot_indices=slot_mapping,
        )
```

**注意**: MLA FP8 量化路径 (`kv_cache_type == "fp8_ds_mla"`) 在当前阶段不实现。

### 3.5 Slot Mapping 生成 — C++ 层适配

**文件**: `rtp_llm/cpp/cuda/ops/CudaFlashInfer.cc:165-232`, `FlashInferMlaParams.cc:480`

**现状**: 在 C++ 端 CPU 计算 slot_mapping，然后 `cudaMemcpyAsync` 到 GPU

**适配**: slot_mapping 的计算逻辑（纯 CPU 端整数运算）设备无关，只需替换 H2D 拷贝：

```cpp
#if USING_ASCEND
    // 方案1: torch tensor copy (推荐)
    buf_d_.copy_(buf_h_);
    // 方案2: aclrtMemcpyAsync
    aclrtMemcpyAsync(buf_d_.data_ptr(), buf_d_.nbytes(),
                     buf_h_.data_ptr(), buf_h_.nbytes(),
                     ACL_MEMCPY_HOST_TO_DEVICE,
                     c10_npu::getCurrentNPUStream());
#endif
```

### 3.6 其他 CUDA API 替换

| 现有 API | 文件 | NPU 替换 |
|----------|------|---------|
| `at::cuda::getCurrentCUDAStream()` | `CudaFlashInfer.cc` | `c10_npu::getCurrentNPUStream()` |
| `at::cuda::currentStreamCaptureStatus()` | `CudaFlashInfer.cc` | 自定义标志位或 `torch.npu` 等价 |
| `cudaMemcpyAsync` H2D | `CudaFlashInfer.cc:234-255` | `aclrtMemcpyAsync` 或 `tensor.copy_()` |

---

## 四、KV Cache 读取适配（替代 FlashInfer Attention）

### 4.1 现状分析

| 场景 | 现有实现 | CUDA 依赖 |
|------|---------|----------|
| MHA Prefill | FlashInfer `BatchPrefillWithPagedKVCacheWrapper` (HND) | FlashInfer CUDA kernel |
| MHA Decode | FlashInfer `BatchDecodeWithPagedKVCacheWrapper` (HND) | FlashInfer CUDA kernel |
| MLA Prefill | FlashInfer `BatchPrefillWithRaggedKVCacheWrapper` (NHD) | FlashInfer + Triton kernel |
| MLA Decode | FlashInfer `BatchMLAPagedAttentionWrapper` | FlashInfer MLA CUDA kernel |

### 4.2 MHA Prefill — NPU 适配方案

**新增文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_prefill.py`

**参考**: vllm-ascend 使用 `torch_npu.npu_fused_infer_attention_score()`

```python
import torch_npu

class AscendPrefillOp:
    def forward(self, q, kv_cache, params):
        # 分离存储 + NHD layout（由 C++ getLayerCache() 保证）
        k_cache = kv_cache.k_cache_base   # [blocks, seq_per_block, kv_heads, head_dim] NHD
        v_cache = kv_cache.v_cache_base   # 同上

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=q,                          # [num_tokens, num_heads, head_dim]
            key=k_cache,
            value=v_cache,
            atten_mask=params.attn_mask,
            block_table=params.block_table,    # 2D [batch_size, max_blocks]
            input_layout="TND",
            block_size=params.block_size,
            actual_seq_lengths=params.actual_seq_lengths_q,
            actual_seq_lengths_kv=params.actual_seq_lengths_kv,
            num_key_value_heads=params.num_kv_heads,
            num_heads=params.num_heads,
            scale=params.scale,
            sparse_mode=3,                     # causal mask
        )
        return attn_output
```

### 4.3 MHA Decode — NPU 适配方案

**新增文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py`

**参考**: vllm-ascend 使用 `torch_npu._npu_paged_attention()`

```python
import torch_npu

class AscendDecodeOp:
    def forward(self, q, kv_cache, params):
        # 分离存储 + NHD layout（由 C++ getLayerCache() 保证）
        k_cache = kv_cache.k_cache_base   # [blocks, seq_per_block, kv_heads, head_dim] NHD
        v_cache = kv_cache.v_cache_base   # 同上

        output = torch.empty_like(q)
        torch_npu._npu_paged_attention(
            query=q,                           # [batch_size, num_heads, head_dim]
            key_cache=k_cache,
            value_cache=v_cache,
            num_kv_heads=params.num_kv_heads,
            num_heads=params.num_heads,
            scale_value=params.scale,
            block_table=params.block_table,    # 2D [batch_size, max_blocks]
            context_lens=params.seq_lens,      # [batch_size]
            out=output,
        )
        return output
```

### 4.4 Block Table 格式适配

**关键问题**: rtp-llm 使用 CSR 格式（`page_indice` + `page_indptr`），vllm-ascend 使用 2D 格式。

**rtp-llm C++ 端**已同时维护两种格式：
- 2D: `kv_cache_block_id_host` `[batch_size, max_batch_blocks]`
- 1D: `page_indice` (展平形式) + `page_indptr` (CSR indptr)

**适配**: 将 2D block table 直接传递给 `torch_npu` 算子。

### 4.5 MLA Attention — NPU 适配方案（后期）

MLA 模型（DeepSeek V2/V3）需要更复杂的适配。

**vllm-ascend 参考**：
- 使用 `torch_npu.npu_fused_infer_attention_score_v2()` 支持 separated RoPE dimensions
- 使用 `torch_npu.npu_kv_rmsnorm_rope_cache()` 融合 RMSNorm + RoPE + cache write
- MLA cache 使用两个独立 tensor：nope_cache (`kv_lora_rank`) 和 rope_cache (`qk_rope_head_dim`)

**rtp-llm MLA 现状**：
- 使用单一 3D tensor `[blocks, block_size, kv_lora_rank + rope_head_dim]`（nope + rope 拼接）
- 需评估是否需拆分为两个独立 tensor 以适配 `npu_fused_infer_attention_score_v2`

**xllm MLA 参考**：
- NPU 上 DeepSeek-V3 + prefix cache 使用 `ACL_FORMAT_FRACTAL_NZ` + 16 字节对齐
- 专用 MLA decode node `decode_mla_node_` 使用自定义 MLA kernel
- Fused MLA 参数: `param.enableFusedMLA = true, param.isNzCache = true`

**建议**: MLA 适配放在 MHA 完成后，单独一个子阶段。

---

## 五、KV Cache Copy Kernel 替换

### 5.1 CUDA Kernel 清单及替换方案

| CUDA Kernel | 源文件 | 功能 | NPU 替换方案 |
|------------|--------|------|-------------|
| `convertOffsetAndSize2IdxKernel` | `kv_cache_kernels.cu` | offset+size 转 index | `torch::Tensor` 算术操作 |
| `reuseCacheKernel` | `kv_cache_kernels.cu` | 复用 prefix cache block | `torch.index_copy_` + `torch.index_select` |
| `concat_and_cache_mla` | `mla_quant_kernel.h` | MLA KV 写入 + FP8 量化 | `torch_npu._npu_reshape_and_cache` + scatter |
| `indexer_k_quant_and_cache` | `mla_quant_kernel.h` | Indexer K 量化写入 | 后期（MLA FP8） |
| `cp_gather_and_upconvert_fp8_kv_cache` | `mla_quant_kernel.h` | FP8 cache 反量化读取 | 后期（MLA FP8） |

### 5.2 reuseCacheKernel 替换

参考 xllm `KVCacheImpl::swap_blocks()` 使用 PyTorch 原生操作：

```python
def reuse_cache_block(src_block_ids, dst_block_ids, kv_cache_tensor):
    selected = torch.index_select(kv_cache_tensor, 0, src_block_ids)
    kv_cache_tensor.index_copy_(0, dst_block_ids, selected)
```

---

## 六、参数结构适配

### 6.1 新增 Ascend Attention 参数类

**新增文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_attn_params.py`

```python
class AscendAttnParams:
    block_table: torch.Tensor       # [batch_size, max_blocks_per_seq] (2D)
    seq_lens: torch.Tensor          # [batch_size]
    slot_mapping: torch.Tensor      # [num_tokens]
    attn_mask: torch.Tensor         # attention mask
    actual_seq_lengths_q: list      # Q 累积序列长度
    actual_seq_lengths_kv: list     # KV 累积序列长度
    num_kv_heads: int
    num_heads: int
    head_dim: int
    block_size: int
    scale: float
```

### 6.2 C++ 端参数生成

rtp-llm 的 `FlashInferAttnParams::fillFlashInfer()` 需要增加 Ascend 路径：
- 保留 `slot_mapping` 和 2D `kv_cache_block_id` 的生成逻辑（设备无关）
- 替换 `cudaMemcpyAsync` 为 NPU H2D
- 替换 `at::cuda::getCurrentCUDAStream()` 为 NPU stream

---

## 七、Factory 注册

修改 `rtp_llm/models_py/modules/factory/attention/attn_factory.py`：

```python
if device_type == "npu":
    KV_CACHE_WRITE_OPS.append(AscendKVCacheWriteOp)
    MLA_KV_CACHE_WRITE_OPS.append(AscendMlaKVCacheWriteOp)
    PREFILL_MHA_IMPS.append(AscendPrefillOp)
    DECODE_MHA_IMPS.append(AscendDecodeOp)
```

---

## 八、可借鉴的设计模式

### 8.1 vllm-ascend Device Adaptor 模式

```python
# vllm_ascend/device/device_op.py
class BaseDeviceAdaptor:       # Ascend 910B
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu._npu_reshape_and_cache(...)

class A5DeviceAdaptor(BaseDeviceAdaptor):  # Ascend A5
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu.npu_scatter_pa_kv_cache(...)  # A5 专用

DeviceOperator = get_device_adaptor()  # 根据设备类型选择
```

rtp-llm 可在 `rtp_llm/utils/device_utils.py` 中封装类似的 NPU 设备适配器。

### 8.2 NPU KV Cache 内存分配经验

**vllm-ascend**:
- 原始 buffer 用 `torch.int8` 按字节分配，再 reshape
- 2MB 内存对齐（PD 分离场景）
- K/V cache 分离为独立 tensor

**xllm**:
- NPU 使用 `torch::empty` + `npu_format_cast(ACL_FORMAT_ND)`（非 `torch::zeros`）
- DeepSeek-V3 + prefix cache 用 `ACL_FORMAT_FRACTAL_NZ` + 16 字节对齐
- XTensor 模式支持虚拟内存管理（rtp-llm 可后期考虑）

### 8.3 xllm ACL Graph 经验

- 标准 `npu_paged_attention` 内部有 `.to(kCPU)` 会打断 ACL Graph
- 提供了 `npu_custom_paged_attention` 变体，通过预计算 `tiling_data` 避免 CPU 同步
- rtp-llm 在 Phase 7 (CUDA Graph → Ascend Graph) 时需参考此方案

---

## 九、完整文件变更清单

| 类别 | 文件 | 变更类型 | 说明 |
|------|------|---------|------|
| **C++ 配置** | `rtp_llm/cpp/cache/CacheConfig.h` | 修改 | 添加 `separate_kv_cache`、`k_block_stride_bytes`、`v_block_stride_bytes` |
| **C++ 数据结构** | `rtp_llm/cpp/cache/BufferTypes.h` | 修改 | 添加 `layers_to_k_buffer_ptrs`、`layers_to_v_buffer_ptrs` |
| **C++ 内存管理** | `rtp_llm/cpp/cache/BlockPool.cc` | 修改 | 设备参数化 + 支持分离分配 K/V tensor |
| | `rtp_llm/cpp/cache/BlockPool.cc` (where()) | 修改 | `is_cuda()` → 设备无关检测 |
| | `rtp_llm/cpp/cache/MemoryEvaluationHelper.cc` | 修改 | `cudaMemGetInfo` → `aclrtGetMemInfo` |
| | `rtp_llm/cpp/cache/KVCacheAllocator.cc` | 修改 | `execBatchCopy` NPU 路径（含分离 K/V 拷贝） |
| | `rtp_llm/cpp/cache/KVCacheManager.cc` | 修改 | `cudaSyncAndCheck` → NPU sync + 分离 layout 传递 |
| **C++ 绑定** | `rtp_llm/models_py/bindings/OpDefs.h` | 修改 | `KVCache`/`LayerKVCache` 添加分离字段 + NHD reshape |
| **C++ 参数生成** | `rtp_llm/cpp/cuda/ops/CudaFlashInfer.cc` | 修改 | stream + memcpy NPU 替换 |
| | `rtp_llm/cpp/cuda/ops/FlashInferMlaParams.cc` | 修改 | stream + memcpy NPU 替换 |
| **Python 类型** | `rtp_llm/ops/librtp_compute_ops/rtp_llm_ops.pyi` | 修改 | `LayerKVCache` 添加 `k_cache_base`、`v_cache_base` |
| **Python 写入** | `ascend_impl/ascend_kv_cache_write_op.py` | **新增** | MHA KV write（使用分离 K/V + NHD） |
| | `ascend_impl/ascend_mla_kv_cache_write_op.py` | **新增** | MLA KV write（使用分离 cache） |
| **Python 读取** | `ascend_impl/ascend_prefill.py` | **新增** | Prefill attention（使用分离 K/V + NHD） |
| | `ascend_impl/ascend_decode.py` | **新增** | Decode attention（使用分离 K/V + NHD） |
| **Python 参数** | `ascend_impl/ascend_attn_params.py` | **新增** | Ascend 参数结构 |
| **Factory** | `attn_factory.py` | 修改 | 注册 Ascend 实现 |
| **工具** | `rtp_llm/utils/device_utils.py` | **新增** | NPU 设备适配器 |

---

## 十、关键风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| KV 分离存储导致内存分配和管理逻辑大幅改动 | BlockPool、KVCacheManager 等核心组件需全面适配 | 利用已有的 `k_block_size()` / `v_block_size()` 预留接口；分离存储作为配置开关，不影响 CUDA 路径 |
| NPU 算子要求 NHD layout，与 rtp-llm HND layout 不兼容 | KV cache 读写失败 | 通过 C++ `getLayerCache()` 直接 reshape 为 NHD，避免 Python 端 permute 开销 |
| `torch_npu._npu_reshape_and_cache` 对 NHD layout 的验证 | 写入正确性不确定 | 优先在子阶段 4.3 做单层 KV write 正确性验证 |
| `npu_fused_infer_attention_score` block_table 格式不兼容 | Prefill 不可用 | rtp-llm C++ 端已有 2D block table，确保传递正确格式 |
| `torch_npu._npu_paged_attention` 性能不达标 | Decode 性能差 | 先跑通功能，Phase 10 优化；备选 aclnn 自定义算子 |
| 分离存储的 block 批量拷贝（Prefix Cache 复用）需要同时拷贝 K/V | 拷贝逻辑复杂化 | 在 `execBatchCopy` 中对 K/V 分别执行拷贝 |
| MLA cache 需适配分离存储模式 | MLA 模型不可用 | MLA cache 使用 k_cache_base 字段，结构与 MHA 的分离模式一致 |
| `torch_npu` API 在某些 CANN 版本不可用 | 功能受阻 | 参考 vllm-ascend Device Adaptor 模式按型号选择 API |
| FP8 MLA per-128-element block quantization 与 Ascend FP8 格式不兼容 | FP8 MLA 不可用 | 先跑 FP16/BF16，FP8 在 Phase 8 统一处理 |

---

## 十一、FP8 KV Cache 适配说明（后期）

rtp-llm FP8 MLA 的 per-token 布局非常特殊：
```
[0..511]  : 512 × float8_e4m3  (quantized NoPE/CKV)
[512..527]: 4 × float32 scale  (one per 128-element block)
[528..655]: 64 × bfloat16      (RoPE part, NOT quantized)
= 656 bytes/token
```

**vllm-ascend 参考**：
- INT8 KV Cache: 使用 `npu_fused_infer_attention_score` 的 `key_antiquant_scale/offset` 参数
- FAQuant for MLA: 使用 `npu_fused_infer_attention_score_v2` 的 `dequant_scale_query/key/value`

**建议**: FP8 KV Cache 放在 Phase 8 (FP8/量化支持) 统一处理，Phase 4 先保证 FP16/BF16 路径跑通。

---

## 十二、里程碑与验证计划

| 子阶段 | 周期 | 交付物 | 验证标准 |
|--------|------|--------|---------|
| 4.1 C++ 内存管理 + KV 分离 | 4 天 | BlockPool NPU 分配 + 分离 K/V tensor + 内存查询 | `BlockPool::init()` 在 NPU 上成功分配分离的 K/V buffer |
| 4.2 Slot Mapping 适配 | 2 天 | C++ 端 H2D 拷贝 + stream 获取 | slot_mapping 正确传递到 NPU |
| 4.3 MHA KV Write | 3 天 | `AscendKVCacheWriteOp`（分离 K/V + NHD） | 单层 KV write 正确性验证 |
| 4.4 MHA Decode | 5 天 | `AscendDecodeOp`（分离 K/V + NHD） | 单请求 decode 生成正确 token |
| 4.5 MHA Prefill | 5 天 | `AscendPrefillOp`（分离 K/V + NHD） | 单请求 prefill 输出正确 |
| 4.6 Factory 注册 | 2 天 | Attention factory NPU 路径 | 端到端推理可跑通（FP16） |
| 4.7 MLA 适配 | 5 天 | MLA write + decode（分离 cache） | DeepSeek V2/V3 可推理 |
| 4.8 Batch Copy 适配 | 2 天 | Block 批量拷贝 NPU 路径（含 K/V 分离拷贝） | Prefix cache 功能正常 |
| **合计** | **~3.5 周** | | |
