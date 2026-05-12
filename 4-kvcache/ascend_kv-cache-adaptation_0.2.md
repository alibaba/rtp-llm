# Phase 4 增强：KV Cache 昇腾适配详细计划 (v0.2)

> 本文档是对 `ascend_kv-cache-adaptation.md` 的修正版本。
> 基于对 rtp-llm **最新代码**（2026-05）的重新审查，修正了文件路径、设备类型、已适配组件等错误描述。
> 变更标记：`[修正]` 表示对 v0.1 的修正，`[新增]` 表示新增内容，`[不变]` 表示方案不变。

---

## 〇、v0.1 方案主要修正汇总

| # | v0.1 问题 | 修正 |
|---|----------|------|
| 1 | 多处使用 `torch::kNPU` | 实际代码使用 `torch::kPrivateUse1` |
| 2 | C++ 层"需要修改"的 5 个组件（2.1.1~2.1.5）描述为未适配状态 | **已全部完成基础 Ascend 适配**，仅需增量修改（分离 K/V） |
| 3 | 文件路径 `rtp_llm/cpp/cuda/ops/CudaFlashInfer.cc` 不存在 | 实际文件为 `rtp_llm/models_py/bindings/cuda/FlashInferMlaParams.cc` |
| 4 | Factory 注册说修改 `attn_factory.py` | 实际注册点在 `__init__.py`，且已有 Ascend 基础注册 |
| 5 | `ascend_impl/` 列为"新增文件" | 目录已存在，含 `torch_sdpa.py`（基础 SDPA 实现） |
| 6 | 方案缺少 RoPE Ascend 替换方案 | 新增 RoPE 适配章节 |
| 7 | 未区分 MHA/MLA 参数生成路径差异 | MHA 主要在 Python 端，MLA 在 C++ 端 |
| 8 | `KVCacheAllocator.cc` 说使用 `cudaMemcpyAsync` | 实际已通过 `runtimeBatchCopy()` 抽象，有 Ascend fallback |
| 9 | stream/memcpy 替换描述为手动替换 | 已通过 `GET_CURRENT_STREAM()` 宏和 `runtimeSyncAndCheck()` 抽象 |

---

## 一、三框架 KV Cache 适配方案对比

> [不变] 此节保持 v0.1 内容，对比表和选型理由仍然成立。

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

> [修正] v0.1 将 5 个组件全部列为"需要修改"，实际 **已全部完成基础 Ascend 适配**。
> 本节重新分类为：**已完成（仅列举现状）** 和 **仍需增量修改** 两类。

### 2.1 已完成基础 Ascend 适配的组件（无需额外修改）

#### 2.1.1 BlockPool 内存分配

**文件**: `rtp_llm/cpp/cache/BlockPool.cc:39-61`

[修正] v0.1 描述为硬编码 `torch::kCUDA`，实际已有 `USING_ASCEND` 分支：

```cpp
// BlockPool.cc:44-52 (当前代码)
#if USING_CUDA
    device = torch::kCUDA;
#elif USING_ASCEND
    device = torch::Device(torch::kPrivateUse1);  // 注意：不是 torch::kNPU
#elif USING_ROCM
    device = torch::kCUDA;
#else
    device = torch::kCPU;
#endif
```

**结论**：设备选择逻辑已完成，无需修改。后续分离 K/V 分配需在此基础上扩展。

#### 2.1.2 GPU 内存查询

**文件**: `rtp_llm/cpp/cache/MemoryEvaluationHelper.cc:60-68`

[修正] v0.1 建议使用 `aclrtGetMemInfo`，实际已封装为 `ascend::getDeviceMemoryInfo`：

```cpp
// MemoryEvaluationHelper.cc:60-68 (当前代码)
#if USING_CUDA
    check_cuda_value(cudaMemGetInfo(&free_gpu_bytes, &total_gpu_bytes));
#elif USING_ROCM
    ROCM_CHECK(hipMemGetInfo(&free_gpu_bytes, &total_gpu_bytes));
#elif USING_ASCEND
    auto [used_bytes, free_bytes] = rtp_llm::ascend::getDeviceMemoryInfo(false);
    free_gpu_bytes  = free_bytes;
    total_gpu_bytes = used_bytes + free_bytes;
#endif
```

**结论**：已完成，无需修改。

#### 2.1.3 Block 批量拷贝

**文件**: `rtp_llm/cpp/cache/KVCacheAllocator.cc:141-192`

[修正] v0.1 说使用 `cudaMemcpyAsync`，实际已通过 `execBatchCopy()` → `runtimeBatchCopy()` 抽象。Ascend 路径使用 `batchCopyFallback()` + `torch::kPrivateUse1`（具体实现在 `rtp_llm/models_py/bindings/core/CudaOps.cc:234-298`）。

**结论**：基础 D2D/H2D/D2H 拷贝已完成。后续分离 K/V 存储后，需扩展支持同时拷贝 K/V 两个独立 tensor。

#### 2.1.4 CUDA 同步调用

**文件**: `rtp_llm/cpp/cache/KVCacheManager.cc:219,507`

[修正] v0.1 建议手动替换为 NPU stream 同步，实际已通过 `cudaSyncAndCheck()` → `runtimeSyncAndCheck()` 抽象：

- **CUDA 路径** (`ExecOps.cc:109-112`): `cudaDeviceSynchronize()` + `check_cuda_error()`
- **Ascend 路径** (`ExecOps.cc:123-126`): `aclrtSynchronizeDevice()` + `ASCEND_CHECK_ERROR()`

此外，`KVCacheManager.cc:196-201` 的 `setKVBlockValue()` 也有 inline `#if USING_ASCEND` guard 处理设备检测：

```cpp
auto dst_device = dst_block.is_cuda ?
#if USING_ASCEND
    torch::Device(torch::kPrivateUse1) :
#else
    torch::kCUDA :
#endif
    torch::kCPU;
```

**结论**：已完成，无需修改。

#### 2.1.5 BlockPool::where() 设备检测

**文件**: `rtp_llm/cpp/cache/BlockPool.cc:513-519`

[修正] 已有 Ascend 分支：

```cpp
MemoryType BlockPool::where() const {
#if USING_ASCEND
    return cache_aligned_buffer_.is_privateuseone() ? MemoryType::MEMORY_NPU : MemoryType::MEMORY_CPU;
#else
    return cache_aligned_buffer_.is_cuda() ? MemoryType::MEMORY_GPU : MemoryType::MEMORY_CPU;
#endif
}
```

**结论**：已完成，无需修改。

#### 2.1.6 Stream 获取抽象

**文件**: `rtp_llm/models_py/bindings/common/Torch_ext.h:28-34`

[修正] v0.1 中多处建议手动替换 stream 获取，实际已通过宏抽象：

```cpp
#if USING_CUDA
#define GET_CURRENT_STREAM() at::cuda::getCurrentCUDAStream(at::cuda::current_device()).stream()
#elif USING_ASCEND
#define GET_CURRENT_STREAM() c10_npu::getCurrentNPUStream().stream()
#elif USING_ROCM
#define GET_CURRENT_STREAM() at::hip::getCurrentHIPStream().stream()
#endif
```

**结论**：已完成。新增的 C++ 代码只需使用 `GET_CURRENT_STREAM()` 宏即可。

### 2.2 无需修改的组件

> [修正] 补充了 MemoryLayoutStrategy 已有的 Ascend 适配。

| 组件 | 文件 | 原因 |
|------|------|------|
| `FullKVCacheGroup` | `FullKVCacheGroup.cc` | 纯 block 计数 + 引用计数，无设备 API |
| `LinearKVCacheGroup` | `LinearKVCacheGroup.cc` | 稀疏 block 分配逻辑，无设备 API |
| `BlockCache` | `BlockCache.cc` | 纯 LRU 数据结构 |
| `BlockRefCounter` | `BlockRefCounter.h` | 纯引用计数数组 |
| `MemoryLayoutStrategy` | `MemoryLayoutStrategy.cc` | 基于 `torch::Tensor` reshape/narrow，已有 Ascend 适配（line 285: `dev.is_cuda() \|\| dev.is_privateuseone()`） |
| `KVCacheSpec` (MHA/MLA/Linear) | `KVCacheSpecBase.h` 等 | 纯 block 大小计算 |
| `CacheConfig` | `CacheConfig.h` | 纯配置结构，无设备 API |

### 2.3 [新增] 仍需增量修改的组件 — KV 分离存储

> 这是 v0.1 第二节的核心增量工作，所有组件的基础 Ascend 适配已完成，
> 但 **K/V 分离存储** 功能尚未实现。

昇腾 NPU 的 `torch_npu` 算子（`_npu_reshape_and_cache`、`_npu_paged_attention`、`npu_fused_infer_attention_score`）要求 `key_cache` 和 `value_cache` 作为**独立的 Tensor** 传入。rtp-llm 当前将 K/V 合并在一个 Tensor 中，通过 `getLayerCache()` reshape 为 `[kernel_block_num, 2, num_kv_heads, kernel_seq_size_per_block, head_dim]`（dim=1 分离 K/V）。

#### 2.3.1 CacheConfig 配置扩展

**文件**: `rtp_llm/cpp/cache/CacheConfig.h`

[修正] 当前文件无 `separate_kv_cache` 字段，确认需要新增。

```cpp
struct CacheConfig {
    // 新增：是否分离 K/V 存储（昇腾 NPU 路径需要）
    bool separate_kv_cache = false;

    // 现有字段保持不变
    size_t kv_block_stride_bytes = 0;  // 合并存储时使用
    size_t kv_scale_stride_bytes = 0;

    // 新增：分离存储时的 K/V stride
    size_t k_block_stride_bytes = 0;
    size_t v_block_stride_bytes = 0;
};
```

> **注意**: `MHAKVCacheSpec.h:37-42` 已有 `k_block_size()` 和 `v_block_size()` 方法（各返回 `local_head_num_kv * size_per_head * seq_size_per_block`），设计上已预留 K/V 分离能力，只需打通整个链路。

#### 2.3.2 BufferTypes 数据结构扩展

**文件**: `rtp_llm/cpp/cache/BufferTypes.h`

当前仅有合并的 `layers_to_kv_buffer_ptrs`，需新增分离字段：

```cpp
struct CacheLayerLayout {
    // 现有字段（合并存储）
    std::vector<int>            layer_to_groups;
    std::vector<CacheGroupType> group_types;
    std::vector<CacheGroupType> layer_attn_types;
    std::vector<torch::Tensor>  layers_to_kv_buffer_ptrs;
    std::vector<torch::Tensor>  layers_to_scale_buffer_ptrs;

    // 新增字段（分离存储 — 昇腾 NPU 路径使用）
    std::vector<torch::Tensor>  layers_to_k_buffer_ptrs;   // K cache per layer
    std::vector<torch::Tensor>  layers_to_v_buffer_ptrs;   // V cache per layer
};
```

#### 2.3.3 BlockPool 内存分配支持分离

**文件**: `rtp_llm/cpp/cache/BlockPool.h` / `BlockPool.cc`

在已有的 `initializeCacheBuffer()`（line 39-61）基础上增加分离分配路径：

```cpp
void BlockPool::initializeCacheBuffer() {
    // 设备选择逻辑已完成（line 40-53），保持不变
    torch::Device device = torch::kCPU;
    // ... (现有 #if USING_ASCEND 等分支保持不变)

    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(device);

    if (separate_kv_cache_) {
        // 新增路径：分离分配 K/V cache
        k_cache_buffer_ = torch::empty(
            {static_cast<int64_t>(config_.total_size_bytes / 2)}, options);
        v_cache_buffer_ = torch::empty(
            {static_cast<int64_t>(config_.total_size_bytes / 2)}, options);
        // NPU: torch::empty 自动分配在 NPU 上，无需 npu_format_cast
        //      因为 KV cache 使用 ND 格式，非 5HD/NZ 格式
    } else {
        // 现有路径：合并分配（line 55）
        cache_aligned_buffer_ = torch::empty(
            {static_cast<int64_t>(config_.total_size_bytes)}, options);
    }
    // ... pin_memory 逻辑保持不变
}
```

#### 2.3.4 KVCache / LayerKVCache C++ 绑定扩展

**文件**: `rtp_llm/models_py/bindings/OpDefs.h`

当前 `LayerKVCache`（line 25-30）和 `KVCache`（line 34-115）均为合并存储：

```cpp
// 当前 OpDefs.h:25-30
struct LayerKVCache {
    torch::Tensor kv_cache_base;     // 合并存储（现有）
    torch::Tensor kv_scale_base;
    int           seq_size_per_block = 0;
    int           layer_id           = -1;
};

// 当前 OpDefs.h:34-47
struct KVCache {
    std::vector<torch::Tensor> kv_cache_base_by_layer;     // 合并存储（现有）
    std::vector<torch::Tensor> kv_scale_base_by_layer;
    // ... 其他字段
};
```

扩展方案：

```cpp
struct LayerKVCache {
    torch::Tensor kv_cache_base;     // 合并存储（CUDA 路径）
    torch::Tensor kv_scale_base;
    int           seq_size_per_block = 0;
    int           layer_id           = -1;

    // 新增：分离存储（Ascend 路径）
    torch::Tensor k_cache_base;      // 分离的 K cache [blocks, seq_per_block, kv_heads, head_dim] NHD
    torch::Tensor v_cache_base;      // 分离的 V cache [blocks, seq_per_block, kv_heads, head_dim] NHD
};

struct KVCache {
    // 现有字段保持不变
    std::vector<torch::Tensor> kv_cache_base_by_layer;
    std::vector<torch::Tensor> kv_scale_base_by_layer;
    int                        seq_size_per_block        = 0;
    int                        kernel_seq_size_per_block = 0;
    int                        num_kv_heads              = 0;
    int                        head_dim                  = 0;
    bool                       use_mla                   = false;
    int                        kv_lora_rank              = 0;
    int                        rope_head_dim             = 0;
    std::vector<rtp_llm::CacheGroupType> layer_attn_types;

    // 新增：分离存储字段
    bool                       separate_kv_cache = false;
    std::vector<torch::Tensor> k_cache_base_by_layer;
    std::vector<torch::Tensor> v_cache_base_by_layer;
};
```

#### 2.3.5 getLayerCache 方法适配

**文件**: `rtp_llm/models_py/bindings/OpDefs.h:49-114`

当前 `getLayerCache()` 的 MHA 路径（line 85-91）返回 HND layout：

```cpp
// 当前代码 (OpDefs.h:87-91)
layer_cache.kv_cache_base = base.reshape({kernel_block_num,
                                          2,
                                          (int64_t)num_kv_heads,
                                          (int64_t)kernel_seq_size_per_block,
                                          (int64_t)head_dim});
// 结果: [kblocks, 2, kv_heads, kpage, dim] = K/V merged HND
```

新增 Ascend 路径：

```cpp
LayerKVCache getLayerCache(int idx) {
    LayerKVCache layer_cache;
    layer_cache.layer_id = idx;

    // ... (现有边界检查和 is_full 判断保持不变, line 54-62)

    if (!is_full) {
        // Linear/SSM: 不变
        // ...
    } else {
        layer_cache.seq_size_per_block =
            kernel_seq_size_per_block > 0 ? kernel_seq_size_per_block : seq_size_per_block;
        const int64_t kernel_blocks_per_kv_block =
            kernel_seq_size_per_block > 0 ? (int64_t)seq_size_per_block / (int64_t)kernel_seq_size_per_block : 1;

        if (base.defined() && base.dim() == 2) {
            const int64_t physical_block_num = base.size(0);
            const int64_t kernel_block_num   = physical_block_num * kernel_blocks_per_kv_block;

            if (separate_kv_cache) {
                // ===== 新增：Ascend 分离存储路径 =====
                auto k_base = k_cache_base_by_layer[idx];
                auto v_base = v_cache_base_by_layer[idx];

                if (use_mla && kv_lora_rank > 0 && rope_head_dim > 0) {
                    // MLA: k_base = compressed KV, v_base = RoPE key
                    layer_cache.k_cache_base = k_base.reshape(
                        {kernel_block_num, (int64_t)kernel_seq_size_per_block,
                         (int64_t)kv_lora_rank});
                    layer_cache.v_cache_base = v_base.reshape(
                        {kernel_block_num, (int64_t)kernel_seq_size_per_block,
                         (int64_t)rope_head_dim});
                } else if (num_kv_heads > 0 && head_dim > 0) {
                    // MHA: NHD layout [blocks, seq_per_block, kv_heads, head_dim]
                    layer_cache.k_cache_base = k_base.reshape(
                        {kernel_block_num, (int64_t)kernel_seq_size_per_block,
                         (int64_t)num_kv_heads, (int64_t)head_dim});
                    layer_cache.v_cache_base = v_base.reshape(
                        {kernel_block_num, (int64_t)kernel_seq_size_per_block,
                         (int64_t)num_kv_heads, (int64_t)head_dim});
                }
            } else {
                // ===== 现有：合并存储路径 (CUDA/ROCm) =====
                if (use_mla && kv_lora_rank > 0 && rope_head_dim > 0) {
                    layer_cache.kv_cache_base = base.reshape({kernel_block_num,
                                                              (int64_t)kernel_seq_size_per_block,
                                                              (int64_t)(kv_lora_rank + rope_head_dim)});
                } else if (num_kv_heads > 0 && head_dim > 0) {
                    layer_cache.kv_cache_base = base.reshape({kernel_block_num,
                                                              2,
                                                              (int64_t)num_kv_heads,
                                                              (int64_t)kernel_seq_size_per_block,
                                                              (int64_t)head_dim});
                } else {
                    layer_cache.kv_cache_base = base;
                }
            }
        } else {
            layer_cache.kv_cache_base = base;
        }

        // scale 处理保持不变
        // ...
    }
    return layer_cache;
}
```

> **设计要点**: 分离存储模式下，`getLayerCache()` 直接将 K/V reshape 为昇腾 NPU 需要的 NHD layout `[blocks, seq_per_block, kv_heads, head_dim]`，避免在 Python 端再做 permute。

#### 2.3.6 配置传递链路

```
启动参数 / 配置文件
    │
    └─ KVCacheConfig.separate_kv_cache = true  (昇腾 NPU 路径)
         │
         ├─ CacheConfig.separate_kv_cache + k_block_stride_bytes + v_block_stride_bytes
         │
         ├─ BlockPool::initializeCacheBuffer() → 分配 k_cache_buffer_ + v_cache_buffer_
         │
         ├─ MemoryLayoutStrategy → 生成分离的 per-layer K/V views
         │
         ├─ KVCacheManager::getMainModelCacheLayerLayout()
         │   → 返回含 layers_to_k_buffer_ptrs / layers_to_v_buffer_ptrs 的 layout
         │
         ├─ PyWrappedModel → 填充 KVCache.k_cache_base_by_layer / v_cache_base_by_layer
         │
         └─ KVCache.getLayerCache() → 返回含 k_cache_base / v_cache_base 的 LayerKVCache
```

#### 2.3.7 Python 类型 Stub 更新

**文件**: `rtp_llm/ops/librtp_compute_ops/__init__.pyi`

需要同步更新 Python 类型定义：

```python
class LayerKVCache:
    kv_cache_base: torch.Tensor
    kv_scale_base: torch.Tensor
    seq_size_per_block: int
    layer_id: int
    # 新增
    k_cache_base: torch.Tensor
    v_cache_base: torch.Tensor

class KVCache:
    kv_cache_base_by_layer: list[torch.Tensor]
    kv_scale_base_by_layer: list[torch.Tensor]
    # ... 现有字段
    # 新增
    separate_kv_cache: bool
    k_cache_base_by_layer: list[torch.Tensor]
    v_cache_base_by_layer: list[torch.Tensor]
```

#### 2.3.8 KVCacheManager::getMainModelCacheLayerLayout() 扩展

**文件**: `rtp_llm/cpp/cache/KVCacheManager.cc:255-298`

当前方法（line 259-260）仅使用 `layers_to_kv_buffer_ptrs`，需增加分离路径：

```cpp
CacheLayerLayout KVCacheManager::getMainModelCacheLayerLayout() const {
    CacheLayerLayout layout;
    auto all_layout = allocator_->allLayerCacheBase();

    if (config_.separate_kv_cache) {
        // 新增：分离存储路径
        auto& all_k_tensors = all_layout.layers_to_k_buffer_ptrs;
        auto& all_v_tensors = all_layout.layers_to_v_buffer_ptrs;
        layout.layers_to_k_buffer_ptrs.resize(config_.layer_num);
        layout.layers_to_v_buffer_ptrs.resize(config_.layer_num);
        for (int i = 0; i < static_cast<int>(config_.layer_num); ++i) {
            layout.layers_to_k_buffer_ptrs[i] = all_k_tensors[i];
            layout.layers_to_v_buffer_ptrs[i] = all_v_tensors[i];
        }
    } else {
        // 现有：合并存储路径
        auto& all_layer_tensors = all_layout.layers_to_kv_buffer_ptrs;
        layout.layers_to_kv_buffer_ptrs.resize(config_.layer_num);
        for (int i = 0; i < static_cast<int>(config_.layer_num); ++i) {
            layout.layers_to_kv_buffer_ptrs[i] = all_layer_tensors[i];
        }
    }
    // ... 其余逻辑保持不变
}
```

#### 2.3.9 Batch Copy 适配（Prefix Cache 复用）

**文件**: `rtp_llm/cpp/cache/KVCacheAllocator.cc`

基础 batch copy 已完成 Ascend 适配（通过 `runtimeBatchCopy` 抽象）。分离 K/V 存储后，`execBatchCopy` 需要对 K/V 分别执行拷贝：

```cpp
// 在 execBatchCopy 调用处，分离模式下需调用两次
if (separate_kv_cache_) {
    execBatchCopy(k_copy_params);  // K cache 拷贝
    execBatchCopy(v_copy_params);  // V cache 拷贝
} else {
    execBatchCopy(copy_params);    // 合并拷贝（现有逻辑）
}
```

---

## 三、KV Cache 写入适配（替代 FlashInfer）

### 3.1 现状分析

> [修正] 补充了准确的文件路径和代码行号。

rtp-llm 的 KV Cache 写入依赖 FlashInfer 库：

**MHA 写入** (`rtp_llm/models_py/modules/factory/attention/cuda_impl/kv_cache_write_op.py:51-72`)：
```python
# 从合并的 kv_cache_base 中分离 K/V（HND layout）
k_cache = kv_cache.kv_cache_base[:, 0, :, :, :]  # [num_pages, kv_heads, page_size, head_dim]
v_cache = kv_cache.kv_cache_base[:, 1, :, :, :]

flashinfer.page.append_paged_kv_cache(
    key, value,
    params.batch_indice_d, params.positions_d,
    (k_cache, v_cache),
    params.page_indice_d,
    params.decode_page_indptr_d,
    params.paged_kv_last_page_len_d,
    "HND",
)
```

**MLA 写入** (`rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/mla_kv_cache_write_op.py:44-56`)：
```python
compute_ops.concat_and_cache_mla(
    append_ckv_t, key_pe,
    kv_cache.kv_cache_base,   # [blocks, block_size, kv_lora_rank + rope_head_dim]
    fmha_params.slot_mapping,
    self.kv_cache_type,
    self.scale,
)
```

### 3.2 MHA KV Cache 写入 — NPU 适配方案

**参考**: vllm-ascend 使用 `torch_npu._npu_reshape_and_cache()` 实现相同功能。

[修正] `ascend_impl/` 目录已存在，新增文件追加到该目录。

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

> [不变] 此节分析仍然成立，保持 v0.1 内容。

| 框架 | KV Cache Shape | Layout 说明 |
|------|---------------|------------|
| rtp-llm (MHA 现有) | `[num_pages, 2, kv_heads, page_size, head_dim]` | K/V 在 dim=1 合并，HND |
| rtp-llm (MHA 昇腾) | `[num_pages, page_size, kv_heads, head_dim]` × 2 | K/V 独立存储，NHD |
| vllm-ascend | `[2, num_blocks, block_size, num_kv_heads, head_size]` | K/V 在 dim=0 分离，NHD |
| xllm | `[n_blocks, block_size, n_kv_heads, head_dim]` × 2 | K/V 分开存储，NHD |

#### 3.3.1 问题根源

rtp-llm MHA 默认使用 **HND layout**（`[pages, kv_heads, seq, dim]`），而昇腾 NPU 的 `torch_npu` 算子均要求 **NHD layout**（`[pages, seq, kv_heads, dim]`）。

#### 3.3.2 解决方案：KV 分离 + NHD Layout

通过 KV 分离存储适配（见 2.3 节），在 C++ 端 `getLayerCache()` 中直接将分离的 K/V reshape 为 NHD layout，避免 Python 端额外 permute。

```python
# ascend_kv_cache_write_op.py — 直接使用分离后的 NHD tensor
k_cache = kv_cache.k_cache_base   # 已经是 NHD [pages, seq, kv_heads, dim]
v_cache = kv_cache.v_cache_base   # 同上
```

#### 3.3.3 NHD Layout 对 slot_mapping 和 block_table 的影响

**不需要修改** `slot_mapping` 和 `block_table` 的计算逻辑。Layout 只影响 slot 内部的数据排布，不影响逻辑索引。

#### 3.3.4 需要修改 Layout 配置的地方

| 修改项 | 文件 | 说明 |
|--------|------|------|
| KV Cache reshape 维度顺序 | `OpDefs.h` `getLayerCache()` | 昇腾路径使用 `[blocks, seq, kv_heads, dim]` |
| Attention wrapper layout 参数 | `ascend_prefill.py` / `ascend_decode.py` | `input_layout="TND"` 传给 `npu_fused_infer_attention_score` |
| `slot_mapping` 计算 | **不需要修改** | 纯 CPU 整数运算，设备无关 |
| `block_table` 计算 | **不需要修改** | 同上 |

### 3.4 MLA KV Cache 写入 — NPU 适配方案

[修正] 文件路径从 v0.1 的新增改为追加到已存在的 `ascend_impl/`。

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

        # MLA cache: 使用分离的 k_cache_base
        cache_base = kv_cache.k_cache_base
        # [blocks, block_size, kv_lora_rank + rope_head_dim]

        # 使用 torch_npu scatter 写入
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

> [修正] v0.1 引用不存在的 `CudaFlashInfer.cc`，实际文件为 `FlashInferMlaParams.cc`。

**文件**: `rtp_llm/models_py/bindings/cuda/FlashInferMlaParams.cc:468-496`

**现状**: slot_mapping 的计算逻辑（纯 CPU 端整数运算）设备无关：
```cpp
// FlashInferMlaParams.cc:478-485 (设备无关的 CPU 计算)
const int32_t batch_id     = batch_indice_ptr[i];
const int32_t position     = positions_ptr[i];
const int32_t block_index  = position / seq_size_per_block;
const int32_t block_offset = position % seq_size_per_block;
const int32_t block_number = block_table_ptr[batch_id * max_blocks + block_index];
slot_mapping_ptr[i]        = static_cast<int64_t>(block_number) * seq_size_per_block + block_offset;
```

**需替换**: H2D 拷贝（line 489-490 当前使用 `cudaMemcpyAsync`）：

```cpp
// 当前代码 (FlashInferMlaParams.cc:489-490)
cudaMemcpyAsync(slot_mapping_d_.data_ptr(), slot_mapping_h_.data_ptr(),
                total_bytes, cudaMemcpyHostToDevice, stream);

// Ascend 替换方案:
// 方案1: torch tensor copy (推荐，与 runtimeBatchCopy 风格一致)
slot_mapping_d_.copy_(slot_mapping_h_);
// 方案2: aclrtMemcpyAsync
aclrtMemcpyAsync(slot_mapping_d_.data_ptr(), slot_mapping_d_.nbytes(),
                 slot_mapping_h_.data_ptr(), slot_mapping_h_.nbytes(),
                 ACL_MEMCPY_HOST_TO_DEVICE,
                 c10_npu::getCurrentNPUStream());
```

> **重要区分**: `FlashInferMlaParams.cc` 是 **MLA 参数生成**。MHA 的参数生成主要在 Python 端（通过 FlashInfer Python wrapper），不在 C++ 端。因此：
> - **MHA 路径**：需在 Python 端构建 slot_mapping 并传递给 Ascend 实现
> - **MLA 路径**：需在 C++ 端 `FlashInferMlaParams.cc` 替换 `cudaMemcpyAsync`

### 3.6 [修正] 其他 CUDA API 替换

> v0.1 列出多个手动替换项，实际已通过宏和抽象函数处理。

| API | 文件 | 状态 | 说明 |
|-----|------|------|------|
| `GET_CURRENT_STREAM()` | `Torch_ext.h:28-34` | **已抽象** | Ascend 返回 `c10_npu::getCurrentNPUStream().stream()` |
| `cudaSyncAndCheck()` | `ExecOps.cc:109-126` | **已抽象** | Ascend 路径用 `aclrtSynchronizeDevice()` |
| `cudaMemcpyAsync` (int32 buf) | `FlashInferMlaParams.cc:358` | **需替换** | MLA int32 buffer H2D |
| `cudaMemcpyAsync` (slot_mapping) | `FlashInferMlaParams.cc:489` | **需替换** | MLA slot_mapping H2D |
| `cudaMemcpyAsync` (sparse mla) | `SparseMlaParams.cc:141,255,260` | **需替换** | Sparse MLA 多个 H2D |
| `at::cuda::getCurrentCUDAStream` | `mla_quant_kernel.cu:228,288,396,883` | **需替换** | MLA CUDA kernels 中的 stream 获取 |

---

## 四、KV Cache 读取适配（替代 FlashInfer Attention）

### 4.1 现状分析

> [不变] 此节分析仍然成立。

| 场景 | 现有实现 | CUDA 依赖 |
|------|---------|----------|
| MHA Prefill | FlashInfer `BatchPrefillWithPagedKVCacheWrapper` (HND) | FlashInfer CUDA kernel |
| MHA Decode | FlashInfer `BatchDecodeWithPagedKVCacheWrapper` (HND) | FlashInfer CUDA kernel |
| MLA Prefill | FlashInfer `BatchPrefillWithRaggedKVCacheWrapper` (NHD) | FlashInfer + Triton kernel |
| MLA Decode | FlashInfer `BatchMLAPagedAttentionWrapper` | FlashInfer MLA CUDA kernel |

### 4.2 [修正] 现有 Ascend Attention Baseline

> v0.1 未提及已有的 Ascend 实现。当前已有基础 SDPA 实现：

**文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/torch_sdpa.py` (126 lines)

| 实现 | support 条件 | 功能 | 限制 |
|------|------------|------|------|
| `AscendSDPAPrefillImpl` | `is_prefill == True` | `F.scaled_dot_product_attention` | 无 paged KV cache，无 RoPE，仅单请求 |
| `AscendSDPADecodeImpl` | `is_prefill == False` | `store_kv()`/`fetch_kv()` + SDPA | 无 paged attention，无 RoPE，仅单请求 |

**注册位置**: `rtp_llm/models_py/modules/factory/attention/__init__.py:48-57`

```python
elif device_type == DeviceType.Ascend:
    from rtp_llm.models_py.modules.factory.attention.ascend_impl.torch_sdpa import (
        AscendSDPAPrefillImpl,
        AscendSDPADecodeImpl,
    )
    PREFILL_MHA_IMPS.append(AscendSDPAPrefillImpl)
    DECODE_MHA_IMPS.append(AscendSDPADecodeImpl)
```

**关键缺口**:
- `PREFILL_MLA_IMPS` 和 `DECODE_MLA_IMPS` 在 Ascend 下为**空**，MLA 模型会直接报错
- 无 RoPE 实现（所有 RoPE 依赖 `flashinfer.rope`）

### 4.3 MHA Prefill — NPU 适配方案

**新增文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_prefill.py`

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

### 4.4 MHA Decode — NPU 适配方案

**新增文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py`

```python
import torch_npu

class AscendDecodeOp:
    def forward(self, q, kv_cache, params):
        # 分离存储 + NHD layout
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

### 4.5 Block Table 格式适配

> [修正] 补充了 2D block table 的来源和当前格式分析。

**关键问题**: rtp-llm 使用 CSR 格式（`page_indice` + `page_indptr`），vllm-ascend 使用 2D 格式。

**rtp-llm C++ 端** 的 2D block table 来源：
- `t_kv_cache_block_id_host`: 2D `[batch_size, max_blocks]` (int32) — 输入到 `FlashInferMlaParams.cc:244`
- 该 2D 表在 `FlashInferMlaParams.cc:328-338` 中被 flatten 为 CSR 格式给 FlashInfer 用

**Ascend 适配**: 将原始 2D block table 直接传递给 `torch_npu` 算子，**不经过 CSR flatten**：
- 方案 A: 在 Python 端从 `attn_inputs.kv_cache_block_id_host` 直接获取 2D 表
- 方案 B: 在 C++ 端新增一个接口返回未 flatten 的 2D block table

推荐方案 A（在 Python 端直接使用已有的 2D block table）。

### 4.6 MLA Attention — NPU 适配方案（后期）

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
- 专用 MLA decode node 使用自定义 MLA kernel

**建议**: MLA 适配放在 MHA 完成后，单独一个子阶段。

---

## 四点五、[新增] RoPE Ascend 适配方案

> v0.1 缺失此部分。当前所有 RoPE 操作依赖 FlashInfer CUDA kernel。

### 4.5.1 现状

rtp-llm 有两个 RoPE 实现，均依赖 `flashinfer.rope`：

| 实现 | 文件 | API |
|------|------|-----|
| MHA RoPE | `cuda_impl/base_rotary_embedding_op.py:86-95` | `flashinfer.rope._apply_rope_pos_ids_cos_sin_cache` |
| MHA RoPE (fallback) | `cuda_impl/base_rotary_embedding_op.py:100-102` | `flashinfer.apply_rope_pos_ids_inplace` |
| MLA RoPE | `cuda_mla_impl/rope_emb_new.py:32-43` | `flashinfer.rope._apply_rope_pos_ids_cos_sin_cache` |

### 4.5.2 Ascend RoPE 替换方案

**方案 A: 纯 PyTorch 实现（推荐，快速验证）**

```python
# ascend_impl/ascend_rope.py
import torch

def apply_rope_pos_ids(q, k, cos_sin_cache, pos_ids, interleave=False):
    """Pure PyTorch RoPE implementation for Ascend NPU."""
    rope_dim = cos_sin_cache.shape[-1]
    q_rope = q[..., :rope_dim]
    k_rope = k[..., :rope_dim]

    cos = cos_sin_cache[pos_ids].unsqueeze(-2)  # [tokens, 1, rope_dim]
    sin = cos_sin_cache[pos_ids].unsqueeze(-2)  # assuming cache stores [cos, sin] interleaved

    # Split into even/odd for rotation
    q1, q2 = q_rope.chunk(2, dim=-1)
    k1, k2 = k_rope.chunk(2, dim=-1)

    if interleave:
        # GPT-NeoX style
        q_rotated = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_rotated = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    else:
        # LLaMA style
        q_rotated = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k_rotated = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

    q_out = q.clone()
    k_out = k.clone()
    q_out[..., :rope_dim] = q_rotated
    k_out[..., :rope_dim] = k_rotated
    return q_out, k_out
```

**方案 B: torch_npu 融合算子（后期优化）**

vllm-ascend 使用 `torch_npu.npu_kv_rmsnorm_rope_cache()` 融合 RMSNorm + RoPE + cache write，可后期参考。

### 4.5.3 RoPE 类结构适配

创建 Ascend 专用 RoPE 类，继承 `BaseRotaryEmbeddingOp` 或独立实现：

```python
# ascend_impl/ascend_rope_emb.py
class AscendRotaryEmbeddingOp:
    """Ascend RoPE using pure PyTorch implementation."""

    def __init__(self, head_size, cos_sin_cache, token_per_block, is_neox_style):
        self.head_size = head_size
        self.cos_sin_cache = cos_sin_cache
        self.token_per_block = token_per_block
        self.is_neox_style = is_neox_style

    def forward(self, query, key, rope_params):
        return apply_rope_pos_ids(
            query, key,
            self.cos_sin_cache,
            rope_params.positions_d,
            interleave=self.is_neox_style,
        )
```

---

## 五、KV Cache Copy Kernel 替换

> [不变] 此节分析仍然成立。

### 5.1 CUDA Kernel 清单及替换方案

| CUDA Kernel | 源文件 | 功能 | NPU 替换方案 |
|------------|--------|------|-------------|
| `convertOffsetAndSize2IdxKernel` | `kv_cache_kernels.cu` | offset+size 转 index | `torch::Tensor` 算术操作 |
| `reuseCacheKernel` | `kv_cache_kernels.cu` | 复用 prefix cache block | `torch.index_copy_` + `torch.index_select` |
| `concat_and_cache_mla` | `mla_quant_kernel.cu` | MLA KV 写入 + FP8 量化 | `torch_npu._npu_reshape_and_cache` + scatter |
| `indexer_k_quant_and_cache` | `mla_quant_kernel.h` | Indexer K 量化写入 | 后期（MLA FP8） |
| `cp_gather_and_upconvert_fp8_kv_cache` | `mla_quant_kernel.h` | FP8 cache 反量化读取 | 后期（MLA FP8） |

### 5.2 reuseCacheKernel 替换

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

### 6.2 [修正] C++ 端参数生成

> v0.1 将 MHA/MLA 参数生成混在一起。实际上：
> - **MHA**: 参数生成主要在 **Python 端**（FlashInfer Python wrapper 构建 page table）
> - **MLA**: 参数生成在 **C++ 端**（`FlashInferMlaParams.cc` 和 `SparseMlaParams.cc`）

**MHA 路径** (Python 端):
- FlashInfer 的 `BatchPrefillWithPagedKVCacheWrapper.plan()` 在 Python 端构建参数
- Ascend 替换：在 Python 端构建 `AscendAttnParams`，从 `attn_inputs` 获取 2D block table

**MLA 路径** (C++ 端):
- `FlashInferMlaParams.cc` 的 `fillParams()` 和 `fillParamsInternal()` 在 C++ 端生成参数
- 需替换: `cudaMemcpyAsync` → NPU H2D (见 3.5 节)
- 需替换: `GET_CURRENT_STREAM()` 已处理 stream，无需额外修改
- `SparseMlaParams.cc` 同理，需替换 `cudaMemcpyAsync` 调用

---

## 七、[修正] Factory 注册

> v0.1 说修改 `attn_factory.py`。实际注册点在 `__init__.py`，且已有 Ascend 基础注册。

**修改文件**: `rtp_llm/models_py/modules/factory/attention/__init__.py`

当前 Ascend 分支 (lines 48-57)：
```python
elif device_type == DeviceType.Ascend:
    from rtp_llm.models_py.modules.factory.attention.ascend_impl.torch_sdpa import (
        AscendSDPAPrefillImpl,
        AscendSDPADecodeImpl,
    )
    PREFILL_MHA_IMPS.append(AscendSDPAPrefillImpl)
    DECODE_MHA_IMPS.append(AscendSDPADecodeImpl)
```

扩展后（新实现优先于 SDPA baseline）：
```python
elif device_type == DeviceType.Ascend:
    # Paged attention implementations (优先使用)
    from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_prefill import (
        AscendPrefillImpl,
    )
    from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_decode import (
        AscendDecodeImpl,
    )

    PREFILL_MHA_IMPS.append(AscendPrefillImpl)    # 新增，优先于 SDPA
    DECODE_MHA_IMPS.append(AscendDecodeImpl)       # 新增，优先于 SDPA

    # SDPA baseline (fallback)
    from rtp_llm.models_py.modules.factory.attention.ascend_impl.torch_sdpa import (
        AscendSDPAPrefillImpl,
        AscendSDPADecodeImpl,
    )
    PREFILL_MHA_IMPS.append(AscendSDPAPrefillImpl)
    DECODE_MHA_IMPS.append(AscendSDPADecodeImpl)

    # MLA implementations (新增 — 当前为空，MLA 模型会报错)
    from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_mla_impl import (
        AscendMlaPrefillImpl,
        AscendMlaDecodeImpl,
    )
    PREFILL_MLA_IMPS.append(AscendMlaPrefillImpl)
    DECODE_MLA_IMPS.append(AscendMlaDecodeImpl)
```

> **注册机制说明**: 工厂按注册顺序遍历实现列表，调用 `support()` 方法检查是否支持。
> 注册在前 = 优先级更高。因此新的 Paged Attention 实现应注册在 SDPA baseline 之前。

---

## 八、可借鉴的设计模式

> [不变] 此节内容仍然成立。

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

### 8.3 xllm ACL Graph 经验

- 标准 `npu_paged_attention` 内部有 `.to(kCPU)` 会打断 ACL Graph
- 提供了 `npu_custom_paged_attention` 变体，通过预计算 `tiling_data` 避免 CPU 同步
- rtp-llm 在 Phase 7 (CUDA Graph → Ascend Graph) 时需参考此方案

---

## 九、[修正] 完整文件变更清单

| 类别 | 文件 | 变更类型 | 说明 | 修正 |
|------|------|---------|------|------|
| **C++ 配置** | `rtp_llm/cpp/cache/CacheConfig.h` | 修改 | 添加 `separate_kv_cache`、`k_block_stride_bytes`、`v_block_stride_bytes` | 不变 |
| **C++ 数据结构** | `rtp_llm/cpp/cache/BufferTypes.h` | 修改 | 添加 `layers_to_k_buffer_ptrs`、`layers_to_v_buffer_ptrs` | 不变 |
| **C++ 内存管理** | `rtp_llm/cpp/cache/BlockPool.cc` | 修改 | 支持分离分配 K/V tensor | [修正] 设备选择已完成，仅增加分离分配路径 |
| | `rtp_llm/cpp/cache/MemoryEvaluationHelper.cc` | ~~修改~~ | ~~`cudaMemGetInfo` → `aclrtGetMemInfo`~~ | [修正] **已完成，无需修改** |
| | `rtp_llm/cpp/cache/KVCacheAllocator.cc` | 修改 | 扩展 `execBatchCopy` 支持分离 K/V 拷贝 | [修正] 基础 batch copy 已适配，仅扩展分离拷贝 |
| | `rtp_llm/cpp/cache/KVCacheManager.cc` | 修改 | `getMainModelCacheLayerLayout()` 增加分离路径 | [修正] sync 已适配，仅增加分离 layout 传递 |
| **C++ 绑定** | `rtp_llm/models_py/bindings/OpDefs.h` | 修改 | `KVCache`/`LayerKVCache` 添加分离字段 + NHD reshape | 不变 |
| **C++ MLA 参数** | `rtp_llm/models_py/bindings/cuda/FlashInferMlaParams.cc` | 修改 | `cudaMemcpyAsync` → NPU H2D | [修正] 文件路径从 `CudaFlashInfer.cc` 修正 |
| | `rtp_llm/models_py/bindings/cuda/SparseMlaParams.cc` | 修改 | `cudaMemcpyAsync` → NPU H2D | [新增] v0.1 未列出 |
| **C++ MLA Kernel** | `rtp_llm/models_py/bindings/cuda/kernels/mla_quant_kernel.cu` | 修改 | stream 获取 + 替换 CUDA API | [新增] v0.1 未列出 |
| **Python 类型** | `rtp_llm/ops/librtp_compute_ops/__init__.pyi` | 修改 | `LayerKVCache` 添加 `k_cache_base`、`v_cache_base` | 不变 |
| **Python RoPE** | `ascend_impl/ascend_rope.py` | **新增** | 纯 PyTorch RoPE 实现 | [新增] v0.1 缺失 |
| | `ascend_impl/ascend_rope_emb.py` | **新增** | Ascend RoPE Embedding Op | [新增] v0.1 缺失 |
| **Python 写入** | `ascend_impl/ascend_kv_cache_write_op.py` | **新增** | MHA KV write（使用分离 K/V + NHD） | 不变 |
| | `ascend_impl/ascend_mla_kv_cache_write_op.py` | **新增** | MLA KV write（使用分离 cache） | 不变 |
| **Python 读取** | `ascend_impl/ascend_prefill.py` | **新增** | Prefill attention（使用分离 K/V + NHD） | 不变 |
| | `ascend_impl/ascend_decode.py` | **新增** | Decode attention（使用分离 K/V + NHD） | 不变 |
| **Python 参数** | `ascend_impl/ascend_attn_params.py` | **新增** | Ascend 参数结构 | 不变 |
| **Factory** | `attention/__init__.py` | 修改 | 注册 Ascend paged + MLA 实现 | [修正] 从 `attn_factory.py` 修正为 `__init__.py` |
| **工具** | `rtp_llm/utils/device_utils.py` | **新增** | NPU 设备适配器 | 不变 |

---

## 十、关键风险与缓解

> [修正] 更新了风险项，反映已完成的适配工作。

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| KV 分离存储导致内存分配和管理逻辑改动 | BlockPool、KVCacheManager 等核心组件需扩展 | 利用已有的 `k_block_size()`/`v_block_size()` 预留接口；分离存储作为配置开关（`separate_kv_cache`），不影响 CUDA 路径 |
| NPU 算子要求 NHD layout，与 rtp-llm HND layout 不兼容 | KV cache 读写失败 | 通过 C++ `getLayerCache()` 直接 reshape 为 NHD，避免 Python 端 permute 开销 |
| `torch_npu._npu_reshape_and_cache` 对 NHD layout 的验证 | 写入正确性不确定 | 优先在子阶段 4.3 做单层 KV write 正确性验证 |
| `npu_fused_infer_attention_score` block_table 格式不兼容 | Prefill 不可用 | rtp-llm Python 端已有 2D block table（`kv_cache_block_id_host`），确保传递正确格式 |
| `torch_npu._npu_paged_attention` 性能不达标 | Decode 性能差 | 先跑通功能，Phase 10 优化；备选 aclnn 自定义算子 |
| 分离存储的 block 批量拷贝需要同时拷贝 K/V | 拷贝逻辑复杂化 | 在 `execBatchCopy` 中对 K/V 分别执行拷贝 |
| MLA cache 需适配分离存储模式 | MLA 模型不可用 | MLA cache 使用 k_cache_base/v_cache_base 字段 |
| `torch_npu` API 在某些 CANN 版本不可用 | 功能受阻 | 参考 vllm-ascend Device Adaptor 模式按型号选择 API |
| FP8 MLA per-128-element block quantization 与 Ascend FP8 格式不兼容 | FP8 MLA 不可用 | 先跑 FP16/BF16，FP8 在 Phase 8 统一处理 |
| [新增] RoPE 依赖 FlashInfer CUDA kernel | 无 RoPE 无法正确推理 | 新增纯 PyTorch RoPE 实现（方案 A），后期可用 torch_npu 融合算子优化 |
| [新增] MLA 模型在 Ascend 下完全不可用 | DeepSeek V2/V3 无法推理 | MLA 适配作为独立子阶段，MHA 先行 |

---

## 十一、FP8 KV Cache 适配说明（后期）

> [不变] 此节保持 v0.1 内容。

rtp-llm FP8 MLA 的 per-token 布局非常特殊：
```
[0..511]  : 512 × float8_e4m3  (quantized NoPE/CKV)
[512..527]: 4 × float32 scale  (one per 128-element block)
[528..655]: 64 × bfloat16      (RoPE part, NOT quantized)
= 656 bytes/token
```

**建议**: FP8 KV Cache 放在 Phase 8 (FP8/量化支持) 统一处理，Phase 4 先保证 FP16/BF16 路径跑通。

---

## 十二、[修正] 里程碑与验证计划

> [修正] 更新了子阶段内容，移除已完成的 C++ 基础适配工作，新增 RoPE 和 MLA 子阶段。

| 子阶段 | 周期 | 交付物 | 验证标准 |
|--------|------|--------|---------|
| 4.1 KV 分离存储 C++ 适配 | 3 天 | `CacheConfig`/`BufferTypes`/`BlockPool`/`OpDefs` 分离 K/V 支持 + `getMainModelCacheLayerLayout()` 分离路径 | `BlockPool::init()` 在 NPU 上成功分配分离的 K/V buffer，`getLayerCache()` 返回 NHD layout |
| 4.2 Slot Mapping + Block Table 适配 | 2 天 | Python 端构建 `AscendAttnParams` + C++ 端 MLA H2D 拷贝替换 | slot_mapping 和 2D block_table 正确传递到 Ascend 算子 |
| 4.3 [新增] RoPE Ascend 适配 | 2 天 | `AscendRotaryEmbeddingOp`（纯 PyTorch RoPE） | MHA RoPE 输出与 FlashInfer 结果一致（数值误差 < 1e-3） |
| 4.4 MHA KV Write + Prefill | 5 天 | `AscendKVCacheWriteOp` + `AscendPrefillImpl`（分离 K/V + NHD） | 单请求 prefill 输出与 CUDA 路径一致 |
| 4.5 MHA Decode | 5 天 | `AscendDecodeImpl`（分离 K/V + NHD + paged attention） | 单请求 decode 生成正确 token |
| 4.6 Factory 注册 + 端到端 | 2 天 | `__init__.py` 注册 Ascend paged 实现 + 端到端测试 | 端到端推理可跑通（FP16/BF16 MHA 模型） |
| 4.7 MLA 适配 | 5 天 | MLA write + decode（分离 cache）+ `FlashInferMlaParams.cc` NPU 适配 | DeepSeek V2/V3 可推理 |
| 4.8 Batch Copy + Prefix Cache | 2 天 | Block 批量拷贝 NPU 路径（含 K/V 分离拷贝） | Prefix cache 功能正常 |
| **合计** | **~3.5 周** | | |
