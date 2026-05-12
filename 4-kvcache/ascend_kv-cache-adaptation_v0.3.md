# Phase 4 增强：KV Cache 昇腾适配详细计划 (v0.3 — MHA 优先)

> 本文档是 v0.2 的完善版本。基于代码审核发现，补充了 block table 格式转换细节、RoPE 实现细节、
> BlockPool 成员变量变更、TP 交互分析、验证方案等内容。
> 
> **范围调整**: v0.3 聚焦 MHA 模型（Qwen3-4B 等），MLA 适配延期到 Phase 4 后续子阶段。
> 变更标记：`[v0.3新增]` 表示 v0.3 新增/修改的内容，`[修正]` 和 `[不变]` 沿用 v0.2 标记。

---

## 〇、版本演进摘要

| 版本 | 主要变更 |
|------|---------|
| v0.1 | 初版，存在文件路径、设备类型等错误 |
| v0.2 | 修正代码路径（`kPrivateUse1`、`FlashInferMlaParams.cc`、`__init__.py` 等），补充 RoPE 章节 |
| v0.3 | **MHA 聚焦**（移除 MLA），补充 block table 2D🚹CSR 转换细节、RoPE 准确实现、BlockPool 成员变量、TP 交互分析、验证模型与测试方法 |

---

## 〇點五、[v0.3新增] v0.2 审核后的关键补充

基于代码验证，v0.2 方案整体准确，以下为关键补充：

| # | 补充项 | 说明 |
|---|--------|------|
| 1 | Block table 2D → CSR 转换逻辑 | C++ `fillParams` 将 `[batch, max_blocks]` 2D 表 flatten 为 CSR 三件套（`page_indice`/`page_indptr`/`last_page_len`）。Ascend 算子直接使用 2D 表，需在 Python 端绕过 CSR flatten。 |
| 2 | block_table 中的 -1 值 | `kv_cache_block_id_host` 包含 `-1` 表示未使用的 block slot。需验证 `torch_npu` 算子对 `-1` 的处理行为。 |
| 3 | RoPE cos_sin_cache 精确格式 | shape `[max_pos, rope_dim]`，非交错格式 `[cos₀..cos_{d/2-1}, sin₀..sin_{d/2-1}]`，由 C++ `RopeCache.genBaseCache` 生成。 |
| 4 | BlockPool.h 需新增成员变量 | `separate_kv_cache_`、`k_cache_buffer_`、`v_cache_buffer_`（当前仅有 `cache_aligned_buffer_`） |
| 5 | TP 交互结论 | KV cache 的 `num_kv_heads` 已是 `local_head_num_kv`（TP 后）。分离 K/V 不改变每个 rank 的 head 维度，无需额外 TP 适配。 |
| 6 | 验证模型与方法 | 推荐 Qwen3-4B（纯 MHA）、Qwen2-7B 作为验证模型；需逐 token 对比 CUDA 路径输出。 |

---

## 一、三框架 KV Cache 适配方案对比

> [不变] 此节保持 v0.1/v0.2 内容，对比表和选型理由仍然成立。

通过分析 vllm-ascend、xllm、rtp-llm 三个项目的 KV Cache 实现，总结各框架的 NPU 适配策略差异：

| 维度 | vllm-ascend | xllm | rtp-llm (目标) |
|------|------------|------|---------------|
| **API 层级** | `torch_npu` Python API | ATB C++ 算子 | `torch_npu` Python API |
| **KV Cache 写入** | `torch_npu._npu_reshape_and_cache` | `atb::npu_reshape_and_cache` | `torch_npu._npu_reshape_and_cache` |
| **Prefill Attention** | `torch_npu.npu_fused_infer_attention_score` | `atb::npu_flash_attention` | `torch_npu.npu_fused_infer_attention_score` |
| **Decode Attention** | `torch_npu._npu_paged_attention` | `atb::npu_paged_attention` | `torch_npu._npu_paged_attention` |
| **内存格式** | NCHW/ND (标准) | ND / FRACTAL_NZ (DeepSeek) | ND |
| **MLA 支持** | `npu_fused_infer_attention_score_v2` + `npu_kv_rmsnorm_rope_cache` | ATB monolithic decoder op | **v0.3 不涵盖** |
| **ACL Graph** | `torch.npu.NPUGraph` | `atb::npu_custom_paged_attention` | `torch.npu.NPUGraph`（Phase 7） |
| **Block 大小** | 128 | 128 | 128 (保持不变) |
| **KV Cache Shape** | `[2, blocks, block_size, kv_heads, head_dim]` K/V dim0 分离 | `[blocks, block_size, kv_heads, head_dim]` × 2 独立 | `[blocks, seq_size, kv_heads, head_dim]` × 2 独立(NHD) |
| **K/V 存储方式** | K/V 独立 tensor | K/V 独立 tensor | **需从合并改为独立** |
| **Layout** | NHD | NHD (标准) | **需从 HND 改为 NHD** |

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
#elif USING_ASCEND
    auto [used_bytes, free_bytes] = rtp_llm::ascend::getDeviceMemoryInfo(false);
    free_gpu_bytes  = free_bytes;
    total_gpu_bytes = used_bytes + free_bytes;
#endif
```

**结论**：已完成，无需修改。

#### 2.1.3 Block 批量拷贝

**文件**: `rtp_llm/cpp/cache/KVCacheAllocator.cc:141-192`

[修正] v0.1 说使用 `cudaMemcpyAsync`，实际已通过 `execBatchCopy()` → `runtimeBatchCopy()` 抽象。Ascend 路径使用 `batchCopyFallback()` + `torch::kPrivateUse1`（实现在 `rtp_llm/models_py/bindings/core/CudaOps.cc:257-298`）。(TODO: 拷贝算子是否缺失？)

**结论**：基础 D2D/H2D/D2H 拷贝已完成。后续分离 K/V 存储后，需扩展支持同时拷贝 K/V 两个独立 tensor。

#### 2.1.4 CUDA 同步调用

**文件**: `rtp_llm/cpp/cache/KVCacheManager.cc:219`

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

**文件**: `rtp_llm/models_py/bindings/common/Torch_ext.h:29-35`

[修正] v0.1 中多处建议手动替换 stream 获取，实际已通过宏抽象：

```cpp
// Torch_ext.h:29-35 (Ascend 分支)
#elif USING_ASCEND
#define GET_CURRENT_STREAM() c10_npu::getCurrentNPUStream().stream()
```

**结论**：已完成。新增的 C++ 代码只需使用 `GET_CURRENT_STREAM()` 宏即可。

---

### 2.2 无需修改的组件

| 组件 | 文件 | 原因 |
|------|------|------|
| `FullKVCacheGroup` | `FullKVCacheGroup.cc` | 纯 block 计数 + 引用计数，无设备 API |
| `LinearKVCacheGroup` | `LinearKVCacheGroup.cc` | 稀疏 block 分配逻辑，无设备 API |
| `BlockCache` | `BlockCache.cc` | 纯 LRU 数据结构 |
| `BlockRefCounter` | `BlockRefCounter.h` | 纯引用计数数组 |
| `MemoryLayoutStrategy` | `MemoryLayoutStrategy.cc:285` | 已有 `dev.is_cuda() \|\| dev.is_privateuseone()` 适配 |
| `KVCacheSpec` (MHA/MLA/Linear) | `KVCacheSpecBase.h` 等 | 纯 block 大小计算 |
| `CacheConfig` | `CacheConfig.h` | 纯配置结构，无设备 API |

---

### 2.3 仍需增量修改的组件 — KV 分离存储

> 这是 v0.1/v0.2 第二节的核心增量工作，所有组件的基础 Ascend 适配已完成，
> 但 **K/V 分离存储** 功能尚未实现。

昇腾 NPU 的 `torch_npu` 算子（`_npu_reshape_and_cache`、`_npu_paged_attention`、`npu_fused_infer_attention_score`）要求 `key_cache` 和 `value_cache` 作为**独立的 Tensor** 传入。rtp-llm 当前将 K/V 合并在一个 Tensor 中，通过 `getLayerCache()` reshape 为 `[kernel_block_num, 2, num_kv_heads, kernel_seq_size_per_block, head_dim]`（dim=1 分离 K/V）。

#### 2.3.1 [v0.3新增] 数据流全景

```
                        当前（合并存储 + HND + FlashInfer）:
                        ═══════════════════════════════════
BlockPool 分配 ──→ [total_size_bytes] uint8 buffer
                   │
MemoryLayout      │ narrow + view
Strategy          │
                   ▼
                   [block_num, kv_block_stride_elems] per layer (含 K+V 拼接)
                   │
OpDefs            │ getLayerCache() reshape
getLayerCache()   │
                   ▼
kv_cache_base = [kernel_blocks, 2, kv_heads, kernel_seq, head_dim]  HND
                   │
Python 端          │ kv_cache_base[:,0] → k_cache  │ dim=1 slice
KVCacheWriteOp    │ kv_cache_base[:,1] → v_cache  │ 出 K/V
                   ▼
flashinfer.page.append_paged_kv_cache(k_cache, v_cache, layout="HND")


                        目标（分离存储 + NHD + torch_npu）:
                        ════════════════════════════════════════
BlockPool 分配 ──→ k_cache_buffer_ [total_size_bytes/2] + v_cache_buffer_ [total_size_bytes/2]
                   │
MemoryLayout      │ narrow + view (K 和 V 独立)
Strategy          │
                   ▼
                   k_base = [block_num, k_block_stride_elems] per layer
                   v_base = [block_num, v_block_stride_elems] per layer
                   │
OpDefs            │ getLayerCache() reshape to NHD
getLayerCache()   │
                   ▼
k_cache_base = [kernel_blocks, kernel_seq, kv_heads, head_dim]  NHD ✅
v_cache_base = [kernel_blocks, kernel_seq, kv_heads, head_dim]  NHD ✅
                   │
Python 端          │ 直接使用 k_cache_base / v_cache_base
AscendOps          │ 无需 slice
                   ▼
torch_npu._npu_reshape_and_cache(key, value, k_cache_base, v_cache_base, slot_mapping)
torch_npu.npu_fused_infer_attention_score(query, k_cache_base, v_cache_base, block_table, ...)
torch_npu._npu_paged_attention(query, k_cache_base, v_cache_base, block_table, ...)
```

#### 2.3.2 CacheConfig 配置扩展

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

#### 2.3.3 BufferTypes 数据结构扩展

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

#### 2.3.4 BlockPool 内存分配支持分离

**文件**: `rtp_llm/cpp/cache/BlockPool.h` / `BlockPool.cc`

[v0.3新增] 当前 `BlockPool.h:132-136` 仅有 `cache_aligned_buffer_` + `cache_base_ptr_` 相关的合并存储成员。分离模式需新增：

**BlockPool.h 新增成员变量**（插入到 `cache_aligned_buffer_` 附近）：

```cpp
// BlockPool.h — 在 cache_aligned_buffer_ (line 132) 上方或下方新增:
    torch::Tensor               k_cache_buffer_;        // 分离模式: K cache 原始 buffer
    torch::Tensor               v_cache_buffer_;        // 分离模式: V cache 原始 buffer
    bool                        separate_kv_cache_ = false;  // 是否启用分离存储
```

**BlockPool.cc `initializeCacheBuffer()` 修改**：

在已有的设备选择逻辑（line 40-53）基础上增加分离分配路径：

```cpp
void BlockPool::initializeCacheBuffer() {
    // 设备选择逻辑已完成（line 40-53），保持不变
    torch::Device device = torch::kCPU;
    // ... (现有 #if USING_ASCEND 等分支保持不变)

    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(device);

    if (separate_kv_cache_) {
        // 新增路径：分离分配 K/V cache
        // total_size_bytes 是 K+V 总和，各分配一半
        size_t per_buffer_bytes = config_.total_size_bytes / 2;
        k_cache_buffer_ = torch::empty(
            {static_cast<int64_t>(per_buffer_bytes)}, options);
        v_cache_buffer_ = torch::empty(
            {static_cast<int64_t>(per_buffer_bytes)}, options);
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

> [v0.3新增] 分离模式下 `cache_aligned_buffer_` 保留但为空 tensor。`allLayerCacheBase()` 在分离模式下返回 K/V 各自按 layer 切分后的列表。

#### 2.3.5 [v0.3新增] processMemoryLayout 适配分离模式

**文件**: `rtp_llm/cpp/cache/BlockPool.cc:processMemoryLayout()`（约 line 180）

当前 `processMemoryLayout()` 从 `cache_aligned_buffer_` narrow 出 `global_layer_kv_tensors_`。分离模式下需要从 `k_cache_buffer_` 和 `v_cache_buffer_` 分别 narrow。关键变化：

```
合并模式: cache_aligned_buffer_ → narrow → global_layer_kv_tensors_[layer] (含 K+V 拼接)
分离模式: k_cache_buffer_ → narrow → global_layer_k_tensors_[layer]   (仅 K)
          v_cache_buffer_ → narrow → global_layer_v_tensors_[layer]   (仅 V)
```

需新增 `BlockPool.h` 成员变量：
```cpp
std::vector<torch::Tensor> global_layer_k_tensors_;       // 分离: per-layer K views
std::vector<torch::Tensor> global_layer_v_tensors_;       // 分离: per-layer V views
```

#### 2.3.6 KVCache / LayerKVCache C++ 绑定扩展

**文件**: `rtp_llm/models_py/bindings/OpDefs.h`

当前 `LayerKVCache`（line 25-30）和 `KVCache`（line 34-47）均为合并存储：

```cpp
// 当前 OpDefs.h:25-30
struct LayerKVCache {
    torch::Tensor kv_cache_base;     // 合并存储（现有）
    torch::Tensor kv_scale_base;
    int           seq_size_per_block = 0;
    int           layer_id           = -1;
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
    torch::Tensor k_cache_base;      // K cache [blocks, seq_per_block, kv_heads, head_dim] NHD
    torch::Tensor v_cache_base;      // V cache [blocks, seq_per_block, kv_heads, head_dim] NHD
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

#### 2.3.7 getLayerCache 方法适配

**文件**: `rtp_llm/models_py/bindings/OpDefs.h:49-115`

当前 `getLayerCache()` 的 MHA 路径（line 87-91）返回 HND layout：

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

                // MHA: NHD layout [blocks, seq_per_block, kv_heads, head_dim]
                layer_cache.k_cache_base = k_base.reshape(
                    {kernel_block_num, (int64_t)kernel_seq_size_per_block,
                     (int64_t)num_kv_heads, (int64_t)head_dim});
                layer_cache.v_cache_base = v_base.reshape(
                    {kernel_block_num, (int64_t)kernel_seq_size_per_block,
                     (int64_t)num_kv_heads, (int64_t)head_dim});
            } else {
                // ===== 现有：合并存储路径 (CUDA/ROCm) =====
                if (num_kv_heads > 0 && head_dim > 0) {
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
> 
> [v0.3新增] 注意：分离路径省略了 MLA 分支（`use_mla && kv_lora_rank > 0`），因为 v0.3 不涵盖 MLA。

#### 2.3.8 配置传递链路

```
启动参数 / 配置文件
    │
    └─ KVCacheConfig.separate_kv_cache = true  (昇腾 NPU 路径)
         │
         ├─ CacheConfig.separate_kv_cache + k_block_stride_bytes + v_block_stride_bytes
         │
         ├─ BlockPool::initializeCacheBuffer() → 分配 k_cache_buffer_ + v_cache_buffer_
         │
         ├─ BlockPool::processMemoryLayout() → 生成分离的 per-layer K/V views
         │
         ├─ BlockPool::allLayerCacheBase() → 返回含 layers_to_k_buffer_ptrs / layers_to_v_buffer_ptrs 的 layout
         │
         ├─ KVCacheManager::getMainModelCacheLayerLayout()
         │   → 返回含 layers_to_k_buffer_ptrs / layers_to_v_buffer_ptrs 的 layout
         │
         ├─ PyWrappedModel → 填充 KVCache.k_cache_base_by_layer / v_cache_base_by_layer
         │
         └─ KVCache.getLayerCache() → 返回含 k_cache_base / v_cache_base 的 LayerKVCache
```

#### 2.3.9 Python 类型 Stub 更新

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

#### 2.3.10 KVCacheManager::getMainModelCacheLayerLayout() 扩展

**文件**: `rtp_llm/cpp/cache/KVCacheManager.cc:255-298`

当前方法（line 259）仅使用 `layers_to_kv_buffer_ptrs`，需增加分离路径：

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

#### 2.3.11 [v0.3新增] setKVBlockValue 分离适配

**文件**: `rtp_llm/cpp/cache/KVCacheManager.cc:153-221`

当前 `setKVBlockValue()` 将 k_buffer 写入 block 的 offset 0，v_buffer 写入 offset `expected_k_bytes`。分离存储模式下，K/V 各有独立 buffer，无需 offset 偏移：

```cpp
bool KVCacheManager::setKVBlockValue(int block_index, int layer_id,
                                     const torch::Tensor& k_buffer,
                                     const torch::Tensor& v_buffer) {
    if (config_.separate_kv_cache) {
        // 分离模式: k_buffer → k cache, v_buffer → v cache (各自独立写入)
        auto dst_blocks = allocator_->convertIndexToBuffer(layer_id, block_index);
        // dst_blocks[0] = K block, dst_blocks[1] = V block
        return copyFunc(k_buffer, dst_blocks[0], 0) && copyFunc(v_buffer, dst_blocks[1], 0);
    } else {
        // 合并模式: 现有逻辑 (k_buffer → offset 0, v_buffer → offset expected_k_bytes)
        // ... (lines 158-219 保持不变)
    }
}
```

#### 2.3.12 Batch Copy 适配（Prefix Cache 复用）

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

rtp-llm 的 KV Cache 写入依赖 FlashInfer 库：

**MHA 写入** (`rtp_llm/models_py/modules/factory/attention/cuda_impl/kv_cache_write_op.py:51-72`)：
```python
# 从合并的 kv_cache_base 中分离 K/V（HND layout）
k_cache = kv_cache.kv_cache_base[:, 0, :, :, :]  # [pages, kv_heads, page_size, head_dim]
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

### 3.2 MHA KV Cache 写入 — NPU 适配方案

**新增文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_kv_cache_write_op.py`

```python
import torch_npu

class AscendKVCacheWriteOp:
    """MHA KV Cache write using torch_npu._npu_reshape_and_cache."""
    
    def __init__(self, num_kv_heads, head_size, token_per_block):
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.token_per_block = token_per_block

    def forward(self, key, value, kv_cache, fmha_params):
        if kv_cache is None:
            return

        # 分离存储 + NHD layout（由 C++ getLayerCache() 保证）
        k_cache = kv_cache.k_cache_base   # [blocks, seq_per_block, kv_heads, head_dim] NHD
        v_cache = kv_cache.v_cache_base   # 同上

        torch_npu._npu_reshape_and_cache(
            key=key,
            value=value,
            key_cache=k_cache,
            value_cache=v_cache,
            slot_indices=fmha_params.slot_mapping,
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

### 3.4 [v0.3新增] 与 FlashInfer CSR 参数的对比

当前 FlashInfer 路径使用 CSR 格式参数：`page_indice` (flat), `page_indptr` (row pointers), `last_page_len`。这些由 C++ `fillParams()` 从 2D `kv_cache_block_id_host` 计算得出。

Ascend `torch_npu` 算子直接使用 2D `block_table` `[batch_size, max_blocks]`。`kv_cache_block_id_host` 即为该 2D 表（`PyAttentionInputs` 中的字段，见 `OpDefs.h:172`）。

```
FlashInfer 路径:
  kv_cache_block_id_host [B, max_blocks] 
    ──C++ fillParams()──→ page_indice [total_pages] + page_indptr [B+1] + last_page_len [B]
    ──→ flashinfer.page.append_paged_kv_cache(...)

Ascend 路径:
  kv_cache_block_id_host [B, max_blocks] 
    ──直接传入──→ torch_npu._npu_reshape_and_cache(key, value, k_cache, v_cache, slot_mapping)
                   torch_npu.npu_fused_infer_attention_score(q, k_cache, v_cache, block_table=..., ...)
                   torch_npu._npu_paged_attention(q, k_cache, v_cache, block_table=..., ...)
```

> [v0.3新增] 注意 `kv_cache_block_id_host` 中可能包含 `-1` 值（表示无效/未使用的 block slot）。需在集成验证时确认 `torch_npu` 算子对 `-1` 的处理行为。如不支持，需在传入前将 `-1` 替换为有效占位值。

### 3.5 [v0.3新增] slot_mapping ≠ kv_cache_block_id_host

这两个是不同概念，容易混淆：

- **`slot_mapping`**: `[num_tokens]` tensor，每个 token 在 KV cache 中的**线性索引**。计算方式：`block_number * page_size + block_offset`。由 `FlashInferMlaParams.cc:478-485` 在 CPU 端计算。
- **`kv_cache_block_id_host`**: `[batch_size, max_blocks]` 2D tensor，每个请求使用的 **block ID 列表**。供 attention 算子查找 K/V 数据。

Ascend 的 `_npu_reshape_and_cache` 使用 `slot_mapping`（写入时知道每个 token 写到哪个 slot），而 `npu_fused_infer_attention_score` / `_npu_paged_attention` 使用 `block_table`（读取时根据 block ID 查找 K/V）。

---

## 四、KV Cache 读取适配（替代 FlashInfer Attention）

### 4.1 现状分析

| 场景 | 现有实现 | CUDA 依赖 |
|------|---------|----------|
| MHA Prefill | FlashInfer `BatchPrefillWithPagedKVCacheWrapper` (HND) | FlashInfer CUDA kernel |
| MHA Decode | FlashInfer `BatchDecodeWithPagedKVCacheWrapper` (HND) | FlashInfer CUDA kernel |

### 4.2 现有 Ascend Attention Baseline

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

### 4.3 MHA Prefill — NPU 适配方案

**新增文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_prefill.py`

**关键设计**: 需要从 `attn_inputs` 构建 Ascend 算子需要的参数。

#### 4.3.1 [v0.3新增] 参数构造逻辑

从 FlashInfer prefill 的参数构造方式推导 Ascend 版本：

```
FlashInfer 参数构造:
  PyFlashinferPrefillPagedAttnOp.prepare()
    → fmha_params.fill_params(prefix_lengths, sequence_lengths, input_lengths,
                                kv_cache_kernel_block_id_host, seq_size_per_block)
    → 内部生成: page_indice, page_indptr, last_page_len
    → prefill_wrapper.plan(page_indptr, page_indice, last_page_len, ...)

Ascend 参数构造:
  AscendPrefillImpl.__init__()
    → 从 attn_inputs 直接获取:
        - kv_cache_block_id_host   → block_table [batch, max_blocks]
        - input_lengths             → actual_seq_lengths_q (cumsum)
        - prefix_lengths            → actual_seq_lengths_kv (cumsum)
    → actual_seq_lengths = cumsum(seq_lens) 形式
```

#### 4.3.2 实现代码

```python
import torch
import torch_npu

class AscendPrefillImpl(FMHAImplBase):
    """Ascend MHA Prefill using npu_fused_infer_attention_score."""

    def __init__(self, attn_configs, attn_inputs, weights,
                 cos_sin_cache=None, fmha_config=None,
                 parallelism_config=None, **kwargs):
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.parallelism_config = parallelism_config
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.scale = attn_configs.scale if attn_configs.scale else \
                     self.head_dim ** -0.5
        self.page_size = attn_inputs.kv_cache.seq_size_per_block if \
                         attn_inputs.kv_cache else 128
        
        # 缓存 block table（后续 prepare 中设置）
        self.block_table = None
        self.actual_seq_q = None
        self.actual_seq_kv = None

    @staticmethod
    def support(attn_configs, attn_inputs):
        return attn_inputs.is_prefill and \
               not attn_configs.use_mla and \
               attn_inputs.kv_cache is not None and \
               attn_inputs.kv_cache.separate_kv_cache

    def prepare(self, attn_inputs):
        """构建 Ascend 算子所需的参数。"""
        # 2D block table: [batch_size, max_blocks]
        self.block_table = attn_inputs.kv_cache_block_id_host
        
        # 累积序列长度 (cu_seq_lens 形式)
        seq_lens_q = attn_inputs.input_lengths
        seq_lens_kv = attn_inputs.prefix_lengths + attn_inputs.input_lengths
        self.actual_seq_q = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=seq_lens_q.device),
            torch.cumsum(seq_lens_q, dim=0)
        ])
        self.actual_seq_kv = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=seq_lens_kv.device),
            torch.cumsum(seq_lens_kv, dim=0)
        ])

    def forward(self, qkv, kv_cache, layer_idx=0):
        q, k, v = qkv.chunk(3, dim=-1)
        # q/k/v: [total_tokens, num_heads, head_dim]

        k_cache = kv_cache.k_cache_base  # NHD
        v_cache = kv_cache.v_cache_base  # NHD

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=q,
            key=k_cache,
            value=v_cache,
            block_table=self.block_table,
            input_layout="TND",
            block_size=self.page_size,
            actual_seq_lengths=self.actual_seq_q,
            actual_seq_lengths_kv=self.actual_seq_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,  # causal mask
        )
        return attn_output
```

> [v0.3新增] `sparse_mode=3` 对应 causal mask。如果模型不支持 causal（如双向 attention），需要根据 `attn_configs.is_causal` 调整。`atten_mask` 参数在当前设计中暂不传递（causal 场景不需要）。
> 
> [v0.3新增] `actual_seq_lengths_q/kv` 是累积形式的 `cu_seq_lens`，而非每个请求的单独长度。FlashInfer 内部也是转换为累积形式使用。

### 4.4 MHA Decode — NPU 适配方案

**新增文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py`

```python
import torch
import torch_npu

class AscendDecodeImpl(FMHAImplBase):
    """Ascend MHA Decode using torch_npu._npu_paged_attention."""

    def __init__(self, attn_configs, attn_inputs, weights,
                 cos_sin_cache=None, fmha_config=None,
                 parallelism_config=None, **kwargs):
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.scale = attn_configs.scale if attn_configs.scale else \
                     self.head_dim ** -0.5

        # 缓存参数（后续 prepare 中设置）
        self.block_table = None
        self.context_lens = None

    @staticmethod
    def support(attn_configs, attn_inputs):
        return not attn_inputs.is_prefill and \
               not attn_configs.use_mla and \
               attn_inputs.kv_cache is not None and \
               attn_inputs.kv_cache.separate_kv_cache

    def prepare(self, attn_inputs):
        """构建 Ascend 算子所需的参数。"""
        # 2D block table: [batch_size, max_blocks]
        self.block_table = attn_inputs.kv_cache_block_id_host
        
        # 当前已存储的序列长度（decode 阶段每个请求的 KV 长度）
        self.context_lens = attn_inputs.prefix_lengths + attn_inputs.input_lengths

    def forward(self, qkv, kv_cache, layer_idx=0):
        q, k, v = qkv.chunk(3, dim=-1)
        # q: [batch_size, num_heads, head_dim] (decode 阶段 tokens=1)
        # k, v: 当前 token 的 K/V（需写入 cache 后忽略）

        k_cache = kv_cache.k_cache_base  # NHD
        v_cache = kv_cache.v_cache_base  # NHD

        output = torch.empty_like(q)
        torch_npu._npu_paged_attention(
            query=q,
            key_cache=k_cache,
            value_cache=v_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=self.block_table,
            context_lens=self.context_lens,
            out=output,
        )
        return output
```

### 4.5 Block Table 适配详解

#### 4.5.1 CSR vs 2D 格式对比

| 格式 | 数据结构 | 使用者 | 说明 |
|------|---------|--------|------|
| CSR | `page_indice` [total_pages] + `page_indptr` [B+1] + `last_page_len` [B] | FlashInfer | 稀疏存储，padding 排除在 indices 外 |
| 2D | `block_table` [B, max_blocks] | torch_npu | 稠密矩阵，-1 表示无效 block |

#### 4.5.2 [v0.3新增] 当前 CSR 转换逻辑（供参考）

`FlashInferMlaParams.cc:262-342` 中的转换循环：

```cpp
// 对每个 batch item i:
//   current_page_num = ceil(seq_len / page_size)
//   从 kv_cache_block_id_host 取 current_page_num 个值放进 page_indice
//   decode_page_indptr[i+1] = decode_page_indptr[i] + current_page_num
//   paged_kv_last_page_len[i] = (seq_len - 1) % page_size + 1
```

Ascend 路径**不需要**这个转换——直接使用 2D `kv_cache_block_id_host`。

#### 4.5.3 [v0.3新增] -1 值处理预案

`kv_cache_block_id_host` 中 `-1` 表示 invalid/unused block slot，出现在以下场景：
1. 请求使用不足 `max_blocks` 个 block，尾部填充 `-1`
2. block 已被回收但尚未从表中清理

**预案**（按优先级排序）：
1. **首选**: 验证 `torch_npu` 算子在 block_table 中遇到 `-1` 时的行为——有效范围外的值可能被自动忽略
2. **备选 A**: 在 Python 端传入 block_table 前，将 `-1` 替换为 `0` 并配合 `context_lens` 确保不会访问到无效 block
3. **备选 B**: 在 C++ 端生成 block_table 时就填充有效占位值（如 block 0）

### 4.6 [v0.3新增] TP 交互确认

KV cache 的 `num_kv_heads` 在构造 KVCacheSpec 时已除以 `attn_tp_size`：

```cpp
// MHAKVCacheSpec.h:24-28
local_head_num_kv = attn_config.kv_head_num / parallelism_config.get_attn_tp_size();
```

因此：
- **分离 K/V 存储不改变 TP 行为**：每个 rank 的 K cache 维度 `[blocks, page_size, local_kv_heads, head_dim]` 中 `local_kv_heads` 已是 TP 后的值
- **block_table 在所有 rank 上相同**：block 分配是全局的（通过 all-gather 取最小值），每个 rank 独立持有完整的 block 列表
- **无需修改 TP 通信逻辑**：attention 的输出通过 `all_reduce` 聚合（已有逻辑），与 KV cache 的 layout 无关

---

## 四点五、RoPE Ascend 适配方案

> v0.2 新增此章节，v0.3 补充精确的实现细节。

### 4.5.1 现状

rtp-llm 的 MHA RoPE 实现依赖 `flashinfer.rope`：

**文件**: `rtp_llm/models_py/modules/factory/attention/cuda_impl/base_rotary_embedding_op.py:86-95`

```python
rope._apply_rope_pos_ids_cos_sin_cache(
    q=query,            # [total_tokens, num_heads, head_dim]
    k=key,              # [total_tokens, num_kv_heads, head_dim]
    q_rope=query,       # ← 原地修改: 输入输出同一 tensor
    k_rope=key,         # ← 原地修改: 输入输出同一 tensor
    cos_sin_cache=self.cos_sin_cache,
    pos_ids=rope_params.positions_d,
    interleave=self.is_neox_style,
)
```

### 4.5.2 [v0.3修正] cos_sin_cache 精确格式

由 C++ `RopeCache.genBaseCache()`（`rtp_llm/cpp/model_utils/RopeCache.cc:16`）生成：

- **Shape**: `[max_position_embeddings, rope_dim]`
- **Layout** (FlashInfer 使用 `interleave=False`): `[cos₀, cos₁, ..., cos_{d/2-1}, sin₀, sin₁, ..., sin_{d/2-1}]`
  - 前半部分: `rope_dim/2` 个 cosine 值
  - 后半部分: `rope_dim/2` 个 sine 值
- **dtype**: `torch.float32`
- **生成方式**: 
  ```
  inv_freq = 1.0 / (rope_theta ^ (arange(0, rope_dim, 2) / rope_dim))
  freqs = outer(arange(max_positions * scale), inv_freq)
  cos_sin = cat([freqs.cos(), freqs.sin()], dim=1)
  ```

### 4.5.3 [v0.3修正] Ascend RoPE 实现

**新增文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_rope.py`

基于准确的 cos_sin_cache 格式（非交错 `[cos..., sin...]`），纯 PyTorch 实现：

```python
import torch


def apply_rope_pos_ids_nhd(q, k, cos_sin_cache, pos_ids, is_neox_style=False):
    """
    Apply RoPE to query and key tensors using a precomputed cos/sin cache.

    Args:
        q: Query tensor  [tokens, num_heads, head_dim]
        k: Key tensor     [tokens, num_kv_heads, head_dim]
        cos_sin_cache:    [max_pos, rope_dim]  non-interleaved [cos_0..cos_{d/2-1}, sin_0..sin_{d/2-1}]
        pos_ids:          [tokens] int32 position indices
        is_neox_style:    interleave mode (False for LLaMA/Qwen, True for GPT-NeoX)

    Returns:
        (q_out, k_out): RoPE-transformed tensors (in-place on input tensors)
    """
    rope_dim = cos_sin_cache.shape[-1]
    half_dim = rope_dim // 2
    
    # 索引 cos_sin_cache 获取每 token 的 cos/sin [tokens, rope_dim]
    embedding = cos_sin_cache[pos_ids]  # [tokens, rope_dim]
    cos = embedding[:, :half_dim]       # [tokens, half_dim]
    sin = embedding[:, half_dim:]       # [tokens, half_dim]
    
    # 提取 Q/K 的 rope 部分: [tokens, heads, half_dim*2] → 两半各 half_dim
    q_rope = q[..., :rope_dim]
    k_rope = k[..., :rope_dim]
    
    # 旋转: 将后半旋转前半、前半旋转后半
    # 注意: -sin 应用于后半 → 前半；+sin 应用于前半 → 后半
    if is_neox_style:
        # GPT-NeoX: 相邻对旋转 (x[0],x[1]), (x[2],x[3]), ...
        q_rope_2 = q_rope.reshape(*q_rope.shape[:-1], -1, 2)
        k_rope_2 = k_rope.reshape(*k_rope.shape[:-1], -1, 2)
        
        q_neg = q_rope_2[..., 1]
        q_pos = q_rope_2[..., 0]
        k_neg = k_rope_2[..., 1]
        k_pos = k_rope_2[..., 0]
        
        cos_exp = cos.unsqueeze(-2).unsqueeze(-1)  # [tokens, 1, half_dim/2, 1?]
        sin_exp = sin.unsqueeze(-2).unsqueeze(-1)
        
        q_rot = torch.stack([
            q_pos * cos - q_neg * sin,
            q_pos * sin + q_neg * cos,
        ], dim=-1).flatten(start_dim=-2)
        k_rot = torch.stack([
            k_pos * cos - k_neg * sin,
            k_pos * sin + k_neg * cos,
        ], dim=-1).flatten(start_dim=-2)
    else:
        # LLaMA/Qwen: 前半与后半旋转
        q1, q2 = q_rope[..., :half_dim], q_rope[..., half_dim:]
        k1, k2 = k_rope[..., :half_dim], k_rope[..., half_dim:]
        
        cos = cos.unsqueeze(-2)  # [tokens, 1, half_dim]
        sin = sin.unsqueeze(-2)  # [tokens, 1, half_dim]
        
        q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    
    # 原地修改（与 FlashInfer 行为一致）
    q[..., :rope_dim] = q_rot
    k[..., :rope_dim] = k_rot
    
    return q, k
```

### 4.5.4 Ascend RoPE Embedding Op

**新增文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_rope_emb.py`

```python
from rtp_llm.models_py.modules.factory.attention.cuda_impl.base_rotary_embedding_op import BaseRotaryEmbeddingOp
from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_rope import apply_rope_pos_ids_nhd


class AscendRotaryEmbeddingOp(BaseRotaryEmbeddingOp):
    """Ascend RoPE using pure PyTorch implementation (replaces flashinfer.rope)."""

    def __init__(self, head_size, cos_sin_cache, token_per_block, is_neox_style,
                 rope_config=None, max_position_embeddings=32768):
        super().__init__(head_size, cos_sin_cache, token_per_block, is_neox_style,
                         rope_config, max_position_embeddings)

    def _apply_rope(self, query, key, rope_params):
        """Override to use pure PyTorch RoPE instead of flashinfer.rope."""
        if self.cos_sin_cache is not None:
            apply_rope_pos_ids_nhd(
                query, key,
                self.cos_sin_cache,
                rope_params.positions_d,
                is_neox_style=self.is_neox_style,
            )
        else:
            # Fallback: 不应该走到这里（Ascend 上 cos_sin_cache 总是存在）
            raise RuntimeError("AscendRotaryEmbeddingOp requires cos_sin_cache")
```

> [v0.3新增] `AscendRotaryEmbeddingOp` 继承 `BaseRotaryEmbeddingOp`，仅覆盖 `_apply_rope()` 方法。`_prepare_warmup_cache_indices()` 等 warmup 逻辑仍使用基类实现（其中涉及 `flashinfer.get_batch_indices_positions` — 这部分在 warmup/JIT 路径下，**需另外确认是否需要替换**；如果 Ascend 不需要 JIT warmup 则可以直接跳过）。

### 4.5.5 [v0.3新增] flashinfer.get_batch_indices_positions 替换

`BaseRotaryEmbeddingOp._prepare_warmup_cache_indices()` 使用了 `flashinfer.get_batch_indices_positions` 和 `flashinfer.get_seq_lens`。这两个函数在 Ascend 上不可用。

**方案**: 在 `AscendRotaryEmbeddingOp` 中覆盖 `_prepare_warmup_cache_indices()`，使用纯 PyTorch 实现：

```python
def _prepare_warmup_cache_indices(self, num_tokens, device):
    """纯 PyTorch 实现，替代 flashinfer.get_batch_indices_positions。"""
    # 简单实现: 假设单 batch、连续 tokens
    batch_indices = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    positions = torch.arange(num_tokens, dtype=torch.int32, device=device)
    
    max_num_pages = (num_tokens + self.token_per_block - 1) // self.token_per_block
    kv_page_indices = positions // self.token_per_block
    kv_page_indptr = torch.tensor([0, max_num_pages], dtype=torch.int32, device=device)
    
    last_page_len = num_tokens % self.token_per_block
    if last_page_len == 0:
        last_page_len = self.token_per_block
    kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)
    
    return batch_indices, positions, kv_page_indices, kv_page_indptr, kv_last_page_len, max_num_pages
```

---

## 五、KV Cache Copy Kernel 替换

### 5.1 CUDA Kernel 清单及替换方案

| CUDA Kernel | 源文件 | 功能 | NPU 替换方案 |
|------------|--------|------|-------------|
| `convertOffsetAndSize2IdxKernel` | `kv_cache_kernels.cu` | offset+size 转 index | `torch::Tensor` 算术操作 |
| `reuseCacheKernel` | `kv_cache_kernels.cu` | 复用 prefix cache block | `torch.index_copy_` + `torch.index_select` |

### 5.2 reuseCacheKernel 替换

```python
def reuse_cache_block(src_block_ids, dst_block_ids, kv_cache_tensor):
    selected = torch.index_select(kv_cache_tensor, 0, src_block_ids)
    kv_cache_tensor.index_copy_(0, dst_block_ids, selected)
```

分离模式下需对 K/V 分别执行：

```python
def reuse_cache_block_separate(src_block_ids, dst_block_ids, k_cache, v_cache):
    k_selected = torch.index_select(k_cache, 0, src_block_ids)
    v_selected = torch.index_select(v_cache, 0, src_block_ids)
    k_cache.index_copy_(0, dst_block_ids, k_selected)
    v_cache.index_copy_(0, dst_block_ids, v_selected)
```

---

## 六、参数结构适配

### 6.1 Ascend Attention 参数类

**新增文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_attn_params.py`

```python
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class AscendAttnParams:
    """Ascend attention 算子参数（对应 torch_npu 接口）。"""
    block_table: Optional[torch.Tensor] = None       # [batch_size, max_blocks_per_seq] (2D)
    seq_lens: Optional[torch.Tensor] = None           # [batch_size] context lengths
    slot_mapping: Optional[torch.Tensor] = None       # [num_tokens]
    actual_seq_lengths_q: Optional[torch.Tensor] = None   # Q cumulative seq lens
    actual_seq_lengths_kv: Optional[torch.Tensor] = None  # KV cumulative seq lens
    num_kv_heads: int = 0
    num_heads: int = 0
    head_dim: int = 0
    block_size: int = 128
    scale: float = 1.0
```

### 6.2 [v0.3修正] 参数填充流程

```python
def build_ascend_params(attn_inputs: PyAttentionInputs, page_size: int) -> AscendAttnParams:
    """从 PyAttentionInputs 构建 AscendAttnParams。"""
    params = AscendAttnParams()
    
    # 2D block table（直接使用，无需 CSR 转换）
    params.block_table = attn_inputs.kv_cache_block_id_host
    
    # Decode context lengths
    if attn_inputs.sequence_lengths.numel() > 0:
        # Decode: 每个请求当前已存储的 token 数
        params.seq_lens = attn_inputs.prefix_lengths + attn_inputs.input_lengths
    
    # Prefill cumulative seq lens
    if attn_inputs.is_prefill:
        prefix_len = attn_inputs.prefix_lengths
        input_len = attn_inputs.input_lengths
        kv_len = prefix_len + input_len
        
        zero = torch.zeros(1, dtype=torch.int32, device=input_len.device)
        params.actual_seq_lengths_q = torch.cat([zero, torch.cumsum(input_len, dim=0)])
        params.actual_seq_lengths_kv = torch.cat([zero, torch.cumsum(kv_len, dim=0)])
    
    params.block_size = page_size
    return params
```

---

## 七、Factory 注册

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

[v0.3修正] 扩展后（新实现优先于 SDPA baseline，移除 MLA 注册）：

```python
elif device_type == DeviceType.Ascend:
    # Paged attention implementations (优先使用 — 基于 torch_npu)
    from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_prefill import (
        AscendPrefillImpl,
    )
    from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_decode import (
        AscendDecodeImpl,
    )

    PREFILL_MHA_IMPS.append(AscendPrefillImpl)    # 新增，优先于 SDPA
    DECODE_MHA_IMPS.append(AscendDecodeImpl)       # 新增，优先于 SDPA

    # SDPA baseline (fallback — 用于单请求/无 paged KV 场景)
    from rtp_llm.models_py.modules.factory.attention.ascend_impl.torch_sdpa import (
        AscendSDPAPrefillImpl,
        AscendSDPADecodeImpl,
    )
    PREFILL_MHA_IMPS.append(AscendSDPAPrefillImpl)
    DECODE_MHA_IMPS.append(AscendSDPADecodeImpl)
```

> **注册机制说明**: 工厂按注册顺序遍历实现列表，调用 `support()` 方法检查是否支持。
> 注册在前 = 优先级更高。因此新的 Paged Attention 实现应注册在 SDPA baseline 之前。
> `AscendPrefillImpl.support()` 需额外检查 `kv_cache.separate_kv_cache == True`，
> 当 separation 未启用时 fallback 到 SDPA。

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
| **C++ 内存管理** | `rtp_llm/cpp/cache/BlockPool.h` | 修改 | [v0.3扩展] 添加 `separate_kv_cache_`、`k_cache_buffer_`、`v_cache_buffer_`、`global_layer_k_tensors_`、`global_layer_v_tensors_` |
| | `rtp_llm/cpp/cache/BlockPool.cc` | 修改 | `initializeCacheBuffer()` 增加分离分配路径；`processMemoryLayout()` 适配分离模式 |
| | `rtp_llm/cpp/cache/KVCacheAllocator.cc` | 修改 | 扩展 `execBatchCopy` 支持分离 K/V 拷贝 |
| | `rtp_llm/cpp/cache/KVCacheManager.cc` | 修改 | `getMainModelCacheLayerLayout()` 增加分离路径；`setKVBlockValue()` 适配分离模式 |
| **C++ 绑定** | `rtp_llm/models_py/bindings/OpDefs.h` | 修改 | `KVCache`/`LayerKVCache` 添加分离字段 + NHD reshape |
| **Python 类型** | `rtp_llm/ops/librtp_compute_ops/__init__.pyi` | 修改 | `LayerKVCache` 添加 `k_cache_base`、`v_cache_base` |
| **Python RoPE** | `ascend_impl/ascend_rope.py` | **新增** | 纯 PyTorch RoPE 实现（基于 C++ cos_sin_cache 格式） |
| | `ascend_impl/ascend_rope_emb.py` | **新增** | Ascend RoPE Embedding Op（替换 flashinfer.rope） |
| **Python 写入** | `ascend_impl/ascend_kv_cache_write_op.py` | **新增** | MHA KV write（torch_npu._npu_reshape_and_cache） |
| **Python 读取** | `ascend_impl/ascend_prefill.py` | **新增** | Prefill attention（torch_npu.npu_fused_infer_attention_score） |
| | `ascend_impl/ascend_decode.py` | **新增** | Decode attention（torch_npu._npu_paged_attention） |
| **Python 参数** | `ascend_impl/ascend_attn_params.py` | **新增** | Ascend 参数结构 |
| **Factory** | `attention/__init__.py` | 修改 | 注册 Ascend paged 实现（优先于 SDPA） |

---

## 十、关键风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| KV 分离存储导致内存分配和管理逻辑改动 | BlockPool、KVCacheManager 等核心组件需扩展 | 利用已有的 `k_block_size()`/`v_block_size()` 预留接口；分离存储作为配置开关（`separate_kv_cache`），不影响 CUDA 路径 |
| NPU 算子要求 NHD layout，与 rtp-llm HND layout 不兼容 | KV cache 读写失败 | 通过 C++ `getLayerCache()` 直接 reshape 为 NHD，避免 Python 端 permute 开销 |
| `torch_npu._npu_reshape_and_cache` 对 NHD layout 的验证 | 写入正确性不确定 | 优先在子阶段 4.3 做单层 KV write 正确性验证 |
| `npu_fused_infer_attention_score` block_table 格式不兼容 | Prefill 不可用 | rtp-llm Python 端已有 2D block table（`kv_cache_block_id_host`），确保传递正确格式 |
| **`block_table` 中的 `-1` 值可能导致算子报错** | [v0.3新增] prefill/decode 失败 | 验证算子的 `-1` 处理行为；备选：传入前替换为占位值 |
| `torch_npu._npu_paged_attention` 性能不达标 | Decode 性能差 | 先跑通功能，Phase 10 优化；备选 aclnn 自定义算子 |
| 分离存储的 block 批量拷贝需要同时拷贝 K/V | 拷贝逻辑复杂化 | 在 `execBatchCopy` 中对 K/V 分别执行拷贝 |
| `torch_npu` API 在某些 CANN 版本不可用 | 功能受阻 | 参考 vllm-ascend Device Adaptor 模式按型号选择 API |
| RoPE 依赖 FlashInfer CUDA kernel | 无 RoPE 无法正确推理 | 新增纯 PyTorch RoPE 实现（§4.5），后期可用 torch_npu 融合算子优化 |
| **RoPE 精度与 FlashInfer 不一致** | [v0.3新增] 模型输出偏离 | 子阶段 4.3 中做逐 token 对比，容差 < 1e-3 |
| **[v0.3新增] TP 场景下分离存储的 K/V buffer 大小计算** | TP 多卡场景 K/V 内存分布错误 | `local_head_num_kv` 已在 `MHAKVCacheSpec` 中计入 TP；分离模式下 K/V 各用 `k_block_size_bytes()` / `v_block_size_bytes()`，理论上与合并模式一致 |
| **Warmup/JIT 路径可能使用 flashinfer 工具函数** | [v0.3新增] warmup 阶段报错 | `AscendRotaryEmbeddingOp` 覆盖 `_prepare_warmup_cache_indices()`，使用纯 PyTorch 实现 |

---

## 十一、[v0.3新增] 验证方案

### 11.1 验证模型

| 优先级 | 模型 | 类型 | 参数 | head_dim | num_kv_heads | 选择理由 |
|--------|------|------|------|----------|-------------|---------|
| P0 | **Qwen3-4B** | 纯 MHA | 4B | 128 | 4~8 | 模型小，加载快，适合快速迭代 |
| P1 | **Qwen2-7B** | 纯 MHA | 7B | 128 | 4 | 广泛使用，社区对比基准多 |
| P2 | Qwen2.5-1.5B | 纯 MHA | 1.5B | 128 | 2 | 最小验证单元 |

所有模型均为 MHA 架构，无 MLA，与 v0.3 范围一致。

### 11.2 验证维度

| 阶段 | 验证项 | 方法 | 标准 |
|------|--------|------|------|
| 4.1~4.2 单元测试 | `BlockPool::init()` 分离分配 | C++ 单元测试 | K/V buffer 正确分配在 NPU 上，size 正确 |
| | `getLayerCache()` NHD layout | Python 单测 | shape 为 `[blocks, seq, kv_heads, dim]` |
| | `slot_mapping` 正确性 | 与 CUDA 路径对比 | 值完全一致（CPU 整数运算） |
| 4.3 RoPE 验证 | `apply_rope_pos_ids_nhd` 精度 | 相同输入，与 flashinfer.rope 结果对比 | max abs diff < 1e-3 |
| | 端到端 RoPE + attention | 单请求 prefill，对比 attention 输出 | max abs diff < 1e-2（考虑 attention 累积误差） |
| 4.4 KV Write 验证 | `_npu_reshape_and_cache` 写入正确性 | 写入后 `torch_npu._npu_paged_attention` 读取验证 | K/V 值一致 |
| 4.5 Prefill 验证 | `npu_fused_infer_attention_score` 端到端 | 单请求 prefill，与 FlashInfer 对比 | token 级别 max abs diff < 1e-2 |
| | 多请求 batch prefill | batch_size=2,4,8 | 所有请求 attention 输出正确 |
| 4.6 Decode 验证 | `_npu_paged_attention` 端到端 | 单请求多步 decode（×10 steps） | 每步 attention 输出与 FlashInfer 一致 |
| | 多请求 batch decode | batch_size=2,4,8 | 所有请求正确 |
| 4.7 端到端 | 完整推理 pipeline | Qwen3-4B 单请求 >100 tokens 生成 | 生成 token 序列与 CUDA 路径完全一致 |
| | TP=2/4 推理 | Qwen3-4B + TP | 输出与单卡一致 |

### 11.3 测试工具

- **单元测试**: `rtp_llm/cpp/cache/test/` 下的 C++ 单元测试（已有测试框架）
- **逐层对比**: Python 脚本 hook attention 层，分别运行 CUDA 和 Ascend 路径，对比输出
- **端到端**: `rtp_llm/models_py/standalone/auto_model.py` 推理脚本

---

## 十二、里程碑与验证计划

> [v0.3修正] 移除 MLA 子阶段，增加 RoPE 详细验证，调整时间估计。

| 子阶段 | 周期 | 交付物 | 验证标准 |
|--------|------|--------|---------|
| 4.1 KV 分离存储 C++ 适配 | 4 天 | `CacheConfig`/`BufferTypes`/`BlockPool.h`/`BlockPool.cc`/`OpDefs`/`KVCacheManager` 分离 K/V 支持 | `BlockPool::init()` 在 NPU 上成功分配分离的 K/V buffer，`getLayerCache()` 返回 NHD layout |
| 4.2 Slot Mapping + 参数适配 | 2 天 | `AscendAttnParams` + Python 端参数构造 | slot_mapping 计算正确，2D block_table 正确传递 |
| 4.3 RoPE Ascend 适配 | 3 天 | `ascend_rope.py` + `ascend_rope_emb.py`（纯 PyTorch RoPE） | MHA RoPE 输出与 FlashInfer 一致（max abs diff < 1e-3） |
| 4.4 MHA KV Write + Prefill | 5 天 | `ascend_kv_cache_write_op.py` + `ascend_prefill.py` | 单请求 prefill 输出与 CUDA 路径一致（max abs diff < 1e-2） |
| 4.5 MHA Decode | 5 天 | `ascend_decode.py` | 单请求多步 decode 生成正确 token |
| 4.6 Factory 注册 + 端到端 | 3 天 | `__init__.py` 注册 + 端到端推理 | Qwen3-4B 端到端推理跑通，>100 tokens 生成与 CUDA 路径一致 |
| 4.7 Batch Copy + Prefix Cache | 2 天 | 分离 K/V block 批量拷贝 | Prefix cache 功能正常 |
| 4.8 TP 多卡验证 | 2 天 | TP=2/4 端到端推理 | TP 推理输出与单卡一致 |
| **合计** | **~4.5 周** | | |

> [v0.3修正] 相比 v0.2 的 3.5 周，v0.3 增加了 1 周 buffer（RoPE +1天，C++ +1天，端到端 +1天，新增 TP 验证 2天）。

---

## 十三、[v0.3新增] 实施建议

### 13.1 实施顺序建议

```
Phase 4.1 (C++ 分离存储)
    │
    ├─→ Phase 4.2 (Slot Mapping)  ←── 可与 4.1 并行（Python 端，依赖少）
    │
    ├─→ Phase 4.3 (RoPE)          ←── 可与 4.1 并行（Python 端，仅需 cos_sin_cache）
    │
    └─→ Phase 4.4 (KV Write + Prefill)  ←── 依赖 4.1 + 4.2 + 4.3
         │
         └─→ Phase 4.5 (Decode)   ←── 依赖 4.4
              │
              └─→ Phase 4.6 (端到端)  ←── 依赖 4.5
                   │
                   ├─→ Phase 4.7 (Batch Copy)
                   └─→ Phase 4.8 (TP 验证)
```

### 13.2 每日验证脚本建议

建议在实施过程中维护以下验证脚本：

```bash
# 1. C++ 层级 K/V buffer 分配验证
cd build && ctest -R BlockPoolTest -V

# 2. Python 层 getLayerCache NHD layout 验证
python -c "
from rtp_llm.ops import KVCache
# ... 构造 KVCache with separate_kv_cache=True
# ... assert shape == [blocks, page_size, kv_heads, head_dim]
"

# 3. RoPE 精度验证
python tests/ascend/test_rope_precision.py  # 新增脚本

# 4. 单层 attention 对比
python tests/ascend/test_attention_layer.py --model Qwen3-4B --layer 0

# 5. 端到端推理
python rtp_llm/models_py/standalone/auto_model.py --model Qwen3-4B --device ascend
```
