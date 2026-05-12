# xLLM NPU KV Cache Adaptation

> **Workspace:** c:\WorkingSpace\Code\xllm

---

## User
> xLLM NPU KV cache适配方案

## Assistant
我来研究一下 xLLM 项目中 KV cache 的现有实现以及 NPU 相关的适配情况。

让我深入研究 KV cache 的核心实现和 NPU 适配细节。

现在我已经对 xLLM 的 NPU KV cache 架构有了全面的了解。下面是详细的适配方案分析。

---

## xLLM NPU KV Cache 适配方案详解

### 一、整体架构概览

xLLM 的 KV cache 在 NPU (Ascend) 上的适配涉及以下核心层次：

```
┌─────────────────────────────────────────────────────┐
│                  Scheduler Layer                      │
│  (Block 分配/回收, Prefix Cache, PD 调度)             │
├─────────────────────────────────────────────────────┤
│               KV Cache Manager                       │
│  (Block 粒度管理, Swap, Transfer)                     │
├─────────────────────────────────────────────────────┤
│            KV Cache Core (kv_cache/)                 │
│  KVCache → KVCacheImpl / Indexed / Quantized / Linear│
│  KVCacheShape (NPU Layout) / KVCacheCreateOptions    │
├─────────────────────────────────────────────────────┤
│         KV Cache Transfer (kv_cache_transfer/)       │
│  LlmDataDist / Mooncake(Default/XTensor) / Spec      │
├─────────────────────────────────────────────────────┤
│           NPU Attention Kernels (kernels/npu/)        │
│  reshape_and_cache / flash_attention / paged_attention│
├─────────────────────────────────────────────────────┤
│           NPU Memory & Format Layer                   │
│  aclrtMalloc / ACL_FORMAT_ND / ACL_FORMAT_FRACTAL_NZ │
│  npu_format_cast / XTensor / GlobalXTensor            │
└─────────────────────────────────────────────────────┘
```

---

### 二、KV Cache 核心实现

#### 2.1 类继承体系

| 类 | 用途 | NPU 适配要点 |
|---|---|---|
| [KVCacheImpl](file:///c:/WorkingSpace/Code/xllm/xllm/core/framework/kv_cache/kv_cache_impl.h) | 标准分页 KV cache | `torch::empty` + `npu_format_cast` |
| [IndexedKVCacheImpl](file:///c:/WorkingSpace/Code/xllm/xllm/core/framework/kv_cache/indexed_kv_cache_impl.h) | Lightning Indexer | index_cache 也需 NPU format cast |
| [QuantizedKVCacheImpl](file:///c:/WorkingSpace/Code/xllm/xllm/core/framework/kv_cache/quantized_kv_cache_impl.h) | INT8 量化 KV cache | **当前仅 MLU 支持** |
| [LinearAttentionKVCacheImpl](file:///c:/WorkingSpace/Code/xllm/xllm/core/framework/kv_cache/linear_attention_kv_cache_impl.h) | 线性注意力 (Qwen3.5) | conv_cache/ssm_cache 用 `ACL_FORMAT_ND` |

#### 2.2 NPU 内存格式适配

核心逻辑在 [kv_cache_utils.cpp](file:///c:/WorkingSpace/Code/xllm/xllm/core/framework/kv_cache/kv_cache_utils.cpp#L34-L49)：

```cpp
// NPU: 使用 torch::empty + npu_format_cast (而非 torch::zeros)
tensors.key_cache = at_npu::native::npu_format_cast(
    torch::empty(kv_cache_shape.key_cache_shape(),
                 torch::dtype(create_options.dtype())
                     .device(create_options.device())),
    npu_format_type);  // ACL_FORMAT_ND 或 ACL_FORMAT_FRACTAL_NZ
```

**关键差异**：
- **GPU/MLU**：使用 `torch::zeros` 初始化
- **NPU**：使用 `torch::empty` + `npu_format_cast`，因为 NPU 的 5HD/NZ 格式需要显式转换

#### 2.3 格式选择策略

在 [kv_cache_utils.cpp:155-160](file:///c:/WorkingSpace/Code/xllm/xllm/core/framework/kv_cache/kv_cache_utils.cpp#L155-L160)：

```cpp
aclFormat get_npu_kv_cache_format(const std::string& model_type) {
  return model_type == "deepseek_v3" && FLAGS_enable_prefix_cache
             ? ACL_FORMAT_FRACTAL_NZ
             : ACL_FORMAT_ND;
}
```

| 场景 | 格式 | 原因 |
|---|---|---|
| 通用模型 | `ACL_FORMAT_ND` | N-D 维度标准布局，兼容性好 |
| DeepSeek-V3 + Prefix Cache | `ACL_FORMAT_FRACTAL_NZ` | NZ 格式对 MLA 的低秩 KV 更友好，对齐到 16 字节 |

---

### 三、KV Cache Shape 适配

#### 3.1 标准 KV Cache Shape

在 [kv_cache_shape.cpp](file:///c:/WorkingSpace/Code/xllm/xllm/core/framework/kv_cache/kv_cache_shape.cpp) 中定义：

**非 MLA 模型**（如 Qwen2/3, LLaMA, GLM4）：
```
key_cache:   [n_blocks, block_size, n_kv_heads, head_dim]
value_cache: [n_blocks, block_size, n_kv_heads, head_dim]
```

**MLA 模型 (DeepSeek-V2/V3)**：
```
key_cache:   [n_blocks, block_size, 1, kv_lora_rank]
value_cache: [n_blocks, block_size, 1, qk_rope_head_dim]
```

#### 3.2 NPU NZ 格式特殊 Shape

DeepSeek-V3 启用 prefix cache 时，shape 需要对齐到 16 字节（NZ 格式要求）：

```cpp
// kv_cache_shape.cpp 中的 NPU NZ 适配
key_cache_shape_ = {n_blocks, ceil_div(kv_lora_rank, 16), block_size, 16};
value_cache_shape_ = {n_blocks, ceil_div(qk_rope_head_dim, 16), block_size, 16};
```

这种布局将 head_dim 维度拆分为 `(ceil_div(dim, 16), 16)`，满足 Ascend NPU 的 NZ 格式内存对齐要求。

#### 3.3 设备布局差异

```cpp
// kv_cache_shape.cpp - apply_device_layout()
// NPU: 保持 [n_blocks, block_size, n_head, head_dim] 不变
// MLU/ILU: 转置为 [n_blocks, n_head, block_size, head_dim]
```

---

### 四、NPU Attention Kernel 适配

在 [attention.cpp](file:///c:/WorkingSpace/Code/xllm/xllm/core/kernels/npu/attention.cpp) 中，KV cache 的读写通过三个核心 kernel 完成：

#### 4.1 Prefill 阶段 - `reshape_paged_cache` + `batch_prefill`

```cpp
// 写入 KV cache
void reshape_paged_cache(key, value, k_cache, v_cache, slot_mapping) {
  atb::npu_reshape_and_cache(key, value, k_cache, v_cache, slot_mapping);
}

// Prefill Attention
void batch_prefill(query, key, value, mask, seq_len, scale, output) {
  atb::npu_flash_attention(query, key, value, mask, seq_len, 
                           scale, num_heads, num_kv_heads, output);
}
```

#### 4.2 Decode 阶段 - `batch_decode` / `batch_decode_acl_graph`

```cpp
// 标准 Decode (非 ACL Graph)
void batch_decode(query, k_cache, v_cache, scale, block_table, seq_lens, output) {
  atb::npu_paged_attention(q, k_cache, v_cache, num_kv_heads, 
                           num_heads, scale, block_table, seq_lens, o);
}

// ACL Graph 兼容 Decode (避免 .to(kCPU) 打断图捕获)
void batch_decode_acl_graph(query, k_cache, v_cache, scale, 
                            block_table, seq_lens, tiling_data, output) {
  atb::npu_custom_paged_attention(q, k_cache, v_cache, num_kv_heads,
                                  num_heads, scale, block_table, 
                                  seq_lens, tiling_data, o);
}
```

**ACL Graph 适配要点**：标准 `npu_paged_attention` 内部有 `.to(kCPU)` 操作会打断 ACL Graph 捕获，因此提供了 `npu_custom_paged_attention` 变体，通过预计算的 `tiling_data` 避免 CPU 同步。

---

### 五、KV Cache Transfer 适配（PD 分离部署）

xLLM 在 NPU 上支持三种 KV Cache 传输方案：

#### 5.1 LlmDataDist Transfer

[llm_data_dist_transfer.h](file:///c:/WorkingSpace/Code/xllm/xllm/core/framework/kv_cache_transfer/llm_data_dist_transfer.h)

- 使用华为 LlmDataDist 库进行 KV cache 传输
- 通过 `aclrtMalloc(ACL_MEM_MALLOC_HUGE_ONLY)` 分配 NPU 设备内存
- 支持 MLA 模型的特殊处理
- 用于 Prefill-Decode 分离部署场景

#### 5.2 Mooncake Transfer (Default)

[mooncake_kv_cache_transfer.h](file:///c:/WorkingSpace/Code/xllm/xllm/core/framework/kv_cache_transfer/mooncake_kv_cache_transfer.h)

- 基于 Mooncake 传输引擎
- 同样使用 `aclrtMalloc` 分配内存
- 将 NPU 设备地址转换为 `torch::Tensor` 进行注册
- 支持逐层 KV cache push（配合 `NPULayerSynchronizer`）

#### 5.3 Mooncake Transfer (XTensor)

- **PD 分离部署专用**，通过 `FLAGS_enable_xtensor` 启用
- 使用 `GlobalXTensor` 统一管理 NPU 内存
- KV cache 分配在 XTensor 的连续虚拟地址空间中
- 支持跨节点 KV cache 迁移（通过 offset 映射）
- 格式统一使用 `ACL_FORMAT_ND`

---

### 六、XTensor 模式适配

在 [kv_cache.cpp:139-152](file:///c:/WorkingSpace/Code/xllm/xllm/core/framework/kv_cache/kv_cache.cpp#L139-L152)：

```cpp
if (create_options.enable_xtensor()) {
  XTensorAllocator& allocator = XTensorAllocator::get_instance();
  auto k_tensors = allocator.create_k_tensors(model_id, key_shape, dtype, num_layers);
  auto v_tensors = allocator.create_v_tensors(model_id, value_shape, dtype, num_layers);
  
  for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    // NPU: 需要显式 format cast
    k_tensor = at_npu::native::npu_format_cast(k_tensors[layer_idx], ACL_FORMAT_ND);
    v_tensor = at_npu::native::npu_format_cast(v_tensors[layer_idx], ACL_FORMAT_ND);
    kv_caches.emplace_back(KVCacheTensors{k_tensor, v_tensor});
  }
}
```

**XTensor 模式限制**：
- 仅支持 key/value cache（不支持 index_cache, conv_cache, ssm_cache）
- 不支持 linear attention
- 需要 `model_id` 用于区分不同模型的内存区域

---

### 七、NPU Decoder Layer 中的 KV Cache 使用

以 [NpuQwen3DecoderLayerImpl](file:///c:/WorkingSpace/Code/xllm/xllm/core/layers/npu/npu_qwen3_decoder_layer_impl.cpp) 为例：

```cpp
// build_node_variant_pack 中绑定 KV cache 到 ATB 算子
node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) = 
    atb_speed::Utils::AtTensor2Tensor(kv_cache.get_k_cache());
node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5) = 
    atb_speed::Utils::AtTensor2Tensor(kv_cache.get_v_cache());
node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 9) = 
    atb_speed::Utils::AtTensor2Tensor(input_params.block_tables);
node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) = 
    atb_speed::Utils::AtTensor2Tensor(input_params.new_cache_slots);
```

**关键点**：
- KV cache tensor 通过 `atb_speed::Utils::AtTensor2Tensor` 转换为 ATB 格式
- Prefill 和 Decode 使用不同的 ATB Node（不同的 param 配置）
- Block copy 在 decoder layer forward 中自动执行（prefix cache swap）

---

### 八、适配方案总结

| 维度 | GPU/CUDA | NPU (Ascend) | 适配要点 |
|---|---|---|---|
| **内存分配** | `torch::zeros` | `torch::empty` + `npu_format_cast` | NPU 格式需显式转换 |
| **内存格式** | NCHW/ND | ND / FRACTAL_NZ | DeepSeek-V3+prefix cache 用 NZ |
| **Shape 布局** | `[n_blocks, block_size, n_head, head_dim]` | 同左 (NPU) | MLU/ILU 需转置 dim1/dim2 |
| **MLA 对齐** | 无特殊要求 | NZ 格式需 16 字节对齐 | `ceil_div(dim, 16)` |
| **Prefill Attention** | Flash Attention | `npu_flash_attention` | ATB 封装 |
| **Decode Attention** | Paged Attention | `npu_paged_attention` | ATB 封装 |
| **ACL Graph** | N/A | `npu_custom_paged_attention` | 需预计算 tiling_data |
| **KV Cache 写入** | `reshape_and_cache` | `npu_reshape_and_cache` | ATB 封装 |
| **KV Cache 传输** | Mooncake | LlmDataDist / Mooncake / XTensor | 三种方案可选 |
| **量化 KV Cache** | 支持 (INT8) | **暂不支持** | 仅 MLU 后端实现 |
| **Block Copy** | CUDA memcpy | `NpuBlockCopy` | ATB 算子实现 |

### 九、新增模型 NPU KV Cache 适配 Checklist

若需要为新模型适配 NPU KV Cache，需关注：

1. **确认模型类型**：是否为 MLA 模型？是否需要 prefix cache？
2. **Shape 定义**：在 `KVCacheShape::init_key_cache_shape` 中添加 NPU 分支
3. **格式选择**：在 `get_npu_kv_cache_format` 中添加模型格式策略
4. **Decoder Layer**：实现 NPU 版 decoder layer，正确绑定 KV cache tensor 到 ATB 算子
5. **Attention Kernel**：确认 prefill/decode kernel 对 KV cache shape 的兼容性
6. **ACL Graph**：若需图模式加速 decode，需适配 `npu_custom_paged_attention` 的 tiling_data
7. **KV Cache Transfer**：若需 PD 分离，确认 LlmDataDist/Mooncake 对新 shape 的支持

## User
> xLLM KV初始化、更新、使用流程

## Assistant
让我深入追踪 xLLM KV Cache 从初始化到更新、使用的完整流程。

现在让我看看模型前向传播中 KV cache 的实际读写路径：

现在我已经掌握了完整的 KV Cache 生命周期。让我整理输出。

---

## xLLM KV Cache 初始化、更新、使用完整流程

### 总体流程图

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. 初始化阶段                                                        │
│   Engine → Worker → allocate_kv_cache() → KVCacheShape → KVCache    │
├──────────────────────────────────────────────────────────────────────┤
│ 2. 请求调度阶段                                                      │
│   Scheduler → Block分配 → Sequence.kv_state → KVCacheState           │
├──────────────────────────────────────────────────────────────────────┤
│ 3. 输入构建阶段                                                      │
│   BatchInputBuilder → slot_ids + block_tables → ModelInputParams     │
├──────────────────────────────────────────────────────────────────────┤
│ 4. 前向执行阶段                                                      │
│   Worker.step() → Executor.forward() → Model.forward() → Layer      │
│     ├─ Block Swap (prefix cache / eviction)                          │
│     ├─ Prefill: reshape_and_cache → flash_attention                  │
│     └─ Decode: paged_attention (with block_table)                    │
├──────────────────────────────────────────────────────────────────────┤
│ 5. Block 回收阶段                                                    │
│   Request 完成 → Scheduler 回收 Block → KVCache 对应区域可重用       │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 一、初始化阶段

#### 1.1 入口：Worker 初始化

[worker_impl.cpp:173-226](file:///c:/WorkingSpace/Code/xllm/xllm/core/runtime/worker_impl.cpp#L173-L226)

```
Engine
  └─ WorkerImpl::allocate_kv_cache(kv_cache_shape)
       ├─ 构建 KVCacheCreateOptions
       │    ├─ device, dtype, num_layers
       │    ├─ model_type (决定 NPU 格式)
       │    ├─ enable_xtensor (决定内存分配方式)
       │    ├─ enable_lighting_indexer (决定是否创建 index_cache)
       │    └─ enable_linear_attention (决定是否创建 conv/ssm_cache)
       └─ allocate_kv_caches(kv_caches_, kv_cache_shape, create_options)
```

#### 1.2 KVCacheShape 构建

[kv_cache_shape.cpp](file:///c:/WorkingSpace/Code/xllm/xllm/core/framework/kv_cache/kv_cache_shape.cpp) 根据 `KVCacheCapacity` + `ModelArgs` + `world_size` 计算每层 KV cache 的 shape：

| 模型类型 | key_cache_shape | value_cache_shape |
|---|---|---|
| 标准 (Qwen2/3, LLaMA) | `[n_blocks, block_size, n_kv_heads, head_dim]` | 同 key |
| MLA (DeepSeek-V2/V3) | `[n_blocks, block_size, 1, kv_lora_rank]` | `[n_blocks, block_size, 1, qk_rope_head_dim]` |
| MLA + NZ (DS-V3+prefix) | `[n_blocks, ceil_div(kv_lora_rank,16), block_size, 16]` | `[n_blocks, ceil_div(qk_rope_head_dim,16), block_size, 16]` |

#### 1.3 KVCache 内存分配

[kv_cache.cpp:122-170](file:///c:/WorkingSpace/Code/xllm/xllm/core/framework/kv_cache/kv_cache.cpp#L122-L170) 有两条路径：

**路径 A：标准分配（非 XTensor）**

```
allocate_kv_caches()
  └─ for each layer:
       └─ KVCache(kv_cache_shape, create_options, layer_idx)
            └─ create_kv_cache_impl()  // 根据 options 选择 Impl 类型
                 └─ KVCacheImpl(shape, options)
                      └─ create_kv_cache_tensors(shape, options)
                           ├─ NPU: torch::empty() + npu_format_cast(ACL_FORMAT_ND/NZ)
                           └─ GPU: torch::zeros()
```

**路径 B：XTensor 分配（PD 分离部署）**

```
allocate_kv_caches()
  └─ XTensorAllocator::create_k_tensors(model_id, shape, dtype, num_layers)
     XTensorAllocator::create_v_tensors(model_id, shape, dtype, num_layers)
       └─ for each layer:
            ├─ NPU: npu_format_cast(tensor, ACL_FORMAT_ND)
            └─ KVCache(KVCacheTensors{k_tensor, v_tensor})
```

**路径 C：Transfer 分配（LlmDataDist / Mooncake）**

```
WorkerImpl::allocate_kv_cache_with_transfer()
  └─ KVCacheTransferFactory::create()
       └─ transfer->allocate_kv_cache(kv_caches, num_layers, shape, dtype)
            ├─ LlmDataDist: aclrtMalloc() + convert_to_torch_tensor()
            ├─ Mooncake(Default): aclrtMalloc() + register
            └─ Mooncake(XTensor): GlobalXTensor + ACL_FORMAT_ND
```

#### 1.4 最终数据结构

初始化完成后，`WorkerImpl::kv_caches_` 是一个 `std::vector<KVCache>`，长度等于 `num_layers`，每个 `KVCache` 持有该层的 `key_cache_` 和 `value_cache_` 两个 tensor。

---

### 二、请求调度阶段（Block 分配）

#### 2.1 Block 概念

[block.h](file:///c:/WorkingSpace/Code/xllm/xllm/core/framework/block/block.h)：Block 是 KV cache 管理的最小单元，每个 Block 包含 `block_size` 个 token 的 KV cache 空间。

```
Block:
  ├─ id_: block 编号 (对应 KV cache tensor 的第 0 维索引)
  ├─ size_: block_size (每个 block 存储的 token 数)
  └─ ref_count_: 引用计数 (支持 prefix cache 共享)
```

#### 2.2 Block 分配流程

```
Scheduler::schedule()
  └─ for each sequence in batch:
       └─ KVCacheManager::allocate(sequence)
            └─ BlockManager::allocate(num_blocks)
                 └─ 返回 Block 列表 (每个 Block 有唯一 id)
       └─ sequence.kv_state().add_kv_blocks(blocks)
            └─ blocks_.insert(blocks)  // 添加到序列的 block 列表
```

#### 2.3 KVCacheState 追踪

[sequence_kv_state.h](file:///c:/WorkingSpace/Code/xllm/xllm/core/framework/request/sequence_kv_state.h)：每个 Sequence 维护自己的 KV cache 状态：

```
KVCacheState:
  ├─ kv_cache_tokens_num_: 已缓存的 token 数
  ├─ blocks_: 分配的 Block 列表
  ├─ src_blocks_: swap 源 blocks (prefix cache / eviction)
  ├─ transfer_kv_info_: PD 分离传输信息
  └─ num_owned_shared_blocks_: 共享 block 数量
```

**Slot 计算**（[sequence_kv_state.cpp:112-126](file:///c:/WorkingSpace/Code/xllm/xllm/core/framework/request/sequence_kv_state.cpp#L112-L126)）：

```cpp
std::vector<int32_t> KVCacheState::kv_cache_slots(int32_t pos_start, int32_t pos_end) {
  for (int32_t i = pos_start; i < pos_end; ++i) {
    const int32_t block_id = blocks_[i / block_size].id();
    const int32_t block_offset = i % block_size;
    slots.push_back(block_id * block_size + block_offset);
  }
  return slots;
}
```

**Slot 的含义**：`slot = block_id * block_size + offset_in_block`，是 KV cache tensor 中的线性位置，用于 `reshape_and_cache` 算子将新 token 的 K/V 写入正确的位置。

---

### 三、输入构建阶段

#### 3.1 BatchInputBuilder 构建

[batch_input_builder.cpp:454-503](file:///c:/WorkingSpace/Code/xllm/xllm/core/framework/batch/batch_input_builder.cpp#L454-L503)：

```
BatchInputBuilder::process_single_sequence()
  ├─ sequence.kv_state().incr_kv_cache_tokens_num(q_seq_len)
  ├─ slot_ids = sequence.kv_state().kv_cache_slots(n_kv_cache_tokens, seq_len)
  │    └─ 计算 new_token_slot_ids (写入位置)
  ├─ blocks = sequence.kv_state().kv_blocks()
  │    └─ 收集 block_ids → block_tables_vec
  ├─ 计算 write_block_ids (需要写入的 block)
  └─ 处理 transfer_kv_info (PD 分离)
```

#### 3.2 关键输出：ModelInputParams

```
ModelInputParams:
  ├─ new_cache_slots: [n_tokens] — 每个 token 写入 KV cache 的 slot 位置
  ├─ block_tables: [n_seq, max_n_blocks] — 每个序列的 block_id 表
  ├─ kv_seq_lens: [n_seq] — 每个序列的 KV cache 长度
  ├─ q_seq_lens: [n_seq] — 每个序列的 query 长度
  ├─ swap_blocks: 需要交换的 block 对 (prefix cache / eviction)
  ├─ src/dst_block_indices: block copy kernel 的源/目标
  └─ graph_buffer.tiling_data: ACL graph 模式的 tiling 数据
```

---

### 四、前向执行阶段

#### 4.1 整体调用链

```
WorkerImpl::step(input)
  ├─ [PD Push] kv_cache_transfer_->push_kv_blocks_async(...)
  ├─ model_executor_->forward(token_ids, positions, kv_caches_, input_params)
  │    └─ CausalLM::forward(tokens, positions, kv_caches, params)
  │         └─ Model::forward(tokens, positions, kv_caches, params)
  │              └─ for each layer:
  │                   ├─ [Block Copy] block_copy_(kv_cache.get_k/v_cache(), ...)
  │                   └─ decoder_layer_(x, cos, sin, mask, kv_caches[i], ...)
  └─ [PD Push Wait] folly::collectAll(futures)
```

#### 4.2 Block Swap（KV Cache 块级操作）

[worker_impl.cpp:539-590](file:///c:/WorkingSpace/Code/xllm/xllm/core/runtime/worker_impl.cpp#L539-L590)：

```
WorkerImpl::apply_kv_block_swaps(input_params)
  ├─ NPU + enable_block_copy_kernel: 使用 NpuBlockCopy ATB 算子
  └─ 否则: 逐层 swap_blocks()
       └─ for each layer:
            kv_caches_[layer_id].swap_blocks(src_tensor, dst_tensor)
              └─ torch::index_select(key_cache_, 0, src)  // 读取源 block
              └─ key_cache_.index_copy_(0, dst, selected)  // 写入目标 block
```

**用途**：
- **Prefix Cache**：将共享前缀的 KV cache block 从 cache store 换入
- **Eviction**：将不活跃的 block 换出到 host/cache store

#### 4.3 NPU Decoder Layer 中的 KV Cache 使用

以 [NpuQwen3DecoderLayerImpl](file:///c:/WorkingSpace/Code/xllm/xllm/core/layers/npu/npu_qwen3_decoder_layer_impl.cpp) 为例：

```
decoder_layer_.forward(x, cos, sin, mask, kv_cache, input_params, ...)
  └─ build_node_variant_pack(node, x, cos, sin, mask, kv_cache, input_params, ...)
       ├─ inTensors[WEIGHT_COUNT + 4] = kv_cache.get_k_cache()  // K cache
       ├─ inTensors[WEIGHT_COUNT + 5] = kv_cache.get_v_cache()  // V cache
       ├─ inTensors[WEIGHT_COUNT + 6] = kv_seq_lens              // 序列长度
       ├─ inTensors[WEIGHT_COUNT + 9] = block_tables             // Block 表
       ├─ inTensors[WEIGHT_COUNT + 10] = new_cache_slots         // 写入 slot
       └─ [chunked_prefill] inTensors[...] = q_seq_lens
```

ATB 算子内部执行：
- **Prefill**: `npu_reshape_and_cache` 写入 + `npu_flash_attention` 读取
- **Decode**: `npu_paged_attention` 直接从 paged KV cache 读取

#### 4.4 KV Cache 写入：reshape_and_cache

[kernels/npu/attention.cpp:22-27](file:///c:/WorkingSpace/Code/xllm/xllm/core/kernels/npu/attention.cpp#L22-L27)：

```
reshape_paged_cache(key, value, k_cache, v_cache, slot_mapping)
  └─ atb::npu_reshape_and_cache(key, value, k_cache, v_cache, slot_mapping)
```

**写入逻辑**：
```
对于每个 token t:
  slot = slot_mapping[t]  // = block_id * block_size + offset
  block_id = slot / block_size
  offset = slot % block_size
  k_cache[block_id, offset, :, :] = key[t, :, :]  // 写入 K
  v_cache[block_id, offset, :, :] = value[t, :, :]  // 写入 V
```

#### 4.5 KV Cache 读取：Attention

**Prefill 阶段**（[attention.cpp:30-40](file:///c:/WorkingSpace/Code/xllm/xllm/core/kernels/npu/attention.cpp#L30-L40)）：
```
batch_prefill(query, key, value, mask, seq_len, scale, output)
  └─ atb::npu_flash_attention(...)
       // key/value 是当前 token 的 K/V（尚未写入 cache）
       // 同时 reshape_and_cache 将其写入 k_cache/v_cache
```

**Decode 阶段**（[attention.cpp:42-62](file:///c:/WorkingSpace/Code/xllm/xllm/core/kernels/npu/attention.cpp#L42-L62)）：
```
batch_decode(query, k_cache, v_cache, scale, block_table, seq_lens, output)
  └─ atb::npu_paged_attention(q, k_cache, v_cache, num_kv_heads,
                               num_heads, scale, block_table, seq_lens, o)
       // 通过 block_table 索引 k_cache/v_cache 中的 block
       // 计算注意力: output = softmax(Q * K^T / sqrt(d)) * V
```

**ACL Graph 模式**（[attention.cpp:64-82](file:///c:/WorkingSpace/Code/xllm/xllm/core/kernels/npu/attention.cpp#L64-L82)）：
```
batch_decode_acl_graph(query, k_cache, v_cache, scale,
                       block_table, seq_lens, tiling_data, output)
  └─ atb::npu_custom_paged_attention(...)
       // 使用预计算的 tiling_data 避免 CPU 同步
       // 适配 ACL Graph 捕获
```

#### 4.6 AttentionMetadata 构建

[attention_metadata_builder.cpp:52](file:///c:/WorkingSpace/Code/xllm/xllm/core/layers/common/attention_metadata_builder.cpp#L52)：

```
AttentionMetadataBuilder::build(params, model_args)
  └─ attn_metadata.slot_mapping = params.new_cache_slots  // 写入位置
     attn_metadata.block_table = params.block_tables       // Block 索引表
     attn_metadata.kv_seq_lens = params.kv_seq_lens       // 序列长度
     attn_metadata.paged_kv_indptr = params.paged_kv_indptr
     attn_metadata.paged_kv_indices = params.paged_kv_indices
     [NPU] attn_metadata.kv_seq_lens_host = tensor(kv_seq_lens_vec)  // CPU 端
     [NPU+Graph] attn_metadata.paged_attention_tiling_data = tiling_data
```

---

### 五、Block 回收阶段

```
Request 完成
  └─ Scheduler::deallocate(request)
       └─ for each sequence:
            └─ KVCacheManager::deallocate(sequence)
                 └─ BlockManager::free(sequence.kv_blocks())
                      └─ 减少引用计数, ref_count==0 时回收 block
            └─ sequence.kv_state().reset()
                 └─ kv_cache_tokens_num_ = 0, blocks_.clear()
```

**注意**：Block 回收只是释放逻辑管理权，KV cache tensor 的物理内存不会被释放（预分配的），回收后的 block 可以分配给新的序列使用，其对应的 KV cache 区域会被新 token 覆盖。

---

### 六、完整数据流总结

```
                    ┌─────────────────────┐
                    │   KV Cache Tensor   │
                    │  [n_blocks, bs,     │
                    │   n_heads, head_dim]│
                    └──────┬──────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐       ┌─────▼─────┐     ┌─────▼─────┐
   │ 写入 K/V │       │ 读取 K/V  │     │ Block Swap │
   │ (Prefill)│       │ (Decode)  │     │ (Prefix/   │
   │          │       │           │     │  Eviction) │
   │ reshape  │       │ paged     │     │ index_     │
   │ and      │       │ attention │     │ select +   │
   │ cache    │       │           │     │ index_copy │
   └────┬─────┘       └─────┬─────┘     └─────┬─────┘
        │                   │                 │
        │  slot_mapping     │ block_table     │ src/dst
        │  (写入位置)        │ (Block索引)     │ block_ids
        │                   │                 │
   ┌────▼───────────────────▼─────────────────▼─────┐
   │              ModelInputParams                    │
   │  new_cache_slots / block_tables / swap_blocks   │
   └────────────────────┬────────────────────────────┘
                        │
                   ┌────▼────┐
                   │Scheduler │
                   │Block分配  │
                   └────┬────┘
                        │
                   ┌────▼────┐
                   │Sequence  │
                   │kv_state()│
                   │blocks_   │
                   │tokens_num│
                   └─────────┘
```

### 七、NPU 特有关键路径总结

| 阶段 | NPU 特殊处理 |
|---|---|
| **初始化** | `torch::empty` + `npu_format_cast(ACL_FORMAT_ND/NZ)`；XTensor 模式用 GlobalXTensor 分配 |
| **Block Swap** | NPU 使用 `NpuBlockCopy` ATB 算子（`FLAGS_enable_block_copy_kernel`）|
| **写入** | `atb::npu_reshape_and_cache`，slot_mapping 由 ATB 算子内部处理 |
| **Prefill 读取** | `atb::npu_flash_attention`，传入 K/V tensor + mask + seq_len |
| **Decode 读取** | `atb::npu_paged_attention`，传入 k/v_cache + block_table + seq_lens |
| **ACL Graph** | `atb::npu_custom_paged_attention` + 预计算 tiling_data，避免 `.to(kCPU)` |
| **PD 传输** | LlmDataDist / Mooncake(Default/XTensor)，逐层同步 push |
| **DeepSeek-V3 NZ** | `ACL_FORMAT_FRACTAL_NZ` + 16 字节对齐 shape |

