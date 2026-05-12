# rtp-llm KV Cache管理

> **Workspace:** c:\WorkingSpace\Code\rtp-llm

---

## User
> KV cache排布

## Assistant
根据代码分析，rtp-llm 项目中的 KV cache 支持两种主要的排布方式：

## 1. MHA (Multi-Head Attention) KV Cache 排布

**5D 张量格式**：`[num_pages, 2, num_kv_heads, page_size, head_dim]`

这是 **HND (Head, Num_pages, Dim)** layout，具体含义：
- `num_pages`: 页（block）的数量
- `2`: K 和 V 的维度，索引 0 是 K cache，索引 1 是 V cache
- `num_kv_heads`: KV 头的数量
- `page_size`: 每页的 token 数量（`kernel_tokens_per_block`）
- `head_dim`: 每个头的维度

**关键代码位置**：
- [kv_cache_write_op.py:49](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_impl/kv_cache_write_op.py#L49) - 写入操作
- [base_attention_test.py:264](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_impl/test/base_attention_test.py#L264) - 测试用例

```python
# 从 kv_cache_write_op.py
k_cache = kv_cache.kv_cache_base[:, 0, :, :, :]  # [num_pages, num_kv_heads, page_size, head_dim]
v_cache = kv_cache.kv_cache_base[:, 1, :, :, :]  # [num_pages, num_kv_heads, page_size, head_dim]
```

## 2. MLA (Multi-Latent Attention) KV Cache 排布

**3D 张量格式**：`[kernel_block_num, kernel_seq_size_per_block, kv_lora_rank + rope_head_dim]`

这是压缩的 KV cache 布局，用于 DeepSeek V3/R1 等模型：
- `kernel_block_num`: kernel 块数量
- `kernel_seq_size_per_block`: 每块的序列长度
- `kv_lora_rank + rope_head_dim`: 压缩后的 KV 维度（包含 LoRA rank 和 RoPE 维度）

**关键代码位置**：
- [OpDefs.h:81-82](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/bindings/OpDefs.h#L81) - C++ 绑定定义
- [flashinfer_mla.py:482](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/flashinfer_mla.py#L482) - MLA 实现

```python
# 从 flashinfer_mla.py
compressed_kv = kv_cache.kv_cache_base.view(
    -1, self.token_per_block, self.kv_lora_rank + self.qk_rope_head_dim
)
```

## 3. 底层存储格式

在底层，KV cache 以 **2D 打包格式**存储：
- `[block_num, kv_block_stride_elems]`
- 其中 `kv_block_stride_elems = 2 * num_kv_heads * tokens_per_block * head_dim`（MHA）

**reshape 逻辑**在 [OpDefs.h:82-88](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/bindings/OpDefs.h#L82)：

```cpp
// MHA layout: [kernel_block_num, 2, num_kv_heads, kernel_seq_size_per_block, head_dim]
layer_cache.kv_cache_base = base.reshape({kernel_block_num,
                                          2,
                                          (int64_t)num_kv_heads,
                                          (int64_t)kernel_seq_size_per_block,
                                          (int64_t)head_dim});
```

## 4. 不同 Backend 的 Layout 使用

| Backend | Layout | 用途 |
|---------|--------|------|
| FlashInfer | HND | Prefill & Decode |
| TRT-LLM | HND | Prefill & Decode |
| FlashInfer MLA | NHD | MLA Prefill |
| Aiter (ROCm) | HND/NHD | 可配置 |

## 5. 关键类和数据结构

- `LayerKVCache`: 单层 KV cache 视图，包含 `kv_cache_base` 和 `kv_scale_base`（用于 FP8）
- `KVCache`: 整个模型的 KV cache，包含所有层的 cache
- `PyAttentionInputs`: 注意力计算的输入参数，包含 block id、sequence lengths 等

## User
> `c:\WorkingSpace\Code\rtp-llm\rtp_llm\models_py\modules\factory\attention\cuda_impl\kv_cache_write_op.py#L71-71` NHD和HND的区别

## Assistant
## NHD 与 HND 的区别

这是 FlashInfer 定义的两种 KV cache 内存排布格式，区别在于 **head 维度和 token 维度的顺序**不同：

### HND (Head-Number-Dimension)

```
[num_pages, num_kv_heads, page_size, head_dim]
```

- 每个 page 内，**head 维度在外层**，token 维度在内层
- 同一个 head 的所有 token 在内存中连续存储
- 内存视图：`page[head][token][dim]`

```
Page 0:
  Head 0: [token_0, token_1, ..., token_{page_size-1}]  ← 连续
  Head 1: [token_0, token_1, ..., token_{page_size-1}]  ← 连续
  ...
```

### NHD (Number-Head-Dimension)

```
[num_pages, page_size, num_kv_heads, head_dim]
```

- 每个 page 内，**token 维度在外层**，head 维度在内层
- 同一个 token 的所有 head 在内存中连续存储
- 内存视图：`page[token][head][dim]`

```
Page 0:
  Token 0: [head_0, head_1, ..., head_{num_kv_heads-1}]  ← 连续
  Token 1: [head_0, head_1, ..., head_{num_kv_heads-1}]  ← 连续
  ...
```

### 在 rtp-llm 中的使用情况

| 场景 | Layout | 代码位置 |
|------|--------|----------|
| MHA Prefill (FlashInfer) | **HND** | [py_flashinfer_mha.py:94](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_impl/py_flashinfer_mha.py#L94) |
| MHA Decode (FlashInfer) | **HND** | [py_flashinfer_mha.py:646](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_impl/py_flashinfer_mha.py#L646) |
| MHA KV Cache Write | **HND** | [kv_cache_write_op.py:71](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_impl/kv_cache_write_op.py#L71) |
| MLA Prefill (FlashInfer) | **NHD** | [flashinfer_mla.py:206](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/flashinfer_mla.py#L206) |
| 参考实现 (attention_ref) | **NHD** | [attention_ref.py:75](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_impl/test/attention_ref.py#L75) |
| XQA Decode | **HND** | [xqa.py:286](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_impl/xqa.py#L286) |

### 性能影响

- **HND** 更适合 attention 计算：同一个 head 的所有 token 连续存储，GPU 读取时 coalescing 更好，对 GEMM/attention kernel 更友好
- **NHD** 更适合 token 级别的操作（如写入单个 token 的 KV）：写入一个 token 的所有 head 数据是连续的

rtp-llm 中 MHA 场景统一使用 **HND**，因为 attention kernel 是计算瓶颈，HND 排布对 kernel 性能更优。MLA 场景使用 **NHD** 是因为 MLA 的 compressed KV 维度特殊（`kv_lora_rank + rope_head_dim`），其 attention 实现路径不同。

## User
> scheduler是如何进行KV资源的管理的

## Assistant
现在我对 rtp-llm 的 Scheduler 如何管理 KV 资源有了全面的理解。下面是详细的分析：

---

## Scheduler 的 KV 资源管理架构

rtp-llm 的 KV Cache 资源管理采用**分层架构**，从上到下依次为：

```
Scheduler → KVCacheManager → KVCacheAllocator → KVCacheGroup → BlockPool
                                                        ↘ BlockCache (LRU)
                                                        ↘ BlockRefCounter
```

### 1. 顶层：FIFOScheduler — 请求调度与准入控制

[FIFOScheduler.h](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h) 是核心调度器，管理三个请求队列：

| 队列 | 状态 | 含义 |
|------|------|------|
| `waiting_streams_` | WAITING | 等待 KV 资源分配 |
| `loading_cache_streams_` | LOADING_CACHE | 正在从远端/内存加载 KV cache |
| `running_streams_` | RUNNING | 正在执行推理 |

**准入控制逻辑**（[FIFOScheduler.cc:122](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.cc#L122)）：

```cpp
bool FIFOScheduler::checkInputLength(const GenerateStreamPtr& stream) {
    // 条件1: 输入长度不超过 KV Cache 最大可用 token 数
    if (stream->inputLength() > cache_manager_->maxAvailableTokensNum()) {
        // 拒绝请求
    }
    // 条件2: input_len * batch_size 不超过 max_batch_tokens_size
    else if (stream->inputLength() * stream->currentBatchSize() > max_batch_tokens_size_) {
        // 拒绝请求
    }
}
```

**内存评估**（[FIFOScheduler.cc:137](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.cc#L137)）：`evaluateRunningMemory` 判断新请求能否加入当前批次：
- 运行中流数 + 新流数 ≤ `max_generate_batch_size`
- `max_token_size * (streams.size() + 1) + running_streams_.size() < max_batch_tokens_size_`

### 2. 中层：KVCacheManager — KV Cache 生命周期管理

[KVCacheManager.h](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/cache/KVCacheManager.h) 是 KV 资源管理的核心入口，提供三个关键操作：

```
malloc(MallocInfo) → 分配 KV cache blocks
free(FreeInfo)     → 释放 KV cache blocks
insertIntoCache()  → 将 block 插入 LRU 缓存（用于 prefix caching）
```

**分配流程**（[KVCacheManager.cc](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/cache/KVCacheManager.cc)）：

```
KVCacheManager::malloc()
  ├── 计算/更新 cache keys（基于 token 序列的哈希）
  └── allocator_->malloc()
        ├── 首次分配 (curBlocksNum == 0) → initMalloc()
        │     ├── initMallocForCommonLen()  ← Prefix Cache 匹配 + 公共序列分配
        │     └── incrMalloc()             ← 增量分配（不同 batch 不同长度）
        └── 增量分配 (curBlocksNum > 0) → incrMalloc()
```

### 3. 分配器层：KVCacheAllocator — 两种策略

根据模型类型选择不同的分配器（[KVCacheManager.cc:68](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/cache/KVCacheManager.cc#L68)）：

| 分配器 | 适用场景 | 特点 |
|--------|----------|------|
| `SingleTypeKVCacheAllocator` | 纯 Full Attention 模型 | 单一 `FullKVCacheGroup` |
| `HybridTypeKVCacheAllocator` | 混合 Full+Linear Attention | 多个 `KVCacheGroup`（Full + Linear） |

### 4. 缓存组层：KVCacheGroup — 实际的 Block 分配与回收

#### FullKVCacheGroup（[FullKVCacheGroup.cc](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/cache/FullKVCacheGroup.cc)）

用于标准 MHA/MLA 注意力：

- **malloc**: 计算所需 block 数 → 从 BlockPool 分配 → 追加到 block_ids
- **match**: 逐 key 查找 BlockCache（LRU），返回可复用的 block 列表
- **free**: 调用 `block_pool_->requestFree()` 释放 block
- **reference**: 引用已存在的 block（用于 prefix cache 复用和 batch 内共享）

```cpp
// 需要的 block 数计算
int needBlocksNum = (seq_len + reserve_step + seq_size_per_block - 1) / seq_size_per_block - current_blocks;
```

#### LinearKVCacheGroup（[LinearKVCacheGroup.cc](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/cache/LinearKVCacheGroup.cc)）

用于线性注意力（如 Mamba/RetNet），采用**稀疏缓存策略**：

- 只保留 `linear_step` 整数倍位置的 block 和尾部 block
- 中间的 block 设为 `NULL_BLOCK_IDX`（不分配实际显存）
- `removeSkippedBlocks()`: 释放不再需要的中间 block

### 5. 底层：BlockPool — 物理显存管理与引用计数

[BlockPool.h](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/cache/BlockPool.h) 是物理 block 的管理者：

**初始化**：
```
BlockPool::init()
  ├── validateConfig()
  ├── initializeCacheBuffer()    ← 分配一大块连续 GPU 显存
  ├── initializeLayerMappings()  ← 建立全局层 → 本地层映射
  ├── initializeLayoutStrategies() ← 按 MemoryLayout 切分显存
  └── initFreeBlocks()           ← block 0 保留，其余加入 free_block_ids_
```

**分配**（[BlockPool.cc:215](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/cache/BlockPool.cc#L215)）：
```cpp
BlockIndicesType BlockPool::malloc(int num_blocks) {
    // 从 free_block_ids_ 中取出 num_blocks 个 block
    // 增加 request_ref_counter
    // 增加 req_con_ref_counter
    // 增加 req_cache_ref_counter
}
```

**引用计数系统**（四重引用计数）：

| 引用计数器 | 含义 |
|------------|------|
| `request_ref_counter_` | 请求持有引用（正在推理的请求） |
| `connector_ref_counter_` | 连接器持有引用（PD 分离场景跨节点传输） |
| `block_cache_ref_counter_` | BlockCache 持有引用（prefix cache） |
| `req_con_ref_counter_` | request + connector 联合计数 |

**释放条件**（[BlockPool.cc:293](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/cache/BlockPool.cc#L293)）：
```cpp
void BlockPool::tryFreeBlocks(const BlockIndicesType& block_ids) {
    for (const auto& block_id : block_ids) {
        // 只有当 request 和 block_cache 引用都为 0 时才真正释放
        if (req_con_ref_counter_.getRefCounter(block_id) == 0
            && block_cache_ref_counter_.getRefCounter(block_id) == 0) {
            free_block_ids_.insert(block_id);
        }
    }
}
```

### 6. Prefix Cache：BlockCache（LRU 缓存）

[BlockCache.h](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/cache/BlockCache.h) 实现了基于 LRU 的 prefix cache：

- **Key**: `(cache_key, group_id)` — cache_key 由 token 序列哈希得到
- **Value**: `CacheItem { cache_key, group_id, block_index, is_resident }`
- **匹配**: 逐 block 查找，返回最长前缀匹配的 block 列表
- **淘汰**: `selectAndEvict(min_blocks)` 淘汰 LRU 条目直到释放足够 block

### 7. 完整的 KV 资源分配流程

```
请求到达 → FIFOScheduler.enqueue()
  │
  ├─ checkInputLength() → 检查 KV cache 容量是否足够
  │
  ├─ schedule() → evaluateRunningMemory() → 判断能否加入当前批次
  │
  └─ 请求进入 running 状态
       │
       ├─ KVCacheManager.malloc()
       │    ├─ 计算 cache_keys（token 哈希）
       │    ├─ initMallocForCommonLen()
       │    │    ├─ BlockCache.match() → 查找可复用的 prefix blocks
       │    │    ├─ FullKVCacheGroup.reference() → 引用复用 blocks
       │    │    └─ FullKVCacheGroup.malloc() → 分配新 blocks
       │    └─ incrMalloc() → 为不同长度的 batch 增量分配
       │
       ├─ 推理执行（使用分配的 KV cache blocks）
       │
       ├─ KVCacheManager.insertIntoCache() → 将 blocks 插入 LRU 缓存
       │
       └─ KVCacheManager.free() → 请求完成，释放 blocks
            └─ BlockPool.requestFree() → 减引用计数，可能真正释放
```

### 8. 关键设计特点

1. **Paged KV Cache**: 将 KV cache 划分为固定大小的 block（page），按需分配，避免预分配整条序列的显存
2. **Prefix Cache 复用**: 通过 LRU BlockCache，相同前缀的请求可以复用已缓存的 KV blocks，只需增加引用计数
3. **Hybrid Attention 支持**: Full + Linear 混合注意力模型使用不同的缓存策略（Full 保留所有 block，Linear 稀疏保留）
4. **四重引用计数**: 确保 block 在被任何持有者（请求/连接器/缓存）使用时不会被释放
5. **Reserve Blocks**: 预留一定比例的 block，避免运行中的请求因新请求占用资源而无法扩展


## User
> rtp-llm kv_cache初始化、更新、使用流程

## Assistant

现在我已经收集了足够的信息，下面详细总结 rtp-llm 的 KV Cache 初始化、更新、使用流程：

---

## rtp-llm KV Cache 完整流程

### 一、初始化阶段

#### 1.1 显存分配

```
Engine 启动
    │
    └─ KVCacheManager::KVCacheManager()
         │
         ├─ allocateAndSync() → 计算可用显存，确定 block_num
         │
         └─ KVCacheManager::init()
              │
              ├─ 创建 Allocator（根据是否 hybrid 选择）
              │    ├─ SingleTypeKVCacheAllocator（纯 Full Attention）
              │    └─ HybridTypeKVCacheAllocator（Full + Linear 混合）
              │
              └─ Allocator::init()
                   │
                   └─ BlockPool::init()
                        ├─ validateConfig()
                        ├─ initializeCacheBuffer()    ← 分配一大块连续 GPU 显存
                        ├─ initializeLayerMappings()  ← 建立层映射
                        ├─ initializeLayoutStrategies() ← 按 layout 切分显存
                        └─ initFreeBlocks()           ← 初始化空闲 block 列表
```

**BlockPool 初始化**（[BlockPool.cc:176](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/cache/BlockPool.cc#L176)）：

```cpp
bool BlockPool::init() {
    validateConfig();
    initializeCacheBuffer();      // 分配 GPU 显存: torch::empty({total_size_bytes}, device=CUDA)
    initializeLayerMappings();    // 建立 global_layer → local_layer 映射
    initializeLayoutStrategies(); // 按 MemoryLayout 切分显存给各层
    initFreeBlocks();             // block 0 保留，其余加入 free_block_ids_
    return true;
}
```

**显存布局**：
- 底层存储：2D 张量 `[block_num, kv_block_stride_bytes]`
- 按 layer 切分，每层有自己的 view

#### 1.2 传递给 Python 模型

```
GptModel::init() (C++)
    │
    ├─ KVCacheManager::getMainModelCacheLayerLayout()
    │    └─ 返回 CacheLayerLayout {
    │         layers_to_kv_buffer_ptrs:   [layer0_tensor, layer1_tensor, ...]
    │         layers_to_scale_buffer_ptrs: [scale0_tensor, scale1_tensor, ...]
    │         layer_to_groups:             [group_id for each layer]
    │         layer_attn_types:            [FULL/LINEAR for each layer]
    │       }
    │
    └─ PyWrappedModel::PyWrappedModel()
         │
         └─ 构建 KVCache 结构，传递给 Python
              │
              └─ py_model_.attr("initialize")(init_resources)
                   │
                   └─ Python 模型保存 self.kv_cache = KVCache(...)
```

**C++ → Python 数据结构**（[OpDefs.h:34](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/bindings/OpDefs.h#L34)）：

```cpp
struct KVCache {
    std::vector<torch::Tensor> kv_cache_base_by_layer;  // 每层的 KV cache tensor
    std::vector<torch::Tensor> kv_scale_base_by_layer;  // 每层的 scale tensor (INT8/FP8)
    int seq_size_per_block;
    int kernel_seq_size_per_block;
    int num_kv_heads;
    int head_dim;
    bool use_mla;
    int kv_lora_rank;
    int rope_head_dim;
    std::vector<CacheGroupType> layer_attn_types;  // FULL 或 LINEAR
};
```

---

### 二、请求处理阶段

#### 2.1 Block 分配

```
请求到达 → FIFOScheduler::enqueue()
    │
    └─ schedule() → 选择可执行的请求
         │
         └─ StreamCacheResource::malloc()
              │
              ├─ 首次分配 (curBlocksNum == 0)
              │    │
              │    └─ KVCacheManager::malloc()
              │         │
              │         ├─ initCacheKeys() → 计算 cache keys（基于 token 序列哈希）
              │         │
              │         └─ Allocator::malloc() → initMalloc()
              │              │
              │              ├─ initMallocForCommonLen()
              │              │    ├─ BlockCache::match() → 查找可复用的 prefix blocks
              │              │    ├─ KVCacheGroup::reference() → 引用复用 blocks
              │              │    └─ KVCacheGroup::malloc() → 分配新 blocks
              │              │
              │              └─ incrMalloc() → 为不同长度的 batch 增量分配
              │
              └─ 增量分配 (curBlocksNum > 0)
                   │
                   └─ KVCacheManager::malloc()
                        │
                        └─ updateCacheKeys() + Allocator::malloc() → incrMalloc()
```

**Block 分配核心逻辑**（[FullKVCacheGroup.cc:23](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/cache/FullKVCacheGroup.cc#L23)）：

```cpp
bool FullKVCacheGroup::malloc(BlockIds& block_ids, int seq_len, bool enable_reuse_cache, int reserve_step) {
    int need_blocks_num = (seq_len + reserve_step + seq_size_per_block - 1) / seq_size_per_block - current_blocks;
    if (need_blocks_num == 0) return true;
    
    // 确保有足够的空闲 blocks
    if (freeBlocksNum() < need_blocks_num) {
        if (!ensureFreeBlocks(need_blocks_num)) return false;
    }
    
    // 从 BlockPool 分配
    auto result = block_pool_->malloc(need_blocks_num);
    block_ids.add(result);
    return true;
}
```

**BlockPool::malloc**（[BlockPool.cc:215](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/cache/BlockPool.cc#L215)）：

```cpp
BlockIndicesType BlockPool::malloc(int num_blocks) {
    std::scoped_lock lock(ref_mu_, free_mu_);
    if (free_block_ids_.size() < num_blocks) return {};  // 不足
    
    // 从 free_block_ids_ 取出 blocks
    auto first = free_block_ids_.begin();
    auto last  = std::next(first, num_blocks);
    block_ids.assign(first, last);
    free_block_ids_.erase(first, last);
    
    // 增加引用计数
    request_ref_counter_.incrementRefCounter(block_ids);
    req_con_ref_counter_.incrementRefCounter(block_ids);
    req_cache_ref_counter_.incrementRefCounter(block_ids);
    
    return block_ids;
}
```

---

### 三、推理执行阶段

#### 3.1 获取每层的 KV Cache

```
Model::forward(inputs, fmha_impl)
    │
    └─ for each layer:
         │
         ├─ select_block_map_for_layer(attention_inputs, layer_id)
         │    └─ 为当前层选择正确的 block 映射
         │
         └─ DecoderLayer::forward(hidden_states, fmha_impl, kv_cache)
              │
              └─ CausalAttention::forward(hidden_states, fmha_impl, kv_cache)
                   │
                   └─ kv_cache = self.kv_cache.get_layer_cache(layer_id)
```

**get_layer_cache 实现**（[OpDefs.h:49](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/bindings/OpDefs.h#L49)）：

```cpp
LayerKVCache KVCache::getLayerCache(int idx) {
    LayerKVCache layer_cache;
    auto base = kv_cache_base_by_layer[idx];
    
    // 根据层类型决定 reshape 方式
    if (is_full_attention) {
        if (use_mla) {
            // MLA: [kernel_block_num, kernel_seq_size_per_block, kv_lora_rank + rope_head_dim]
            layer_cache.kv_cache_base = base.reshape({kernel_block_num, kernel_seq_size, kv_lora_rank + rope_head_dim});
        } else {
            // MHA: [kernel_block_num, 2, num_kv_heads, kernel_seq_size_per_block, head_dim]
            layer_cache.kv_cache_base = base.reshape({kernel_block_num, 2, num_kv_heads, kernel_seq_size, head_dim});
        }
    } else {
        // Linear: 保持原始 2D 格式
        layer_cache.kv_cache_base = base;
    }
    return layer_cache;
}
```

#### 3.2 KV Cache 写入

**MHA 写入流程**（[kv_cache_write_op.py:35](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_impl/kv_cache_write_op.py#L35)）：

```python
def forward(self, key: torch.Tensor, value: torch.Tensor, kv_cache: Optional[LayerKVCache]) -> None:
    if kv_cache is not None:
        k_cache = kv_cache.kv_cache_base[:, 0, :, :, :]  # [num_pages, num_kv_heads, page_size, head_dim]
        v_cache = kv_cache.kv_cache_base[:, 1, :, :, :]
        
        # 使用 FlashInfer 的 append_paged_kv_cache 写入
        page.append_paged_kv_cache(
            key, value,
            params.batch_indice_d,
            params.positions_d,
            (k_cache, v_cache),
            params.page_indice_d,
            params.decode_page_indptr_d,
            params.paged_kv_last_page_len_d,
            "HND",  # kv_layout
        )
```

**MLA 写入流程**（[mla_kv_cache_write_op.py:35](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/mla_kv_cache_write_op.py#L35)）：

```python
def forward(self, append_ckv_t, key_pe, kv_cache, fmha_params, total_global_ids=None) -> None:
    if kv_cache is not None:
        compute_ops.concat_and_cache_mla(
            append_ckv_t,    # [num_tokens, kv_lora_rank]
            key_pe,          # [num_tokens, rope_head_dim]
            kv_cache.kv_cache_base,
            fmha_params.slot_mapping,
            self.kv_cache_type,
            self.scale,
        )
```

#### 3.3 Attention 计算（读取 KV Cache）

**MHA Prefill**（[py_flashinfer_mha.py:212](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_impl/py_flashinfer_mha.py#L212)）：

```python
def forward(self, q: torch.Tensor, kv_cache: Optional[LayerKVCache]) -> torch.Tensor:
    paged_kv_cache = kv_cache.kv_cache_base
    # FlashInfer prefill wrapper 直接使用 paged KV cache
    return self.prefill_wrapper.run(q, paged_kv_cache)
```

**MHA Decode**（[py_flashinfer_mha.py:688](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_impl/py_flashinfer_mha.py#L688)）：

```python
def forward(self, q: torch.Tensor, kv_cache: Optional[LayerKVCache], params) -> torch.Tensor:
    paged_kv_cache = kv_cache.kv_cache_base
    return self.decode_wrapper.run(q, paged_kv_cache)
```

**MLA Decode**（[flashinfer_mla.py:471](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/flashinfer_mla.py#L471)）：

```python
def forward(self, q_nope, q_pe, kv_cache, layer_id) -> torch.Tensor:
    # reshape 为 MLA 格式: [kernel_block_num, kernel_seq_size, kv_lora_rank + rope_head_dim]
    compressed_kv = kv_cache.kv_cache_base.view(-1, self.token_per_block, self.kv_lora_rank + self.qk_rope_head_dim)
    
    # 分离 compressed KV 和 position encoding
    compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    
    # 使用 MLA wrapper 计算 attention
    self.mla_wrapper.run(q_nope, q_pe, compressed_kv, k_pe, attn_output)
```

---

### 四、请求完成阶段

#### 4.1 插入 Prefix Cache（可选）

```
StreamCacheResource::release()
    │
    ├─ insertIntoCache()（如果启用 prefix cache）
    │    │
    │    └─ KVCacheManager::insertIntoCache()
    │         │
    │         └─ Allocator::insertIntoCache()
    │              │
    │              └─ KVCacheGroup::insertIntoCache()
    │                   │
    │                   └─ BlockCache::put() → 插入 LRU 缓存
    │                        └─ block_pool_->blockCacheReference() → 增加引用计数
    │
    └─ free() → 释放 blocks
```

#### 4.2 释放 Blocks

```
KVCacheManager::free(free_info)
    │
    └─ Allocator::free(free_info)
         │
         └─ KVCacheGroup::free(block_indices)
              │
              └─ BlockPool::requestFree(block_indices)
                   │
                   └─ tryFreeBlocks()
                        │
                        └─ 只有当 req_con_ref_counter 和 block_cache_ref_counter 都为 0 时
                           才真正将 block 放回 free_block_ids_
```

**释放逻辑**（[BlockPool.cc:293](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/cache/BlockPool.cc#L293)）：

```cpp
void BlockPool::tryFreeBlocks(const BlockIndicesType& block_ids) {
    for (const auto& block_id : block_ids) {
        // 只有当 request 和 block_cache 引用都为 0 时才真正释放
        if (req_con_ref_counter_.getRefCounter(block_id) == 0
            && block_cache_ref_counter_.getRefCounter(block_id) == 0) {
            free_block_ids_.insert(block_id);
        }
    }
}
```

---

### 五、完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              初始化阶段                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Engine 启动                                                                 │
│      │                                                                       │
│      ├─ KVCacheManager::init()                                              │
│      │      └─ BlockPool::init()                                            │
│      │             └─ 分配 GPU 显存 [block_num, kv_block_stride_bytes]       │
│      │                                                                       │
│      └─ PyWrappedModel::PyWrappedModel()                                    │
│             └─ getMainModelCacheLayerLayout() → 传递给 Python               │
│                    └─ self.kv_cache = KVCache(kv_cache_base_by_layer)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              请求处理阶段                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  请求到达                                                                    │
│      │                                                                       │
│      ├─ FIFOScheduler::schedule()                                           │
│      │                                                                       │
│      └─ StreamCacheResource::malloc()                                       │
│             │                                                                │
│             ├─ 计算 cache_keys (token 哈希)                                 │
│             ├─ BlockCache::match() → prefix cache 复用                      │
│             └─ BlockPool::malloc() → 分配新 blocks                          │
│                    └─ 增加 request_ref_counter                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              推理执行阶段                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Model::forward(inputs)                                                      │
│      │                                                                       │
│      └─ for each layer:                                                      │
│             │                                                                │
│             ├─ kv_cache = self.kv_cache.get_layer_cache(layer_id)           │
│             │      └─ reshape 为 [blocks, 2, heads, seq_per_block, dim]     │
│             │                                                                │
│             ├─ CausalAttention::forward(hidden_states, fmha_impl, kv_cache) │
│             │      │                                                         │
│             │      ├─ qkv = qkv_proj(hidden_states)                         │
│             │      ├─ q, k, v = rope(qkv)                                   │
│             │      ├─ KVCacheWriteOp::forward(k, v, kv_cache)  ← 写入 cache │
│             │      │      └─ FlashInfer append_paged_kv_cache               │
│             │      │                                                         │
│             │      └─ fmha_impl.forward(q, kv_cache)  ← 读取 cache          │
│             │             └─ FlashInfer prefill/decode wrapper              │
│             │                                                                │
│             └─ mlp(hidden_states)                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              请求完成阶段                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  StreamCacheResource::release()                                              │
│      │                                                                       │
│      ├─ insertIntoCache() (可选)                                            │
│      │      └─ BlockCache::put() → 插入 LRU 缓存                            │
│      │             └─ 增加 block_cache_ref_counter                          │
│      │                                                                       │
│      └─ free()                                                               │
│             └─ BlockPool::requestFree()                                     │
│                    └─ 减少 request_ref_counter                              │
│                    └─ tryFreeBlocks() → 如果所有引用为 0，放回 free_list    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 六、关键数据结构总结

| 结构 | 位置 | 作用 |
|------|------|------|
| `BlockPool` | C++ | 物理显存管理，维护空闲 block 列表和引用计数 |
| `BatchKVCacheResource` | C++ | 每个 batch 的 block 分配信息，包含 block_ids 和 cache_keys |
| `KVCache` | C++/Python | 全局 KV cache，包含每层的 tensor view |
| `LayerKVCache` | Python | 单层的 KV cache view，reshape 为正确的 layout |
| `BlockCache` | C++ | LRU 缓存，用于 prefix cache 复用 |
| `BlockRefCounter` | C++ | 四重引用计数，控制 block 生命周期 |

## User
> 为什么好像没有看到类似vLLM里面的slot_mapping、block_table

## Assistant
rtp-llm **确实有** `slot_mapping` 和 `block_table`，只是命名略有不同。让我解释它们的对应关系：

## rtp-llm vs vLLM 概念对应

| vLLM 概念 | rtp-llm 对应 | 说明 |
|-----------|-------------|------|
| `block_table` | `kv_cache_block_id_host` / `page_indice` | 存储 token → 物理 block 的映射 |
| `slot_mapping` | `slot_mapping` | 存储 token → 物理 slot 位置的映射 |

## 具体实现

### 1. Block Table（`page_indice` / `kv_cache_block_id_host`）

**vLLM**：
```python
block_table: [batch_size, max_blocks]  # 每个 seq 使用的物理 block id 列表
```

**rtp-llm**：
- `kv_cache_block_id_host`: `[batch_size, max_blocks]` — 2D 形式，与 vLLM 相同
- `page_indice`: 展平后的 1D 形式，配合 `page_indptr` 使用

**生成逻辑**（[CudaFlashInfer.cc:226](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/cuda/ops/CudaFlashInfer.cc#L226)）：

```cpp
// 从 kv_cache_block_id_host (2D) 展平为 page_indice (1D)
for (int i = 0; i < batch_size; i++) {
    int page_num = (seq_len + tokens_per_block - 1) / tokens_per_block;
    for (int j = 0; j < page_num; j++) {
        auto page_idx = kv_cache_block_id[i * max_batch_blocks + j];
        page_indice[total_page_idx++] = page_idx;
    }
    page_indptr[i + 1] = total_page_idx;  // 类似 CSR 格式的 indptr
}
```

### 2. Slot Mapping

**vLLM**：
```python
slot_mapping = block_number * block_size + block_offset
# 每个 token 在 KV cache 中的物理位置
```

**rtp-llm**：完全相同的计算方式

**生成逻辑**（[FlashInferMlaParams.cc:480](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/bindings/cuda/FlashInferMlaParams.cc#L480)）：

```cpp
for (int64_t i = 0; i < input_token_num; ++i) {
    const int32_t batch_id     = batch_indice_ptr[i];
    const int32_t position     = positions_ptr[i];
    const int32_t block_index  = position / seq_size_per_block;      // 逻辑 block 索引
    const int32_t block_offset = position % seq_size_per_block;      // block 内偏移
    const int32_t block_number = block_table_ptr[batch_id * max_blocks + block_index];  // 物理 block id
    slot_mapping_ptr[i] = static_cast<int64_t>(block_number) * seq_size_per_block + block_offset;
}
```

### 3. 使用场景

**MHA 写入 KV Cache**（[kv_cache_write_op.py:68](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_impl/kv_cache_write_op.py#L68)）：

```python
page.append_paged_kv_cache(
    key, value,
    params.batch_indice_d,    # 每个 token 属于哪个 batch
    params.positions_d,       # 每个 token 在序列中的位置
    (k_cache, v_cache),
    params.page_indice_d,     # block_table (展平形式)
    params.decode_page_indptr_d,
    params.paged_kv_last_page_len_d,
    "HND",
)
```

**MLA 写入 KV Cache**（[mla_kv_cache_write_op.py:43](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/mla_kv_cache_write_op.py#L43)）：

```python
compute_ops.concat_and_cache_mla(
    append_ckv_t,
    key_pe,
    kv_cache.kv_cache_base,
    fmha_params.slot_mapping,  # 直接使用 slot_mapping
    self.kv_cache_type,
    self.scale,
)
```

## 为什么感觉"没看到"？

1. **命名差异**：rtp-llm 使用 `page_indice` 而非 `block_table`，使用 `kv_cache_block_id_host` 作为 2D 形式

2. **封装层级**：rtp-llm 将这些参数封装在 `FlashInferAttnParams` / `FlashInferMlaAttnParams` 类中，通过 `fmha_params.page_indice_d`、`fmha_params.slot_mapping` 访问

3. **FlashInfer 接口**：rtp-llm 使用 FlashInfer 作为底层 attention kernel，FlashInfer 使用 CSR 格式（`page_indice` + `page_indptr`）而非 vLLM 的 2D `block_table`

## 完整参数对比

| 参数 | vLLM | rtp-llm | 用途 |
|------|------|---------|------|
| Block table (2D) | `block_tables` | `kv_cache_block_id_host` | 每个 seq 使用的物理 block id |
| Block table (1D) | - | `page_indice_d` | 展平的 block table |
| Block indptr | - | `decode_page_indptr_d` | CSR 格式的 indptr |
| Slot mapping | `slot_mapping` | `slot_mapping` | token → 物理 slot 位置 |
| Positions | `positions` | `positions_d` | 每个 token 在序列中的位置 |
| Batch indices | - | `batch_indice_d` | 每个 token 属于哪个 batch |

## User
> MHA如果想支持NHD格式的KV排布，需要进行哪些配置

## Assistant
根据代码分析，rtp-llm 中 **MHA 默认使用 HND layout**，NHD layout 主要用于 MLA 和 ragged prefill 场景。如果要让 MHA 支持 NHD 格式，需要进行以下修改：

## 当前状态

| 场景 | Layout | 代码位置 |
|------|--------|----------|
| MHA Paged Prefill | **HND** | [py_flashinfer_mha.py:94](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_impl/py_flashinfer_mha.py#L94) |
| MHA Paged Decode | **HND** | [py_flashinfer_mha.py:646](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_impl/py_flashinfer_mha.py#L646) |
| MHA KV Cache Write | **HND** | [kv_cache_write_op.py:71](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_impl/kv_cache_write_op.py#L71) |
| MLA Prefill | **NHD** | [flashinfer_mla.py:206](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/flashinfer_mla.py#L206) |
| Ragged Prefill (测试) | **NHD** | [attention_ref.py:75](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_impl/test/attention_ref.py#L75) |

## 支持 NHD 需要修改的地方

### 1. FlashInfer Wrapper 初始化

**当前代码**（[py_flashinfer_mha.py:94](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_impl/py_flashinfer_mha.py#L94)）：
```python
self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
    self.g_workspace_buffer,
    "HND",  # ← 硬编码为 HND
    backend=backend,
)
```

**需要修改为**：
```python
self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
    self.g_workspace_buffer,
    "NHD",  # ← 改为 NHD
    backend=backend,
)
```

### 2. KV Cache Write 操作

**当前代码**（[kv_cache_write_op.py:71](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_impl/kv_cache_write_op.py#L71)）：
```python
page.append_paged_kv_cache(
    key, value,
    self.params.batch_indice_d,
    self.params.positions_d,
    (k_cache, v_cache),
    self.params.page_indice_d,
    self.params.decode_page_indptr_d,
    self.params.paged_kv_last_page_len_d,
    "HND",  # ← 硬编码为 HND
)
```

**需要修改为**：
```python
page.append_paged_kv_cache(
    key, value,
    self.params.batch_indice_d,
    self.params.positions_d,
    (k_cache, v_cache),
    self.params.page_indice_d,
    self.params.decode_page_indptr_d,
    self.params.paged_kv_last_page_len_d,
    "NHD",  # ← 改为 NHD
)
```

### 3. KV Cache Tensor Shape

**HND 格式**：
```python
# [num_pages, 2, num_kv_heads, page_size, head_dim]
k_cache = kv_cache.kv_cache_base[:, 0, :, :, :]  # [num_pages, num_kv_heads, page_size, head_dim]
v_cache = kv_cache.kv_cache_base[:, 1, :, :, :]
```

**NHD 格式**：
```python
# [num_pages, 2, page_size, num_kv_heads, head_dim]
k_cache = kv_cache.kv_cache_base[:, 0, :, :, :]  # [num_pages, page_size, num_kv_heads, head_dim]
v_cache = kv_cache.kv_cache_base[:, 1, :, :, :]
```

### 4. C++ 层 KV Cache Reshape

**当前代码**（[OpDefs.h:86](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/bindings/OpDefs.h#L86)）：
```cpp
// MHA layout: [kernel_block_num, 2, num_kv_heads, kernel_seq_size_per_block, head_dim]
layer_cache.kv_cache_base = base.reshape({kernel_block_num,
                                          2,
                                          (int64_t)num_kv_heads,
                                          (int64_t)kernel_seq_size_per_block,
                                          (int64_t)head_dim});
```

**需要修改为**：
```cpp
// NHD layout: [kernel_block_num, 2, kernel_seq_size_per_block, num_kv_heads, head_dim]
layer_cache.kv_cache_base = base.reshape({kernel_block_num,
                                          2,
                                          (int64_t)kernel_seq_size_per_block,
                                          (int64_t)num_kv_heads,
                                          (int64_t)head_dim});
```

### 5. 配置化方案（推荐）

如果想要通过配置切换 layout，可以：

**1. 添加配置项**（在 `AttentionConfigs` 或 `HWKernelConfig` 中）：
```python
@dataclass
class AttentionConfigs:
    # ... existing fields ...
    kv_layout: str = "HND"  # "HND" or "NHD"
```

**2. 修改相关代码使用配置**：
```python
# py_flashinfer_mha.py
self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
    self.g_workspace_buffer,
    attn_configs.kv_layout,  # 使用配置
    backend=backend,
)

# kv_cache_write_op.py
page.append_paged_kv_cache(
    # ...
    self.kv_layout,  # 使用配置
)
```

## 注意事项

1. **FlashInfer 支持**：FlashInfer 同时支持 HND 和 NHD layout，只需在创建 wrapper 时指定即可

2. **性能差异**：
   - **HND**：同一个 head 的所有 token 连续存储，对 attention kernel 更友好
   - **NHD**：同一个 token 的所有 head 连续存储，对 token 级别操作更友好

3. **MLA 场景**：MLA 使用 NHD 是因为其 compressed KV 维度特殊（`kv_lora_rank + rope_head_dim`），与标准 MHA 不同

4. **Decode 场景**：Decode 时每个 step 只处理 1 个 token，NHD layout 可能更高效（所有 head 数据连续）

## User
> Layout修改之后，slot_mapping和block_table的计算是否需要修改

## Assistant
**不需要修改**。`slot_mapping` 和 `block_table` 的计算与 KV Cache 的 Layout（HND vs NHD）无关。

## 原因分析

### 1. slot_mapping 计算逻辑

```python
slot_mapping = block_number * block_size + block_offset
```

这个公式计算的是**线性 slot 索引**，表示每个 token 在整个 KV cache 中的位置。无论 HND 还是 NHD：

- 每个 block 仍然包含 `block_size` 个 token
- `block_number` 是物理 block ID
- `block_offset` 是 token 在 block 内的偏移

**Layout 只影响 slot 内部的数据排布，不影响 slot 的索引计算。**

### 2. block_table 计算逻辑

```python
# block_table 存储每个序列使用的物理 block ID 列表
block_table[batch_id] = [block_id_0, block_id_1, ...]
```

这完全与 layout 无关，只是记录逻辑序列到物理 block 的映射。

### 3. Layout 的作用范围

| 概念 | 含义 | 与 Layout 关系 |
|------|------|----------------|
| `block_table` | 序列 → 物理 block 映射 | **无关** |
| `slot_mapping` | token → 线性 slot 索引 | **无关** |
| `slot` 内部数据排布 | 一个 token 的 KV 数据如何存储 | **相关** |

**HND vs NHD 的区别**在于同一个 slot 内部的数据排布：

```
HND: slot 数据 = [num_kv_heads, head_dim]  ← head 维度在外
NHD: slot 数据 = [num_kv_heads, head_dim]  ← head 维度在内（但内存中连续的是 head 维度）
```

## 验证：FlashInfer 的处理方式

FlashInfer 在创建 wrapper 时指定 layout，内部自动处理不同的访问模式：

```python
# HND layout
wrapper_hnd = BatchPrefillWithPagedKVCacheWrapper(workspace, "HND")
# 内部访问: kv_cache[block_id, head_id, offset, dim]

# NHD layout  
wrapper_nhd = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
# 内部访问: kv_cache[block_id, offset, head_id, dim]
```

两种情况下，`page_indice`（block_table）和 `slot_mapping` 的含义完全相同，只是 kernel 内部访问内存的索引计算不同。

## 结论

| 修改项 | 是否需要修改 |
|--------|-------------|
| `slot_mapping` 计算 | ❌ 不需要 |
| `block_table` / `page_indice` 计算 | ❌ 不需要 |
| `kv_cache_base` tensor shape | ✅ 需要 |
| FlashInfer wrapper 初始化的 layout 参数 | ✅ 需要 |
| `append_paged_kv_cache` 的 layout 参数 | ✅ 需要 |

**核心原理**：Layout 只影响物理内存排布，不影响逻辑索引。FlashInfer kernel 内部会根据指定的 layout 参数，使用不同的索引公式访问内存，但输入的 `page_indice` 和 `slot_mapping` 含义保持不变。

## User
> 当前kv_cache是放在一个Tensor内的，要接入的新的Attention算子，要求k_cache和v_cache单独存放，适配需要做哪些修改

## Assistant
根据代码分析，当前 rtp-llm 的 KV cache 存储方式是 **K 和 V 合并在一个 Tensor 中**：

```
底层存储: [block_num, kv_block_stride_bytes]
MHA reshape: [kernel_block_num, 2, num_kv_heads, kernel_seq_size_per_block, head_dim]
                                                    ↑ dim=1 表示 K/V 合并
```

如果要适配要求 **k_cache 和 v_cache 分开存储** 的新 Attention 算子，需要修改以下内容：

---

## 一、C++ 层修改

### 1. CacheConfig 配置结构

**文件**: [CacheConfig.h](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/cache/CacheConfig.h)

```cpp
struct CacheConfig {
    // 新增：是否分离 K/V 存储
    bool separate_kv_cache = false;
    
    // 现有字段
    size_t kv_block_stride_bytes = 0;  // 合并存储时使用
    size_t k_block_stride_bytes = 0;   // 新增：分离存储时的 K stride
    size_t v_block_stride_bytes = 0;   // 新增：分离存储时的 V stride
};
```

### 2. BufferTypes 数据结构

**文件**: [BufferTypes.h](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/cache/BufferTypes.h)

```cpp
struct CacheLayerLayout {
    // 现有字段（合并存储）
    std::vector<torch::Tensor> layers_to_kv_buffer_ptrs;
    
    // 新增字段（分离存储）
    std::vector<torch::Tensor> layers_to_k_buffer_ptrs;   // K cache per layer
    std::vector<torch::Tensor> layers_to_v_buffer_ptrs;   // V cache per layer
};

struct KVCacheBuffer {
    // 现有字段
    torch::Tensor kv_blocks;
    
    // 新增字段
    torch::Tensor k_blocks;  // 分离的 K cache
    torch::Tensor v_blocks;  // 分离的 V cache
};
```

### 3. BlockPool 内存分配

**文件**: [BlockPool.h](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/cache/BlockPool.h) / [BlockPool.cc](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/cache/BlockPool.cc)

```cpp
class BlockPool {
private:
    // 现有
    torch::Tensor cache_aligned_buffer_;  // 合并的 KV buffer
    
    // 新增
    torch::Tensor k_cache_buffer_;  // 分离的 K buffer
    torch::Tensor v_cache_buffer_;  // 分离的 V buffer
    bool separate_kv_cache_ = false;

public:
    void initializeCacheBuffer() {
        if (separate_kv_cache_) {
            // 分离分配
            k_cache_buffer_ = torch::empty({block_num, k_block_stride_bytes}, ...);
            v_cache_buffer_ = torch::empty({block_num, v_block_stride_bytes}, ...);
        } else {
            // 合并分配（现有逻辑）
            cache_aligned_buffer_ = torch::empty({block_num, kv_block_stride_bytes}, ...);
        }
    }
};
```

### 4. KVCache / LayerKVCache 结构

**文件**: [OpDefs.h](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/bindings/OpDefs.h)

```cpp
struct KVCache {
    // 现有字段（合并存储）
    std::vector<torch::Tensor> kv_cache_base_by_layer;
    
    // 新增字段（分离存储）
    std::vector<torch::Tensor> k_cache_base_by_layer;
    std::vector<torch::Tensor> v_cache_base_by_layer;
    
    bool separate_kv_cache = false;
};

struct LayerKVCache {
    // 现有字段
    torch::Tensor kv_cache_base;
    
    // 新增字段
    torch::Tensor k_cache_base;  // 分离的 K cache
    torch::Tensor v_cache_base;  // 分离的 V cache
};
```

### 5. getLayerCache 方法

**文件**: [OpDefs.h](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/bindings/OpDefs.h#L49)

```cpp
LayerKVCache getLayerCache(int idx) {
    LayerKVCache layer_cache;
    
    if (separate_kv_cache) {
        // 分离存储模式
        layer_cache.k_cache_base = k_cache_base_by_layer[idx].reshape(
            {kernel_block_num, num_kv_heads, kernel_seq_size_per_block, head_dim});
        layer_cache.v_cache_base = v_cache_base_by_layer[idx].reshape(
            {kernel_block_num, num_kv_heads, kernel_seq_size_per_block, head_dim});
    } else {
        // 合并存储模式（现有逻辑）
        layer_cache.kv_cache_base = kv_cache_base_by_layer[idx].reshape(
            {kernel_block_num, 2, num_kv_heads, kernel_seq_size_per_block, head_dim});
    }
    
    return layer_cache;
}
```

---

## 二、Python 层修改

### 1. LayerKVCache 类型定义

**文件**: `rtp_llm/ops/librtp_compute_ops/rtp_llm_ops.pyi`

```python
class LayerKVCache:
    # 现有属性
    @property
    def kv_cache_base(self) -> torch.Tensor: ...
    
    # 新增属性
    @property
    def k_cache_base(self) -> torch.Tensor: ...
    
    @property
    def v_cache_base(self) -> torch.Tensor: ...
```

### 2. Attention 实现适配

**文件**: [py_flashinfer_mha.py](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_impl/py_flashinfer_mha.py)

```python
def forward(self, q: torch.Tensor, kv_cache: Optional[LayerKVCache]) -> torch.Tensor:
    if kv_cache.k_cache_base is not None and kv_cache.v_cache_base is not None:
        # 分离存储模式
        k_cache = kv_cache.k_cache_base  # [num_pages, num_kv_heads, page_size, head_dim]
        v_cache = kv_cache.v_cache_base
        paged_kv_cache = (k_cache, v_cache)
    else:
        # 合并存储模式（现有逻辑）
        paged_kv_cache = kv_cache.kv_cache_base
    
    return self.prefill_wrapper.run(q, paged_kv_cache)
```

### 3. KV Cache Write 操作适配

**文件**: [kv_cache_write_op.py](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/factory/attention/cuda_impl/kv_cache_write_op.py)

```python
def forward(self, key: torch.Tensor, value: torch.Tensor, kv_cache: Optional[LayerKVCache]) -> None:
    if kv_cache.k_cache_base is not None and kv_cache.v_cache_base is not None:
        # 分离存储模式
        k_cache = kv_cache.k_cache_base
        v_cache = kv_cache.v_cache_base
    else:
        # 合并存储模式（现有逻辑）
        k_cache = kv_cache.kv_cache_base[:, 0, :, :, :]
        v_cache = kv_cache.kv_cache_base[:, 1, :, :, :]
    
    page.append_paged_kv_cache(key, value, ..., (k_cache, v_cache), ...)
```

---

## 三、配置化方案（推荐）

### 1. 添加配置项

**文件**: `rtp_llm/cpp/config/ConfigModules.h` 或 `AttentionConfigs`

```cpp
// C++ 配置
struct KVCacheConfig {
    bool separate_kv_cache = false;  // 是否分离 K/V 存储
    // ...
};
```

```python
# Python 配置
@dataclass
class AttentionConfigs:
    # ... existing fields ...
    separate_kv_cache: bool = False  # 是否分离 K/V 存储
```

### 2. 配置传递链路

```
配置文件 / 启动参数
    │
    └─ KVCacheConfig.separate_kv_cache
         │
         ├─ CacheConfig.separate_kv_cache
         │
         ├─ BlockPool::initializeCacheBuffer() → 决定分配方式
         │
         ├─ KVCacheManager::getMainModelCacheLayerLayout() → 决定返回结构
         │
         └─ PyWrappedModel → KVCache 结构 → Python LayerKVCache
```

---

## 四、总结：修改清单

| 层级 | 文件 | 修改内容 |
|------|------|----------|
| **C++ 配置** | `CacheConfig.h` | 添加 `separate_kv_cache`、`k_block_stride_bytes`、`v_block_stride_bytes` |
| **C++ 内存** | `BlockPool.cc` | 支持分配两个独立的 tensor |
| **C++ 数据结构** | `BufferTypes.h` | 添加 `layers_to_k_buffer_ptrs`、`layers_to_v_buffer_ptrs` |
| **C++ 绑定** | `OpDefs.h` | `KVCache` 和 `LayerKVCache` 添加分离字段 |
| **Python 类型** | `rtp_llm_ops.pyi` | `LayerKVCache` 添加 `k_cache_base`、`v_cache_base` |
| **Python Attention** | `py_flashinfer_mha.py` | 适配分离的 k_cache/v_cache |
| **Python Write** | `kv_cache_write_op.py` | 适配分离写入 |

**关键点**：代码中已经有 `k_block_size()` 和 `v_block_size()` 方法（[MHAKVCacheSpec.h:37-42](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/cache/MHAKVCacheSpec.h#L37-42)），说明设计上已经预留了 K/V 分离的能力，主要是需要打通整个链路。

