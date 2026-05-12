# KV Cache 昇腾适配 v0.3 代码审核报告

> 审核目标：对照 `ascend_kv-cache-adaptation_v0.3.md` 计划，审核已完成修改的代码。

---

## 一、审核范围

| 类别 | 文件 | 状态 |
|------|------|------|
| C++ 配置 | `CacheConfig.h` | 已修改 |
| C++ 数据结构 | `BufferTypes.h` | 已修改 |
| C++ 内存管理 | `BlockPool.h` / `BlockPool.cc` | 已修改 |
| C++ 分配器 | `KVCacheAllocator.cc` | 已修改 |
| C++ 管理器 | `KVCacheManager.cc` | 已修改 |
| C++ 配置创建 | `SingleConfigCreator.cc` / `HybridConfigCreator.cc` | 已修改 |
| C++ 绑定 | `OpDefs.h` / `OpDefs.cc` / `PyWrappedModel.h` | 已修改 |
| Python 类型 | `librtp_compute_ops/__init__.pyi` | 已修改 |
| Python Factory | `attention/__init__.py` | 已修改 |
| Python 新增 | `ascend_prefill.py` / `ascend_decode.py` | 新增 |
| Python 新增 | `ascend_kv_cache_write_op.py` | 新增 |
| Python 新增 | `ascend_rope.py` / `ascend_rope_emb.py` | 新增 |
| Python 新增 | `ascend_attn_params.py` | 新增 |

---

## 二、致命问题 (Blocker)

### 🔴 #1 — KVCacheManager.cc:256-305 死代码（重影代码）

**文件**: `rtp_llm/cpp/cache/KVCacheManager.cc:256-305`

**问题描述**:  
`setKVBlockValue` 函数在 line 254 的 `return true;` 闭合大括号之后，lines 256-305 存在一段重复的/残留的合并路径代码，**永远不可能被执行**。且该段代码包含严重 bug：K 和 V 都写入 `dst[0]` 的同一个 block 偏移不同位置，但分离模式下 `dst[0]` 只是 K block 的地址，V 应该写入 dst[1]（如果有的话）。

```cpp
// line 254: return true;
}   // <-- setKVBlockValue 在此结束

// ===== 以下代码永远不可达 =====
    auto dst = allocator_->convertIndexToBuffer(layer_id, block_index);  // line 256
    // ... (重复的 copyFunc 定义)
    if (!copyFunc(k_buffer, dst[0], 0)) {       // line 295
        return false;
    }
    if (!copyFunc(v_buffer, dst[0], expected_k_bytes)) {  // line 299 ← BUG
        return false;
    }
    cudaSyncAndCheck();
    return true;
}
```

**修复建议**: 删除 lines 256-305。

---

### 🔴 #2 — convertIndexToBuffer 在分离模式下返回值不足

**文件**: `rtp_llm/cpp/cache/KVCacheManager.cc:157-160` + `rtp_llm/cpp/cache/MemoryLayoutStrategy.cc:212-230`

**问题描述**:  
分离路径 `setKVBlockValue` 要求 `convertIndexToBuffer` 返回 `>= 2` 个 `BlockInfo`（dst[0]=K, dst[1]=V）:

```cpp
auto dst = allocator_->convertIndexToBuffer(layer_id, block_index);
RTP_LLM_CHECK_WITH_INFO(dst.size() >= 2,
    "convertIndexToBuffer returned fewer than 2 blocks for separate K/V cache");
```

但 `MemoryLayoutStrategy::createBasicBlockInfo()` 在无 scale 且无 partition 时只返回 **1 个** `BlockInfo`（仅 kv）。这会导致 `RTP_LLM_CHECK_WITH_INFO` 失败崩溃。

同样，`KVCacheAllocator::blockBatchCopy` 分离路径中使用 `convertIndexToAddr`（而非 `convertIndexToBuffer`），只获取到 `kv_addr`，无法获取 V buffer 中的独立地址。

**修复建议**:  
两种方案选一：

- **方案 A**: 修改 `MemoryLayoutStrategy::createBasicBlockInfo()`，在分离模式下返回 K/V 两个独立的 `BlockInfo`（分别指向 K buffer 和 V buffer 中对应的 block 地址）。
- **方案 B**: 在 `BlockPool` / `KVCacheManager` 层面新增独立的 `convertIndexToKBuffer` / `convertIndexToVBuffer` 接口。

---

### 🔴 #3 — KVCacheAllocator::blockBatchCopy 分离模式 V copy 地址错误

**文件**: `rtp_llm/cpp/cache/KVCacheAllocator.cc:188-193`

**问题描述**:  
分离模式下 V copy 的源/目标地址计算使用了 `kv_addr + k_block_size_bytes`：

```cpp
k_copy_params.add(dst_addr_info.kv_addr, src_addr_info.kv_addr, k_block_size_bytes, copy_type);
v_copy_params.add(
    static_cast<char*>(dst_addr_info.kv_addr) + k_block_size_bytes,  // ← 错: 仍在K buffer
    static_cast<char*>(src_addr_info.kv_addr) + k_block_size_bytes,  // ← 错: 仍在K buffer
    v_block_size_bytes, copy_type);
```

分离模式下 `kv_addr` 指向的是 **K buffer**。`kv_addr + k_block_size_bytes` 指向的是 **K buffer 中下一个 block 的起始位置**，而非 **V buffer 中对应 V block 的位置**。

**正确逻辑**: V copy 应从 `v_cache_buffer_` 中对应 block 的地址读取/写入，而不是从 K buffer 偏移。

**修复建议**:  
需要 `BlockAddrInfo` 新增 `v_addr` 字段，或新增独立方法返回 V buffer 中 block 的地址。

---

### 🔴 #4 — 配置链路未闭合，`separate_kv_cache` 永远为 false

**文件**: `rtp_llm/cpp/cache/CacheConfig.h:48` / `BlockPool.h:143` / `SingleConfigCreator.cc` / `HybridConfigCreator.cc`

**问题描述**:  

- `CacheConfig::separate_kv_cache` 默认 `false`
- `BlockPool::separate_kv_cache_` 默认 `false`
- `SingleConfigCreator::createSingleConfig()` 和 `HybridConfigCreator::setupPhysicalSizes()` 虽然计算了 `k_block_stride_bytes` / `v_block_stride_bytes`，但**没有将 `separate_kv_cache` 设置为 `true`**
- 没有任何入口点将 `separate_kv_cache` flag 从配置传入 `CacheConfig` 和 `BlockPool`

结果：所有分离存储的代码虽然写了，但永远不会被执行。

**修复建议**:  
在配置链路中增加 `separate_kv_cache` 的设置：

1. 在 `KVCacheConfig`（Python 启动参数配置）中增加 `separate_kv_cache` 选项
2. 在 `CacheConfig` 构建时传递该 flag
3. 在 `BlockPool` 构造函数/init 中从 `CacheConfig` 读取并设置 `separate_kv_cache_`

**数据流**:
```
启动参数/配置 → KVCacheConfig.separate_kv_cache → CacheConfig.separate_kv_cache → BlockPool.separate_kv_cache_
```

---

## 三、严重问题

### 🟡 #5 — blockBatchCopy 分离路径用了 `config_.cache_specs[0]`

**文件**: `rtp_llm/cpp/cache/KVCacheAllocator.cc:173`

**问题描述**:  
```cpp
auto& spec = config_.cache_specs[0];
```
对于 HybridAttention（多个 group/spec 并存），`config_.cache_specs[0]` 可能不是正确的 spec。分离模式下 K/V block size 需要与当前组的 spec 匹配。

**修复建议**: 确认当前场景是否为单一 spec（MHA only）。如果是 HybridAttention + 分离 K/V，需要按 group 分别处理。

---

### 🟡 #6 — ascend_prefill.py 中 tensor 加法缺少设备/dtype 检查

**文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_prefill.py:39`

**问题描述**:  
```python
seq_lens_kv = attn_inputs.prefix_lengths + attn_inputs.input_lengths
```

如果 `prefix_lengths` 和 `input_lengths` 在不同设备上（如一个在 CPU，一个在 NPU），或 dtype 不同（int32 vs int64），直接加法会抛出异常。

**修复建议**:  
```python
seq_lens_kv = attn_inputs.prefix_lengths.to(device=attn_inputs.input_lengths.device) + attn_inputs.input_lengths
```
或确保驱动层传入时已经统一设备和 dtype。

---

### 🟡 #7 — ascend_prefill.py 中 qkv.chunk 的 layout 假设未验证

**文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_prefill.py:50`

**问题描述**:  
```python
q, k, v = qkv.chunk(3, dim=-1)
```
假设 `qkv` 是 `[tokens, num_heads * 3 * head_dim]` 在最后一维拼接。如果模型使用 Q/KV separation（如 GQA，Q 和 KV shape 不同），layout 可能不同。

**修复建议**: 验证目标模型（Qwen3-4B）的 qkv 拼接格式，或增加 layout 检测/配置。

---

## 四、中等问题

### 🟠 #8 — 分离模式下 scale tensor 从 k_cache_buffer_ 分配

**文件**: `rtp_llm/cpp/cache/BlockPool.cc:97`

**问题描述**:  
```cpp
torch::Tensor full_tensor = separate_kv_cache_ ? k_cache_buffer_ : cache_aligned_buffer_;
```
分离模式下，scale tensor（如果存在）也从 `k_cache_buffer_` narrow 出来。这意味着 scale 占用 K buffer 的空间。但 V buffer 中没有对应的 scale 副本。在 `processLayerTensors` 中创建的 V view（line 197-204）假设 V buffer 中对应 offset 位置是 V 数据，如果 scale offset 恰好落在 V buffer 对应的地址范围内，会产生错误。

**修复建议**: 确认 scale 数据是否需要同时在 K 和 V buffer 中分配；或者将 scale 独立分配。当前 MHA 场景大概率无 scale（`kv_scale_stride_bytes == 0`），暂时影响有限。

---

### 🟠 #9 — ascend_kv_cache_write_op.py 接口与计划不一致

**文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_kv_cache_write_op.py`

**问题描述**:  
- 计划文档: `forward(key, value, kv_cache, fmha_params)`
- 实际实现: `set_params(params)` + `forward(key, value, kv_cache)`，slot_mapping 从 `self.params.slot_mapping` 获取

需要确认调用方（Factory 或 AttentionWrapper）使用的是哪种接口模式，确保兼容。

---

### 🟠 #10 — ascend_decode.py 中每次 forward 分配 output tensor

**文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py:41`

**问题描述**:  
```python
output = torch.empty_like(q)
```
Decode 阶段每次 forward 都会在 NPU 上分配新的 output tensor。频繁分配会影响性能。

**修复建议**: 在 `prepare()` 中预分配 output buffer，forward 时复用。

---

## 五、轻微问题 / 建议

### 🔵 #11 — BlockPool::from_blob 创建的 V view 生命周期依赖

**文件**: `rtp_llm/cpp/cache/BlockPool.cc:198-203`

`torch::from_blob` 创建的 V tensor views 不拥有内存，仅引用 `v_cache_buffer_` 中的指针。当前设计正确（v_cache_buffer_ 是 BlockPool 成员变量），但需要注意 `v_cache_buffer_` 释放前所有 pool 层面 view 必须已失效。

---

### 🔵 #12 — ascend_rope.py neox_style 分支效率

**文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_rope.py:37-43`

```python
cos_exp = cos.unsqueeze(-2).unsqueeze(-1)
sin_exp = sin.unsqueeze(-2).unsqueeze(-1)

q_rot = torch.stack([...], dim=-1).flatten(start_dim=-2)
```

`stack + flatten` 组合比直接 `cat` 或 reshape 更慢。另外 `squeeze(-1)` 在对 `cos_exp` 使用时（后面紧接 `[..., 0]` 的索引操作）显得多余。建议简化为与 non-neox 分支一致的 `cat` 操作。

---

### 🔵 #13 — ascend_rope_emb.py warmup 仅支持单 batch

**文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_rope_emb.py:32`

```python
kv_page_indptr = torch.tensor([0, max_num_pages], dtype=torch.int32, device=device)
```

硬编码了单 batch 假设。如果 Ascend 路径不启用 JIT warmup 则无影响；否则多 batch 场景会出错。

---

### 🔵 #14 — `__init__.pyi` 字段顺序与 C++ 不一致

**文件**: `rtp_llm/ops/librtp_compute_ops/__init__.pyi`

.pyi 文件中 `KVCache` 新增字段（`separate_kv_cache`, `k_cache_base_by_layer`, `v_cache_base_by_layer`）放在 `rope_head_dim` 之后。C++ 定义中这些字段在 `layer_attn_types` 之前。顺序不影响运行，但可能造成维护困惑。

---

### 🔵 #15 — BlockPool::where() 分离模式在 buffer 未初始化时可能异常

**文件**: `rtp_llm/cpp/cache/BlockPool.cc:559-560`

```cpp
if (separate_kv_cache_) {
    return k_cache_buffer_.is_privateuseone() ? MemoryType::MEMORY_NPU : MemoryType::MEMORY_CPU;
}
```

如果 `where()` 在 `initializeCacheBuffer()` 之前被调用（如早期内存评估），`k_cache_buffer_` 可能未定义，`.is_privateuseone()` 可能抛异常。

---

## 六、汇总

| 严重级别 | 数量 | Issue # |
|----------|------|---------|
| 🔴 致命 (Blocker) | 4 | #1, #2, #3, #4 |
| 🟡 严重 | 3 | #5, #6, #7 |
| 🟠 中等 | 3 | #8, #9, #10 |
| 🔵 轻微 | 5 | #11~#15 |

### 优先修复次序

```
1. #1  — 删除 KVCacheManager.cc 死代码（line 256-305）
2. #2  — 修复 convertIndexToBuffer 在分离模式下返回 K/V 两个 BlockInfo
3. #3  — 修复 blockBatchCopy 分离模式下 V copy 地址
4. #4  — 闭合 separate_kv_cache 配置链路（入口点设置 flag）
5. #6  — ascend_prefill.py tensor 加法设备/dtype 检查
6. #7  — ascend_prefill.py qkv.chunk layout 验证
7. #5  — 确认 hybrid attention 兼容性
8. #8-#15 — 优化项
```
