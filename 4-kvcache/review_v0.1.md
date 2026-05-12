# Phase 4 KV Cache 昇腾适配代码审核报告 (v0.1)

> 审核范围：基于 `ascend_kv-cache-adaptation_v0.3.md` 计划文档的代码修改
> 审核时间：2026-05-11
> 涉及文件：15 个文件，+334 / -16 行变更

---

## 一、严重问题（P0 — 编译失败或运行时错误）

### 1. `KVCacheManager.cc:256-305` — 函数外悬空代码（编译错误）

`setKVBlockValue(int, int, Tensor, Tensor)` 在第 254 行 `}` 正确关闭后，第 256-305 行存在一段**不属于任何函数的代码**。这是原始函数体的残留副本，位于命名空间作用域内，**无法编译通过**。

**文件**: `rtp_llm/cpp/cache/KVCacheManager.cc`

**现状**:
```cpp
// 第 254 行: setKVBlockValue(int, int, Tensor, Tensor) 函数正确关闭
}

    auto dst = allocator_->convertIndexToBuffer(layer_id, block_index);  // 第 256 行: 悬空代码开始
    RTP_LLM_CHECK_WITH_INFO(
        !dst.empty(), "convertIndexToBuffer returned empty for layer %d, block %d", layer_id, block_index);
    // ... 约 50 行悬空代码 ...
    return true;
}  // 第 305 行: 悬空代码结束

bool KVCacheManager::setKVBlockValue(int block_index, ...  // 第 307 行: 下一个函数
```

**修复**: 删除第 255-305 行的全部悬空代码。

---

### 2. `BlockPool.separate_kv_cache_` 从未被设为 `true`（功能失效）

**文件**: `rtp_llm/cpp/cache/BlockPool.h`, `BlockPool.cc`, `BlockPoolConfigHelper.h`

**问题链**:
- `BlockPoolConfig` 没有 `separate_kv_cache` 字段
- `BlockPool` 构造函数从 `BlockPoolConfig` 获取配置，但该结构体不携带此标志
- `BlockPool.separate_kv_cache_` 默认 `false`，**没有任何代码将其设为 `true`**

**结果**: `BlockPool.cc` 中所有 `if (separate_kv_cache_)` 分支都是死代码，分离分配永远不会执行。

**修复建议**:
1. 在 `BlockPoolConfig` 中新增 `bool separate_kv_cache = false` 字段
2. 在 `BlockPoolConfigHelper::createConfig()` 中从 `CacheConfig.separate_kv_cache` 传递该值
3. 在 `BlockPool` 构造函数中从 `BlockPoolConfig` 读取该值赋给 `separate_kv_cache_`

---

### 3. `CacheConfig.separate_kv_cache` 从未被设为 `true`（配置链断裂）

**文件**: `rtp_llm/cpp/cache/SingleConfigCreator.cc`, `HybridConfigCreator.cc`

**现状**: `SingleConfigCreator` 和 `HybridConfigCreator` 新增了 `k_block_stride_bytes`/`v_block_stride_bytes` 赋值，但 **没有设置 `config.separate_kv_cache = true`**。

**结果**: 整个配置传递链断裂：
```
CacheConfig.separate_kv_cache = false (从未设置)
  → BlockPoolConfig.separate_kv_cache (字段不存在)
    → BlockPool.separate_kv_cache_ = false (永远不会变)
      → 所有分离路径代码不执行
```

**修复建议**: 在 Ascend 设备检测路径中设置 `CacheConfig.separate_kv_cache = true`。可能在以下位置之一：
- `SingleConfigCreator::createSingleConfig()` 中根据编译宏 `#if USING_ASCEND` 设置
- 或者在 `KVCacheManager` 构造时根据 runtime 设备类型设置

---

### 4. `BlockPoolConfigHelper` 池大小计算在分离模式下错误（内存越界）

**文件**: `rtp_llm/cpp/cache/BlockPoolConfigHelper.h:133-134`

**现状**:
```cpp
cfg.kv_block_pool_size_bytes =
    static_cast<size_t>(layer_num) * static_cast<size_t>(cfg.block_num) * cfg.kv_block_stride_bytes;
```

`kv_block_stride_bytes` 是 **K+V 合并步长**（例如 128 * 4 * 128 * 2 = K+V 合计）。

**问题**: 在分离模式下：
- `k_cache_buffer_` 只有 `total_size_bytes / 2` 大小
- `initializeLayoutStrategies()` 将 `k_cache_buffer_` 作为 `full_tensor` 传入 `processMemoryLayout()`
- `processMemoryLayout()` 对 `k_cache_buffer_` 做 narrow 时使用 `kv_block_pool_size_bytes`（基于 K+V 合并大小计算）
- narrow 操作会**超出 k_cache_buffer_ 的边界**，导致崩溃或数据损坏

**修复建议**: 分离模式下：
- `kv_block_pool_size_bytes` 应使用 `k_block_stride_bytes` 计算（而非 `kv_block_stride_bytes`）
- `total_size_bytes` 的计算需要考虑 K/V 分离后的总大小
- `kv_block_stride_bytes` 在分离模式下含义变为 K-only 步长

---

### 5. `KVCacheAllocator.cc:190-192` — V 地址计算错误（数据损坏）

**文件**: `rtp_llm/cpp/cache/KVCacheAllocator.cc:189-193`

**现状**:
```cpp
k_copy_params.add(dst_addr_info.kv_addr, src_addr_info.kv_addr, k_block_size_bytes, copy_type);
v_copy_params.add(
    static_cast<char*>(dst_addr_info.kv_addr) + k_block_size_bytes,   // ← 错误!
    static_cast<char*>(src_addr_info.kv_addr) + k_block_size_bytes,   // ← 错误!
    v_block_size_bytes, copy_type);
```

**问题**:
- `convertIndexToAddr()` 返回的地址位于 **K buffer** 内（因为 `processMemoryLayout()` 使用 `k_cache_buffer_` 作为 full_tensor）
- 加上 `k_block_size_bytes` 偏移后仍在 K buffer 内部
- V 数据位于独立的 `v_cache_buffer_` 中，地址完全不同
- 这会导致 V 数据拷贝到 K buffer 的错误位置，造成数据损坏

**修复建议**: 需要独立获取 V buffer 的地址。方案：
1. 在 `MemoryLayoutStrategy` 中增加对 V buffer 的地址映射支持
2. 或在 `BlockPool` 中提供 `convertIndexToVAddr()` 方法，基于 `v_cache_buffer_` 的偏移计算
3. 参考 `processLayerTensors()` 中 V view 的计算方式（通过 K offset 推算 V offset）

---

## 二、中等问题（P1 — 功能不完整）

### 6. `KVCacheManager::setKVBlockValue` 分离路径 — convertIndexToBuffer 期望返回 2 个 block

**文件**: `rtp_llm/cpp/cache/KVCacheManager.cc:158-160`

**现状**:
```cpp
auto dst = allocator_->convertIndexToBuffer(layer_id, block_index);
RTP_LLM_CHECK_WITH_INFO(dst.size() >= 2,
                        "convertIndexToBuffer returned fewer than 2 blocks for separate K/V cache");
```

**问题**:
- `convertIndexToBuffer()` 通过 `MemoryLayoutStrategy` 返回 block info，该策略只基于 K buffer 初始化
- 它不会返回 K 和 V 两个独立 block，通常只返回 1 个 block
- 这个 `RTP_LLM_CHECK_WITH_INFO` 检查会直接触发断言失败

**修复建议**: 
- 为分离模式实现新的地址查询接口，能同时返回 K 和 V buffer 的 block info
- 或者在分离模式下分别调用 K 和 V 的 convertIndexToBuffer

---

### 7. `ascend_prefill.py` / `ascend_decode.py` — 缺少 KV Cache Quantization (scale) 支持

**文件**: 
- `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_prefill.py`
- `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py`

**问题**: 两个实现都没有处理 KV cache 量化（fp8 scale）。如果模型启用了 `enable_kv_scale`：
- 现有代码不会传递 scale 参数
- 读写的数据精度不匹配，导致结果错误

**修复建议**: 后续补充 scale 参数传递。需要确认 `torch_npu` 的 `_npu_reshape_and_cache` 和 `npu_fused_infer_attention_score` 是否支持 fp8 quantized KV cache。

---

## 三、修复优先级汇总

| 优先级 | 编号 | 问题 | 影响范围 | 修复难度 |
|--------|------|------|---------|---------|
| P0 | #1 | KVCacheManager.cc 悬空代码 | 编译失败 | 低 — 删除代码 |
| P0 | #2 | BlockPool.separate_kv_cache_ 未传递 | 整个功能不生效 | 中 — 需修改配置链路 |
| P0 | #3 | CacheConfig.separate_kv_cache 未设置 | 整个功能不生效 | 低 — 添加赋值 |
| P0 | #4 | BlockPoolConfigHelper 池大小计算错误 | 内存越界 | 中 — 需重新计算池大小 |
| P0 | #5 | KVCacheAllocator V 地址计算错误 | 数据损坏 | 中 — 需实现 V 地址查询 |
| P1 | #6 | setKVBlockValue 分离路径 block 查询 | 运行时断言 | 中 — 需实现双 buffer 查询 |
| P2 | #7 | 缺少 KV Cache Quantization 支持 | fp8 模型结果错误 | 高 — 依赖 torch_npu API |

---

## 四、修复顺序建议

```
步骤 1: 删除 KVCacheManager.cc 悬空代码 (#1) — 立即可修

步骤 2: 打通配置传递链 (#2 + #3)
  ├─ BlockPoolConfig 新增 separate_kv_cache 字段
  ├─ BlockPoolConfigHelper::createConfig() 传递标志
  ├─ SingleConfigCreator / HybridConfigCreator 中设置 CacheConfig.separate_kv_cache = true (Ascend)
  └─ BlockPool 构造函数读取并赋值

步骤 3: 修正池大小计算 (#4)
  ├─ 分离模式下 kv_block_pool_size_bytes 使用 k_block_stride_bytes
  └─ 验证 total_size_bytes = K pool + V pool

步骤 4: 修正 V 地址查询 (#5 + #6)
  ├─ MemoryLayoutStrategy 或 BlockPool 增加 V buffer 地址映射
  ├─ KVCacheAllocator::blockBatchCopy 使用正确的 V 地址
  └─ KVCacheManager::setKVBlockValue 使用正确的双 buffer 查询

步骤 5: 编译验证
  └─ USING_ASCEND=ON 编译通过，无死代码警告
```

---

## 五、已确认无问题的部分

| 文件 | 修改内容 | 状态 |
|------|---------|------|
| `BufferTypes.h` | 新增 `layers_to_k_buffer_ptrs` / `layers_to_v_buffer_ptrs` | ✅ 正确 |
| `CacheConfig.h` | 新增 `separate_kv_cache` / `k_block_stride_bytes` / `v_block_stride_bytes` 字段 | ✅ 结构正确，但需设置 |
| `OpDefs.h` / `OpDefs.cc` | `LayerKVCache`/`KVCache` 新增分离字段 + `getLayerCache()` NHD reshape | ✅ 逻辑正确 |
| `OpDefs.h` | `getLayerCache()` 分离路径 NHD reshape `[blocks, seq, kv_heads, dim]` | ✅ 正确 |
| `PyWrappedModel.h` | 从 layout 填充 `k_cache_base_by_layer` / `v_cache_base_by_layer` | ✅ 正确 |
| `SingleTypeKVCacheAllocator.cc` | `allLayerCacheBase()` 传递 K/V tensors | ✅ 正确 |
| `HybridTypeKVCacheAllocator.cc` | `allLayerCacheBase()` 传递 K/V tensors | ✅ 正确 |
| `KVCacheManager::getMainModelCacheLayerLayout()` | 分离路径传递 K/V layout | ✅ 正确 |
| `__init__.pyi` | Python 类型 stub 更新 | ✅ 正确 |
| `__init__.py` | Factory 注册 Ascend paged attention | ✅ 正确 |
| `ascend_prefill.py` | Prefill attention 实现 | ⚠️ 代码正确，但缺少 scale 支持 |
| `ascend_decode.py` | Decode attention 实现 | ⚠️ 代码正确，但缺少 scale 支持 |
| `ascend_kv_cache_write_op.py` | KV cache 写入实现 | ✅ 正确 |
| `ascend_rope.py` | 纯 PyTorch RoPE 实现 | ✅ 正确 |
| `ascend_rope_emb.py` | Ascend RoPE Embedding Op | ✅ 正确 |
| `ascend_attn_params.py` | Ascend 参数结构 | ✅ 正确 |
