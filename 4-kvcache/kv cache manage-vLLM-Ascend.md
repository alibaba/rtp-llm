# KV Cache 流程详解

> **Workspace:** c:\WorkingSpace\Code\vllm-ascend

---

## User
> KV cache初始化、使用、更新的详细流程

## Assistant
我来帮你分析vLLM Ascend中KV cache的初始化、使用和更新的详细流程。让我先搜索相关代码。

现在我来为你详细解释 vLLM Ascend 中 KV cache 的初始化、使用和更新的完整流程。

---

## KV Cache 详细流程

### 一、KV Cache 初始化

KV cache 的初始化发生在模型加载完成后，主要在 [model_runner_v1.py:2652](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2652) 中进行。

#### 1.1 入口函数

```python
def initialize_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
    # 1. 分配原始内存
    kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
    # 2. Reshape成正确的形状
    kv_caches = self._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
    # 3. 设置跨层KV cache共享
    for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
        kv_caches[layer_name] = kv_caches[target_layer_name]
    # 4. 绑定到attention层
    bind_kv_cache(kv_caches, self.compilation_config.static_forward_context, self.kv_caches, num_attn_module)
    return kv_caches
```

#### 1.2 内存分配 (`_allocate_kv_cache_tensors`)

KV cache 内存分配的核心逻辑在 [model_runner_v1.py:2718](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2718)：

```
┌─────────────────────────────────────────────────────────────┐
│                    KV Cache 内存布局                          │
├─────────────────────────────────────────────────────────────┤
│  对于普通 Attention:                                         │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │    K Cache       │  │    V Cache       │                 │
│  │  (int8 tensor)   │  │  (int8 tensor)   │                 │
│  └──────────────────┘  └──────────────────┘                 │
│                                                             │
│  对于 MLA (Multi-Latent Attention):                         │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │   Nope Cache     │  │   Rope Cache     │                 │
│  │  (kv_lora_rank)  │  │(qk_rope_head_dim)│                 │
│  └──────────────────┘  └──────────────────┘                 │
│                                                             │
│  对于 DeepSeek Sparse Attention:                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐       │
│  │K Cache  │ │V Cache  │ │DSA K    │ │DSA K Scale  │       │
│  └─────────┘ └─────────┘ └─────────┘ └─────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

关键代码：
```python
# 分离 K/V cache 以支持 prefill disaggregation
k_tensor = torch.zeros(k_tensor_size, dtype=torch.int8, device=self.device)
v_tensor = torch.zeros(v_tensor_size, dtype=torch.int8, device=self.device)

# 2MB 内存对齐 (用于 KV Transfer)
if self.vllm_config.kv_transfer_config is not None:
    k_tensor = self._align_memory(k_tensor, alignment)[:k_tensor_size]
    v_tensor = self._align_memory(v_tensor, alignment)[:v_tensor_size]
```

#### 1.3 内存 Reshape (`_reshape_kv_cache_tensors`)

将原始内存 reshape 成正确的形状，定义在 [attention_v1.py:95](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L95)：

```python
@staticmethod
def get_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_type: str = "",
) -> tuple[int, ...]:
    # 形状: (2, num_blocks, block_size, num_kv_heads, head_size)
    # 2 表示 K 和 V
    return (2, num_blocks, block_size, num_kv_heads, head_size)
```

最终 KV cache 的形状：
- **K Cache**: `(num_blocks, block_size, num_kv_heads, head_size)`
- **V Cache**: `(num_blocks, block_size, num_kv_heads, head_size)`

---

### 二、KV Cache 使用

KV cache 的使用涉及两个核心数据结构：`block_table` 和 `slot_mapping`。

#### 2.1 核心数据结构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Block Table 结构                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Request 0: [Block 5, Block 12, Block 3, ...]                     │
│   Request 1: [Block 7, Block 2, Block 9, ...]                      │
│   Request 2: [Block 1, Block 8, ...]                               │
│                                                                     │
│   block_table: shape = (max_num_reqs, max_blocks_per_req)          │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                        Slot Mapping 计算                             │
│                                                                     │
│   slot_mapping = block_number * block_size + block_offset          │
│                                                                     │
│   例如: block_size=128, position=300                               │
│   block_idx = 300 // 128 = 2                                        │
│   block_offset = 300 % 128 = 44                                     │
│   block_number = block_table[req_idx][block_idx]                   │
│   slot_mapping = block_number * 128 + 44                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 2.2 Slot Mapping 计算

在 [block_table.py:99](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/block_table.py#L99) 中：

```python
def compute_slot_mapping(self, req_indices: np.ndarray, positions: np.ndarray) -> None:
    # 计算逻辑块索引
    logical_block_idx = positions // self.block_size
    
    # 从 block_table 中获取物理块编号
    block_table_indices = (
        req_indices * self.max_num_blocks_per_req * self.blocks_per_phys_block + logical_block_idx
    )
    block_numbers = self.block_table.np.ravel()[block_table_indices]
    
    # 计算块内偏移
    block_offsets = positions % self.block_size
    
    # 最终 slot_mapping = 块号 * block_size + 块内偏移
    np.add(block_numbers * self.block_size, block_offsets, 
           out=self.slot_mapping.np[: req_indices.shape[0]])
```

#### 2.3 Attention Forward 中的 KV Cache 读取

在 [attention_v1.py:730](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L730) 中：

```python
def _get_fia_params(self, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata):
    if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
        # Decode 阶段: 从 KV cache 读取
        num_block, block_size, _, _ = self.key_cache.shape
        key = self.key_cache.view(num_block, block_size, -1)
        value = self.value_cache.view(num_block, block_size, -1)
        block_table = attn_metadata.block_tables
        actual_seq_lengths_kv = attn_metadata.seq_lens_list
    # ...
    return key, value, block_size, block_table, actual_seq_lengths_kv
```

---

### 三、KV Cache 更新

KV cache 的更新发生在每次 forward pass 中，通过 `reshape_and_cache` 函数完成。

#### 3.1 更新流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KV Cache 更新流程                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 计算 K, V 投影                                                   │
│     key = k_proj(query)    # shape: [num_tokens, num_kv_heads, head_size]
│     value = v_proj(query)  # shape: [num_tokens, num_kv_heads, head_size]
│                                                                     │
│  2. 调用 reshape_and_cache                                           │
│     ┌─────────────────────────────────────────────────────────┐     │
│     │  reshape_and_cache(key, value, kv_cache, slot_mapping)  │     │
│     └─────────────────────────────────────────────────────────┘     │
│                                                                     │
│  3. 写入 KV Cache                                                    │
│     key_cache[slot_mapping] = key                                  │
│     value_cache[slot_mapping] = value                              │
│                                                                     │
│  4. 执行 Attention 计算                                              │
│     output = attention(query, key_cache, value_cache, block_table)  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 3.2 reshape_and_cache 实现

在 [attention_v1.py:884](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L884) 和 [device_op.py:31](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/device/device_op.py#L31)：

```python
def reshape_and_cache(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: tuple[torch.Tensor],
    attn_metadata: AscendMetadata,
    output: torch.Tensor,
):
    if len(kv_cache) > 1:
        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        
        slots = attn_metadata.slot_mapping
        
        # 调用 NPU 算子写入 KV cache
        DeviceOperator.reshape_and_cache(
            key=key[: attn_metadata.num_actual_tokens],
            value=value[: attn_metadata.num_actual_tokens],
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            slot_mapping=slots[: attn_metadata.num_actual_tokens],
        )
    return query, key, value, output
```

#### 3.3 底层 NPU 算子

在 [device_op.py:31-35](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/device/device_op.py#L31)：

```python
# Ascend 910B/C (BaseDeviceAdaptor)
class BaseDeviceAdaptor:
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu._npu_reshape_and_cache(
            key=key, 
            value=value, 
            key_cache=key_cache, 
            value_cache=value_cache, 
            slot_indices=slot_mapping
        )

# Ascend A5 (A5DeviceAdaptor)
class A5DeviceAdaptor(BaseDeviceAdaptor):
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu.npu_scatter_pa_kv_cache(
            key=key.contiguous(),
            value=value.contiguous(),
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping.contiguous(),
        )
```

---

### 四、完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KV Cache 完整生命周期                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         1. 初始化阶段                                 │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  Worker.initialize_from_config()                                     │   │
│  │       ↓                                                              │   │
│  │  ModelRunner.initialize_kv_cache()                                   │   │
│  │       ↓                                                              │   │
│  │  _allocate_kv_cache_tensors()  → 分配 int8 原始内存                   │   │
│  │       ↓                                                              │   │
│  │  _reshape_kv_cache_tensors()  → Reshape 成正确形状                    │   │
│  │       ↓                                                              │   │
│  │  bind_kv_cache()  → 绑定到各 Attention 层                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         2. 调度阶段                                   │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  Scheduler.schedule()  → 分配 blocks 给请求                          │   │
│  │       ↓                                                              │   │
│  │  BlockTable.add_row(block_ids)  → 记录 block 映射                     │   │
│  │       ↓                                                              │   │
│  │  compute_slot_mapping()  → 计算每个 token 的存储位置                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         3. Forward 阶段                               │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  Attention.forward()                                                 │   │
│  │       ↓                                                              │   │
│  │  ┌─────────────────────────────────────────────────────────────┐     │   │
│  │  │  a) 更新 KV Cache (Prefill/Chunked Prefill)                  │     │   │
│  │  │     reshape_and_cache(key, value, kv_cache, slot_mapping)    │     │   │
│  │  │         ↓                                                    │     │   │
│  │  │     torch_npu._npu_reshape_and_cache()  → 写入 NPU 内存       │     │   │
│  │  └─────────────────────────────────────────────────────────────┘     │   │
│  │       ↓                                                              │   │
│  │  ┌─────────────────────────────────────────────────────────────┐     │   │
│  │  │  b) 执行 Attention 计算                                       │     │   │
│  │  │     - Prefill: npu_fused_infer_attention_score()             │     │   │
│  │  │     - Decode:  _npu_paged_attention() 或 FIA                 │     │   │
│  │  └─────────────────────────────────────────────────────────────┘     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         4. 释放阶段                                   │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  Request 完成 → Scheduler 释放 blocks                                │   │
│  │       ↓                                                              │   │
│  │  BlockPool.free_blocks(block_ids)  → 回收内存                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 五、关键代码位置总结

| 功能 | 文件位置 |
|------|----------|
| KV Cache 初始化入口 | [model_runner_v1.py:2652](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2652) |
| 内存分配 | [model_runner_v1.py:2718](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2718) |
| 内存 Reshape | [model_runner_v1.py:2855](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2855) |
| KV Cache 形状定义 | [attention_v1.py:95](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L95) |
| Slot Mapping 计算 | [block_table.py:99](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/block_table.py#L99) |
| KV Cache 更新 | [attention_v1.py:884](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L884) |
| NPU 算子调用 | [device_op.py:31](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/device/device_op.py#L31) |
| Attention 计算 | [attention_v1.py:730](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L730) |

## User
> 为什么KV cache更新在Attention计算前面，KV是在哪里计算得到的

## Assistant
这是一个很好的问题！让我详细解释 KV 的计算来源和为什么更新在 Attention 计算之前。

## K/V 的计算来源

### 1. 普通 Attention（非 MLA）

K 和 V 是在 **Attention 层之外** 通过投影层计算得到的，然后作为参数传入 `forward` 函数：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Transformer 层的前向传播流程                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  hidden_states (输入)                                                        │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    QKV 投影层 (在 Attention 层外)                     │   │
│  │                                                                     │   │
│  │   query = q_proj(hidden_states)   # [num_tokens, num_heads, head_size]│  │
│  │   key = k_proj(hidden_states)     # [num_tokens, num_kv_heads, head_size]│
│  │   value = v_proj(hidden_states)   # [num_tokens, num_kv_heads, head_size]│
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Attention.forward()                               │   │
│  │                                                                     │   │
│  │   def forward(self, layer, query, key, value, kv_cache, ...):      │   │
│  │       # 1. 先将 K/V 写入 cache                                       │   │
│  │       reshape_and_cache(key, value, kv_cache, slot_mapping)         │   │
│  │                                                                     │   │
│  │       # 2. 再执行 attention 计算                                     │   │
│  │       output = attention_op(query, key_cache, value_cache, ...)     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  output (输出)                                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

从 [attention_v1.py:936](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L936) 可以看到：

```python
def forward(
    self,
    layer: AttentionLayer,
    query: torch.Tensor,    # 已经计算好的 Q
    key: torch.Tensor,      # 已经计算好的 K
    value: torch.Tensor,    # 已经计算好的 V
    kv_cache: tuple[torch.Tensor],
    attn_metadata: AscendMetadata,
    output: torch.Tensor | None = None,
    ...
) -> torch.Tensor:
    """Forward pass with Ascend attention.
    Args:
        query: shape = [num_tokens, num_heads, head_size]
        key: shape = [num_tokens, num_kv_heads, head_size]
        value: shape = [num_tokens, num_kv_heads, head_size]
    """
```

### 2. MLA (Multi-Latent Attention) 的特殊情况

MLA 的 K/V 计算更复杂，在 [mla_v1.py:1550](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/mla_v1.py#L1550) 中：

```python
def _mla_preprocess(self, layer_name, hidden_states, kv_cache, attn_metadata, need_gather_q_kv):
    # 1. 计算 latent representation
    qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
    q_c, kv_no_split = qkv_lora.split([self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
    
    # 2. 处理 decode tokens - 同时计算 K/V 并写入 cache
    if has_decode:
        decode_preprocess_res = self.mla_preprocess_decode(q_c, kv_no_split, kv_cache, attn_metadata)
    
    # 3. 处理 prefill tokens - 同时计算 K/V 并写入 cache
    if has_prefill:
        prefill_preprocess_res = self.mla_preprocess_prefill(q_c, kv_no_split, kv_cache, attn_metadata)
```

---

## 为什么 KV Cache 更新在 Attention 计算之前？

这是设计上的必然选择，原因如下：

### 1. 当前 Token 的 K/V 必须参与 Attention 计算

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Attention 计算的数据来源                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Prefill 阶段 (处理 prompt):                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Query: 当前所有 prompt tokens                                        │   │
│  │  Key/Value: 当前所有 prompt tokens 的 K/V (刚计算出来的)              │   │
│  │                                                                     │   │
│  │  Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d)) @ V               │   │
│  │                                                                     │   │
│  │  这里 K 和 V 可以直接使用刚计算出来的值，不需要从 cache 读取           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Decode 阶段 (生成新 token):                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Query: 当前新生成的 1 个 token                                       │   │
│  │  Key/Value: 历史所有 tokens 的 K/V (从 cache 读取) + 当前 token 的 K/V│   │
│  │                                                                     │   │
│  │  所以必须先将当前 token 的 K/V 写入 cache，                           │   │
│  │  然后 attention 计算时才能读取完整的 KV cache                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. 代码流程解析

从 [attention_v1.py:730](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L730) 可以看到不同阶段的处理：

```python
def _get_fia_params(self, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata):
    if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
        # Prefill 无 cache hit: 直接使用传入的 K/V，不需要从 cache 读取
        block_table = None
        actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
        
    elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
        # Decode: 必须从 KV cache 读取所有历史 K/V
        key = self.key_cache.view(num_block, block_size, -1)   # 从 cache 读取
        value = self.value_cache.view(num_block, block_size, -1)
        block_table = attn_metadata.block_tables
        
    elif attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
        # Chunked Prefill: 从 cache 读取已有的 K/V
        key = self.key_cache.view(num_block, block_size, -1)
        value = self.value_cache.view(num_block, block_size, -1)
```

### 3. 完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    为什么先更新 KV Cache 再计算 Attention                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  时间线:                                                                     │
│                                                                             │
│  T1: 计算 Q/K/V 投影                                                         │
│      │                                                                      │
│      │   hidden_states ──┬──> q_proj ──> Query                              │
│      │                   ├──> k_proj ──> Key                                │
│      │                   └──> v_proj ──> Value                              │
│      ▼                                                                      │
│  T2: 写入 KV Cache (reshape_and_cache)                                      │
│      │                                                                      │
│      │   Key  ──> key_cache[slot_mapping] = Key                            │
│      │   Value ──> value_cache[slot_mapping] = Value                       │
│      │                                                                      │
│      │   此时 KV cache 包含: 历史所有 K/V + 当前 token 的 K/V                │
│      ▼                                                                      │
│  T3: 执行 Attention 计算                                                     │
│      │                                                                      │
│      │   对于 Decode:                                                       │
│      │   Attention(Query, key_cache, value_cache)                          │
│      │                                                                      │
│      │   对于 PrefillNoCache:                                               │
│      │   Attention(Query, Key, Value)  # 直接使用刚计算的 K/V               │
│      │                                                                      │
│      ▼                                                                      │
│  T4: 输出结果                                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4. 关键理解

| 阶段 | K/V 来源 | 是否需要先写入 Cache |
|------|----------|---------------------|
| PrefillNoCache | 直接使用刚计算的 K/V | 否（但会写入供后续使用） |
| PrefillCacheHit | 部分从 cache，部分刚计算 | 是（新 token 的 K/V） |
| ChunkedPrefill | 从 cache 读取 | 是（新 chunk 的 K/V） |
| DecodeOnly | 完全从 cache 读取 | **必须**（当前 token 的 K/V） |

**总结**：KV cache 更新在 attention 计算之前是因为：
1. **Decode 阶段必须如此**：当前 token 的 K/V 必须先写入 cache，attention 计算才能读取完整的 KV 序列
2. **统一处理**：为了代码一致性，所有阶段都采用相同的顺序
3. **PrefillNoCache 是特例**：虽然可以直接使用刚计算的 K/V，但为了后续 decode 使用，仍需写入 cache

## User
> scheduler是如何进行KV资源的管理的

## Assistant
让我搜索 scheduler 相关的 KV 资源管理代码。

根据代码分析，让我详细解释 Scheduler 是如何进行 KV 资源管理的：

## Scheduler KV 资源管理架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Scheduler KV 资源管理架构                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                           Scheduler                                      │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │   │
│  │  │   waiting 队列   │  │   running 队列   │  │   finished_req_ids     │  │   │
│  │  │  (等待调度)      │  │  (正在执行)      │  │  (已完成请求)           │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │   │
│  │                              │                                          │   │
│  │                              ▼                                          │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │                    KVCacheManager                                │    │   │
│  │  │  ┌─────────────────────┐  ┌─────────────────────────────────┐   │    │   │
│  │  │  │      BlockPool      │  │   SingleTypeKVCacheManager      │   │    │   │
│  │  │  │  (块池管理)          │  │   (单类型 KV cache 管理)         │   │    │   │
│  │  │  │  - blocks[]         │  │   - req_to_blocks               │   │    │   │
│  │  │  │  - free_block_queue │  │   - find_longest_cache_hit      │   │    │   │
│  │  │  └─────────────────────┘  └─────────────────────────────────┘   │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 1. 核心数据结构

### BlockPool (块池)

```python
# 位于 vllm.v1.core.block_pool
class BlockPool:
    def __init__(self, num_gpu_blocks, enable_prefix_caching, block_size, enable_kv_cache_events):
        self.num_gpu_blocks = num_gpu_blocks
        self.blocks: list[KVCacheBlock] = [...]  # 所有 block 的列表
        self.free_block_queue = FreeBlockQueue()  # 空闲 block 队列
```

### KVCacheManager 核心方法

从 [recompute_scheduler.py:246](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/core/recompute_scheduler.py#L246) 可以看到调度流程：

```python
def schedule(self) -> RecomputeSchedulerOutput:
    # 1. 每个调度步骤开始时，通知 KVCacheManager
    self.kv_cache_manager.new_step_starts()
    
    # 2. 调度 RUNNING 请求
    while req_index < len(self.running) and token_budget > 0:
        # 3. 为请求分配 KV cache slots
        new_blocks = self.kv_cache_manager.allocate_slots(
            request,
            num_new_tokens,
            num_lookahead_tokens=self.num_lookahead_tokens,
        )
        
        if new_blocks is None:
            # 4. 分配失败，需要抢占低优先级请求
            preempted_req = self.running.pop()
            self._preempt_request(preempted_req, scheduled_timestamp)
```

## 2. KV 资源分配流程

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           KV 资源分配详细流程                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Step 1: 新调度步骤开始                                                          │
│  ─────────────────────────                                                      │
│  self.kv_cache_manager.new_step_starts()                                       │
│  │                                                                              │
│  │  作用: 重置内部状态，准备新的调度周期                                           │
│  │                                                                              │
│  ▼                                                                              │
│  Step 2: 检查 Prefix Cache 命中                                                  │
│  ────────────────────────────────                                               │
│  new_computed_blocks, num_computed_tokens = self.kv_cache_manager.get_computed_blocks(request)
│  │                                                                              │
│  │  作用: 查找已缓存的 KV blocks，避免重复计算                                     │
│  │  流程:                                                                        │
│  │    1. 计算 token 的 block hash                                                │
│  │    2. 在 block_pool 中查找匹配的 cached blocks                                 │
│  │    3. 返回匹配的 blocks 和 token 数量                                          │
│  │                                                                              │
│  ▼                                                                              │
│  Step 3: 分配新的 KV blocks                                                      │
│  ───────────────────────────────                                                │
│  new_blocks = self.kv_cache_manager.allocate_slots(                            │
│      request,                                                                   │
│      num_new_tokens,                    # 需要分配的 token 数量                   │
│      num_new_computed_tokens,            # 已缓存的 token 数量                   │
│      new_computed_blocks,                # 已缓存的 blocks                       │
│      num_lookahead_tokens,               # 推测解码的 lookahead tokens           │
│  )                                                                              │
│  │                                                                              │
│  │  作用: 为请求分配新的 KV cache 空间                                            │
│  │  流程:                                                                        │
│  │    1. 计算需要的新 block 数量                                                  │
│  │    2. 从 free_block_queue 中获取空闲 blocks                                    │
│  │    3. 如果空闲 blocks 不足，返回 None (触发抢占)                                 │
│  │    4. 将 blocks 关联到 request                                                │
│  │                                                                              │
│  ▼                                                                              │
│  Step 4: 请求完成时释放资源                                                       │
│  ─────────────────────────────                                                  │
│  self.kv_cache_manager.free(request)                                           │
│  │                                                                              │
│  │  作用: 释放请求占用的 KV cache blocks                                          │
│  │  流程:                                                                        │
│  │    1. 获取 request 关联的所有 blocks                                           │
│  │    2. 减少 block 的引用计数 (ref_cnt)                                          │
│  │    3. 如果 ref_cnt == 0，将 block 放回 free_block_queue                        │
│  │    4. 如果启用 prefix caching，保留 block 的 hash 用于后续命中                   │
│  │                                                                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 3. 抢占机制

当 KV cache 资源不足时，Scheduler 会抢占低优先级请求：

```python
# 从 [recompute_scheduler.py:292-322](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/core/recompute_scheduler.py#L292-322)
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(...)
    
    if new_blocks is not None:
        break  # 分配成功
    
    # 分配失败，需要抢占
    if self.policy == SchedulingPolicy.PRIORITY:
        # 优先级调度：抢占优先级最低的请求
        preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
    else:
        # FCFS：抢占最后加入的请求
        preempted_req = self.running.pop()
    
    self._preempt_request(preempted_req, scheduled_timestamp)
    preempted_reqs.append(preempted_req)
```

## 4. Prefix Caching 机制

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Prefix Caching 工作原理                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  请求 A: "The quick brown fox jumps over the lazy dog"                          │
│          └── Block 0 ── Block 1 ── Block 2 ── Block 3                          │
│              hash: h0    hash: h1    hash: h2    hash: h3                       │
│                                                                                 │
│  请求 B: "The quick brown fox runs fast"                                        │
│          └── Block 0 ── Block 1 ── Block 4                                     │
│              hash: h0    hash: h1    hash: h4 (新)                              │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  get_computed_blocks(请求 B)                                             │   │
│  │                                                                          │   │
│  │  1. 计算 token hash: [h0, h1, h4]                                        │   │
│  │  2. 在 block_pool 中查找:                                                │   │
│  │     - h0: 命中 (Block 0, 来自请求 A)                                     │   │
│  │     - h1: 命中 (Block 1, 来自请求 A)                                     │   │
│  │     - h4: 未命中                                                          │   │
│  │  3. 返回: ([Block 0, Block 1], 2 * block_size tokens)                   │   │
│  │                                                                          │   │
│  │  结果: 请求 B 可以跳过前 2 个 block 的计算，直接复用请求 A 的 KV cache      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 5. 完整调度流程图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Scheduler 完整调度流程                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  schedule()                                                                     │
│  │                                                                              │
│  ├─► new_step_starts()                    # 重置状态                            │
│  │                                                                              │
│  ├─► 调度 RUNNING 请求                                                          │
│  │    │                                                                         │
│  │    ├─► 计算 num_new_tokens             # 需要调度的 token 数                  │
│  │    │                                                                         │
│  │    ├─► allocate_slots(request, num_new_tokens)                               │
│  │    │    │                                                                    │
│  │    │    ├─► 成功: 加入 scheduled_running_reqs                                 │
│  │    │    │                                                                    │
│  │    │    └─► 失败: 抢占低优先级请求                                            │
│  │    │         │                                                               │
│  │    │         ├─► _preempt_request()                                          │
│  │    │         │    └─► kv_cache_manager.free(preempted_req)                   │
│  │    │         │                                                               │
│  │    │         └─► 重试 allocate_slots()                                       │
│  │    │                                                                         │
│  │    └─► 更新 token_budget                                                     │
│  │                                                                              │
│  ├─► 调度 WAITING 请求                                                          │
│  │    │                                                                         │
│  │    ├─► get_computed_blocks(request)    # 检查 prefix cache 命中              │
│  │    │                                                                         │
│  │    ├─► connector.get_num_new_matched_tokens()  # 检查外部 KV cache           │
│  │    │                                                                         │
│  │    ├─► allocate_slots(request, num_new_tokens, ...)                          │
│  │    │                                                                         │
│  │    └─► 加入 scheduled_new_reqs 或 scheduled_resumed_reqs                     │
│  │                                                                              │
│  ├─► 构建 SchedulerOutput                                                       │
│  │    │                                                                         │
│  │    └─► 包含: scheduled_new_reqs, scheduled_running_reqs, preempted_reqs      │
│  │                                                                              │
│  └─► update_from_output()                 # 处理执行结果                         │
│       │                                                                         │
│       ├─► 更新 request 状态                                                     │
│       │                                                                         │
│       └─► 对于完成的请求:                                                        │
│            └─► _free_request(request)                                           │
│                 └─► kv_cache_manager.free(request)                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 6. 关键 API 总结

| 方法 | 作用 | 调用时机 |
|------|------|----------|
| `new_step_starts()` | 重置调度状态 | 每个调度步骤开始 |
| `get_computed_blocks(request)` | 获取 prefix cache 命中的 blocks | 新请求首次调度 |
| `allocate_slots(request, num_tokens, ...)` | 分配 KV cache blocks | 每次调度需要新空间 |
| `free(request)` | 释放请求的 KV cache | 请求完成或被抢占 |
| `cache_blocks(request, num_tokens)` | 缓存 blocks 用于 prefix caching | KV transfer 完成后 |
| `get_block_ids(request_id)` | 获取请求的 block IDs | 需要访问 KV cache 时 |
| `take_events()` | 获取 KV cache 事件 | 每个调度步骤结束 |

**总结**：Scheduler 通过 `KVCacheManager` 和 `BlockPool` 管理 KV 资源，核心机制包括：
1. **按需分配**：根据 token 数量动态分配 blocks
2. **抢占机制**：资源不足时抢占低优先级请求
3. **Prefix Caching**：复用已计算的 KV cache，减少重复计算
4. **引用计数**：通过 `ref_cnt` 管理 block 生命周期，支持多请求共享

## User
> NPU Attention操作过程中，是如何进行KV cache的写入和读取的

## Assistant
让我详细分析 NPU Attention 操作中 KV cache 的写入和读取流程。

根据代码分析，让我详细解释 NPU Attention 操作中 KV cache 的写入和读取流程：

## NPU Attention KV Cache 写入和读取流程

### 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    NPU Attention KV Cache 写入与读取架构                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Attention.forward()                              │   │
│  │                                                                          │   │
│  │   输入: query, key, value (刚通过投影层计算)                              │   │
│  │                                                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │   │  Step 1: reshape_and_cache()  ─────────►  写入 KV Cache         │   │   │
│  │   │                                                                 │   │   │
│  │   │  torch_npu._npu_reshape_and_cache(                             │   │   │
│  │   │      key, value,           # 当前 token 的 K/V                  │   │   │
│  │   │      key_cache,            # 全局 K cache [num_blocks, ...]     │   │   │
│  │   │      value_cache,          # 全局 V cache                       │   │   │
│  │   │      slot_mapping          # token -> slot 的映射               │   │   │
│  │   │  )                                                              │   │   │
│  │   └─────────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                          │   │
│  │                              ▼                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │   │  Step 2: forward_impl()  ──────────────►  读取 KV Cache         │   │   │
│  │   │                                                                 │   │   │
│  │   │  根据 attn_state 选择读取方式:                                   │   │   │
│  │   │  - DecodeOnly: npu_paged_attention()                            │   │   │
│  │   │  - Prefill/Chunked: npu_fused_infer_attention_score()           │   │   │
│  │   └─────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2. KV Cache 写入流程

#### 2.1 核心数据结构

```python
# KV Cache 形状定义 - [attention_v1.py:93](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L93)
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
    return (2, num_blocks, block_size, num_kv_heads, head_size)
    #      │   │           │            │              │
    #      │   │           │            │              └─ 每个 head 的维度
    #      │   │           │            └─ KV head 数量
    #      │   │           └─ 每个 block 的 token 数量
    #      │   └─ block 总数
    #      └─ K 和 V 两个 tensor
```

#### 2.2 Slot Mapping 计算

```python
# [block_table.py:106-145](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/block_table.py#L106-145)
def compute_slot_mapping(self, req_indices, positions):
    """
    计算 slot_mapping: token 位置 -> KV cache slot
    
    例如: block_size = 16, positions = [0, 1, 32, 33]
    block_tables = [[5, 7, ...], ...]  # block 5, 7 被分配给这个请求
    
    计算过程:
    1. logical_block_idx = positions // block_size
       -> [0, 0, 2, 2]
    
    2. block_numbers = block_table[req_idx, logical_block_idx]
       -> [5, 5, 7, 7]  # 从 block_table 查找物理 block 编号
    
    3. block_offsets = positions % block_size
       -> [0, 1, 0, 1]  # 在 block 内的偏移
    
    4. slot_mapping = block_numbers * block_size + block_offsets
       -> [80, 81, 112, 113]  # 最终的 slot 位置
    """
    logical_block_idx = positions // self.block_size
    block_table_indices = req_indices * self.max_num_blocks_per_req + logical_block_idx
    block_numbers = self.block_table.np.ravel()[block_table_indices]
    block_offsets = positions % self.block_size
    slot_mapping = block_numbers * self.block_size + block_offsets
```

#### 2.3 写入操作

```python
# [attention_v1.py:884-918](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L884-918)
def reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
    if len(kv_cache) > 1:
        # 记录 NPU event 用于同步 (KV producer 场景)
        if self.is_kv_producer:
            attn_metadata.reshape_cache_event = torch.npu.Event()
        
        # 获取 KV cache 引用
        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        
        slots = attn_metadata.slot_mapping
        
        # 调用 NPU 算子写入 KV cache
        DeviceOperator.reshape_and_cache(
            key=key[:attn_metadata.num_actual_tokens],      # 当前 token 的 K
            value=value[:attn_metadata.num_actual_tokens],  # 当前 token 的 V
            key_cache=self.key_cache,                       # 全局 K cache
            value_cache=self.value_cache,                   # 全局 V cache
            slot_mapping=slots[:attn_metadata.num_actual_tokens],  # slot 映射
        )
        
        if self.is_kv_producer:
            attn_metadata.reshape_cache_event.record()
```

```python
# [device_op.py:31-34](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/device/device_op.py#L31-34)
# 实际调用 torch_npu 的底层算子
def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
    torch_npu._npu_reshape_and_cache(
        key=key, 
        value=value, 
        key_cache=key_cache, 
        value_cache=value_cache, 
        slot_indices=slot_mapping
    )
```

### 3. KV Cache 读取流程

#### 3.1 根据场景选择读取方式

```python
# [attention_v1.py:920-935](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L920-935)
def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
    num_tokens = query.shape[0]
    
    if (attn_metadata.attn_state == AscendAttentionState.DecodeOnly
        and using_paged_attention(num_tokens, self.vllm_config)
        and self.sliding_window is None):
        # Decode 场景: 使用 Paged Attention
        output = self.forward_paged_attention(query, attn_metadata, output)
    else:
        # Prefill/Chunked 场景: 使用 Fused Infer Attention
        output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output)
```

#### 3.2 Decode 场景 - Paged Attention

```python
# [attention_v1.py:860-872](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L860-872)
def forward_paged_attention(self, query, attn_metadata, output):
    """
    Decode 场景专用，每个请求只有 1 个新 token
    
    通过 block_table 直接访问 KV cache:
    - block_table: [batch_size, max_blocks_per_seq] 
    - 每行存储一个请求的所有 block IDs
    """
    torch_npu._npu_paged_attention(
        query=query,                              # [batch_size, num_heads, head_size]
        key_cache=self.key_cache,                 # [num_blocks, block_size, num_kv_heads, head_size]
        value_cache=self.value_cache,             # [num_blocks, block_size, num_kv_heads, head_size]
        num_kv_heads=self.num_kv_heads,
        num_heads=self.num_heads,
        scale_value=self.scale,
        block_table=attn_metadata.block_tables,   # 物理块映射表
        context_lens=attn_metadata.seq_lens,      # 每个请求的序列长度
        out=output,
    )
    return output
```

#### 3.3 Prefill/Chunked 场景 - Fused Infer Attention

```python
# [attention_v1.py:700-730](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L700-730)
def _get_fia_params(self, key, value, attn_metadata):
    """
    根据不同场景准备 KV cache 参数
    """
    if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
        # Prefill 无缓存: 直接使用刚计算的 K/V
        block_table = None
        actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
        
    elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
        # Decode: 从 KV cache 读取
        num_block, block_size, _, _ = self.key_cache.shape
        key = self.key_cache.view(num_block, block_size, -1)
        value = self.value_cache.view(num_block, block_size, -1)
        block_table = attn_metadata.block_tables
        actual_seq_lengths_kv = attn_metadata.seq_lens_list
        
    else:  # ChunkedPrefill
        # Chunked Prefill: 从 KV cache 读取已有部分
        num_block, block_size, _, _ = self.key_cache.shape
        key = self.key_cache.view(num_block, block_size, -1)
        value = self.value_cache.view(num_block, block_size, -1)
        block_table = attn_metadata.block_tables
        actual_seq_lengths_kv = attn_metadata.seq_lens_list
        
    return key, value, block_size, block_table, actual_seq_lengths_kv
```

```python
# [attention_v1.py:755-805](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L755-805)
def forward_fused_infer_attention(self, query, key, value, attn_metadata, output):
    key, value, block_size, block_table, actual_seq_lengths_kv = \
        self._get_fia_params(key, value, attn_metadata)
    
    # 调用 NPU FlashAttention 算子
    attn_output, _ = torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key,                              # 可能是刚计算的，也可能从 cache 读取
        value=value,
        atten_mask=attn_metadata.attn_mask,   # 注意力掩码
        block_table=block_table,              # 物理块映射表
        input_layout="TND",                   # Token-Num-Head-Dim 布局
        block_size=block_size,
        actual_seq_lengths=attn_metadata.actual_seq_lengths_q,      # Q 序列长度
        actual_seq_lengths_kv=actual_seq_lengths_kv,                # KV 序列长度
        num_key_value_heads=self.num_kv_heads,
        num_heads=self.num_heads,
        scale=self.scale,
        sparse_mode=3,                        # causal mask 模式
    )
```

### 4. 完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    NPU Attention KV Cache 完整流程                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                    准备阶段 (在 ModelRunner 中)                            │ │
│  │                                                                           │ │
│  │  1. Scheduler 分配 blocks ──► block_table 记录请求的物理块映射              │ │
│  │  2. compute_slot_mapping() ──► 计算 slot_mapping (token -> slot)         │ │
│  │  3. commit_block_table() / commit_slot_mapping() ──► 拷贝到 GPU           │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                         │
│                                      ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                    Attention Forward 执行阶段                              │ │
│  │                                                                           │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │ │
│  │  │  输入:                                                              │  │ │
│  │  │  - query: [num_tokens, num_heads, head_size]                        │  │ │
│  │  │  - key: [num_tokens, num_kv_heads, head_size]                       │  │ │
│  │  │  - value: [num_tokens, num_kv_heads, head_size]                     │  │ │
│  │  │  - kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]   │  │ │
│  │  │  - attn_metadata: slot_mapping, block_tables, seq_lens, ...         │  │ │
│  │  └─────────────────────────────────────────────────────────────────────┘  │ │
│  │                                      │                                    │ │
│  │                                      ▼                                    │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Step 1: 写入 KV Cache (reshape_and_cache)                          │  │ │
│  │  │                                                                     │  │ │
│  │  │  ┌─────────────────────────────────────────────────────────────┐   │  │ │
│  │  │  │  slot_mapping 示例:                                          │   │  │ │
│  │  │  │  token 0 -> slot 80  (block 5, offset 0)                     │   │  │ │
│  │  │  │  token 1 -> slot 81  (block 5, offset 1)                     │   │  │ │
│  │  │  │  token 2 -> slot 112 (block 7, offset 0)                     │   │  │ │
│  │  │  │  ...                                                         │   │  │ │
│  │  │  │                                                             │   │  │ │
│  │  │  │  torch_npu._npu_reshape_and_cache(                          │   │  │ │
│  │  │  │      key[0] -> key_cache[slot_mapping[0]]                   │   │  │ │
│  │  │  │      key[1] -> key_cache[slot_mapping[1]]                   │   │  │ │
│  │  │  │      ...                                                    │   │  │ │
│  │  │  │  )                                                          │   │  │ │
│  │  │  └─────────────────────────────────────────────────────────────┘   │  │ │
│  │  └─────────────────────────────────────────────────────────────────────┘  │ │
│  │                                      │                                    │ │
│  │                                      ▼                                    │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Step 2: 读取 KV Cache 并计算 Attention                             │  │ │
│  │  │                                                                     │  │ │
│  │  │  ┌───────────────────────────────────────────────────────────────┐ │  │ │
│  │  │  │  Decode 场景:                             │ │  │ │
│  │  │  │                                                               │ │  │ │
│  │  │  │  Query: [batch_size, 1, num_heads, head_size]  # 每请求1token │ │  │ │
│  │  │  │  KV Cache: 通过 block_table 访问所有历史 K/V                   │ │  │ │
│  │  │  │                                                               │ │  │ │
│  │  │  │  block_table 示例:                                            │ │  │ │
│  │  │  │  req 0: [5, 7, 12, ...]  # block IDs for request 0           │ │  │ │
│  │  │  │  req 1: [3, 8, 15, ...]  # block IDs for request 1           │ │  │ │
│  │  │  │                                                               │ │  │ │
│  │  │  │  算子根据 block_table 自动从 key_cache/value_cache 读取       │ │  │ │
│  │  │  └───────────────────────────────────────────────────────────────┘ │  │ │
│  │  │                                                                     │  │ │
│  │  │  ┌───────────────────────────────────────────────────────────────┐ │  │ │
│  │  │  │  Prefill 场景: (npu_fused_infer_attention_score)             │ │  │ │
│  │  │  │                                                               │ │  │ │
│  │  │  │  - PrefillNoCache: 直接使用刚计算的 key/value                 │ │  │ │
│  │  │  │  - ChunkedPrefill: 从 cache 读取已有部分 + 新计算的 K/V       │ │  │ │
│  │  │  │                                                               │ │  │ │
│  │  │  │  actual_seq_lengths: Q 的序列长度 [len_q1, len_q1+len_q2,..] │ │  │ │
│  │  │  │  actual_seq_lengths_kv: KV 的序列长度                         │ │  │ │
│  │  │  └───────────────────────────────────────────────────────────────┘ │  │ │
│  │  └─────────────────────────────────────────────────────────────────────┘  │ │
│  │                                      │                                    │ │
│  │                                      ▼                                    │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │ │
│  │  │  输出:                                                              │  │ │
│  │  │  output: [num_tokens, num_heads, head_size]                         │  │ │
│  │  └─────────────────────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5. 关键数据结构总结

| 数据结构 | 形状 | 作用 |
|---------|------|------|
| `key_cache` | `[num_blocks, block_size, num_kv_heads, head_size]` | 存储所有 K |
| `value_cache` | `[num_blocks, block_size, num_kv_heads, head_size]` | 存储所有 V |
| `slot_mapping` | `[num_tokens]` | token → slot 位置映射 |
| `block_tables` | `[batch_size, max_blocks_per_seq]` | 请求 → 物理块映射 |
| `seq_lens` | `[batch_size]` | 每个请求的序列长度 |

### 6. 写入与读取的关键区别

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    KV Cache 写入 vs 读取                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  写入 (reshape_and_cache):                                                      │
│  ─────────────────────────                                                      │
│  • 使用 slot_mapping 索引: token 位置 → 全局 slot                               │
│  • 一次写入当前所有新 token 的 K/V                                               │
│  • 算子: torch_npu._npu_reshape_and_cache                                       │
│  • 数据流: key/value (小) → key_cache/value_cache (大)                          │
│                                                                                 │
│  读取: │
│  ─────────────────────────────────────────────────────────────                  │
│  • 使用 block_tables 索引: 请求 → 物理块列表                                     │
│  • 读取请求的所有历史 K/V                                                        │
│  • Decode: torch_npu._npu_paged_attention                                       │
│  • Prefill: torch_npu.npu_fused_infer_attention_score                           │
│  • 数据流: key_cache/value_cache (大) → attention 计算                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**总结**：NPU Attention 通过两个关键步骤管理 KV cache：
1. **写入**：使用 `slot_mapping` 将新计算的 K/V 写入对应 slot
2. **读取**：使用 `block_tables` 从 cache 中读取历史 K/V 进行 attention 计算

## User
> slot_mapping和block_table的关系

## Assistant
让我详细分析 `slot_mapping` 和 `block_table` 之间的关系。

## slot_mapping 和 block_table 的关系

### 1. 核心概念

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    slot_mapping 与 block_table 的层次关系                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  层次结构:                                                                       │
│                                                                                 │
│  Request (请求) ──────► Block Table ──────► Physical Blocks (物理块)            │
│       │                    [请求ID]              [block_id, block_id, ...]       │
│       │                                                                         │
│       ▼                                                                         │
│  Token (词元) ──────► Slot Mapping ──────► KV Cache Slot (缓存槽位)             │
│       │                  [token位置]              [具体存储位置]                 │
│       │                                                                         │
│       └─────────── slot_mapping 由 block_table 计算得出 ───────────┘            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2. 数据结构定义

```python
# block_table: 请求 -> 物理块列表
# 形状: [batch_size, max_blocks_per_seq]
block_table = [
    [5, 7, 12, 3, ...],   # 请求 0 的物理块 IDs
    [2, 8, 15, 9, ...],   # 请求 1 的物理块 IDs
    [1, 4, 6, 11, ...],   # 请求 2 的物理块 IDs
    ...
]

# slot_mapping: token -> KV cache slot
# 形状: [num_tokens]
slot_mapping = [80, 81, 112, 113, 32, 33, 128, ...]
#              └──────────────────┘  └──────┘  └──┘
#                 请求 0 的 tokens    请求 1    请求 2
```

### 3. 计算关系

从 [block_table.py:106-145](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/block_table.py#L106-145) 的核心计算逻辑：

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    slot_mapping 计算公式                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  输入:                                                                          │
│    - req_indices: [num_tokens]  每个 token 属于哪个请求                          │
│    - positions: [num_tokens]    每个 token 在请求中的位置                        │
│    - block_table: [batch_size, max_blocks_per_req]  请求的物理块映射             │
│    - block_size: 128  每个 block 包含的 token 数量                               │
│                                                                                 │
│  计算步骤:                                                                       │
│                                                                                 │
│  Step 1: 计算 token 所在的逻辑块索引                                             │
│  ────────────────────────────────────                                           │
│  logical_block_idx = positions // block_size                                    │
│                                                                                 │
│  Step 2: 在 block_table 中查找物理块编号                                         │
│  ────────────────────────────────────────                                       │
│  block_table_index = req_indices * max_blocks_per_req + logical_block_idx       │
│  block_number = block_table[block_table_index]                                  │
│                                                                                 │
│  Step 3: 计算 token 在 block 内的偏移                                            │
│  ─────────────────────────────────────                                           │
│  block_offset = positions % block_size                                          │
│                                                                                 │
│  Step 4: 计算最终的 slot                                                        │
│  ─────────────────────────                                                       │
│  slot_mapping = block_number * block_size + block_offset                        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4. 具体示例

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    具体计算示例                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  假设:                                                                          │
│    - block_size = 16 (每个 block 存储 16 个 token)                              │
│    - max_blocks_per_req = 4 (每个请求最多 4 个 block)                            │
│                                                                                 │
│  block_table:                                                                   │
│    请求 0: [5, 7, 12, 3]   # 物理块 IDs                                         │
│    请求 1: [2, 8, 15, 9]                                                       │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Token 数据:                                                             │   │
│  │                                                                         │   │
│  │  req_indices = [0, 0, 0, 0, 0, 0, 1, 1, 1]                              │   │
│  │  positions    = [0, 1, 2, 3, 16, 17, 0, 1, 32]                          │   │
│  │                └────请求 0────┘  └──请求 1──┘                            │   │
│  │                token 0-3     token 16-17  token 0-1, 32                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  计算 slot_mapping:                                                             │
│  ───────────────────                                                            │
│                                                                                 │
│  Token 0 (req=0, pos=0):                                                       │
│    logical_block_idx = 0 // 16 = 0                                             │
│    block_number = block_table[0, 0] = 5                                        │
│    block_offset = 0 % 16 = 0                                                   │
│    slot = 5 * 16 + 0 = 80                                                      │
│                                                                                 │
│  Token 1 (req=0, pos=1):                                                       │
│    logical_block_idx = 1 // 16 = 0                                             │
│    block_number = block_table[0, 0] = 5                                        │
│    block_offset = 1 % 16 = 1                                                   │
│    slot = 5 * 16 + 1 = 81                                                      │
│                                                                                 │
│  Token 4 (req=0, pos=16):  # 跨越第一个 block                                   │
│    logical_block_idx = 16 // 16 = 1                                            │
│    block_number = block_table[0, 1] = 7                                        │
│    block_offset = 16 % 16 = 0                                                  │
│    slot = 7 * 16 + 0 = 112                                                     │
│                                                                                 │
│  Token 6 (req=1, pos=0):  # 请求 1 的第一个 token                               │
│    logical_block_idx = 0 // 16 = 0                                             │
│    block_number = block_table[1, 0] = 2                                        │
│    block_offset = 0 % 16 = 0                                                   │
│    slot = 2 * 16 + 0 = 32                                                      │
│                                                                                 │
│  最终 slot_mapping = [80, 81, 82, 83, 112, 113, 32, 33, 128, ...]              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5. KV Cache 存储视图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    KV Cache 物理存储与映射关系                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  KV Cache 形状: [num_blocks, block_size, num_kv_heads, head_size]               │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Physical Block 0 (slot 0-15)                                           │   │
│  │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐    │   │
│  │  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │12 │13 │14 │15 │    │   │
│  │  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│  ...                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Physical Block 2 (slot 32-47)  ◄── 请求 1 的第一个 block               │   │
│  │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐    │   │
│  │  │32 │33 │34 │...│   │   │   │   │   │   │   │   │   │   │   │47 │    │   │
│  │  │▲  │▲  │   │   │   │   │   │   │   │   │   │   │   │   │   │   │    │   │
│  │  ││  ││  │   │   │   │   │   │   │   │   │   │   │   │   │   │   │    │   │
│  │  │└──┼──┘   │   │   │   │   │   │   │   │   │   │   │   │   │   │    │   │
│  │  │请求1    │   │   │   │   │   │   │   │   │   │   │   │   │   │    │   │
│  │  │token 0,1│   │   │   │   │   │   │   │   │   │   │   │   │   │    │   │
│  │  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│  ...                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Physical Block 5 (slot 80-95)  ◄── 请求 0 的第一个 block               │   │
│  │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐    │   │
│  │  │80 │81 │82 │83 │84 │...│   │   │   │   │   │   │   │   │   │95 │    │   │
│  │  │▲  │▲  │▲  │▲  │   │   │   │   │   │   │   │   │   │   │   │   │    │   │
│  │  ││  ││  ││  ││  │   │   │   │   │   │   │   │   │   │   │   │   │    │   │
│  │  │└──┴──┴──┴──┘   │   │   │   │   │   │   │   │   │   │   │   │   │    │   │
│  │  │请求0 token 0-3  │   │   │   │   │   │   │   │   │   │   │   │   │    │   │
│  │  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│  ...                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Physical Block 7 (slot 112-127)  ◄── 请求 0 的第二个 block             │   │
│  │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐    │   │
│  │  │112│113│114│...│   │   │   │   │   │   │   │   │   │   │   │127│    │   │
│  │  │▲  │▲  │   │   │   │   │   │   │   │   │   │   │   │   │   │   │    │   │
│  │  ││  ││  │   │   │   │   │   │   │   │   │   │   │   │   │   │   │    │   │
│  │  │└──┘   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │    │   │
│  │  │请求0  │   │   │   │   │   │   │   │   │   │   │   │   │   │   │    │   │
│  │  │token16,17│   │   │   │   │   │   │   │   │   │   │   │   │   │    │   │
│  │  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6. 使用场景对比

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    slot_mapping vs block_table 使用场景                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  slot_mapping - 用于写入 KV Cache                                       │   │
│  │                                                                         │   │
│  │  场景: 将新计算的 K/V 写入 cache                                        │   │
│  │  算子: torch_npu._npu_reshape_and_cache()                               │   │
│  │                                                                         │   │
│  │  # 写入当前所有 token 的 K/V                                            │   │
│  │  for i, slot in enumerate(slot_mapping):                                │   │
│  │      key_cache[slot] = key[i]    # token i 的 K 写入 slot              │   │
│  │      value_cache[slot] = value[i]                                       │   │
│  │                                                                         │   │
│  │  特点:                                                                   │   │
│  │  - token 级别的精确映射                                                  │   │
│  │  - 每个 token 对应一个 slot                                             │   │
│  │  - 用于分散写入操作                                                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  block_table - 用于读取 KV Cache                                        │   │
│  │                                                                         │   │
│  │  场景: 从 cache 读取历史 K/V 进行 attention 计算                         │   │
│  │  算子: torch_npu._npu_paged_attention() 或                              │   │
│  │        torch_npu.npu_fused_infer_attention_score()                      │   │
│  │                                                                         │   │
│  │  # 读取请求的所有历史 K/V                                               │   │
│  │  for req_id in range(batch_size):                                       │   │
│  │      block_ids = block_table[req_id]  # 获取该请求的所有 block          │   │
│  │      for block_id in block_ids:                                         │   │
│  │          # 从 block_id 读取 K/V 用于 attention                          │   │
│  │          ...                                                            │   │
│  │                                                                         │   │
│  │  特点:                                                                   │   │
│  │  - 请求级别的块映射                                                      │   │
│  │  - 每个请求对应一组 blocks                                              │   │
│  │  - 用于批量读取操作                                                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 7. 关系总结

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    slot_mapping 与 block_table 关系总结                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────┐     │
│  │                        生成流程                                        │     │
│  │                                                                       │     │
│  │  Scheduler 分配 blocks                                                │     │
│  │        │                                                              │     │
│  │        ▼                                                              │     │
│  │  block_table.add_row(block_ids, req_idx)  ──► 更新 block_table       │     │
│  │        │                                                              │     │
│  │        ▼                                                              │     │
│  │  block_table.compute_slot_mapping(req_indices, positions)             │     │
│  │        │                                                              │     │
│  │        └──────────────────────────────────────► 计算 slot_mapping    │     │
│  │                                                                       │     │
│  └───────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────┐     │
│  │                        使用流程                                        │     │
│  │                                                                       │     │
│  │  Attention Forward:                                                   │     │
│  │                                                                       │     │
│  │  1. 写入阶段:                                                          │     │
│  │     reshape_and_cache(key, value, kv_cache, slot_mapping)             │     │
│  │     └── 使用 slot_mapping 定位每个 token 的存储位置                    │     │
│  │                                                                       │     │
│  │  2. 读取阶段:                                                          │     │
│  │     npu_paged_attention(query, kv_cache, block_table, seq_lens)       │     │
│  │     └── 使用 block_table 定位每个请求的所有历史 K/V                    │     │
│  │                                                                       │     │
│  └───────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
│  关键公式:                                                                       │
│  ─────────                                                                      │
│  slot_mapping[i] = block_table[req_idx, logical_block_idx] * block_size        │
│                    + block_offset                                               │
│                                                                                 │
│  其中:                                                                          │
│    - logical_block_idx = position // block_size                                │
│    - block_offset = position % block_size                                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**总结**：
- **block_table** 是请求级别的映射，记录每个请求分配了哪些物理块
- **slot_mapping** 是 token 级别的映射，由 block_table 计算得出，记录每个 token 应该存储在哪个 slot
- **写入时**使用 slot_mapping（精确到每个 token）
- **读取时**使用 block_table（批量读取整个请求的 K/V）

## User
> `c:\WorkingSpace\Code\vllm-ascend\vllm_ascend\attention\attention_v1.py#L884-911` 第一次Prefill计算得到的KV是什么时候写入KV_cache的

## Assistant
让我分析第一次 Prefill 时 KV 写入 KV_cache 的完整流程。

## 第一次 Prefill 时 KV 写入 KV_cache 的完整流程

### 1. 完整调用链

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    第一次 Prefill 时 KV 写入流程                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Step 1: 模型前向传播 (在 Attention 层之外)                               │   │
│  │                                                                         │   │
│  │  hidden_states ──┬──> q_proj ──> query                                  │   │
│  │                  ├──> k_proj ──> key      ◄── 新计算的 K                │   │
│  │                  └──> v_proj ──> value    ◄── 新计算的 V                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Step 2: Attention.forward() [attention_v1.py:936-975]                  │   │
│  │                                                                         │   │
│  │  def forward(self, layer, query, key, value, kv_cache, ...):           │   │
│  │      # attn_state = PrefillNoCache (第一次 Prefill)                     │   │
│  │                                                                         │   │
│  │      if key is not None and value is not None:                          │   │
│  │          # ┌─────────────────────────────────────────────────────────┐  │   │
│  │          # │ Step 2.1: 写入 KV cache                                │  │   │
│  │          # └─────────────────────────────────────────────────────────┘  │   │
│  │          query, key, value, output = self.reshape_and_cache(            │   │
│  │              query, key, value, kv_cache, attn_metadata, output         │   │
│  │          )                                                              │   │
│  │                                                                         │   │
│  │      # ┌─────────────────────────────────────────────────────────────┐  │   │
│  │      # │ Step 2.2: 执行 Attention 计算                               │  │   │
│  │      # └─────────────────────────────────────────────────────────────┘  │   │
│  │      attn_output = self.forward_impl(                                   │   │
│  │          query, key, value, kv_cache, attn_metadata, output             │   │
│  │      )                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2. reshape_and_cache 写入过程

```python
# [attention_v1.py:884-918](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L884-918)
def reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
    if len(kv_cache) > 1:  # 确保 kv_cache 已初始化
        # 记录 NPU event (用于 KV producer 场景的同步)
        if self.is_kv_producer:
            attn_metadata.reshape_cache_event = torch.npu.Event()
        
        # 获取 KV cache 的引用
        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        
        slots = attn_metadata.slot_mapping  # token -> slot 映射
        
        # ★★★ 核心写入操作 ★★★
        DeviceOperator.reshape_and_cache(
            key=key[:attn_metadata.num_actual_tokens],      # 当前所有 token 的 K
            value=value[:attn_metadata.num_actual_tokens],  # 当前所有 token 的 V
            key_cache=self.key_cache,                       # 全局 K cache
            value_cache=self.value_cache,                   # 全局 V cache
            slot_mapping=slots[:attn_metadata.num_actual_tokens],  # slot 映射
        )
        
        if self.is_kv_producer:
            attn_metadata.reshape_cache_event.record()
    
    return query, key, value, output
```

### 3. PrefillNoCache 状态下的 Attention 计算

```python
# [attention_v1.py:691-697](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L691-697)
def _get_fia_params(self, key, value, attn_metadata):
    if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
        # ★ 第一次 Prefill: 不从 cache 读取，直接使用刚计算的 K/V
        block_size = 128
        block_table = None  # 不需要 block_table
        actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
        # key 和 value 保持原样（刚计算出来的）
        
    elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
        # 有缓存命中: 从 cache 读取部分 K/V
        key = self.key_cache.view(...)
        value = self.value_cache.view(...)
        block_table = attn_metadata.block_tables
        ...
```

```python
# [attention_v1.py:784-792](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L784-792)
def forward_fused_infer_attention(self, query, key, value, attn_metadata, output):
    key, value, block_size, block_table, actual_seq_lengths_kv = \
        self._get_fia_params(key, value, attn_metadata)
    
    num_tokens = attn_metadata.actual_seq_lengths_q[-1]
    query = query[:num_tokens]
    
    # PrefillNoCache 时，直接使用刚计算的 K/V
    if (attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
        and self.attn_type != AttentionType.ENCODER_DECODER):
        key = key[:num_tokens]      # 使用刚计算的 key
        value = value[:num_tokens]  # 使用刚计算的 value
    
    # 执行 attention 计算
    attn_output, _ = torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key,          # 刚计算的 K（已写入 cache）
        value=value,      # 刚计算的 V（已写入 cache）
        block_table=None, # PrefillNoCache 不需要
        ...
    )
```

### 4. 时间线总结

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    第一次 Prefill 时间线                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  T1: 计算 Q/K/V 投影                                                            │
│      │                                                                          │
│      │   hidden_states ──> q_proj/k_proj/v_proj ──> query, key, value          │
│      │                                                                          │
│      ▼                                                                          │
│  T2: 进入 Attention.forward()                                                   │
│      │                                                                          │
│      │                                                                          │
│      ├─► T2.1: reshape_and_cache()  ──────────────► 写入 KV Cache              │
│      │    │                                                                     │
│      │    │   key[i]  ──> key_cache[slot_mapping[i]]                           │
│      │    │   value[i] ──> value_cache[slot_mapping[i]]                        │
│      │    │                                                                     │
│      │    └── 此时 KV cache 已包含当前 token 的 K/V                             │
│      │                                                                          │
│      │                                                                          │
│      ├─► T2.2: forward_impl() ───────────────────► Attention 计算              │
│      │    │                                                                     │
│      │    │   对于 PrefillNoCache:                                              │
│      │    │   - block_table = None (不从 cache 读取)                            │
│      │    │   - 直接使用刚计算的 key/value                                       │
│      │    │   - 但这些 K/V 已经在 T2.1 写入了 cache                              │
│      │    │                                                                     │
│      │    └── 输出 attention 结果                                               │
│      │                                                                          │
│      ▼                                                                          │
│  T3: 返回输出，继续后续层的计算                                                   │
│                                                                                 │
│  ───────────────────────────────────────────────────────────────────────────    │
│  关键点:                                                                         │
│  ───────────────────────────────────────────────────────────────────────────    │
│                                                                                 │
│  1. 写入发生在 Attention 计算之前                                                │
│  2. PrefillNoCache 时，虽然写入了 cache，但计算时直接用刚计算的 K/V              │
│  3. 写入是为了后续 Decode 阶段能够读取这些 K/V                                   │
│  4. slot_mapping 在 Scheduler 阶段就已经计算好了                                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5. 为什么 PrefillNoCache 不从 cache 读取？

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PrefillNoCache vs 其他状态的读取策略                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  PrefillNoCache (第一次 Prefill)                                        │   │
│  │                                                                         │   │
│  │  写入: ✅ 写入所有 token 的 K/V 到 cache                                 │   │
│  │  读取: ❌ 不从 cache 读取，直接使用刚计算的 K/V                           │   │
│  │                                                                         │   │
│  │  原因:                                                                   │   │
│  │  - 当前 token 的 K/V 刚计算出来，已经在内存中                            │   │
│  │  - 直接使用比从 cache 读取更高效                                         │   │
│  │  - 写入是为了后续 Decode 使用                                            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  DecodeOnly (Decode 阶段)                                               │   │
│  │                                                                         │   │
│  │  写入: ✅ 写入当前 1 个 token 的 K/V 到 cache                            │   │
│  │  读取: ✅ 从 cache 读取所有历史 K/V                                      │   │
│  │                                                                         │   │
│  │  原因:                                                                   │   │
│  │  - 需要所有历史 token 的 K/V 进行 attention                              │   │
│  │  - 历史 K/V 只能从 cache 读取                                           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  ChunkedPrefill (分块 Prefill)                                          │   │
│  │                                                                         │   │
│  │  写入: ✅ 写入当前 chunk 的 K/V 到 cache                                 │   │
│  │  读取: ✅ 从 cache 读取之前 chunk 的 K/V                                 │   │
│  │                                                                         │   │
│  │  原因:                                                                   │   │
│  │  - 之前 chunk 的 K/V 已在 cache 中                                      │   │
│  │  - 需要合并历史和当前 chunk 的 K/V                                       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**总结**：第一次 Prefill 时，KV 在 `Attention.forward()` 中通过 `reshape_and_cache()` 写入 KV_cache，发生在 Attention 计算之前。虽然写入了 cache，但由于是 PrefillNoCache 状态，Attention 计算直接使用刚计算的 K/V，而不是从 cache 读取。写入的目的是为后续 Decode 阶段提供历史 K/V。

## User
> `c:\WorkingSpace\Code\vllm-ascend\vllm_ascend\worker\model_runner_v1.py#L2608-2608` kv_cache初始化详细过程介绍

## Assistant
让我详细分析 KV cache 初始化的完整过程。

## KV Cache 初始化详细过程

### 1. 整体流程概览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    KV Cache 初始化完整流程                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  initialize_kv_cache(kv_cache_config)                                          │
│  │                                                                              │
│  ├─► Step 1: 预处理配置                                                         │
│  │    - deepcopy(kv_cache_config)                                              │
│  │    - may_add_encoder_only_layers_to_kv_cache_config()                       │
│  │    - maybe_add_kv_sharing_layers_to_kv_cache_groups()                       │
│  │                                                                              │
│  ├─► Step 2: 初始化 Attention Backend                                          │
│  │    - initialize_attn_backend(kv_cache_config)                               │
│  │    - 确定 use_hybrid_blocks, need_accepted_tokens                           │
│  │                                                                              │
│  ├─► Step 3: 重新初始化 Input Batch (如果需要)                                   │
│  │    - may_reinitialize_input_batch(kv_cache_config)                          │
│  │                                                                              │
│  ├─► Step 4: 初始化 KV Cache Tensors ★核心步骤★                                 │
│  │    │                                                                         │
│  │    ├─► 4.1: _allocate_kv_cache_tensors()  ──► 分配内存                      │
│  │    │                                                                         │
│  │    └─► 4.2: _reshape_kv_cache_tensors()   ──► 调整形状                      │
│  │                                                                              │
│  ├─► Step 5: 绑定 KV Cache 到 Attention 层                                      │
│  │    - bind_kv_cache(kv_caches, ...)                                          │
│  │                                                                              │
│  └─► Step 6: 其他初始化                                                         │
│       - Drafter attention group 初始化 (推测解码)                                │
│       - KV Transfer Group 注册 (分布式 KV 传输)                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2. Step 4.1: 分配内存 (_allocate_kv_cache_tensors)

```python
# [model_runner_v1.py:2688-2818](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2688-2818)
def _allocate_kv_cache_tensors(self, kv_cache_config):
    """
    分配 KV cache 的原始内存缓冲区
    返回: dict[layer_name, tensor] 或 dict[layer_name, (k_tensor, v_tensor)]
    """
    kv_cache_raw_tensors = {}
    alignment = 2 * 1024 * 1024  # 2MB 对齐 (用于 prefill disaggregation)
    
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        for layer_name in kv_cache_tensor.shared_by:
            
            # ┌─────────────────────────────────────────────────────────────────┐
            # │  情况 1: Mamba Linear Attention 或 Hybrid (attn + mamba)        │
            # └─────────────────────────────────────────────────────────────────┘
            if "linear_attn" in layer_name or self.hybrid_with_attn_and_mamba:
                # 分配单个连续的 int8 tensor
                tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
                kv_cache_raw_tensors[layer_name] = tensor
            
            # ┌─────────────────────────────────────────────────────────────────┐
            # │  情况 2: 普通 Attention (分离 K 和 V cache)                      │
            # └─────────────────────────────────────────────────────────────────┘
            elif "attn" in layer_name and not use_mamba:
                # 计算 K 和 V 的尺寸比例
                k_dim, v_dim = self._get_attention_kv_cache_dims(layer_name, kv_cache_spec)
                k_tensor_split_factor, v_tensor_split_factor = calc_split_factor([k_dim, v_dim])
                
                k_tensor_size = kv_cache_tensor.size // k_tensor_split_factor
                v_tensor_size = kv_cache_tensor.size // v_tensor_split_factor
                
                # 分离分配 K 和 V tensor
                k_tensor = torch.zeros(k_tensor_size, dtype=torch.int8, device=self.device)
                v_tensor = torch.zeros(v_tensor_size, dtype=torch.int8, device=self.device)
                
                # 存储为 tuple
                kv_cache_raw_tensors[layer_name] = (k_tensor, v_tensor)
    
    return kv_cache_raw_tensors
```

### 3. Step 4.2: 调整形状 (_reshape_kv_cache_tensors)

```python
# [model_runner_v1.py:2820-3043](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2820-3043)
def _reshape_kv_cache_tensors(self, kv_cache_config, kv_cache_raw_tensors):
    """
    将原始内存缓冲区 reshape 为正确的形状和数据类型
    """
    kv_caches = {}
    
    for group in self._kv_cache_spec_attn_group_iterator():
        attn_backend = group.backend
        
        for layer_name in group.layer_names:
            current_kv_cache_spec = layer_kv_cache_spec[layer_name]
            
            # ┌─────────────────────────────────────────────────────────────────┐
            # │  情况 1: AttentionSpec (普通 Attention)                         │
            # └─────────────────────────────────────────────────────────────────┘
            if isinstance(current_kv_cache_spec, AttentionSpec):
                raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[layer_name]
                
                # 计算 KV cache 形状
                # 形状: [num_blocks, block_size, num_kv_heads, head_size]
                kv_cache_shape = attn_backend.get_kv_cache_shape(
                    num_blocks,
                    current_kv_cache_spec.block_size,
                    current_kv_cache_spec.num_kv_heads,
                    current_kv_cache_spec.head_size,
                )
                
                # MLA 特殊处理
                if isinstance(current_kv_cache_spec, MLAAttentionSpec):
                    # K cache: nope_cache, V cache: rope_cache
                    k_shape = (num_blocks, block_size, num_kv_heads, k_dim)
                    v_shape = (num_blocks, block_size, num_kv_heads, v_dim)
                else:
                    k_shape = v_shape = kv_cache_shape[1:]
                
                # Reshape tensor
                k_cache = raw_k_tensor.view(dtype).view(k_shape)
                v_cache = raw_v_tensor.view(dtype).view(v_shape)
                
                kv_caches[layer_name] = (k_cache, v_cache)
            
            # ┌─────────────────────────────────────────────────────────────────┐
            # │  情况 2: MambaSpec (Mamba State)                                │
            # └─────────────────────────────────────────────────────────────────┘
            elif isinstance(current_kv_cache_spec, MambaSpec):
                raw_tensor = kv_cache_raw_tensors[layer_name]
                
                # Mamba 有多个 state tensors (conv_state, ssm_state)
                state_tensors = []
                for shape, dtype in zip(current_kv_cache_spec.shapes, current_kv_cache_spec.dtypes):
                    target_shape = (num_blocks, *shape)
                    tensor = raw_tensor[start_idx:target_idx].view(dtype).view(target_shape)
                    state_tensors.append(tensor)
                
                kv_caches[layer_name] = state_tensors
    
    return kv_caches
```

### 4. 详细流程图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    KV Cache 内存分配与形状调整                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  输入: kv_cache_config                                                          │
│  ├── kv_cache_tensors: [KVCacheTensor(size, shared_by), ...]                   │
│  └── kv_cache_groups: [KVCacheGroupSpec(layer_names, kv_cache_spec), ...]      │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  _allocate_kv_cache_tensors() - 分配原始内存                             │   │
│  │                                                                         │   │
│  │  遍历 kv_cache_tensors:                                                  │   │
│  │                                                                         │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐ │   │
│  │  │  Attention 类型:                                                   │ │   │
│  │  │                                                                   │ │   │
│  │  │  k_tensor_size = total_size * k_dim / (k_dim + v_dim)            │ │   │
│  │  │  v_tensor_size = total_size * v_dim / (k_dim + v_dim)            │ │   │
│  │  │                                                                   │ │   │
│  │  │  k_tensor = torch.zeros(k_tensor_size, dtype=int8, device=npu)   │ │   │
│  │  │  v_tensor = torch.zeros(v_tensor_size, dtype=int8, device=npu)   │ │   │
│  │  │                                                                   │ │   │
│  │  │  kv_cache_raw_tensors[layer_name] = (k_tensor, v_tensor)         │ │   │
│  │  └───────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                         │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐ │   │
│  │  │  Mamba 类型:                                                       │ │   │
│  │  │                                                                   │ │   │
│  │  │  tensor = torch.zeros(total_size, dtype=int8, device=npu)        │ │   │
│  │  │  kv_cache_raw_tensors[layer_name] = tensor                        │ │   │
│  │  └───────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  _reshape_kv_cache_tensors() - 调整形状                                  │   │
│  │                                                                         │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐ │   │
│  │  │  Attention 类型:                                                   │ │   │
│  │  │                                                                   │ │   │
│  │  │  kv_cache_shape = [num_blocks, block_size, num_kv_heads, head_dim]│ │   │
│  │  │                                                                   │ │   │
│  │  │  k_cache = k_tensor.view(bf16).view(kv_cache_shape)              │ │   │
│  │  │  v_cache = v_tensor.view(bf16).view(kv_cache_shape)              │ │   │
│  │  │                                                                   │ │   │
│  │  │  kv_caches[layer_name] = (k_cache, v_cache)                       │ │   │
│  │  │                                                                   │ │   │
│  │  │  最终形状:                                                         │ │   │
│  │  │  k_cache: [num_blocks, block_size, num_kv_heads, head_size]      │ │   │
│  │  │  v_cache: [num_blocks, block_size, num_kv_heads, head_size]      │ │   │
│  │  └───────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                         │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐ │   │
│  │  │  Mamba 类型:                                                       │ │   │
│  │  │                                                                   │ │   │
│  │  │  state_tensors = []                                               │ │   │
│  │  │  for shape, dtype in spec.shapes, spec.dtypes:                   │ │   │
│  │  │      tensor = raw_tensor[...].view(dtype).view((num_blocks, *shape))│ │   │
│  │  │      state_tensors.append(tensor)                                 │ │   │
│  │  │                                                                   │ │   │
│  │  │  kv_caches[layer_name] = state_tensors  # [conv_state, ssm_state]│ │   │
│  │  └───────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  bind_kv_cache() - 绑定到 Attention 层                                   │   │
│  │                                                                         │   │
│  │  for layer_name, kv_cache in kv_caches.items():                        │   │
│  │      attention_layer = static_forward_context[layer_name]              │   │
│  │      attention_layer.kv_cache = kv_cache                               │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5. 关键数据结构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    KV Cache 数据结构                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  kv_cache_config (输入配置):                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  KVCacheConfig(                                                          │   │
│  │      num_blocks = 10000,              # 总 block 数量                    │   │
│  │      kv_cache_tensors = [                                                │   │
│  │          KVCacheTensor(                                                  │   │
│  │              size = 1073741824,       # 字节数                           │   │
│  │              shared_by = ["layer0.attn", "layer1.attn", ...]            │   │
│  │          ),                                                              │   │
│  │          ...                                                             │   │
│  │      ],                                                                  │   │
│  │      kv_cache_groups = [                                                 │   │
│  │          KVCacheGroupSpec(                                               │   │
│  │              layer_names = ["layer0.attn", ...],                        │   │
│  │              kv_cache_spec = AttentionSpec(                             │   │
│  │                  block_size = 128,                                      │   │
│  │                  num_kv_heads = 8,                                      │   │
│  │                  head_size = 128,                                       │   │
│  │                  dtype = torch.bfloat16,                                │   │
│  │              )                                                           │   │
│  │          ),                                                              │   │
│  │          ...                                                             │   │
│  │      ]                                                                   │   │
│  │  )                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  kv_caches (最终输出):                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  {                                                                       │   │
│  │      "layer0.attn": (                                                    │   │
│  │          k_cache: [num_blocks, block_size, num_kv_heads, head_size],   │   │
│  │          v_cache: [num_blocks, block_size, num_kv_heads, head_size],   │   │
│  │      ),                                                                  │   │
│  │      "layer1.attn": (k_cache, v_cache),                                 │   │
│  │      ...                                                                 │   │
│  │      "layer0.linear_attn": [conv_state, ssm_state],  # Mamba           │   │
│  │  }                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6. 特殊情况处理

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    特殊情况处理                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. MLA (Multi-Latent Attention):                                               │
│  ─────────────────────────────────                                              │
│  • K cache 存储 nope_cache (compressed latent)                                  │
│  • V cache 存储 rope_cache (position embedding)                                │
│  • 形状不同: k_shape = [..., kv_lora_rank], v_shape = [..., qk_rope_head_dim]  │
│                                                                                 │
│  2. Sparse Attention (DeepSeek V3):                                             │
│  ─────────────────────────────────────                                          │
│  • 额外分配 dsa_k_cache (dense-sparse attention)                                │
│  • 可能还有 dsa_k_scale_cache (C8 量化)                                         │
│  • kv_caches[layer_name] = (k_cache, v_cache, dsa_k_cache, dsa_k_scale_cache)  │
│                                                                                 │
│  3. KV Transfer (分布式):                                                        │
│  ─────────────────────────────                                                  │
│  • 内存需要 2MB 对齐                                                             │
│  • 使用 _align_memory() 进行地址对齐                                            │
│                                                                                 │
│  4. KV Sharing (层间共享):                                                       │
│  ─────────────────────────────                                                  │
│  • 多个层共享同一个 KV cache tensor                                             │
│  • kv_cache_tensor.shared_by 包含所有共享层名                                   │
│                                                                                 │
│  5. Hybrid Blocks (Mamba + Attention):                                          │
│  ─────────────────────────────────────                                          │
│  • 使用统一的 tensor 格式                                                        │
│  • 需要处理 conv_block_padding_size                                             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 7. 初始化时机

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    KV Cache 初始化时机                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Engine 启动流程:                                                                │
│                                                                                 │
│  1. EngineCore.__init__()                                                       │
│     │                                                                           │
│     ├─► 计算可用 GPU 内存                                                        │
│     │                                                                           │
│     ├─► 确定 num_blocks (基于内存和模型配置)                                     │
│     │                                                                           │
│     └─► 创建 KVCacheConfig                                                      │
│                                                                                 │
│  2. ModelRunner.initialize_kv_cache(kv_cache_config)                            │
│     │                                                                           │
│     ├─► 分配内存 (_allocate_kv_cache_tensors)                                   │
│     │                                                                           │
│     ├─► 调整形状 (_reshape_kv_cache_tensors)                                    │
│     │                                                                           │
│     └─► 绑定到层 (bind_kv_cache)                                                │
│                                                                                 │
│  3. 后续使用:                                                                    │
│     │                                                                           │
│     ├─► Scheduler 分配 blocks 给请求                                            │
│     │                                                                           │
│     ├─► Attention 层通过 kv_cache 参数访问                                      │
│     │                                                                           │
│     └─► reshape_and_cache() 写入数据                                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**总结**：KV Cache 初始化分为三个核心步骤：
1. **分配内存**：根据配置为每层分配 int8 原始内存缓冲区
2. **调整形状**：将原始内存 reshape 为正确的形状和数据类型
3. **绑定到层**：将 KV cache 绑定到对应的 Attention 层，供后续前向传播使用

## User
> `c:\WorkingSpace\Code\vllm-ascend\vllm_ascend\attention\attention_v1.py#L940-940` 这里的kv_cache是在哪里初始化的，是怎么和当前的Attention层绑定在一起的

## Assistant
让我详细分析 kv_cache 的初始化和绑定过程。

现在我已经收集了足够的信息来详细解释 kv_cache 的初始化和绑定过程。

## KV Cache 初始化与 Attention 层绑定过程

### 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    KV Cache 初始化与绑定架构                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Engine 启动阶段                                                         │   │
│  │                                                                         │   │
│  │  1. ModelRunner.__init__()                                              │   │
│  │     │                                                                   │   │
│  │     └─► 加载模型时，每个 Attention 层注册到 static_forward_context       │   │
│  │         compilation_config.static_forward_context[layer_name] = layer   │   │
│  │                                                                         │   │
│  │  2. ModelRunner.initialize_kv_cache(kv_cache_config)                    │   │
│  │     │                                                                   │   │
│  │     ├─► _allocate_kv_cache_tensors()   ──► 分配内存                     │   │
│  │     ├─► _reshape_kv_cache_tensors()    ──► 调整形状                     │   │
│  │     │                                                                   │   │
│  │     └─► bind_kv_cache()  ★★★ 绑定到 Attention 层 ★★★                   │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  推理阶段                                                                │   │
│  │                                                                         │   │
│  │  Attention.forward()                                                    │   │
│  │     │                                                                   │   │
│  │     └─► 从 layer.kv_cache 获取 kv_cache                                │   │
│  │         调用 impl.forward(layer, query, key, value, kv_cache, ...)      │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2. Step 1: Attention 层注册到 static_forward_context

```python
# vLLM 中 Attention 层初始化时 (在模型加载阶段)
# 每个 Attention 层会将自己注册到 static_forward_context

# 例如在 vllm_ascend/ops/mla.py 中:
class AscendMLA:
    def __init__(self, ...):
        ...
        # 将自己注册到 static_forward_context
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self  # ★ 注册
```

### 3. Step 2: bind_kv_cache 绑定过程

```python
# [patch_qwen3_next_mtp.py:9-51](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_next_mtp.py#L9-51)
def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    """
    将分配的 KV cache 绑定到 ModelRunner 和 forward context
    """
    
    # ┌─────────────────────────────────────────────────────────────────┐
    # │  Step 2.1: 绑定到 ModelRunner.kv_caches (list)                  │
    # │  用于模型推理时按索引访问                                        │
    # └─────────────────────────────────────────────────────────────────┘
    assert len(runner_kv_caches) == 0
    
    # 按 layer_index 排序，转换为 list
    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)
    
    for layer_index in sorted(index2name.keys()):
        layer_name = index2name[layer_index][0]
        runner_kv_caches.append(kv_caches[layer_name])  # 添加到 list
    
    # ┌─────────────────────────────────────────────────────────────────┐
    # │  Step 2.2: 绑定到 Attention 层的 kv_cache 属性                  │
    # │  用于 Attention 层直接访问                                      │
    # └─────────────────────────────────────────────────────────────────┘
    for layer_name, kv_cache in kv_caches.items():
        # ★★★ 核心：将 kv_cache 绑定到 Attention 层 ★★★
        forward_context[layer_name].kv_cache = [kv_cache]
```

### 4. 调用链详解

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    bind_kv_cache 调用链                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  initialize_kv_cache_tensors(kv_cache_config)                                   │
│  │                                                                              │
│  │  返回: kv_caches = {                                                         │
│  │      "model.layers.0.self_attn.attn": (k_cache, v_cache),                   │
│  │      "model.layers.1.self_attn.attn": (k_cache, v_cache),                   │
│  │      ...                                                                     │
│  │  }                                                                           │
│  │                                                                              │
│  ▼                                                                              │
│  bind_kv_cache(                                                                 │
│      kv_caches,                              # dict[layer_name, tensor]        │
│      self.compilation_config.static_forward_context,  # dict[layer_name, layer]│
│      self.kv_caches,                         # list[tensor]                    │
│  )                                                                              │
│  │                                                                              │
│  ├─► Step 1: 遍历 kv_caches                                                    │
│  │    for layer_name, kv_cache in kv_caches.items():                           │
│  │                                                                              │
│  ├─► Step 2: 从 static_forward_context 获取对应的 Attention 层                  │
│  │    attention_layer = forward_context[layer_name]                            │
│  │    # 例如: forward_context["model.layers.0.self_attn.attn"]                 │
│  │    # 返回: Attention 对象实例                                                │
│  │                                                                              │
│  └─► Step 3: 设置 Attention 层的 kv_cache 属性                                  │
│       attention_layer.kv_cache = [kv_cache]                                    │
│       # 现在 Attention 层可以直接通过 self.kv_cache 访问 KV cache               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5. 推理时 kv_cache 的传递

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    推理时 kv_cache 传递流程                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  vLLM Attention.forward() (在 vLLM 源码中)                               │   │
│  │                                                                         │   │
│  │  def forward(self, hidden_states, positions, ...):                     │   │
│  │      # 1. 计算 Q, K, V 投影                                             │   │
│  │      query, key, value = self._compute_qkv(hidden_states)              │   │
│  │                                                                         │   │
│  │      # 2. 从 self.kv_cache 获取 KV cache ★★★                           │   │
│  │      kv_cache = self.kv_cache[0] if self.kv_cache else None            │   │
│  │                                                                         │   │
│  │      # 3. 调用 impl.forward，传入 kv_cache                              │   │
│  │      output = self.impl.forward(                                        │   │
│  │          layer=self,           # Attention 层实例                       │   │
│  │          query=query,                                                   │   │
│  │          key=key,                                                       │   │
│  │          value=value,                                                   │   │
│  │          kv_cache=kv_cache,    # ★ 这里传入 kv_cache                   │   │
│  │          attn_metadata=attn_metadata,                                   │   │
│  │          output=output,                                                 │   │
│  │      )                                                                  │   │
│  │      return output                                                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  AscendAttentionBackendImpl.forward()                                    │   │
│  │  [attention_v1.py:933-984](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L933-984)   │
│  │                                                                         │   │
│  │  def forward(self, layer, query, key, value, kv_cache, ...):            │   │
│  │      # kv_cache: tuple[torch.Tensor] = (k_cache, v_cache)              │   │
│  │      # 形状: [num_blocks, block_size, num_kv_heads, head_size]         │   │
│  │                                                                         │   │
│  │      if key is not None and value is not None:                          │   │
│  │          # 写入 KV cache                                                 │   │
│  │          query, key, value, output = self.reshape_and_cache(            │   │
│  │              query, key, value, kv_cache, attn_metadata, output         │   │
│  │          )                                                              │   │
│  │                                                                         │   │
│  │      # 执行 Attention 计算                                               │   │
│  │      attn_output = self.forward_impl(                                   │   │
│  │          query, key, value, kv_cache, attn_metadata, output             │   │
│  │      )                                                                  │   │
│  │      return attn_output                                                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6. 数据结构关系图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    数据结构关系                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ModelRunner                                                                    │
│  ├── compilation_config.static_forward_context: dict                           │
│  │   │                                                                          │
│  │   ├── "model.layers.0.self_attn.attn" ──► Attention 层实例 0                │
│  │   │                                       └── kv_cache: [(k_cache, v_cache)]│
│  │   │                                                                          │
│  │   ├── "model.layers.1.self_attn.attn" ──► Attention 层实例 1                │
│  │   │                                       └── kv_cache: [(k_cache, v_cache)]│
│  │   │                                                                          │
│  │   └── ...                                                                    │
│  │                                                                              │
│  └── kv_caches: list                                                            │
│      ├── [0] = (k_cache_0, v_cache_0)  # layer 0                               │
│      ├── [1] = (k_cache_1, v_cache_1)  # layer 1                               │
│      └── ...                                                                    │
│                                                                                 │
│  ───────────────────────────────────────────────────────────────────────────    │
│                                                                                 │
│  bind_kv_cache() 的作用:                                                         │
│                                                                                 │
│  1. 将 kv_caches dict 转换为 runner.kv_caches list (按 layer_index 排序)        │
│                                                                                 │
│  2. 将每个 kv_cache 绑定到对应的 Attention 层:                                   │
│     forward_context[layer_name].kv_cache = [kv_cache]                          │
│                                                                                 │
│  这样在推理时:                                                                   │
│  - Attention 层可以通过 self.kv_cache[0] 直接访问                               │
│  - ModelRunner 可以通过 self.kv_caches[layer_idx] 访问                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 7. 总结

**kv_cache 的初始化和绑定流程**：

1. **初始化阶段** (`initialize_kv_cache`):
   - 分配内存 → 调整形状 → 得到 `kv_caches` dict

2. **绑定阶段** (`bind_kv_cache`):
   - 将 `kv_caches` dict 转换为 `ModelRunner.kv_caches` list
   - 将每个 `kv_cache` 绑定到对应的 Attention 层的 `kv_cache` 属性

3. **推理阶段**:
   - vLLM 的 `Attention.forward()` 从 `self.kv_cache[0]` 获取 `kv_cache`
   - 调用 `impl.forward(layer, query, key, value, kv_cache, ...)` 传入

**关键点**：
- `static_forward_context` 是一个全局字典，存储所有 Attention 层的引用
- `bind_kv_cache` 通过 `layer_name` 作为 key，将 `kv_cache` 绑定到对应的 Attention 层
- Attention 层通过 `self.kv_cache[0]` 访问（使用 list 是为了支持 v0 PP virtual engine）

## User
> 初始化时，计算KV可用空间的流程是什么

## Assistant
让我分析初始化时计算 KV 可用空间的流程。

## 初始化时计算 KV 可用空间的流程

### 1. 整体流程概览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    KV 可用空间计算流程                                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Step 1: 初始化内存快照                                                   │   │
│  │                                                                         │   │
│  │  Worker.init_model()                                                    │   │
│  │  │                                                                      │   │
│  │  ├─► gc.collect() + torch.npu.empty_cache()                            │   │
│  │  │                                                                      │   │
│  │  └─► self.init_snapshot = MemorySnapshot()                             │   │
│  │       记录: total_memory, free_memory                                   │   │
│  │                                                                         │   │
│  │  self.requested_memory = total_memory * gpu_memory_utilization         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Step 2: 加载模型权重                                                     │   │
│  │                                                                         │   │
│  │  ModelRunner.__init__()                                                 │   │
│  │  │                                                                      │   │
│  │  └─► self.model = get_model(...)                                       │   │
│  │       self.model_memory_usage = m.consumed_memory                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Step 3: Profile Run (测量运行时内存)                                     │   │
│  │                                                                         │   │
│  │  Worker.determine_available_memory()                                    │   │
│  │  │                                                                      │   │
│  │  ├─► memory_profiling(init_snapshot, weights_memory)                   │   │
│  │  │   │                                                                  │   │
│  │  │   └─► model_runner.profile_run()  ──► 执行 dummy forward            │   │
│  │  │       测量运行时峰值内存                                               │   │
│  │  │                                                                      │   │
│  │  └─► 计算 available_kv_cache_memory_bytes                              │   │
│  │       = requested_memory - non_kv_cache_memory                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Step 4: 计算 num_blocks (在 vLLM EngineCore 中)                         │   │
│  │                                                                         │   │
│  │  根据 available_kv_cache_memory_bytes 和 KVCacheSpec 计算:              │   │
│  │  num_blocks = available_memory / page_size_bytes_per_block              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2. Step 1: 初始化内存快照

```python
# [worker.py:262-278](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/worker.py#L262-278)
def init_model(self):
    # 清理内存
    gc.collect()
    torch.npu.empty_cache()
    
    # ★ 记录初始内存快照
    self.init_snapshot = MemorySnapshot()
    # 包含: total_memory, free_memory
    
    # 计算请求的内存 (基于 gpu_memory_utilization 配置)
    self.requested_memory = self.init_snapshot.total_memory * self.cache_config.gpu_memory_utilization
    
    # 验证空闲内存是否足够
    if self.init_snapshot.free_memory < self.requested_memory:
        raise ValueError(
            f"Free memory on device ({GiB(self.init_snapshot.free_memory)}/"
            f"{GiB(self.init_snapshot.total_memory)} GiB) is less than "
            f"desired GPU memory utilization ({self.cache_config.gpu_memory_utilization}, "
            f"{GiB(self.requested_memory)} GiB)"
        )
```

### 3. Step 2: 加载模型权重

```python
# [model_runner_v1.py:2598-2601](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2598-2601)
def __init__(self, vllm_config, device):
    ...
    with DeviceMemoryProfiler() as m:
        self.model = get_model(vllm_config)
    
    # ★ 记录模型权重占用的内存
    self.model_memory_usage = m.consumed_memory
    logger.info("Loading model weights took %.4f GB", m.consumed_memory / float(2**30))
```

### 4. Step 3: Profile Run 测量运行时内存

```python
# [worker.py:321-355](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/worker.py#L321-355)
def determine_available_memory(self) -> int:
    """
    通过 profile 测量模型运行时的内存使用，确定可用于 KV cache 的内存
    """
    GiB = lambda b: b / GiB_bytes
    
    # ★ 执行 profile run，测量运行时内存
    with memory_profiling(
        self.init_snapshot,
        weights_memory=int(self.model_runner.model_memory_usage),
    ) as profile_result:
        self.model_runner.profile_run()  # 执行 dummy forward
    
    free_gpu_memory = profile_result.after_profile.free_memory
    
    # ★★★ 核心计算: 可用于 KV cache 的内存 ★★★
    self.available_kv_cache_memory_bytes = (
        self.requested_memory - profile_result.non_kv_cache_memory
    )
    
    logger.info_once(
        "Available KV cache memory: %.2f GiB", 
        GiB(self.available_kv_cache_memory_bytes), 
        scope="local"
    )
    
    return int(self.available_kv_cache_memory_bytes)
```

### 5. 内存计算公式

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    内存计算公式详解                                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  GPU 总内存分配                                                          │   │
│  │                                                                         │   │
│  │  total_gpu_memory = NPU 设备总内存                                      │   │
│  │  requested_memory = total_gpu_memory × gpu_memory_utilization          │   │
│  │                   = total_gpu_memory × 0.9 (默认)                       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  non_kv_cache_memory 组成                                                │   │
│  │                                                                         │   │
│  │  non_kv_cache_memory = weights_memory + activation_memory + misc       │   │
│  │                                                                         │   │
│  │  其中:                                                                   │   │
│  │  - weights_memory: 模型权重内存                                          │   │
│  │  - activation_memory: 运行时激活内存 (profile_run 测量)                  │   │
│  │  - misc: 其他杂项内存                                                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  available_kv_cache_memory 计算                                          │   │
│  │                                                                         │   │
│  │  available_kv_cache_memory = requested_memory - non_kv_cache_memory    │   │
│  │                                                                         │   │
│  │  示例:                                                                   │   │
│  │  - total_gpu_memory = 64 GB                                             │   │
│  │  - gpu_memory_utilization = 0.9                                         │   │
│  │  - requested_memory = 57.6 GB                                           │   │
│  │  - weights_memory = 20 GB                                               │   │
│  │  - activation_memory = 5 GB                                             │   │
│  │  - non_kv_cache_memory = 25 GB                                          │   │
│  │  - available_kv_cache_memory = 57.6 - 25 = 32.6 GB                     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6. num_blocks 计算

```python
# 在 vLLM EngineCore 中，根据 available_kv_cache_memory_bytes 计算 num_blocks

# 每个 block 的内存大小 (page_size_bytes)
page_size_bytes = block_size × num_kv_heads × head_size × dtype_size × num_layers

# 计算 num_blocks
num_blocks = available_kv_cache_memory_bytes // page_size_bytes

# 示例:
# - block_size = 128
# - num_kv_heads = 8
# - head_size = 128
# - dtype_size = 2 (bf16)
# - num_layers = 32
# - page_size_bytes = 128 × 8 × 128 × 2 × 32 = 67,108,864 bytes ≈ 64 MB
# - available_kv_cache_memory = 32.6 GB
# - num_blocks = 32.6 GB / 64 MB ≈ 510 blocks
```

### 7. Profile Run 详解

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Profile Run 流程                                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  model_runner.profile_run()                                                     │
│  │                                                                              │
│  ├─► 准备 dummy 输入数据                                                        │
│  │    - input_ids: 随机生成的 token IDs                                        │
│  │    - positions: 位置编码                                                    │
│  │    - num_tokens = max_num_tokens                                            │
│  │                                                                              │
│  ├─► 构建 attention metadata                                                    │
│  │    - 模拟真实的 attention 场景                                               │
│  │                                                                              │
│  └─► 执行模型前向传播                                                            │
│       with set_ascend_forward_context(...):                                    │
│           outputs = self._model_forward(...)                                   │
│                                                                                 │
│  ───────────────────────────────────────────────────────────────────────────    │
│  Profile Run 目的:                                                               │
│  ───────────────────────────────────────────────────────────────────────────    │
│                                                                                 │
│  1. 测量运行时激活内存 (activation memory)                                       │
│     - 中间张量的内存占用                                                         │
│     - Attention 计算的临时内存                                                   │
│     - MoE 专家路由的临时内存                                                     │
│                                                                                 │
│  2. 验证模型可以正常运行                                                          │
│     - 检查是否有 OOM                                                            │
│     - 验证 NPU kernel 是否正常工作                                               │
│                                                                                 │
│  3. 为后续计算 KV cache 可用空间提供准确数据                                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 8. 完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    完整初始化流程                                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Engine 启动                                                                    │
│  │                                                                              │
│  ├─► 1. Worker.init_model()                                                    │
│  │    │                                                                         │
│  │    ├─► MemorySnapshot() ──► 记录 total_memory, free_memory                  │
│  │    │                                                                         │
│  │    ├─► requested_memory = total × gpu_memory_utilization                    │
│  │    │                                                                         │
│  │    └─► ModelRunner.__init__() ──► 加载模型权重                               │
│  │         └─► model_memory_usage = 权重内存                                    │
│  │                                                                              │
│  ├─► 2. Worker.determine_available_memory()                                    │
│  │    │                                                                         │
│  │    ├─► memory_profiling(init_snapshot, weights_memory)                      │
│  │    │    │                                                                    │
│  │    │    └─► model_runner.profile_run()                                      │
│  │    │         └─► 测量运行时峰值内存                                           │
│  │    │                                                                         │
│  │    └─► available_kv_cache_memory = requested - non_kv_cache_memory          │
│  │                                                                              │
│  ├─► 3. EngineCore 计算 num_blocks                                              │
│  │    │                                                                         │
│  │    └─► num_blocks = available_memory / page_size_bytes                      │
│  │                                                                              │
│  └─► 4. ModelRunner.initialize_kv_cache(kv_cache_config)                       │
│       │                                                                         │
│       ├─► _allocate_kv_cache_tensors() ──► 分配内存                             │
│       │                                                                         │
│       ├─► _reshape_kv_cache_tensors() ──► 调整形状                              │
│       │                                                                         │
│       └─► bind_kv_cache() ──► 绑定到 Attention 层                               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**总结**：KV 可用空间的计算分为三个关键步骤：
1. **初始快照**：记录 GPU 总内存和空闲内存
2. **Profile Run**：执行 dummy forward 测量运行时内存占用
3. **计算可用空间**：`available_kv_cache_memory = requested_memory - non_kv_cache_memory`

这个值随后被用于计算可以分配多少 KV cache blocks。

## User
> `c:\WorkingSpace\Code\vllm-ascend\vllm_ascend\worker\model_runner_v1.py#L2972-2972` 每一层的k_cache是个Torch Tensor？其大小是否在动态的变化？num_blocks是最开始就确定好的最大值么

## Assistant
让我详细分析 k_cache 的结构和大小管理。

## k_cache 结构详解

### 1. k_cache 是一个固定的 Torch Tensor

```python
# [model_runner_v1.py:2972](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2972)
k_cache = raw_k_tensor.view(k_cache_dtype).view(k_shape)

# k_shape = (num_blocks, block_size, num_kv_heads, head_size)
# 例如: (10000, 128, 8, 128) 表示:
#   - 10000 个 blocks
#   - 每个 block 存储 128 个 token
#   - 8 个 KV heads
#   - 每个 head 的维度是 128
```

### 2. 大小不会动态变化

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    k_cache 内存结构                                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  初始化时:                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  k_cache: Tensor[num_blocks, block_size, num_kv_heads, head_size]       │   │
│  │                                                                         │   │
│  │  形状固定，内存预分配                                                      │   │
│  │  例如: (10000, 128, 8, 128) × bf16 = ~3.2 GB                           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  运行时:                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Tensor 形状不变，但内容会被动态写入/读取                                   │   │
│  │                                                                         │   │
│  │  Block 0    Block 1    Block 2    ...    Block 9999                    │   │
│  │  ┌──────┐   ┌──────┐   ┌──────┐         ┌──────┐                       │   │
│  │  │ K,V  │   │ K,V  │   │ K,V  │   ...   │ 空   │                       │   │
│  │  │数据  │   │数据  │   │数据  │         │      │                       │   │
│  │  └──────┘   └──────┘   └──────┘         └──────┘                       │   │
│  │                                                                         │   │
│  │  Scheduler 负责管理哪些 blocks 被使用，哪些是空闲的                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3. num_blocks 是初始化时确定的最大值

```python
# [model_runner_v1.py:2908-2918](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2908-2918)
num_blocks = sum_page_size_bytes // current_kv_cache_spec.page_size_bytes

# `num_blocks` is the number of blocks the model runner can use.
# `kv_cache_config.num_blocks` is the number of blocks that
# KVCacheManager may allocate.
# Since different GPUs may have different number of layers and
# different memory capacities, `num_blocks` can be different on
# different GPUs, and `kv_cache_config.num_blocks` is set to
# the min of all `num_blocks`. Verify it here.
assert num_blocks >= kv_cache_config.num_blocks
```

### 4. Block 管理机制

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Block 分配与管理                                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  固定的 Block Pool (在 KVCacheManager 中)                                │   │
│  │                                                                         │   │
│  │  total_blocks = num_blocks (初始化时确定，不变)                          │   │
│  │  free_blocks = 动态变化 (随请求分配/释放)                                 │   │
│  │  used_blocks = 动态变化 (随请求分配/释放)                                 │   │
│  │                                                                         │   │
│  │  约束: free_blocks + used_blocks = total_blocks                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Scheduler.allocate_slots() 请求分配 blocks:                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  1. 检查 free_blocks 是否足够                                            │   │
│  │  2. 如果足够，从 free_block_queue 中取出需要的 blocks                     │   │
│  │  3. 如果不够，执行 preemption (抢占低优先级请求的 blocks)                  │   │
│  │  4. 返回分配的 blocks 给请求                                              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Scheduler.free() 释放 blocks:                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  1. 请求完成或被抢占时调用                                                │   │
│  │  2. 将请求占用的 blocks 放回 free_block_queue                             │   │
│  │  3. 这些 blocks 可以被后续请求重用                                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5. 数据结构关系

```python
# 每层的 kv_cache 结构
kv_caches: dict[str, torch.Tensor] = {
    "model.layers.0.self_attn.attn": (
        k_cache: Tensor[num_blocks, block_size, num_kv_heads, head_size],
        v_cache: Tensor[num_blocks, block_size, num_kv_heads, head_size],
    ),
    "model.layers.1.self_attn.attn": (k_cache, v_cache),
    ...
}

# 所有层共享相同的 num_blocks
# 但每层有独立的 k_cache 和 v_cache tensor
```

### 6. 关键点总结

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    关键点总结                                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. k_cache 是 Torch Tensor                                                    │
│  ────────────────────────────                                                   │
│  • 类型: torch.Tensor                                                           │
│  • 形状: (num_blocks, block_size, num_kv_heads, head_size)                     │
│  • 数据类型: 通常 bf16 或 fp16                                                  │
│  • 设备: NPU (Ascend)                                                           │
│                                                                                 │
│  2. 大小不会动态变化                                                            │
│  ────────────────────────────                                                   │
│  • Tensor 形状在初始化时固定                                                    │
│  • 内存预分配，运行时不增长/缩小                                                 │
│  • 内容会被动态写入和覆盖                                                        │
│                                                                                 │
│  3. num_blocks 是固定的最大值                                                   │
│  ────────────────────────────                                                   │
│  • 初始化时根据可用内存计算                                                      │
│  • 代表最多可存储的 block 数量                                                   │
│  • 所有层共享相同的 num_blocks                                                  │
│  • Scheduler 负责管理 block 的分配和回收                                         │
│                                                                                 │
│  4. Block 是管理单位                                                            │
│  ────────────────────────────                                                   │
│  • 每个 block 存储 block_size 个 token 的 KV                                    │
│  • Scheduler 从固定 pool 中分配/释放 blocks                                     │
│  • Preemption 机制确保高优先级请求能获得 blocks                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**总结**：
- **k_cache 是固定的 Torch Tensor**，形状为 `(num_blocks, block_size, num_kv_heads, head_size)`
- **大小不会动态变化**，内存预分配，运行时只改变内容
- **num_blocks 是初始化时确定的最大值**，代表 KV cache 的容量上限，Scheduler 从这个固定 pool 中动态分配和回收 blocks

