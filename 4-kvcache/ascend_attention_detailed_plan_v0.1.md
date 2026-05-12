# Ascend Attention 适配详细方案 v0.1

## 1. 问题分析

### 当前状态

`AscendPrefillImpl`（ascend_prefill.py）和 `AscendDecodeImpl`（ascend_decode.py）的 `forward()` 方法仅做了：

```python
# prefill
q, k, v = qkv.chunk(3, dim=-1)
k_cache = kv_cache.k_cache_base
v_cache = kv_cache.v_cache_base
attn_output, _ = torch_npu.npu_fused_infer_attention_score(query=q, key=k_cache, value=v_cache, ...)

# decode  
q, k, v = qkv.chunk(3, dim=-1)
k_cache = kv_cache.k_cache_base
v_cache = kv_cache.v_cache_base
torch_npu._npu_paged_attention(query=q, key_cache=k_cache, value_cache=v_cache, ...)
```

**缺失的关键步骤：**
1. ❌ 没有对 K, V 应用 RoPE
2. ❌ 没有将 K, V 写入 KV cache（`_npu_reshape_and_cache`）
3. ❌ 没有处理 `need_rope_kv_cache` 配置开关
4. ❌ 没有处理 `write_cache_store`（cache_store_inputs）
5. ❌ 没有使用 `create_params()` / `prepare()` 模式管理参数

当前直接从 `kv_cache.k_cache_base` / `v_cache_base` 读取数据，假设 cache 已被外部填充。

### CUDA 参考实现

`PyFlashinferPrefillImplBase.forward()`（py_flashinfer_mha.py:497-525）：

```python
def forward(self, qkv, kv_cache, layer_idx=0):
    if self.need_rope_kv_cache:
        if self.rope_impl is not None:
            query, key, value = self.rope_impl.forward(qkv)  # RoPE
        else:
            query, key, value = self._split_qkv(qkv)
        self.kv_cache_write_op.forward(key, value, kv_cache)  # KV Write
        qkv = self._prepare_fmha_input(query, key, value)     # query only for paged
    common.apply_write_cache_store(...)
    return self.fmha_impl.forward(qkv, kv_cache)
```

`PyFlashinferDecodeImpl.forward()`（py_flashinfer_mha.py:796-812）：

```python
def forward(self, qkv, kv_cache, layer_idx=0):
    if self.need_rope_kv_cache:
        qkv = self.rope_impl.forward(qkv, kv_cache, self.rope_params)  # Fused RoPE+KVCache
    common.apply_write_cache_store(...)
    return self.fmha_impl.forward(qkv, kv_cache, self.fmha_params)
```

## 2. 涉及的文件

| 文件 | 内容 | 当前状态 |
|---|---|---|
| `ascend_prefill.py` | `AscendPrefillImpl` - Prefill attention impl | 需重构 |
| `ascend_decode.py` | `AscendDecodeImpl` - Decode attention impl | 需重构 |
| `ascend_kv_cache_write_op.py` | `AscendKVCacheWriteOp` - KV cache write op | **已存在，未使用** |
| `ascend_rope_emb.py` | `AscendRotaryEmbeddingOp` - RoPE op | **已存在，未使用；缺少 `forward()` 实现，不可实例化** |
| `ascend_rope.py` | `apply_rope_pos_ids_nhd` - 底层 RoPE 实现 | 已存在 |
| `ascend_attn_params.py` | `AscendAttnParams` / `build_ascend_params` | 辅助工具类 |
| `torch_sdpa.py` | `AscendSDPAPrefillImpl` / `AscendSDPADecodeImpl` | SDPA 回退实现 |

## 3. 适配方案

### 3.1 方案概述

参考 CUDA `PyFlashinferPrefillImplBase` 架构，为 Ascend 实现：

1. **新建 `ascend_prefill_base.py`** — 公共 Prefill 基类 `AscendPrefillImplBase`
2. **重构 `ascend_prefill.py`** — 继承基类，仅提供 NPU op 实现
3. **重构 `ascend_decode.py`** — 参考 CUDA `PyFlashinferDecodeImpl` 模式
4. **修复 `ascend_kv_cache_write_op.py`** — 对齐接口
5. **修复 `ascend_rope_emb.py`** — 补齐 `forward()` 方法
6. **注册关系保持不变** — `__init__.py` 的注册代码不动

### 3.2 详细架构

```
FMHAImplBase (ABC)
├── AscendPrefillImplBase  (NEW, 增加 RoPE + KVCacheWrite 流程)  
│   ├── AscendPrefillImpl  (已有, 重构 forward)
│   └── AscendSDPAPrefillImpl (已有, 不改)
│
└── AscendDecodeImpl  (已有, 直接重构)
```

#### 3.2.1 AscendPrefillImplBase（新建）

位置：`ascend_impl/ascend_prefill_base.py`

职责：
- 检查 `need_rope_kv_cache`
- 创建 `AscendRotaryEmbeddingOp` + `AscendKVCacheWriteOp`
- `create_params()` — 创建 `FlashInferMlaAttnParams`，填充 params（获取 `slot_mapping`、`positions_d`）
- `forward()` 标准流程：RoPE → KVCacheWrite → write_cache_store → fmha_impl.forward
- 子类只需提供 `_create_fmha_impl()` 和 `_prepare_fmha_input()`

```python
class AscendPrefillImplBase(FMHAImplBase):
    def __init__(self, attn_configs, attn_inputs, weights, 
                 cos_sin_cache=None, fmha_config=None, 
                 parallelism_config=None, **kwargs):
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        
        self.fmha_impl = self._create_fmha_impl(attn_configs, attn_inputs)
        self.rope_impl = self._create_rope_impl(attn_configs, cos_sin_cache)
        self.kv_cache_write_op = AscendKVCacheWriteOp(
            num_kv_heads=attn_configs.kv_head_num,
            head_size=attn_configs.size_per_head,
            token_per_block=attn_inputs.kv_cache.seq_size_per_block,
        )
        self.create_params(attn_inputs)
        self.fmha_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    def _create_rope_impl(self, attn_configs, cos_sin_cache):
        from rtp_llm.ops import RopeStyle
        if attn_configs.rope_config.style == RopeStyle.No:
            return None
        return AscendRotaryEmbeddingOp(
            head_size=attn_configs.size_per_head,
            cos_sin_cache=cos_sin_cache,
            token_per_block=attn_inputs.kv_cache.seq_size_per_block,
            is_neox_style=False,
            rope_config=attn_configs.rope_config,
            max_position_embeddings=attn_configs.max_seq_len,
        )

    def create_params(self, attn_inputs):
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
            attn_inputs.kv_cache.seq_size_per_block,
        )
        self.fmha_impl.set_params(self.fmha_params)
        if self.rope_impl is not None:
            self.rope_impl.set_params(self.fmha_params)
        self.kv_cache_write_op.set_params(self.fmha_params)

    def forward(self, qkv, kv_cache, layer_idx=0):
        if self.need_rope_kv_cache:
            if self.rope_impl is not None:
                query, key, value = self.rope_impl.forward(qkv)
            else:
                query, key, value = self._split_qkv(qkv)
            self.kv_cache_write_op.forward(key, value, kv_cache)
            qkv = self._prepare_fmha_input(query, key, value)
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(qkv, kv_cache)
```

#### 3.2.2 重构 AscendPrefillImpl（ascend_prefill.py）

当前 NPU prefill 使用的是 `torch_npu.npu_fused_infer_attention_score`，它直接接受 `key=k_cache, value=v_cache`（读 cache）。

```python
class AscendPrefillImpl(AscendPrefillImplBase):
    def _create_fmha_impl(self, attn_configs, attn_inputs):
        return AscendPrefillAttnOp(attn_configs, attn_inputs)
    
    def _prepare_fmha_input(self, query, key, value):
        return query  # Paged layout: only query → fmha
    
    @staticmethod
    def support(attn_configs, attn_inputs):
        return attn_inputs.is_prefill and \
               not attn_configs.use_mla and \
               attn_inputs.kv_cache is not None and \
               attn_inputs.kv_cache.separate_kv_cache

class AscendPrefillAttnOp:
    """封装 NPU prefill attention op，只读 cache"""
    def __init__(self, attn_configs, attn_inputs):
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.scale = attn_configs.scale or self.head_dim ** -0.5
        self.page_size = attn_inputs.kv_cache.seq_size_per_block
        self.block_table = None
        self.actual_seq_q = None
        self.actual_seq_kv = None

    def set_params(self, params):
        self.params = params

    def prepare(self, attn_inputs):
        self.block_table = attn_inputs.kv_cache_block_id_host
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

    def forward(self, q, kv_cache):
        k_cache = kv_cache.k_cache_base
        v_cache = kv_cache.v_cache_base
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=q, key=k_cache, value=v_cache,
            block_table=self.block_table,
            input_layout="TND",
            block_size=self.page_size,
            actual_seq_lengths=self.actual_seq_q,
            actual_seq_lengths_kv=self.actual_seq_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        return attn_output
```

#### 3.2.3 重构 AscendDecodeImpl（ascend_decode.py）

参考 CUDA `PyFlashinferDecodeImpl`，decode 步骤：RoPE → KVCacheWrite → write_cache_store → paged_attention

```python
class AscendDecodeImpl(FMHAImplBase):
    def __init__(self, attn_configs, attn_inputs, weights,
                 cos_sin_cache=None, fmha_config=None,
                 parallelism_config=None, **kwargs):
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        
        # --- Decode AttnOp (复用当前的 forward 逻辑) ---
        self.fmha_impl = AscendDecodeAttnOp(attn_configs, attn_inputs)
        self.rope_impl = self._create_rope_impl(attn_configs, cos_sin_cache)
        self.kv_cache_write_op = AscendKVCacheWriteOp(
            num_kv_heads=attn_configs.kv_head_num,
            head_size=attn_configs.size_per_head,
            token_per_block=attn_inputs.kv_cache.seq_size_per_block,
        )
        self.create_params(attn_inputs)
        self.fmha_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    def _create_rope_impl(self, attn_configs, cos_sin_cache):
        from rtp_llm.ops import RopeStyle
        if attn_configs.rope_config.style == RopeStyle.No:
            return None
        return AscendRotaryEmbeddingOp(
            head_size=attn_configs.size_per_head,
            cos_sin_cache=cos_sin_cache,
            token_per_block=attn_inputs.kv_cache.seq_size_per_block,
            is_neox_style=False,
            rope_config=attn_configs.rope_config,
            max_position_embeddings=attn_configs.max_seq_len,
        )

    def create_params(self, attn_inputs):
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
            attn_inputs.kv_cache.seq_size_per_block,
        )
        self.fmha_impl.set_params(self.fmha_params)
        if self.rope_impl is not None:
            self.rope_impl.set_params(self.fmha_params)
        self.kv_cache_write_op.set_params(self.fmha_params)

    def forward(self, qkv, kv_cache, layer_idx=0):
        if self.need_rope_kv_cache:
            if self.rope_impl is not None:
                query, key, value = self.rope_impl.forward(qkv)
            else:
                _, key, value = self._split_qkv(qkv)
            self.kv_cache_write_op.forward(key, value, kv_cache)
            q = query if self.rope_impl is not None else qkv.chunk(3, dim=-1)[0]
        else:
            q = qkv.chunk(3, dim=-1)[0]
        
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(q, kv_cache)

class AscendDecodeAttnOp:
    def __init__(self, attn_configs, attn_inputs):
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.scale = attn_configs.scale or self.head_dim ** -0.5
        self.block_table = None
        self.context_lens = None

    def set_params(self, params):
        self.params = params

    def prepare(self, attn_inputs):
        self.block_table = attn_inputs.kv_cache_block_id_host
        self.context_lens = attn_inputs.prefix_lengths + attn_inputs.input_lengths

    def forward(self, q, kv_cache):
        output = torch.empty_like(q)
        torch_npu._npu_paged_attention(
            query=q,
            key_cache=kv_cache.k_cache_base,
            value_cache=kv_cache.v_cache_base,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=self.block_table,
            context_lens=self.context_lens,
            out=output,
        )
        return output
```

#### 3.2.4 修复 AscendKVCacheWriteOp

当前问题：
- `slot_mapping` 从 `self.params` 获取，需确认 dtype（`_npu_reshape_and_cache` 需要 int32）
- `set_params()` 与 `forward()` 接口分离（沿用 CUDA 模式）

```python
class AscendKVCacheWriteOp:
    def __init__(self, num_kv_heads, head_size, token_per_block):
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.token_per_block = token_per_block
        self.params = None

    def set_params(self, params):
        self.params = params

    def forward(self, key, value, kv_cache):
        if kv_cache is None:
            return
        slot_mapping = self.params.slot_mapping
        # 确保 int32 类型
        if slot_mapping.dtype != torch.int32:
            slot_mapping = slot_mapping.to(torch.int32)
        torch_npu._npu_reshape_and_cache(
            key=key,
            value=value,
            key_cache=kv_cache.k_cache_base,
            value_cache=kv_cache.v_cache_base,
            slot_indices=slot_mapping,
        )
```

#### 3.2.5 修复 AscendRotaryEmbeddingOp

当前 `ascend_rope_emb.py` 的 `AscendRotaryEmbeddingOp` **缺少 `forward()` 方法**，继承自 `BaseRotaryEmbeddingOp` 的抽象方法，无法实例化。需补齐 `forward()`：

```python
class AscendRotaryEmbeddingOp(BaseRotaryEmbeddingOp):
    def __init__(self, head_size, cos_sin_cache, token_per_block, is_neox_style,
                 rope_config=None, max_position_embeddings=32768):
        super().__init__(head_size, cos_sin_cache, token_per_block, is_neox_style,
                         rope_config, max_position_embeddings)
        self.num_heads = None  # 由 set_params 或调用方设置
        self.num_kv_heads = None
        self.params = None

    def set_params(self, params):
        self.params = params

    def set_head_info(self, num_heads, num_kv_heads):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

    def forward(self, qkv):
        """Apply RoPE to QKV.
        
        Args:
            qkv: [total_tokens, (num_heads + 2*num_kv_heads) * head_dim]
            
        Returns:
            (query, key, value) after RoPE, each [total_tokens, num_heads, head_dim]
        """
        qkv = qkv.reshape(qkv.shape[0], -1)
        q, k, v = torch.split(qkv, [
            self.head_size * self.num_heads,
            self.head_size * self.num_kv_heads,
            self.head_size * self.num_kv_heads,
        ], dim=-1)
        query = q.reshape(q.shape[0], self.num_heads, self.head_size)
        key = k.reshape(k.shape[0], self.num_kv_heads, self.head_size)
        value = v.reshape(v.shape[0], self.num_kv_heads, self.head_size)
        
        self._apply_rope(query, key, self.params)
        return query, key, value

    def _apply_rope(self, query, key, rope_params):
        if self.cos_sin_cache is not None:
            apply_rope_pos_ids_nhd(
                query, key, self.cos_sin_cache,
                rope_params.positions_d,
                is_neox_style=self.is_neox_style,
            )
        else:
            raise RuntimeError("AscendRotaryEmbeddingOp requires cos_sin_cache")
```

注意：`forward()` 需要知道 `num_heads` / `num_kv_heads` 才能 split qkv。可在创建时通过 `set_head_info()` 传递，或在 `__init__` 中直接传入 `attn_configs`（更简洁，类似 `MhaRotaryEmbeddingOp` 的做法）。

**推荐方案**：让 `AscendRotaryEmbeddingOp.__init__` 直接接收 `attn_configs`（类似 `MhaRotaryEmbeddingOp`），内部提取 `head_size`, `num_heads`, `num_kv_heads` 等：

```python
class AscendRotaryEmbeddingOp(BaseRotaryEmbeddingOp):
    def __init__(self, attn_configs, cos_sin_cache=None):
        super().__init__(
            attn_configs.size_per_head,
            cos_sin_cache,
            attn_configs.kernel_tokens_per_block,
            is_neox_style=False,
            rope_config=attn_configs.rope_config,
            max_position_embeddings=attn_configs.max_seq_len + attn_configs.gen_num_per_cycle + 1,
        )
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.params = None

    def set_params(self, params):
        self.params = params

    def forward(self, qkv):
        qkv = qkv.reshape(qkv.shape[0], -1)
        q, k, v = torch.split(qkv, [
            self.head_size * self.num_heads,
            self.head_size * self.num_kv_heads,
            self.head_size * self.num_kv_heads,
        ], dim=-1)
        query = q.reshape(q.shape[0], self.num_heads, self.head_size)
        key = k.reshape(k.shape[0], self.num_kv_heads, self.head_size)
        value = v.reshape(v.shape[0], self.num_kv_heads, self.head_size)
        self._apply_rope(query, key, self.params)
        return query, key, value

    def _apply_rope(self, query, key, rope_params):
        if self.cos_sin_cache is not None:
            apply_rope_pos_ids_nhd(query, key, self.cos_sin_cache,
                                   rope_params.positions_d,
                                   is_neox_style=self.is_neox_style)
        else:
            raise RuntimeError("AscendRotaryEmbeddingOp requires cos_sin_cache")
```

### 3.3 slot_mapping 的获取

`slot_mapping` 来自 `FlashInferMlaAttnParams` C++ 对象，通过 `fill_params()` 填充：
```python
self.fmha_params.fill_params(
    prefix_lengths, sequence_lengths, input_lengths,
    kv_cache_block_id_host, seq_size_per_block
)
# 之后访问:
slot_mapping = self.fmha_params.slot_mapping  # [num_tokens], dtype=int64
```

注意：
- `fill_params()` 内部通过 C++ `FlashInferMlaParams.cc:478-485` 计算 slot_mapping，逻辑为 `block_number * page_size + block_offset`，**设备无关的 CPU 计算**
- 输出 dtype 为 `int64`（C++ `static_cast<int64_t>`），但 `_npu_reshape_and_cache` 需要 `int32`
- 需在 `AscendKVCacheWriteOp.forward()` 中 `slot_mapping.to(torch.int32)`

**确认**：`fill_params()` 是否在 Ascend 上正常工作？它内部使用 `GET_CURRENT_STREAM()` 宏（已支持 Ascend）和 `cudaMemcpyAsync`（需替换为 `tensor.copy_()`）。当前 `FlashInferMlaParams.cc` 中 `fillFlashInfer()` 的 H2D 拷贝是 MLA 专用路径；对于 MHA，`fill_params()` 只做 CPU 计算 + H2D 拷贝 slot_mapping。如果在 Ascend 编译时 `USING_ASCEND` 未处理 H2D 替换，此函数可能崩溃。

**缓解**：若 `fill_params()` 在 Ascend 上不可用，则退回到 Python 端直接计算 slot_mapping：
```python
block_table = attn_inputs.kv_cache_block_id_host  # [B, max_blocks]
page_size = attn_inputs.kv_cache.seq_size_per_block
batch_indices = ...  # 每个 token 的 batch_id
positions = ...      # 每个 token 的 position
block_number = block_table[batch_id, position // page_size]
slot_mapping = block_number * page_size + position % page_size
```

### 3.4 关键注意事项

#### 3.4.1 依赖关系：C++ KV 分离存储先行

本方案假设：
- `kv_cache.k_cache_base` 和 `kv_cache.v_cache_base` 已作为**独立 NHD layout tensor** 存在（`[blocks, seq_per_block, kv_heads, head_dim]`）
- `kv_cache.separate_kv_cache == True` 已在配置链中正确设置

**这是 C++ 层 KV 分离存储适配工作的输出**，包括：
- `BlockPool` 分离分配 K/V buffer
- `OpDefs::getLayerCache()` NHD reshape
- `CacheConfig.separate_kv_cache` 配置链路闭合
- 详见 `4-kvcache/ascend_kv-cache-adaptation_v0.3.md` 第二节

**当前 `CacheConfig.separate_kv_cache` 配置链未闭合**（review_v0.2.md #4），需要先修复。

#### 3.4.2 `FlashInferMlaAttnParams.fill_params()` Ascend 兼容性

- MHA 路径下，`fill_params()` 主要用于生成 `slot_mapping` 和 `positions_d`（CPU 计算 + H2D 拷贝）
- Ascend 上 H2D 拷贝需使用 `tensor.copy_()` 或 `aclrtMemcpyAsync`（当前 `FlashInferMlaParams.cc:489` 使用 `cudaMemcpyAsync`）
- 确认 Ascend 编译路径下是否已替换，否则需增加 `#if USING_ASCEND` 分支
- **备选方案**：Python 端直接计算 slot_mapping（见 3.3 节）

#### 3.4.3 `AscendRotaryEmbeddingOp` 缺少 `forward()`

当前 `ascend_rope_emb.py` 只定义了 `_apply_rope()` 和 `_prepare_warmup_cache_indices()`，**缺少 `forward()` 方法**，继承自 `BaseRotaryEmbeddingOp`（`@abstractmethod`），无法实例化。需按 3.2.5 节补充。

#### 3.4.4 KVCache 的 NHD layout

| 框架 | Layout | Shape |
|---|---|---|
| rtp-llm 现有 (FlashInfer) | HND | `[pages, kv_heads, page_size, head_dim]` |
| rtp-llm 目标 (Ascend) | NHD | `[pages, page_size, kv_heads, head_dim]` |
| vLLM-Ascend | NHD | `[blocks, block_size, kv_heads, head_dim]` |

C++ `getLayerCache()` 分离路径直接 reshape 为 NHD，Python 端无需 permute。

#### 3.4.5 Block table 格式

| 框架 | 格式 | 值 |
|---|---|---|
| FlashInfer | CSR: `page_indice + page_indptr` | 紧凑 flat，不含 -1 |
| torch_npu | 2D: `[B, max_blocks]` | 含 -1 表示无效 block |

Ascend 路径直接使用 `attn_inputs.kv_cache_block_id_host`（2D int32 tensor）。

**-1 值处理**：需验证 `torch_npu._npu_paged_attention` 和 `npu_fused_infer_attention_score` 对 block_table 中 -1 的行为。vLLM-Ascend 中 block_table 的 -1 通常配合 `context_lens` 确保不会被访问到。如需替换，可在传入前将 -1 替换为 0。

参考：vLLM-Ascend，无效值填0而不是-1。

#### 3.4.6 `need_rope_kv_cache = False`

当模型不需要 RoPE（如 BERT）时，`attn_configs.need_rope_kv_cache = False`，跳过 RoPE 和 KV cache write 步骤，仅执行 attention op。

### 3.5 实现步骤

#### Step 1: 修复 AscendRotaryEmbeddingOp（必做前置）

- 修改 `ascend_rope_emb.py`，添加 `forward()` 方法
- 建议改为接收 `attn_configs` 的构造函数（类似 `MhaRotaryEmbeddingOp`）

#### Step 2: 确认 FlashInferMlaAttnParams.fill_params() Ascend 兼容性

- 检查 `FlashInferMlaParams.cc` 中 H2D 拷贝是否有 `USING_ASCEND` 分支
- 如没有，增加 `#if USING_ASCEND` 或改用 `tensor.copy_()`
- 如不可行，实现 Python 端 slot_mapping 计算作为备选

#### Step 3: 新建 `ascend_prefill_base.py`

- 创建 `AscendPrefillImplBase` 继承 `FMHAImplBase`
- 实现 `__init__`, `create_params`, `_create_rope_impl`, `_split_qkv`, `forward`

#### Step 4: 重构 `ascend_prefill.py`

- `AscendPrefillImpl` 改为继承 `AscendPrefillImplBase`
- 提取 NPU op 为 `AscendPrefillAttnOp` 独立类
- `_prepare_fmha_input` 返回 `query`（paged layout）

#### Step 5: 重构 `ascend_decode.py`

- `AscendDecodeImpl` + `AscendDecodeAttnOp`
- `forward()`: RoPE → KVCacheWrite → write_cache_store → paged_attention

#### Step 6: 修复 `ascend_kv_cache_write_op.py`

- 添加 `slot_mapping.to(torch.int32)` dtype 转换
- 确保 `set_params` / `forward` 接口模式

#### Step 7: 端到端验证

- 需先确保 C++ 层 KV 分离存储配置链闭合（`separate_kv_cache = true`）

### 3.6 验证方案

1. **单元测试**：RoPE 输出 vs FlashInfer（数值误差 < 1e-3）
2. **单元测试**：`AscendKVCacheWriteOp` 写入后读回正确性
3. **集成测试**：用 Qwen3-4B 跑完整 prefill + decode，对比 CUDA 路径输出
4. **Baseline 验证**：先设 `need_rope_kv_cache=False` 跳过 RoPE + Write，验证 attention op 本身正确；再开启完整流程

## 4. 风险与缓解

| 风险 | 缓解 |
|---|---|
| `FlashInferMlaAttnParams.fill_params()` 在 Ascend 上 H2D 拷贝失败 | Python 端直接计算 slot_mapping（CPU 整数运算，设备无关）|
| `AscendRotaryEmbeddingOp` 缺少 `forward()` 导致实例化失败 | Step 1 优先修复，在开始其他工作前完成 |
| C++ 层 `separate_kv_cache` 配置链未闭合，`k_cache_base`/`v_cache_base` 不存在 | 本方案依赖 4-kvcache C++ 层修改，需先完成或并行推进 |
| `_npu_reshape_and_cache` 不接受 int64 slot_mapping | `slot_mapping.to(torch.int32)` 转换 |
| block_table 中包含 -1 导致 NPU 算子崩溃 | 验证行为 + 备选替换为 0 |
| `npu_fused_infer_attention_score` block_table 参数位置/格式不对 | 参考 vLLM-Ascend `attention_v1.py` 确认参数签名 |
| Decode RoPE+KVCacheWrite 两步骤分离引入额外延迟 | 后续考虑融合算子（aclnn 自定义）或 `npu_kv_rmsnorm_rope_cache` |
