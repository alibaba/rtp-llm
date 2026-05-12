# Ascend Attention 适配详细方案 v0.2

## 1. 问题分析

### 当前状态

`AscendPrefillImpl`（ascend_prefill.py）和 `AscendDecodeImpl`（ascend_decode.py）的 `forward()` 方法仅直接从 `kv_cache.k_cache_base` / `v_cache_base` 读取数据，假设 cache 已被外部填充：

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
1. ❌ 没有对 K, V 应用 RoPE（`need_rope_kv_cache` 配置）
2. ❌ 没有将 K, V 写入 KV cache（`_npu_reshape_and_cache`）
3. ❌ 没有处理 `write_cache_store`（cache_store_inputs）
4. ❌ 没有使用 `create_params()` / `prepare()` 模式管理 RoPE/KVCacheWrite 参数

**AscendRotaryEmbeddingOp 当前状态（基于源码确认）：**
- `ascend_rope_emb.py` 的 `AscendRotaryEmbeddingOp` **已实现 `_apply_rope` 和 `_prepare_warmup_cache_indices` 覆盖**，使用纯 PyTorch `apply_rope_pos_ids_nhd()`
- 但**缺少 `forward()` 方法**：`BaseRotaryEmbeddingOp` 声明了 `@abstractmethod forward()`，导致子类无法实例化
- `__init__` 签名与 `BaseRotaryEmbeddingOp` 一致（接收 `head_size, cos_sin_cache, token_per_block, is_neox_style` 等）

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

| 文件 | 路径 | 内容 | 当前状态 |
|---|---|---|---|
| `ascend_prefill.py` | ascend_impl/ | `AscendPrefillImpl` - Prefill attention impl | **需重构，组合 RoPE + KVCacheWrite + AttnOp** |
| `ascend_decode.py` | ascend_impl/ | `AscendDecodeImpl` - Decode attention impl | **需重构，组合 RoPE + KVCacheWrite + AttnOp** |
| `ascend_kv_cache_write_op.py` | ascend_impl/ | `AscendKVCacheWriteOp` - KV cache write op | 已存在，未使用；**缺 slot_mapping int32 转换** |
| `ascend_rope_emb.py` | ascend_impl/ | `AscendRotaryEmbeddingOp` - RoPE op | **缺少 `forward()`，不可实例化** |
| `ascend_rope.py` | ascend_impl/ | `apply_rope_pos_ids_nhd` - 纯 PyTorch RoPE 实现 | 已存在，正常工作 |
| `ascend_attn_params.py` | ascend_impl/ | `AscendAttnParams` / `build_ascend_params` | 辅助工具类 |
| `base_rotary_embedding_op.py` | cuda_impl/ | `BaseRotaryEmbeddingOp` - 基类（含抽象 forward） | 依赖 flashinfer |
| `fmha_impl_base.py` | attention/ | `FMHAImplBase` - MHA 基类（ABC） | 仅定义 forward/support 接口 |
| `FlashInferMlaParams.cc` | bindings/cuda/ | C++ `fill_params()` slot_mapping/positions 计算 | **含 cudaMemcpyAsync，Ascend 不可用** |

## 3. 适配方案

### 3.1 方案概述

采用**组合模式**（而非继承基类），直接在 `AscendPrefillImpl` 和 `AscendDecodeImpl` 中组合 RoPE、KVCacheWrite、AttentionOp 三个组件。不引入额外的中间基类。

1. **重构 `ascend_prefill.py`** — 组合 `AscendRotaryEmbeddingOp` + `AscendKVCacheWriteOp` + `AscendPrefillAttnOp`
2. **重构 `ascend_decode.py`** — 组合 `AscendRotaryEmbeddingOp` + `AscendKVCacheWriteOp` + `AscendDecodeAttnOp`
3. **修复 `ascend_kv_cache_write_op.py`** — 对齐接口 + int32 转换
4. **修复 `ascend_rope_emb.py`** — 补齐 `forward()` 方法
5. **注册关系保持不变** — `__init__.py` 的注册代码不动

不建基类的理由：
- 只有一个实现类（`AscendPrefillImpl`）需要 RoPE + KVCacheWrite 流程，`AscendSDPAPrefillImpl` 不走这个流程
- `AscendPrefillImpl` 和 `AscendDecodeImpl` 的 RoPE/KVCacheWrite 创建逻辑不完全相同（prefill 用 fmha_params, decode 也是），共用的 `_create_rope_impl` 等代码量很少
- 组合比继承更灵活，避免不必要的抽象层次

### 3.2 详细架构

```
FMHAImplBase (ABC)
├── AscendPrefillImpl  (组合 RoPE + KVCacheWrite + AscendPrefillAttnOp)
├── AscendSDPAPrefillImpl (已有, 不改)
└── AscendDecodeImpl   (组合 RoPE + KVCacheWrite + AscendDecodeAttnOp)
```

#### 3.2.1 重构 AscendPrefillImpl（ascend_prefill.py）

RoPE、KVCacheWrite、FMHA op 三组件在 `__init__` 中创建，`forward()` 按序调用：RoPE → KVCacheWrite → write_cache_store → fmha_impl.forward。

```python
class AscendPrefillImpl(FMHAImplBase):
    def __init__(self, attn_configs, attn_inputs, weights,
                 cos_sin_cache=None, fmha_config=None,
                 parallelism_config=None, **kwargs):
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs

        self.fmha_impl = AscendPrefillAttnOp(attn_configs, attn_inputs)
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

#### 3.2.2 重构 AscendDecodeImpl（ascend_decode.py）

参考 CUDA `PyFlashinferDecodeImpl`，decode 步骤：RoPE → KVCacheWrite → write_cache_store → paged_attention

```python
class AscendDecodeImpl(FMHAImplBase):
    def __init__(self, attn_configs, attn_inputs, weights,
                 cos_sin_cache=None, fmha_config=None,
                 parallelism_config=None, **kwargs):
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs

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

#### 3.2.3 修复 AscendKVCacheWriteOp

当前问题：
- `slot_mapping` 从 `self.params` 获取，`fill_params()` 输出 dtype 为 **int64**（C++ `static_cast<int64_t>`），但 `_npu_reshape_and_cache` 需要 **int32**
- 缺失 `slot_mapping.to(torch.int32)` 转换

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
        # _npu_reshape_and_cache requires int32
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

#### 3.2.4 修复 AscendRotaryEmbeddingOp

**源码现状**（ascend_rope_emb.py）：
- 已覆盖 `_apply_rope()` — 使用 `apply_rope_pos_ids_nhd()`（纯 PyTorch 实现，无 flashinfer 依赖）
- 已覆盖 `_prepare_warmup_cache_indices()` — 纯 PyTorch 实现
- **缺少 `forward()` 方法** — `BaseRotaryEmbeddingOp` 声明 `@abstractmethod forward()`，导致无法实例化

`forward()` 需实现的功能：
1. 接收 `qkv` 张量 `[total_tokens, (num_heads + 2*num_kv_heads) * head_dim]`
2. Split 为 query, key, value 三个张量，reshape 为 NHD layout
3. 调用 `self._apply_rope(query, key, self.params)` 应用 RoPE
4. 返回 `(query, key, value)` 三元组

由于 `AscendRotaryEmbeddingOp.__init__` 不接收 num_heads/num_kv_heads，需要补充：
- **方案 A**：修改 `__init__` 签名，增加 `num_heads` 和 `num_kv_heads` 参数
- **方案 B**：通过 `set_head_info(num_heads, num_kv_heads)` 方法设置
- **方案 C**：修改 `__init__` 直接接收 `attn_configs`（类似 `MhaRotaryEmbeddingOp`）

**推荐方案 C**（更简洁，无需额外的 setter）：

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
```

**备选方案 A**（保持 `__init__` 签名不变，加 setter）：

若希望不破坏现有 `__init__` 兼容性，用 setter 注入 head info：

```python
class AscendRotaryEmbeddingOp(BaseRotaryEmbeddingOp):
    def __init__(self, head_size, cos_sin_cache, token_per_block, is_neox_style,
                 rope_config=None, max_position_embeddings=32768):
        super().__init__(head_size, cos_sin_cache, token_per_block, is_neox_style,
                         rope_config, max_position_embeddings)
        self.num_heads = None
        self.num_kv_heads = None
        self.params = None

    def set_params(self, params):
        self.params = params

    def set_head_info(self, num_heads, num_kv_heads):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

    def forward(self, qkv):
        assert self.num_heads is not None, "set_head_info() must be called before forward()"
        ...
```

`AscendPrefillImpl` 和 `AscendDecodeImpl` 中创建 `AscendRotaryEmbeddingOp` 后需调用 `set_head_info()`：

```python
self.rope_impl = AscendRotaryEmbeddingOp(...)
self.rope_impl.set_head_info(attn_configs.head_num, attn_configs.kv_head_num)
```

**推荐方案 C（attn_configs）**，最简洁。

### 3.3 slot_mapping 的获取（参考 ROCm 处理方式）

`slot_mapping` 来自 `FlashInferMlaAttnParams` C++ 对象，通过 `fill_params()` 填充：

```python
self.fmha_params.fill_params(
    prefix_lengths, sequence_lengths, input_lengths,
    kv_cache_block_id_host, seq_size_per_block
)
slot_mapping = self.fmha_params.slot_mapping  # [num_tokens], dtype=int64
```

**fill_params() 在 Ascend 上的问题（来自 FlashInferMlaParams.cc 源码分析）：**

`fill_params()` 内部（FlashInferMlaParams.cc:468-496）：
1. slot_mapping 计算在 **CPU 上完成**（纯整数运算，设备无关）：
   ```cpp
   slot_mapping_ptr[i] = static_cast<int64_t>(block_number) * seq_size_per_block + block_offset;
   ```
2. **H2D 拷贝使用 `cudaMemcpyAsync`**（FlashInferMlaParams.cc:489），Ascend 上不支持
3. `GET_CURRENT_STREAM()` 宏的 Ascend 兼容性取决于编译配置

**ROCm 平台的处理方式（参考）：**

ROCm 使用 `FusedRopeKVCachePrefillOpBase`（fused_rope_kvcache_op.py），**完全不依赖外部 slot_mapping**：

```
ROCm fused kernel 路径：
  prepare():
    kv_cache_offset = convert_offset_to_block_array(
        attn_inputs.kv_cache_kernel_block_id_device
    )
    # 原始 block table 传给 kernel，内部处理偏移计算
    # 无 slot_mapping 参与

  forward():
    prefill_fused_rope_kvcache(qkv, kv_cache, kv_cache_offset, ...)
    # fused kernel 内部处理 RoPE + 写 cache + 返回 Q
```

核心差异：
| 方案 | slot_mapping | 流程 |
|---|---|---|
| CUDA FlashInfer | 外部计算 (`fill_params`) | RoPE → KVCacheWrite(需 slot_mapping) → FMHA |
| ROCm aiter | **不需要**，kernel 内部处理 | FusedRoPEKVCache(含 RoPE + Write) → FMHA |
| Ascend 当前方案 | 外部计算（Python 备选） | RoPE → KVCacheWrite(需 slot_mapping) → FMHA |

**对 Ascend 方案的启示**：
- ROCm 证明了一条替代路径：fused kernel 可以从外部消除 slot_mapping 依赖
- 当前 Ascend 方案走拆分路径（RoPE + `_npu_reshape_and_cache`），slot_mapping 仍然必要
- 后续可参考 ROCm 实现一个 Ascend 版 fused kernel（基于 aclnn/Triton），彻底消除 slot_mapping 依赖

**备选方案（`fill_params` 不可用时）**：

Python 端直接计算 slot_mapping（设备无关，纯 CPU 整数运算）：

```python
def compute_slot_mapping(block_table, positions, batch_ids, page_size):
    max_blocks = block_table.shape[1]
    block_index = positions // page_size
    block_offset = positions % page_size
    block_number = block_table[batch_ids, block_index]
    return block_number * page_size + block_offset
```

与 C++ `fill_params` 的计算公式完全一致：
```cpp
slot_mapping_ptr[i] = static_cast<int64_t>(block_number) * seq_size_per_block + block_offset;
```

### 3.4 关键注意事项

#### 3.4.1 `apply_rope_pos_ids_nhd` 的实现细节

当前实现（ascend_rope.py）是**纯 PyTorch 实现**，核心逻辑：
1. 用 `pos_ids` 索引 `cos_sin_cache` 获取 `cos` / `sin`
2. 对 q/k 的 `[..., :rope_dim]` 部分应用旋转
3. 支持 `is_neox_style`（interleave）和默认（half-split）两种模式
4. **in-place 修改** q/k 张量

对应 vLLM-Ascend 的做法：
- vLLM-Ascend 使用 `torch_npu._npu_rotary_embedding()`（CANN 内置算子）或 Triton 路径
- 签名：`torch_npu._npu_rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox_style)`
- 是 in-place 操作（直接修改 query/key，无返回值）

rtp-llm Ascend 路径当前使用纯 PyTorch 实现，与 torch_npu CANN 算子等价，不需要额外适配。后续可考虑替换为 `torch_npu._npu_rotary_embedding` 以获得潜在性能提升。

#### 3.4.2 KVCache 的 NHD layout

| 框架 | Layout | Shape |
|---|---|---|
| rtp-llm 现有 (FlashInfer) | HND | `[pages, kv_heads, page_size, head_dim]` |
| rtp-llm 目标 (Ascend) | NHD | `[pages, page_size, kv_heads, head_dim]` |
| vLLM-Ascend | NHD | `[blocks, block_size, kv_heads, head_dim]` |

C++ `getLayerCache()` 分离路径直接 reshape 为 NHD，Python 端无需 permute。

#### 3.4.3 Block table 格式

| 框架 | 格式 | Shape | dtype | 无效值 |
|---|---|---|---|---|
| FlashInfer | CSR `page_indice + page_indptr` | 紧凑 flat | int32 | 无 |
| torch_npu | 2D `[B, max_blocks]` | `[B, max_blocks]` | int32 | 0（来自 vLLM-Ascend 确认）|
| rtp-llm (attn_inputs) | 2D `kv_cache_block_id_host` | `[B, max_blocks]` | int32 | 可能含 -1 |

**无效值处理**（参考 vLLM-Ascend）：
- vLLM-Ascend block_table 无效值填 **0 而不是 -1**（model_runner_v1.py:2102: `blk_table_tensor.fill_(0)`）
- block 0 是合法的物理 block，但 FIA 算子根据 `actual_seq_lengths_kv` / `context_lens` 限制实际访问范围
- 若 rtp-llm 的 block_table 含 -1，需在传入 attention op 前替换为 0，确保与 CANN 算子兼容

#### 3.4.4 `need_rope_kv_cache = False`

当模型不需要 RoPE（如 BERT）时，`attn_configs.need_rope_kv_cache = False`，跳过 RoPE 和 KV cache write 步骤，仅执行 attention op。

#### 3.4.5 依赖关系：C++ KV 分离存储先行

本方案假设：
- `kv_cache.k_cache_base` 和 `kv_cache.v_cache_base` 已作为**独立 NHD layout tensor** 存在（`[blocks, seq_per_block, kv_heads, head_dim]`）
- `kv_cache.separate_kv_cache == True` 已在配置链中正确设置

这是 C++ 层 KV 分离存储适配工作的输出，包括：
- `BlockPool` 分离分配 K/V buffer
- `OpDefs::getLayerCache()` NHD reshape
- `CacheConfig.separate_kv_cache` 配置链路闭合
- 详见 `4-kvcache/ascend_kv-cache-adaptation_v0.3.md` 第二节

**当前 `CacheConfig.separate_kv_cache` 配置链未闭合**（review_v0.2.md #4），需要先修复。

#### 3.4.6 `_npu_reshape_and_cache` 的 slot_mapping 类型要求

| 属性 | 值 |
|---|---|
| `fill_params()` 输出 dtype | int64（C++ `static_cast<int64_t>`） |
| `_npu_reshape_and_cache` 要求 | **int32** |
| 转换方式 | `slot_mapping.to(torch.int32)` |

#### 3.4.7 `AscendRotaryEmbeddingOp` 创建时的 token_per_block 参数

`BaseRotaryEmbeddingOp.__init__` 接收 `token_per_block`，在 `_prepare_warmup_cache_indices` 中使用。
- CUDA 路径：`attn_configs.kernel_tokens_per_block`
- Ascend 路径：`attn_inputs.kv_cache.seq_size_per_block`

在 `AscendPrefillImpl` 和 `AscendDecodeImpl` 中创建时使用 `attn_inputs.kv_cache.seq_size_per_block`。

### 3.5 实现步骤

#### Step 1: 修复 AscendRotaryEmbeddingOp（必做前置）

- 修改 `ascend_rope_emb.py`，添加 `forward()` 方法
- 推荐方案 C：修改 `__init__` 为接收 `attn_configs`（类似 `MhaRotaryEmbeddingOp`）
- 备选方案 A：保持 `__init__` 不变，加 `set_head_info()` 方法

#### Step 2: 确认 FlashInferMlaAttnParams.fill_params() Ascend 兼容性

- `FlashInferMlaParams.cc:489` 使用 `cudaMemcpyAsync`，Ascend 上不可用
- Slot mapping 计算逻辑（.cc:478-484）是纯 CPU 整数运算，设备无关
- **任务**：
  1. 检查 Ascend 编译路径下是否有 `#if USING_ASCEND` 分支处理 H2D 拷贝
  2. 如没有，增加 `#if USING_ASCEND` 分支改用 `tensor.copy_()`
  3. 或实现 Python 端 slot_mapping 计算作为备选（3.3 节）
  4. `positions_d` 也是通过 `fill_params()` 计算，确认在 Ascend 上是否可用

#### Step 3: 重构 `ascend_prefill.py`

- `AscendPrefillImpl` 组合 `AscendRotaryEmbeddingOp`、`AscendKVCacheWriteOp`、`AscendPrefillAttnOp`
- `forward()` 按序执行：RoPE → KVCacheWrite → write_cache_store → fmha_impl.forward
- 提取 NPU attention op 为内部组件 `AscendPrefillAttnOp`

#### Step 4: 重构 `ascend_decode.py`

- `AscendDecodeImpl` + `AscendDecodeAttnOp`
- `forward()`: RoPE → KVCacheWrite → write_cache_store → paged_attention

#### Step 5: 修复 `ascend_kv_cache_write_op.py`

- 添加 `slot_mapping.to(torch.int32)` dtype 转换
- 确保 `set_params` / `forward` 接口模式

#### Step 6: Block table -1 值替换

- 在 `AscendPrefillAttnOp.prepare()` 和 `AscendDecodeAttnOp.prepare()` 中
- 若 block_table 含 -1，替换为 0

#### Step 7: 端到端验证

- 需先确保 C++ 层 KV 分离存储配置链闭合（`separate_kv_cache = true`）

### 3.6 验证方案

1. **单元测试**：RoPE 输出 vs CUDA FlashInfer（数值误差 < 1e-3）— 使用 `apply_rope_pos_ids_nhd` 与 `flashinfer.rope._apply_rope_pos_ids_cos_sin_cache` 对比
2. **单元测试**：`AscendKVCacheWriteOp` 写入后读回正确性
3. **集成测试**：用 Qwen3-4B 跑完整 prefill + decode，对比 CUDA 路径输出
4. **Baseline 验证**：先设 `need_rope_kv_cache=False` 跳过 RoPE + Write，验证 attention op 本身正确；再开启完整流程

## 4. 风险与缓解

| 风险 | 缓解 |
|---|---|
| `FlashInferMlaAttnParams.fill_params()` 在 Ascend 上 H2D 拷贝失败（`cudaMemcpyAsync` `.cc:489`） | Python 端直接计算 slot_mapping（CPU 整数运算，设备无关）|
| `AscendRotaryEmbeddingOp` 缺少 `forward()` 导致实例化失败 | Step 1 优先修复，在开始其他工作前完成 |
| C++ 层 `separate_kv_cache` 配置链未闭合，`k_cache_base`/`v_cache_base` 不存在 | 本方案依赖 4-kvcache C++ 层修改，需先完成或并行推进 |
| `_npu_reshape_and_cache` 不接受 int64 slot_mapping | `slot_mapping.to(torch.int32)` 转换 |
| block_table 中包含 -1 导致 NPU 算子崩溃 | 参考 vLLM-Ascend，替换 -1 为 0 |
| `npu_fused_infer_attention_score` block_table 参数位置/格式不对 | 参考 vLLM-Ascend `attention_v1.py` 确认参数签名 |
| `AscendRotaryEmbeddingOp.__init__` 不接收 `num_heads`/`num_kv_heads`，无法 split qkv | 方案 C: 改为接收 `attn_configs`；方案 A: 增加 `set_head_info()` |
| Decode RoPE+KVCacheWrite 两步骤分离引入额外延迟 | 后续考虑融合算子（aclnn 自定义）或 `npu_kv_rmsnorm_rope_cache` |
| `_npu_rotary_embedding` CANN 算子 vs 纯 PyTorch `apply_rope_pos_ids_nhd` 数值差异 | 当前用纯 PyTorch 实现，数值等价于 CuDA 参考。后续可替换为 CANN 算子以获得性能提升 |

## 5. 参考文档

| 文档 | 内容 |
|---|---|
| `ref_vLLM-Ascend-rope.md` | vLLM-Ascend RoPE 实现：`torch_npu._npu_rotary_embedding`（CANN）和 Triton 路径 |
| `ref_vLLM-Ascend-block_table Shape and Invalid Values.md` | block_table shape `[max_num_reqs, max_blocks]`，无效值填 0 而非 -1 |
| `ref_vLLM-Ascend-embedding.md` | vLLM-Ascend 算子注册和调用的完整链路 |
| `ascend_rope.py` | `apply_rope_pos_ids_nhd` 纯 PyTorch RoPE 实现 |
| `ascend_rope_emb.py` | `AscendRotaryEmbeddingOp` 当前状态（缺 forward） |
| `ascend_kv_cache_write_op.py` | `AscendKVCacheWriteOp` 当前状态（缺 int32 转换） |
| `ascend_prefill.py` | `AscendPrefillImpl` 当前实现（缺 RoPE + KV Write） |
| `ascend_decode.py` | `AscendDecodeImpl` 当前实现（缺 RoPE + KV Write） |
| `base_rotary_embedding_op.py` | `BaseRotaryEmbeddingOp` 抽象基类（含 `@abstractmethod forward`） |
| `FlashInferMlaParams.cc` | C++ `fill_params()` 实现（含 `cudaMemcpyAsync` H2D 拷贝） |
| `ascend_kv-cache-adaptation_v0.3.md` | C++ 层 KV 分离存储适配方案 |
