# Qwen3-Next/Qwen3.5 Hybrid 全链路 + KV cache 复用与 fp32→bf16 NaN 风险审计

本文档把 Qwen3-Next（HybridAttentionType: LINEAR + FULL）从 **请求接入 → C++ 调度
→ Python forward → 每一层 attention dispatch → KV cache 物理布局 → free/realloc
→ PD 分离传输** 的全链路逐段对齐，重点把 **LINEAR fp32 SSM/conv state** 与
**FULL bf16/fp8 K/V** 共享同一份 BlockPool 字节时所有可能 fp32→bf16 错位的交点
全部排查一遍，每一处都标注是否还有残留 NaN 风险。

> 阅读前提：commit `3d9945df8` 已合入 `feat/qwen_36_online`。本文是对该修复的
> 完整背景与覆盖性检验。

---

## 0. 一图概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         一次请求的完整路径                                │
└─────────────────────────────────────────────────────────────────────────┘

HTTP → frontend (uvicorn) → backend RPC → NormalEngine::step()
  │
  ├─ Scheduler::schedule()              FIFOScheduler 驱动 GenerateStateMachine
  │  │
  │  ├─ moveToNext() → handle{Waiting,Running,Loading,Finished}
  │  │   └─ 若 stream 完结 → releaseResource → tryReleaseKVBlock
  │  │       │
  │  │       ├─ insertIntoCache (BlockCache, blockCacheRef++)
  │  │       ├─ storeCacheAsync (memory/remote 二级 cache, connectorRef++)
  │  │       ├─ evictDeviceCacheToMemory (可选 tiered)
  │  │       └─ cache_manager->free
  │  │           └─ HybridTypeKVCacheAllocator::free
  │  │               ├─ LinearKVCacheGroup::free  → ★ zeroLinearWriteRegion + requestFree
  │  │               └─ FullKVCacheGroup::free    → requestFree
  │  │
  │  └─ initKVBlock / incrKVBlock → cache_manager->malloc
  │      └─ HybridTypeKVCacheAllocator::initMalloc / incrMalloc
  │          └─ per-group->malloc → BlockPool::malloc
  │
  ├─ NormalModelInputGatherer::allocateModelInputBuffers
  │  │  分配 model_input.kv_cache_kernel_block_id (torch::zeros) [group, batch, max_kernel_blocks]
  │  └─ copyKvCacheBlocksToModelInput → memcpy 把每个 stream 的真实 block id 写入 row 前缀
  │
  └─ PyWrappedModel.forward(inputs) → Qwen3NextModel.forward
      │
      ├─ build_cp_linear_attn_metadata (CP 模式)
      ├─ prepare_fmha_impl(inputs)        XQAImpl 绑定 group 0 page_table
      │
      └─ for i in range(num_layers):
          ├─ select_block_map_for_layer(attn_inputs, i) ★ 按 gid 切换 page_table
          │   gid = kv_cache_layer_to_group[i]
          │   attn_inputs.kv_cache_kernel_block_id_device =
          │       kv_cache_kernel_block_id_device_by_group[gid]
          │
          └─ Qwen3NextDecoderLayer(i).forward(...)
              │
              ├─ if layer_type == LINEAR:
              │   Qwen3NextGatedDeltaNet.forward
              │     prefill_gdn / decode_gdn / cp_prefill 三种实现
              │
              │     _conv1d:
              │       conv_states (bf16 view, dtype = model_config.data_type) =
              │         LinearCacheConverter.get_conv_state_tensor(kv_cache_base)
              │       causal_conv1d_fn / causal_conv1d_update
              │         读 conv_states[block_map[batch,i]] 的 conv_state 字段
              │         写 同位置（写新 conv state，存入 bf16 字节）
              │       注：conv_state dtype 跟随激活类型，不是 fp32。
              │
              │     _fla:
              │       ssm_states (fp32 view) = LinearCacheConverter.get_ssm_state_tensor(kv_cache_base)
              │       load_initial_state_from_block_map
              │         读 ssm_states[block_map[batch, last_block_offset]] → initial_states (fp32)
              │       chunk_gated_delta_rule / fused_recurrent_gated_delta_rule
              │         以 initial_states 为起点跑 SSM 递推
              │       store_ssm_state_to_block_map
              │         写 final_state（fp32）→ ssm_states[block_map[batch, last_block_offset]]
              │       (decode 路径里 fused_recurrent INPLACE_FINAL_STATE=True，
              │       直接写回 ssm_states)
              │
              │     write_cache_store (cache_store 异步落盘)
              │
              └─ else (FULL ATTN):
                  Qwen3NextAttention.forward → CausalAttention.forward
                    qkv = self.qkv_proj(hidden_states)   bf16/fp8
                    fmha_input = FusedRopeKVCacheDecodeOp.forward(qkv, kv_cache, params)
                      → decode_fused_rope_kvcache(... store_cache=True, store_kv=False)
                        ★ K/V 在此写入 kv_cache.kv_cache_base 的 bf16/fp8 区域
                        通过 page_table=kv_cache_offset 寻址
                    apply_write_cache_store (PD-sep cache_store async)
                    out = XQAWrapper.fmha_impl.forward(fmha_input, kv_cache, fmha_params)
                      → flashinfer.xqa(q, k_cache, v_cache, page_table, seq_lens, ...)
                      ★ XQA 读 K/V (bf16) → softmax → output bf16
```

---

## 1. 模型层布局

`rtp_llm/models/qwen3_next/qwen3_next.py:_parse_hybrid_attention_config`
按 `full_attention_interval` 切：

```python
for i in range(num_layers):
    if (i + 1) % attention_step == 0:
        types.append(HybridAttentionType.NONE)   # FULL attention
    else:
        types.append(HybridAttentionType.LINEAR) # SSM/conv
```

举例 Qwen3.5-MoE，`num_layers=48`，`full_attention_interval=4`：
- LINEAR 层：0,1,2, 4,5,6, 8,9,10, 12,13,14, …, 44,45,46 共 36 层
- FULL  层：3, 7, 11, 15, …, 47                      共 12 层

`HybridConfigCreator::createLayerGroups` (`rtp_llm/cpp/cache/HybridConfigCreator.cc`)：
- `group_layer_num = max(linear_layers_per_group, full_layer_count)`
  对上面例子：full=12, linear=36 → gcd=12, group_layer_num=12
- LINEAR 分成 3 个 group，每个 12 层
- FULL  分成 1 个 group，12 层
- **每个 group 都恰好有 group_layer_num 层 → 每个 group 占满全部物理槽位 [0..11]**

`setupCacheConfigSpecs` 把 group 顺序排列：先所有 FULL groups，再所有 LINEAR groups。
所以 `gid=0` 是 FULL group，`gid=1..3` 是 LINEAR groups。

`setupPhysicalSizes`:
```cpp
config.kv_block_stride_bytes = full_kv_block_stride_bytes;
RTP_LLM_CHECK_WITH_INFO(full_kv_block_stride_bytes >= linear_kv_block_stride_bytes,
                        "not support full attention with padding now");
```
**断言**: `full_block_size_bytes >= linear_block_size_bytes`。

> 实测 Qwen3.5-MoE FP8 + `seq_size_per_block=4096`：
> - LINEAR per block：`ssm_size + conv_size`
>   - ssm = `local_num_v_heads × head_v_dim × head_k_dim × ssm_state_item_size`
>     - 当前线上脚本 `SSM_STATE_DTYPE=fp32` → item_size = 4
>   - conv = `(kernel_dim-1) × qkv_size × conv_state_item_size`
>     - conv_state_dtype = model_config.data_type → 当前 `ACT_TYPE=BF16` → item_size = 2
>   - 合计 ~MB 级
> - FULL  per block：`2 × num_kv_heads × head_dim × tokens_per_block × 1`（fp8）
>   - 4096 tokens/block 下也是 MB 级
>
> 断言能通过。但若 `tokens_per_block` 变小、或 head 数缩小，FULL block 会
> 变得更小，可能违反断言。**脚本固定的 `SEQ_SIZE_PER_BLOCK=4096` 是关键设定。**

---

## 2. BlockPool 物理布局：LINEAR 与 FULL 字节如何共占同一块 row

### 2.1 BlockPool 一次性分配整段连续 buffer

`BlockPool::initializeCacheBuffer` (`rtp_llm/cpp/cache/BlockPool.cc:40`)：

```cpp
cache_aligned_buffer_ = torch::empty({total_size_bytes}, kUInt8.device(kCUDA));
...
cache_aligned_buffer_.zero_();     // ★ 我加的 fix
c10::cuda::getCurrentCUDAStream().synchronize();   // ★
```

`total_size_bytes = sum(layout.kv_block_pool_size_bytes + layout.kv_scale_pool_size_bytes)`
覆盖 main 模型 + 所有 MTP 子模型 layout。

> **重要观察**：原本 `MemoryLayoutStrategy::processKVTensor` 已经在做
> `clearKVTensor(reshaped_tensor)` (即 `fill_(0)`)，理论上也覆盖整块 KV pool。
> 我加这一行的真正价值有两点：
>
> 1. **统一在最上游清零，覆盖所有 layout 拼起来时可能的 padding 字节**
>    （sum 计算如果 hasScale 路径不同会不会留缝隙）。
> 2. **`synchronize()` 保证 zero 已经在 GPU 上落盘**，避免后续如果有任何
>    跨流（cross-stream）拿这片显存做 RDMA register / pin 失败。
>
> 上线生效结合表现，**最关键的是 `synchronize()` 这一步**——保证零化动作真正
> 完成，再被任何下游线程（worker pool / RDMA 注册 / forward kernel）观察到。

### 2.2 hybrid 单一 layout 下的 layer 槽位映射

```
BlockPool layout for hybrid (单一 main layout):
  layer_num = group_layer_num                              (例 12)
  block_num = N                                            (按显存算)
  kv_block_stride_bytes = max(full_block_bytes, linear_block_bytes)  = full

物理 buffer 形如:
  [layer_0][layer_1]...[layer_11]
  每个 layer = block_num × kv_block_stride_bytes

每个 layer 槽位「逻辑共享」于:
  - LINEAR group 0 的第 i 层（layer_ids 在 LINEAR group 0 内的第 i 个）
  - LINEAR group 1 的第 i 层
  - LINEAR group 2 的第 i 层
  - FULL  group   的第 i 层
```

`KVCacheGroup::init` (`rtp_llm/cpp/cache/KVCacheGroup.cc:6`):
```cpp
auto layer_tensors = block_pool_->allLayerCacheBase();
for (int i = 0; i < layer_ids_.size(); ++i) {
    global_layer_to_kv_tensors[layer_ids_[i]] = layer_tensors[i];
}
```
`layer_tensors[i]` = BlockPool 第 i 个物理槽位 tensor。
LINEAR group 0 把全局 layer_id 0/1/2/4/5/6/8/9/10/12/13/14 各自映射到 layer_tensors[0..11]。
LINEAR group 1 把 16/17/18/20/21/22/… 映射到 layer_tensors[0..11]（**复用同样的物理 tensor**）。
FULL  group   把 3/7/11/15/19/23/27/31/35/39/43/47 映射到 layer_tensors[0..11]。

> ⚠ 全局 layer_id 不同的多个层共享一份物理 tensor。但具体某个 block_id 在
> 某一时刻只属于一个 group（由 ref counter 决定）。所以语义上不会冲突；
> 物理上 byte 区间的 *复用* 才是 fp32→bf16 的来源。

### 2.3 单 row 内的 LINEAR vs FULL 字节布局

```
单个物理 row（kv_block_stride_bytes 长，例 ~2 MB）：

┌─────────────────────────────┬──────────────┬─────────────────────────────┐
│  LINEAR ssm_state_size_bytes│ conv_state…  │  …超出 LINEAR 区域的尾部     │
│  (fp32, NaN 风险源)         │ (bf16，跟激活)│  （只有 FULL 会触及）        │
└─────────────────────────────┴──────────────┴─────────────────────────────┘
                              ↑
                              LINEAR 写到这里为止 (linear_block_size_bytes)

注：实际 NaN 来源 *只* 在 ssm_state 这段 fp32 字节。
conv_state dtype = model_config.data_type（当前线上 = bf16），与 FULL K/V 同 dtype，
即使被 FULL 误读也不会因为 dtype 解释错位产生 NaN。
保守起见 zeroLinearWriteRegion 仍清整 row。

┌────────────────────────────────────────────────────────────────────────┐
│  FULL K + V (按 kernel_tokens_per_block × num_kv × head_dim × dtype)   │
│  bf16 / fp8                                                           │
└────────────────────────────────────────────────────────────────────────┘
↑
FULL 写覆盖这整段 (full_block_size_bytes ≥ linear_block_size_bytes)
```

LINEAR view 由 `LinearCacheConverter` 通过 `set_(storage, offset, size, stride)`
**字节级 reinterpret** 出来（`rtp_llm/models_py/utils/typed_storage_view.py`）：
- ssm_state offset=0, dtype = `linear_attention_config.ssm_state_dtype`（线上 fp32）
- conv_state offset=ssm_state_size_bytes, dtype = `model_config.data_type`（线上 bf16）

参见 `rtp_llm/config/model_config.py:843-846`：`conv_state_dtype` 强制等于
`model_config.data_type`，**不受 `kv_cache_config.ssm_state_dtype` 影响**。

LINEAR 的读写都通过这两个 view，访问范围严格在 [0..ssm+conv) 之内。

FULL view 直接通过 `kv_cache.kv_cache_base[:, 0/1, ...]` 访问 K/V，
按 kernel_tokens_per_block 寻址，覆盖整段 row（HND layout）。

**关键不变量**：
- 同一 block_id 在 LINEAR/FULL 间切换归属时，**新归属者必须看到「干净」字节**
  （要么是自己刚写的，要么是 0），否则会读到上一归属者的 ssm_state fp32 字节，
  bf16 重解释概率性 NaN。
- 注：LINEAR 写到 row 里的字节，只有 `[0..ssm_state_size_bytes)` 这段是 fp32
  （会引发 NaN）。`[ssm_state_size_bytes..linear_block_size_bytes)` 这段是
  bf16 conv_state（与 FULL K/V 同 dtype），即使被 FULL 误读也不会 NaN。
- LINEAR write 不覆盖 row 尾部 [linear_size .. full_size)，但 LINEAR read 也
  只读 [0..linear_size)，所以**LINEAR 读自己时不会错位**。
- FULL write 覆盖 [0..full_size)，因为 `full_size ≥ linear_size`，**所以 FULL
  写一遍就把 LINEAR 残留全部用 bf16 覆盖**。但 FULL 写是 *按 token 稀疏写*，
  只写当前 batch 实际写到的 kernel_block 槽位。一个逻辑 block 内未写的
  kernel_block 槽位仍保留旧字节。

---

## 3. 一次请求里 KV cache 的全部 lifecycle

按时序梳理一个 hybrid request 的 KV cache 触点：

### 3.1 initKVBlock（请求开始）

`StreamCacheResource::initKVBlock` → `cache_manager->malloc` →
`HybridTypeKVCacheAllocator::initMalloc`:

1. `initMallocForCommonLen`:
   - `reuseCache(match_keys, kv_resource)` (若开启 device cache)
     - 对每个 FULL group 做前缀匹配
     - 对每个 LINEAR group 做单 key 右→左 join 匹配
     - 复用的 block 通过 `referenceValidBlocks` → `block_pool_->requestReference`
       增加 ref count，从 free pool 摘出
   - 给 batch 0 调 per-group `malloc`
   - 其他 batch reference batch 0 的 common block
2. `incrMalloc`:
   - 给每个 batch 每个 group 递增 malloc
   - 失败时回滚：所有新分到的 block 通过 `zeroLinearGroupBytes` + `requestFree`
     退还（防御性，因为这些 block 可能从一个上游归属者带着 fp32 残留来）

> 注：LINEAR group 的 `malloc` 只在 `linear_step` 倍数位置 + tail 位置实际拿
> block（`rtp_llm/cpp/cache/LinearKVCacheGroup.cc:94-104`），其他位置写
> `NULL_BLOCK_IDX`（即 -1）。所以 LINEAR group 的 `block_indices` 是稀疏的。

### 3.2 forward 期间的 KV cache 读写

每个 forward step（prefill 或 decode）：

`NormalModelInputGatherer::allocateModelInputBuffers`
(`rtp_llm/cpp/normal_engine/NormalModelInputGatherer.cc:184`):

```cpp
model_input.kv_cache_kernel_block_id =
    torch::zeros({groups, total_batch, max_blocks_num × kbpkb}, pinned_i32);
model_input.kv_cache_block_id =
    torch::zeros({groups, total_batch, max_blocks_num}, pinned_i32);
```
**两个 page_table 都用 `torch::zeros` 分配，padding = 0**。

随后 `copyKvCacheBlocksToModelInput`：
- 对每个 stream，把 per-group 的 `kernel_blocks`（int32_t 数组）通过 `memcpy`
  写到对应 batch row 前缀
- LINEAR group 的稀疏槽位 = `NULL_BLOCK_IDX = -1`（来自
  `BlockIds::updateKernelSlotAt`）
- FULL group 的所有槽位都是有效 block id (≥1)
- **稀疏剩下的 row 尾部保持 `0`**

cuda graph 路径同步：`CudaGraphRunner::prepareInputs`
(`rtp_llm/cpp/cuda_graph/cuda_graph_runner.cc:122`)：
```cpp
py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device.fill_(0);
...
for each group g:
    py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device_by_group[g].fill_(0);
    py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host_by_group[g].fill_(0);
```
注释说 *"otherwise it will cause the cache block pollution"*。
所以 cuda graph replay 路径下 padding 也是 0。

`Qwen3NextModel.forward` 主循环：
```python
for i, decoder_layer in enumerate(self.layers):
    select_block_map_for_layer(attention_inputs, i)  # 按 gid 切换 page_table
    hidden_states = decoder_layer(hidden_states, fmha_impl,
                                   kv_cache=self.kv_cache.get_layer_cache(i),
                                   attention_inputs=attention_inputs,
                                   attn_meta=attn_meta)
```

`select_block_map_for_layer` (`rtp_llm/models_py/model_desc/block_map.py`)：
```python
gid = int(attention_inputs.kv_cache_layer_to_group[layer_idx].item())
attention_inputs.kv_cache_kernel_block_id_device =
    attention_inputs.kv_cache_kernel_block_id_device_by_group[gid]
```
**LINEAR 层和 FULL 层用各自 group 的 page_table**。

### 3.3 LINEAR 层 forward（在 layer i, gid = LINEAR group id）

#### 3.3.1 prefill 路径 (`Qwen3NextGatedDeltaNetPrefill.forward`)

```python
kv_cache_tensor = kv_cache.kv_cache_base.reshape(shape[0], -1)   # 该层物理 tensor 的 reshape view
seq_size_per_block = kv_cache.seq_size_per_block

# _conv1d
conv_states = LinearCacheConverter.get_conv_state_tensor(kv_cache_tensor)
              .transpose(1, 2)
                # shape [block_num, qkv_size, kernel_dim - 1], dtype=conv_state_dtype (bf16)
out = causal_conv1d_fn(
    x = mixed_qkv.transpose(0, 1),
    weight = conv_weights,
    conv_states = conv_states,
    query_start_loc = cu_seqlens,
    block_map = attn_inputs.kv_cache_kernel_block_id_device,  # ← LINEAR group 的 page_table
    seq_size_per_block = seq_size_per_block,
    prefix_lengths = attn_inputs.prefix_lengths_d,
    metadata = prefill_conv1d_meta,
).transpose(0, 1)

# _fla
ssm_states = LinearCacheConverter.get_ssm_state_tensor(kv_cache_tensor)
                # shape [block_num, num_v_heads, head_v_dim, head_k_dim], dtype=ssm_state_dtype (fp32)
initial_states = empty(B, H, V, K, dtype=fp32)
load_initial_state_from_block_map(
    prefix_lengths_d, kv_cache_kernel_block_id_device,
    ssm_states, initial_states, seq_size_per_block,
)
   # 内部 kernel:
   # is_zero = (prefix == 0)
   # block_idx = where(is_zero, 0, block_map[batch, (prefix-1)//seq_size_per_block])
   # 注意: prefix==0 时强制读 block 0，并通过 tl.where 把结果换成 zeros
   # block 0 必须为 0 才安全 ★

attn_out, h, final_state = chunk_gated_delta_rule(
    query, key, value, g, beta,
    initial_state = initial_states,
    output_final_state = True,
    cu_seqlens = cu_seqlens,
    use_qk_l2norm_in_kernel = True,
)

store_ssm_state_to_block_map(
    h, final_state, prefix_lengths_d, cu_seqlens,
    kv_cache_kernel_block_id_device, ssm_states,
    seq_size_per_block, chunk_size = 64,
)
   # 内部 kernel:
   # if block_idx <= 0: return     # block 0/-1 都跳过写
```

**LINEAR prefill 的所有读写都经过 fp32 view，对 row 的访问范围严格 ≤ ssm+conv 字节。**

#### 3.3.2 decode 路径 (`Qwen3NextGatedDeltaNetDecode.forward`)

```python
# _conv1d:
conv_states = get_conv_state_tensor(kv_cache_tensor)  # bf16 view (= activation dtype)
out = causal_conv1d_update(
    mixed_qkv, conv_states.transpose(1,2), conv_weights, ...,
    block_map = kv_cache_kernel_block_id_device,
    seq_size_per_block, sequence_lengths_plus_1_d,
)
   # 内部 read：tl.device_assert(read_block_id >= 0)
   # 内部 write：if write_block_id > 0 (tightened from `!= -1` to ban block 0)

# _fla:
ssm_states = get_ssm_state_tensor(kv_cache_tensor)  # fp32 view
core_attn_out, _ = fused_recurrent_gated_delta_rule(
    q, k, v, g, beta,
    initial_state = ssm_states,
    inplace_final_state = True,    ← 直接 inplace 写回 ssm_states 同位置
    block_map = kv_cache_kernel_block_id_device,
    seq_size_per_block, sequence_lengths_plus_1_d, ...
)
   # 内部 kernel (fused_recurrent_gated_delta_rule_fwd_kernel):
   # 读: read_block_id = block_map[batch, last_block_offset]
   #     if read_block_id <= 0: return    ← 不读 block 0/-1
   #     b_h = load(initial_state[block_id]).to(fp32)
   # 写: write_block_id = block_map[batch, current_block_offset]
   #     write_active = (write_block_id > 0)    ← 不写 block 0
   #     safe_block_id = where(write_active, write_block_id, 0)
   #     store(ssm_states[safe_block_id], b_h, mask & write_active)
```

decode 同样只触及 LINEAR fp32 view 的字节范围。

### 3.4 FULL 层 forward（在 layer i, gid = FULL group id = 0）

```python
qkv = self.qkv_proj(hidden_states)   # bf16 / fp8
fmha_input = FusedRopeKVCacheDecodeOp.forward(qkv, kv_cache, rope_params)
   # 内部走 rtp_kernel.fused_rope_kvcache.decode_fused_rope_kvcache(...)
   # 入参: kv_cache=kv_cache.kv_cache_base, kv_cache_offset=convert_offset_to_block_array(
   #     attn_inputs.kv_cache_kernel_block_id_device)
   # 即用 FULL group 的 page_table 把当前 token 的 K/V 写入 cache
   # ★ 这一步把 bf16/fp8 K/V 写到 row 内对应 kernel_block 槽位

apply_write_cache_store(write_cache_store_impl, attn_inputs, kv_cache)
   # PD-sep 的 cache_store async push（CPU 侧 op）

out = XQAWrapper.forward(fmha_input, kv_cache, fmha_params)
   # 内部:
   # k_cache = kv_cache.kv_cache_base[:, 0, ...]    bf16/fp8 view
   # v_cache = kv_cache.kv_cache_base[:, 1, ...]    bf16/fp8 view
   # page_table = kv_cache_kernel_block_id_device  ← FULL group 的 page_table
   # seq_lens = attn_inputs.sequence_lengths
   # flashinfer.xqa(q_4d, k_cache, v_cache, page_table, seq_lens, ...)
   # ★ XQA 用 page_table 索引 k_cache/v_cache 读取，在 fp32 累加器里做 softmax
```

`xqa.py` line 99 / 159（commit `b43276aa9` 改动后）已经把
`torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)` 兜底拆掉，**任何 K/V
NaN 都会原样穿透到 attention 输出**。

### 3.5 tryReleaseKVBlock（请求结束）

`StreamCacheResource::tryReleaseKVBlock`:

```cpp
if (reuseCache && !hasError && status == FINISHED) {
    if (enableDeviceCache) {
        cache_manager->insertIntoCache(insert_info);
        // 把 block 插入 BlockCache, blockCacheRef++
    }
    storeCacheAsync(batch_kv_cache_resource_,
                     enableMemoryCache && !enableTieredMemoryCache,
                     enableRemoteCache);
    // 走 KVCacheConnectorCoordinator::asyncWrite
    // → allocator_->incrKVCacheRef(true)  connectorRef++
    // → connector->asyncWrite (memory connector / remote connector)
    // → 异步把 KV cache 内容拷到 host 内存 / 远端

    if (enableTieredMemoryCache) {
        evictDeviceCacheToMemory();
        // pop 一些 block 出 BlockCache + storeCacheAsync + blockCacheFree
    }
}

cache_manager->free(free_info);
// → HybridTypeKVCacheAllocator::free
// → per-group->free
//   ├─ LinearKVCacheGroup::free
//   │     zeroLinearWriteRegion(valid)   ★ 写零整 row（不只 LINEAR 区域）
//   │     block_pool_->requestFree(valid)
//   └─ FullKVCacheGroup::free
//         block_pool_->requestFree(blocks)   无清零
```

`zeroLinearWriteRegion` 实际把 *整个* row 的 [0..max_stride) 都清零（不只
ssm+conv），覆盖 LINEAR 用过的 fp32 + 没用过的尾部，确保下游 FULL malloc
看到的整块都是 0。

---

## 4. 所有 fp32→bf16 字节复用 / NaN 风险交点

下面把 LINEAR fp32 字节有可能被 FULL bf16/fp8 路径误读的 *所有* 路径列出来，
逐项核对当前的覆盖情况。

### 交点 1：block 重新分配（LINEAR free → FULL malloc）

时序：
1. 请求 A 的 LINEAR group 拿到 block X，写 fp32 ssm/conv state
2. 请求 A 结束，`LinearKVCacheGroup::free` → `zeroLinearWriteRegion`
   把 row 全清零 → `requestFree`
3. block X 回到 BlockPool free pool，状态 = 全零
4. 请求 B 的 FULL group `block_pool_->malloc` 拿到 block X
5. 请求 B 写 bf16/fp8 K/V 到 [0..full_size)
6. XQA 读 → 安全

**状态：✅ 已覆盖**（最近 5 个 commit + 我加的 init zero）

需要担心的边界情况：

| 路径 | 是否会绕过 LINEAR free 的清零 | 状态 |
|---|---|---|
| `LinearKVCacheGroup::free` | 否，自己就是清零者 | ✅ |
| `LinearKVCacheGroup::removeSkippedBlocks` | 否，commit `87e9870d2` 加了清零 | ✅ |
| `HybridTypeKVCacheAllocator::incrMalloc` 回滚 | 否，加了 `zeroLinearGroupBytes` | ✅ |
| `HybridTypeKVCacheAllocator::decrKVCacheRef`（request & connector） | 否，加了 `zeroLinearGroupBytes` | ✅ |
| `KVCacheAllocator::blockCacheFree(BatchKVCacheResourcePtr)` | 否，hook `cleanBlocksBeforeBlockCacheFree` | ✅ |
| `KVCacheGroup::ensureFreeBlocks` 驱逐 | 否，hook `block_cache_free_hook_` | ✅ |
| `FullKVCacheGroup::free` | 不需要（FULL bf16 不会自身造 NaN） | ✅ |
| `KVCacheMemoryConnector::freeBlocks` (host pool) | N/A（独立 host BlockPool，与 GPU 不共享字节） | ✅ |

### 交点 2：block 0 sentinel

block 0 永远不会被 malloc（`BlockPool::initFreeBlocks` 从 1 起始），所以
**永远不会经过 LINEAR free 的清零**。

但 block 0 *会被 read*：
- LINEAR `load_initial_state_from_block_map_kernel`：当 `prefix == 0` 时
  显式 `block_idx = where(is_zero, 0, ...)`，**读 ssm_states[0]**，然后用
  `tl.where(is_zero, zeros, loaded)` 把结果换成 0
  → 如果 block 0 是 NaN 字节，**load 还是会发生**（GPU 不会做条件 load
  优化），后续 where 语义把结果换成 0。
  在 IEEE 754 下，`where` 是按元素选择，不会把 NaN 传给输出。**所以这条
  对 NaN 不会传播**。
- FULL XQA：`page_table` padding = 0 时，理论上 `seq_lens` 把超出范围的位置
  mask 掉，但 IEEE 754 下 `0 × NaN = NaN` 且 `NaN + 0 = NaN`，部分 kernel
  实现里 mask 是「乘 0」而不是「条件 load」，**NaN 还是会传**。

→ **block 0 *必须* 是真 0**。`BlockPool::initializeCacheBuffer` 的
`cache_aligned_buffer_.zero_()` + `synchronize()` 是这一点的保证。

> 原本 `MemoryLayoutStrategy::processKVTensor` 已经会 `clearKVTensor(reshaped_tensor)`，
> 理论上应该已经清零了 block 0。但**没有 `synchronize()`**，意味着这个清零
> kernel 在哪个 stream 上执行、何时实际落盘，是不确定的。我加在
> `initializeCacheBuffer` 末尾的 `synchronize()` 把这件事变得 deterministic。

**状态：✅ 已覆盖**

### 交点 3：FULL 写入稀疏 → 未写 kernel_block 槽位被读出

时序：
1. block X 被 FULL group malloc（之前是 LINEAR 归属，已清零；或者从未被使用）
2. FULL 写当前 batch token 的 K/V 到对应 kernel_block 槽位
3. 一个 logical block 含 64 个 kernel_block 槽位（`SEQ_SIZE_PER_BLOCK / KERNEL_SEQ_SIZE_PER_BLOCK = 4096/64 = 64`）
4. 如果当前 sequence 在该 block 内只有 100 个 token，只写 2 个 kernel_block，剩 62 个未写
5. XQA 读时：seq_lens 限制读到 token 100 为止，理论上不会触及未写槽位
6. **但**：如果 XQA 内部有 warp-level pre-fetch 或 mask-后再 load 的实现，
   未写槽位的字节会被读出。
7. 当前未写槽位的字节 = block 起手时的状态：
   - 起手是 0（init zero） → 0.0 bf16 → softmax 安全 ✅
   - 上一归属者是 LINEAR，且 LINEAR free 已清零 → 0 ✅
   - 上一归属者是 FULL，且 FULL free 未清零 → bf16 K/V 残留（不是 NaN） ✅
   - 上一归属者是 LINEAR，但 LINEAR free 因为某种 bug 没清零 → fp32 字节 → NaN 风险 ❌

**状态：✅ 已覆盖**（init zero + LINEAR free 清零）

### 交点 4：page_table padding 指向 block 0

时序：
1. batch 内有长短不一 sequence
2. `kv_cache_kernel_block_id` 用 `torch::zeros` 分配，所有位置初值 = 0
3. `copyKvCacheBlocksToModelInput` 用 `memcpy` 把每个 stream 的真实 block id
   写到 row 前缀
4. 短 sequence 的 row 尾部仍是 0
5. XQA 用 page_table 索引到 block 0
6. block 0 被 read → 看交点 2

**状态：✅ 已覆盖**（block 0 = 0 之后这条退化为安全）

### 交点 5：cuda graph replay 下的 page_table padding

`CudaGraphRunner::prepareInputs` 每次 replay 前都 `fill_(0)` 把 captured
page_table 清零，再用 strided D2D copy 把当前 step 真实 block id 写入前缀。
和非 cuda graph 路径行为一致 → 退化为交点 4 → ✅ 已覆盖

### 交点 6：PD-sep prefill→decode KV transfer

prefill 端 `Qwen3NextModel.forward` 在每层 attention 后调
`asyncWriteByLayer`（P2P prefill）→ `P2PConnectorWorkerPrefill::writeByLayer`
把当前 layer 的 KV cache 字节按 (cache_key, block_id) 切片，传输 *整段
kv_block_stride_bytes*（不区分 LINEAR/FULL spec 自身的有效区域）。

decode 端通过 `P2PConnectorSchedulerDecode::asyncRead` 拉取，**RDMA 写到自己
的 BlockPool block X 的整段 row**：
- 对 LINEAR 层：prefill 写的是 fp32 ssm+conv，decode 收到 fp32 ssm+conv
  写入 row 的 [0..linear_size)；**尾部 [linear_size..full_size) 保持原有
  decode-side 字节** → 由于 block X 在 decode 端 malloc 时是干净（init zero
  或 LINEAR free 清零），尾部是 0。decode LINEAR layer 只读 [0..linear_size)，
  安全。
- 对 FULL 层：prefill 写的是 bf16/fp8 K/V，decode 收到全 row 写入。

**状态：✅ 已覆盖**

> 前提是 decode 端 block X 在 malloc 时是干净的——这个由 init zero + free 清零
> 保证。

### 交点 7：连接器异步 store 与 free 并发

`tryReleaseKVBlock` 顺序：
1. `insertIntoCache` (BlockCache, blockCacheRef++)
2. `storeCacheAsync` (异步拷到 host/remote, connectorRef++)
3. `cache_manager->free` → LINEAR free → `zeroLinearWriteRegion`(GPU 异步)
   → `requestFree`(CPU 立即)

step 2 的异步拷贝读取的是 GPU buffer 的 fp32 字节（LINEAR 层）；
step 3 的 zero 在 worker thread 当前 stream 排队，可能比 step 2 的拷贝
先完成。

后果：memory cache 拿到的 LINEAR ssm 状态会被零化 → 后续 reload 时复用拿
到的是零（而不是「上一请求的 prefix end-state」）→ **LINEAR 二级缓存复用
在功能上失效**（prefix benefit 退化为 0），**但不会引入 NaN**。

**这是一个 commit `48a878df3` / `b43276aa9` 留下的取舍**——为了正确性
（不出 NaN）牺牲了 LINEAR 复用的有效性。本次不动。

### 交点 8：异步 zero 流序竞态（worker free vs engine forward）

LinearKVCacheGroup::free 在 worker 线程的 `current CUDA stream` 上排
`tensor.zero_()`；engine 线程后续 `block_pool_->malloc` 立刻把 block 交给
forward。

最坏情况：worker stream 的 zero 在 engine stream 写完 K/V 之后才执行 →
**清零 K/V 写入**。

但是！**`tensor.zero_()` 写的是 0 字节，IEEE 754 下 bf16(0) = 0.0**，
softmax 不会出 NaN。最差是 attention 输出幅值偏弱（部分 K/V 被零化）。
**不会产生用户报告的 NaN→乱码现象**。

**状态：⚠ 未修复（不在本次 NaN bug scope 内）**
未来如果发现 attention 输出系统性偏弱再修。可选方案：
- LINEAR free 后 record `cudaEvent`，BlockPool::malloc 在 wait event 后再返回
- 把 zero 推迟到 malloc 时（在新 owner 的 stream 上同步执行）

### 交点 9：LinearKVCacheGroup::malloc 部分失败 block 泄漏

`LinearKVCacheGroup::malloc` (`rtp_llm/cpp/cache/LinearKVCacheGroup.cc:118-135`)
循环中如果 `block_pool_->malloc(1)` 第二次或后续失败，**前面已经分到的
`new_ids` 不会被 `block_ids.add` 接管**，request_ref > 0 但无 owner 跟
踪，永久泄漏。

**这是独立的内存泄漏 bug，不会引起乱码**（泄漏 block stay-out-of free
pool，不会被 FULL malloc 复用）。本次不动，但建议后续单独修。

### 交点 10：`MemoryLayoutStrategy::processScaleTensor` 对 LINEAR 不生效

`processScaleTensor` 只在 `config_.hasScale()` 为 true 时清零 scale 区域。
对纯 LINEAR layout（无 scale），scale 区域 size = 0，不需要清零。
对 FULL FP8/INT8 layout，scale 区域被 `clearScaleTensor` 清零（hasScale=true）。
对 FULL BF16 layout，无 scale，size = 0。

**状态：✅ 没有遗漏**

### 交点 11：blockBatchCopy 用 `cache_specs[0]` 的 block_size_bytes

`KVCacheAllocator::blockBatchCopy` (`rtp_llm/cpp/cache/KVCacheAllocator.cc:140`)
按 `cache_specs[0]->block_size_bytes()` 拷贝。对 hybrid，`cache_specs[0]`
是 FULL spec（FULL groups 排在前面），block_size_bytes 是 FULL 的 size。

如果某层是 LINEAR，`cache_specs[0]` 的 size 不一定等于 LINEAR 的 size。这个
路径用于 beam search 等场景的 block 复制。Qwen3-Next 一般不开 beam search，
影响范围有限。**不在 NaN 范围内**。

### 交点 12：CP（Context Parallel）all-gather 路径

`Qwen3NextGatedDeltaNet._forward_cp_prefill` 走 CP 路径，all-gather 全部 token
的 mixed_qkv，再在每个 CP rank 上重新跑 LINEAR。其 ssm_states 读写也通过
LinearCacheConverter view，路径与正常 prefill 一致。
**状态：✅ 与交点 1-3 同等覆盖**

### 交点 13：MTP 子模型独立 layout

MTP layout 在 BlockPool 中独立分段（`BlockPoolConfigHelper.h:42-72`），
但共用同一份 `cache_aligned_buffer_`。我加的 `cache_aligned_buffer_.zero_()`
覆盖整段（main + 所有 MTP）。MTP 子模型的 SP（speculative decoding）KV cache
也由 init zero 保护。
**状态：✅ 已覆盖**

---

## 5. 还有没有「SSM state fp32 把 NaN 引入 XQA 输出」的隐蔽路径？

把怀疑面再细分：

### A. SSM 计算本身产生 NaN，传播到下游

LINEAR fwd 的输出 = `out_proj(norm(local_attn_out × z))`。这是 bf16/fp8
（取决于 quant_config）。如果 SSM 内部计算 NaN（例如 `1/sqrt(eps)` 边界
case），输出 bf16 NaN。

→ 传到下一层 hidden_states → FULL ATTN 在那一层投影 Q/K/V 也��到 NaN → FULL
写 NaN 到 KV cache → XQA 读 NaN → 输出 NaN。

**这条是真的可能**。验证方法：
- 在 `Qwen3NextGatedDeltaNet.forward` 末尾加 `assert not torch.isnan(attn_output).any()`
- 在 FULL 层入口 `qkv = self.qkv_proj(hidden_states)` 后加 `assert not torch.isnan(qkv).any()`

如果上线乱码再现，先验这条。

> SSM 数值稳定性常见隐患：
> - `chunk_gated_delta_rule` 内 `cumsum` over many chunks 累积漂移
> - `fused_recurrent` 内 `b_h *= exp(b_g)` 当 `b_g` 极大时 `exp` overflow → Inf
> - `b_h += b_k * b_v`，b_h 的范围随时间增大，能否溢出 fp32

### B. fused_gdn_gating 边界

`fused_gdn_gating(self.alog, a, b, self.dt_bias)` 计算 g, beta。
若 `dt_bias` 偏置过大，`exp(-dt)` 会下溢成 0，但 0 不是 NaN，安全。
但 `softplus(x) = log(1 + exp(x))` 当 x 大时近似 x（常用 stable 实现），
不易出 NaN。

### C. `chunk_gated_delta_rule` 内累积器精度

`store_ssm_state_to_block_map` 的 assert：
```python
assert h.dtype == torch.float32 and final_states.dtype == torch.float32
```
chunk_gated_delta_rule 内部已经保证 fp32 累加。**正常情况下不会产生 NaN**。

### D. RMSNormGated（`norm`）

`local_attn_out = self.norm(local_attn_out.reshape(-1, head_v_dim),
                              z.reshape(-1, head_v_dim))`
RMSNormGated 内部 `1 / sqrt(mean + eps)`，eps 一般 1e-6。如果输入全 0，
`mean = 0`，结果是 `1 / sqrt(eps)` 有限值。如果输入有 NaN，输出 NaN。

→ 这一步**只放大已有 NaN，不会无中生有**。

### E. `out_proj` (LinearFactory)

bf16/fp8 GEMM。fp8 GEMM 输入 NaN 会传到输出。但 fp8 本身不能精确表示 NaN
（fp8_e4m3fn 没有 inf/nan 编码！），实际上 fp8 是 saturating，最大值就停。
→ **fp8 路径反而是天然的 NaN 净化器**，但 BF16 量化版仍会传 NaN。

### F. `LinearCacheConverter._build_typed_storage_view` 对齐错位

`stride_bytes` 必须能整除 target dtype 的 item_size，否则 raise ValueError。
但是！如果 `block_size_bytes`（来自 `kv_cache_tensor.stride(0) ×
elem_size`）不是 `sizeof(fp32) = 4` 的整数倍，会 raise。但 cache spec
设计是必然 4 字节对齐的，所以这条不会触发。

**也就是说，view 不会越界、不会读到隔壁 row 的字节。**

### G. `kv_cache.kv_cache_base` 的 reshape view 重新解释

```python
kv_cache_tensor = kv_cache.kv_cache_base.reshape(kv_cache.kv_cache_base.shape[0], -1)
```

`kv_cache_base` 在 BlockPool 端 dtype 是 LINEAR layout 的 dtype（hybrid 单
layout 下 = config.dtype = bf16 / fp8）。把 bf16 view 一路到 LINEAR 层，再
通过 LinearCacheConverter reinterpret 为 fp32。这是字节级重解释，不会有
arithmetic conversion。

但 `kv_cache.kv_cache_base.shape[0]` 是 block_num，`-1` 是 row 内全部元素
（按 bf16 计），shape 形如 `[block_num, kv_block_stride_bytes/2]`。

LinearCacheConverter 用 `kv_cache_tensor.stride(0) × kv_cache_tensor.dtype size`
得到 row stride bytes = `(kv_block_stride_bytes/2) × 2 = kv_block_stride_bytes` ✓

→ 没有错位，view 跑得对。

### H. flashinfer XQA 内部可能的非 mask 推断 read

XQA 实现细节不开源，无法直接看。但**已有兜底：block 0 = 0**。即使 XQA 出现
warp-level over-read，碰到的也是 0 字节 → bf16(0) = 0.0 → softmax 安全。

### I. KV cache write op 的 store_kv=False

`FusedRopeKVCacheDecodeOp.forward` 调用 `decode_fused_rope_kvcache`
（在 `rtp_kernel.fused_rope_kvcache`，外部 package）`store_kv=False`。

但同一调用 `kv_cache=kv_cache.kv_cache_base` + `kv_cache_offset` 都被传，
**`store_cache=True`**（隐含，因为 kv_cache 非空）。这意味着 K/V *是*
被写入 kv_cache_base 的对应 page slot 的。

只是 *不返回* k/v tensor 给上层。这一点在 prefill_fused_rope_kvcache 也
一致：`store_kv` 控制是否同时把 k/v 输出 tensor 化（用于 unpaged FMHA），
`store_cache` 控制是否写入 paged cache。

**结论：FULL ATTN K/V 写入路径正常，没有遗漏。**

---

## 6. 风险等级分类

| 风险来源 | 触发概率 | 后果 | 当前覆盖 |
|---|---|---|---|
| LINEAR fp32 字节经由 block 重分配 → FULL bf16 read | 高（每个请求结束都会发生） | 概率性 NaN 乱码 | ✅ commit `87e9870d2` 系列 + init zero |
| Block 0 未初始化字节被 page_table 0 padding 命中 | 高（任何 batch 短长混排） | 概率性 NaN 乱码 | ✅ commit `3d9945df8`（init zero + sync） |
| cuda graph replay 下 page_table padding | 高 | 同上 | ✅ `cuda_graph_runner.cc:122` 显式 fill_(0) + block 0 = 0 兜底 |
| MTP 子模型 buffer 未初始化 | 中 | 同上（MTP 命中时） | ✅ init zero 覆盖整 buffer |
| PD-sep decode 端 block 拉取后 row 尾部残留 | 低 | NaN（如果尾部之前有 fp32） | ✅ 由 init zero + LINEAR free zero 联合覆盖 |
| 异步 zero 流序竞态（worker free vs engine forward） | 极低（PyTorch 默认共享 stream 0） | 输出幅值偏弱（不是 NaN） | ⚠ 未修；不解释 NaN 问题 |
| LINEAR cache reuse 复用 ssm 状态退化为零 | 100%（设计 trade-off） | LINEAR prefix benefit 失效 | 已知，commit 注释里明说 |
| LINEAR 计算自身产生 NaN（exp overflow / chunk drift） | 未知 | 传播到 FULL → 乱码 | ❌ 未防御；如果上线复发，先查这条 |
| LinearKVCacheGroup::malloc 部分失败 block 泄漏 | 极低 | 内存泄漏（不是乱码） | ⚠ 未修，独立 bug |
| `blockBatchCopy` 用 cache_specs[0] size | 极低（beam search 才走） | 复制错误 | ⚠ 未修，独立 bug |

---

## 7. 复现 / 调试 Hook 推荐

如果上线再次出现乱码，按以下顺序加 assert 定位：

```python
# A. 在 Qwen3NextGatedDeltaNet.forward 末尾
assert not torch.isnan(attn_output).any() and not torch.isinf(attn_output).any(), \
    f"LINEAR layer NaN/Inf detected"

# B. 在每个 FULL ATTN 入口（Qwen3NextAttention.forward 顶部）
assert not torch.isnan(hidden_states).any(), \
    f"NaN propagated INTO full attn from previous layer"

# C. 在 XQAImpl.forward 调 fmha_impl 之前
assert not torch.isnan(fmha_input).any(), \
    f"Q tensor NaN before XQA"

# D. 临时把 nan_to_num 加回 XQAImpl.forward / XQADecodeImpl.forward
out = self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)
nan_count = torch.isnan(out).sum().item()
if nan_count > 0:
    print(f"XQA produced {nan_count} NaN at layer {layer_idx}")
out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
return out

# E. 监控 KV cache buffer 内 fp32 NaN 比例（启动时跑一次）
def check_block_pool_clean(block_pool_tensor):
    fp32_view = block_pool_tensor.view(torch.float32)
    nan_mask = torch.isnan(fp32_view) | torch.isinf(fp32_view)
    print(f"BlockPool NaN/Inf bytes ratio: {nan_mask.float().mean().item():.6f}")
```

逻辑：
- 如果 A 触发 → SSM/conv state 在 LINEAR 层就 NaN（怀疑计算端 / SSM 字节
  污染）
- 如果只 B 触发 → 上一层 LINEAR 输出 NaN（同 A）
- 如果只 C 触发 → Q 投影出问题（很少见）
- 如果只 D 触发 → K/V cache 字节脏（KV cache fp32→bf16 误读 / 前一层污染）
- 如果 E 显示 NaN 字节存在 → init zero / LINEAR free zero 没生效

---

## 8. 总结

经过这一轮全链路审计，LINEAR fp32 SSM/conv state 与 FULL bf16/fp8 K/V
**所有共享同一物理 row 字节的复用路径**都被列出来并核对了覆盖情况：

- **「字节复用 → bf16 误读 NaN」** 这条主线已被堵住（init zero + 全部 free
  路径都接 zero pass + block 0 兜底）。
- **「异步流序」** 不会产生 NaN（最差是零化新写入的 K/V），不解释当前
  bug，本次不动。
- **「LINEAR 计算端自身产生 NaN」** 是未防御的剩余风险——如果上线再现
  乱码，按第 7 节 hook A/B/C 定位。
