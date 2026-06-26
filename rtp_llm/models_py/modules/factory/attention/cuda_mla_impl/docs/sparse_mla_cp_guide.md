# Sparse MLA with Context Parallelism (CP) - Deep Dive

## 0. 阅读前须知

本文档面向完全不了解 MLA 的新手，从基础概念逐步讲到 `SparseMlaFp8CPOp` 的完整实现。

涉及的核心源文件：

| 文件 | 作用 |
|------|------|
| `flashmla_sparse_impl.py` | 非 CP 的 Sparse MLA 基础实现 |
| `flashmla_sparse_cp_impl.py` | CP 变体（本文重点） |
| `mla_kv_cache_write_op.py` | KV Cache 写入 op |
| `fused_qk_rope_cat_cache_mla.py` | 融合的 RoPE + KV Cache 写入 Triton kernel |
| `cp_utils.py` | CP 的 zig-zag Q 分配工具 |
| `block_index_to_global.py` | topk 索引从 request-local 转全局 |
| `triton_kv_scatter.py` | 将局部输出散回全局位置 |

---

## 1. 背景知识

### 1.1 什么是 MLA (Multi-Latent Attention)

MLA 是 DeepSeek-V2/V3 提出的注意力变体，核心思想是**压缩 KV Cache**。

传统 MHA (Multi-Head Attention) 中，每个 token 在 KV Cache 里存 `num_kv_heads * head_dim` 维的 K 和 V。对于大模型（如 128 heads x 128 dim），每个 token 占用大量显存。

MLA 的做法：

```
传统 MHA:
  K = [num_kv_heads, head_dim]  →  缓存完整 K
  V = [num_kv_heads, head_dim]  →  缓存完整 V

MLA:
  compressed_kv = [kv_lora_rank]  →  缓存低秩压缩后的 KV（如 512 维）
  k_pe = [rope_head_dim]          →  缓存 RoPE 位置编码部分（如 64 维）
```

一个 token 只需存 `kv_lora_rank + rope_head_dim = 576` 维（BF16），大幅减少 KV Cache 占用。

### 1.2 "Sparse" 是什么意思

Sparse MLA 在 attention 计算中不遍历所有 KV token，而是通过一个 **indexer**（稀疏索引器）先选出 `top_k` 个最相关的 KV token，只对这些做 attention。这把 attention 的计算量从 O(T*S) 降低到 O(T*topk)。

```
完整 Attention:   Q @ K^T → softmax → V     (遍历所有 S 个 KV token)
Sparse Attention: Q @ K[topk]^T → softmax → V[topk]  (只看 topk 个)
```

`topk_indices` 的形状是 `[T, topk]`，表示每个 Q token 选出的 KV token 在当前 request 内的位置索引。

### 1.3 什么是 Context Parallelism (CP)

当 prefill 的序列特别长（如 256K token）时，单张卡放不下所有 token 的中间激活。CP 把一条长序列切成多段，分到多张卡上并行处理：

```
长序列: [t0, t1, t2, t3, t4, t5, t6, t7]  (8 个 token)

CP_SIZE=2 时:
  Rank 0 处理: [t0, t1, t2, t3]
  Rank 1 处理: [t4, t5, t6, t7]
```

但 attention 需要看完整的 KV：Q token t0 可能需要 attend 到 t7。所以 CP 的关键问题是：
**每张卡只算自己的一部分 Q，但需要看到所有的 KV。**

解决方案：通过 `all_gather` 收集所有卡的 KV，写入完整的 paged KV cache，然后只计算本卡负责的 Q 的 attention。

### 1.4 Zig-Zag 分配

CP 不是简单的前后切分，而是 **zig-zag**（锯齿形）分配 Q token：

```
8 个 token 的序列, CP_SIZE=2:

Rank 0 的 chunk: [t0, t1, t6, t7]  ← 前半 + 后半
Rank 1 的 chunk: [t2, t3, t4, t5]  ← 中间部分

每个 rank 的 chunk 再分两组:
  Rank 0: q0_idx=[t0, t1] (前半)  q1_idx=[t6, t7] (后半)
  Rank 1: q0_idx=[t2, t3] (前半)  q1_idx=[t4, t5] (后半)
```

这样做的好处是**负载均衡**：causal attention 中，越靠后的 token 要看越多的 KV，zig-zag 让每张卡的总计算量大致相等。

`generate_q_indices(chunk_lengths)` 负责这个分配（见 `cp_utils.py:161`）。

### 1.5 Paged KV Cache

KV Cache 不是一整块连续显存，而是分成固定大小的 **page**（如 64 token 一页）。每个 request 有一个 `block_table` 记录它用了哪些 page：

```
block_table[req_i] = [page_3, page_7, page_12, ...]

slot_mapping[token_j] = page_idx * page_size + offset_in_page
```

这样可以灵活分配/回收显存，避免碎片。`slot_mapping == -1` 表示这个 token 不需要写入 cache（如 padding token）。

### 1.6 FP8 KV Cache

为进一步压缩显存，KV Cache 可以用 FP8 格式存储。每个 slot 占 656 字节：

```
656 bytes = 512 bytes (compressed_kv, FP8)
          +  16 bytes (4 个 float32 scale, 每 128 元素一个)
          +  64 bytes (k_pe, BF16, 32 个 BF16 元素)
          +  64 bytes (对齐 padding)
```

compressed_kv 做分块 FP8 量化（每 128 元素一个 scale），k_pe 保持 BF16。

---

## 2. 类继承关系

```
SparseMlaOp                  ← BF16 稀疏 MLA 算子（最基础）
  └── SparseMlaFp8Op          ← FP8 稀疏 MLA 算子（增加 FP8 cache 支持）
       └── SparseMlaFp8CPOp   ← FP8 + CP 算子（增加 all-gather + Q 切分）

MlaImplBase                   ← MLA 层的抽象基类
  └── SparseMlaImpl           ← 完整的 MLA 层（RoPE + KV Write + BMM + Attention）
       └── SparseMlaCpImpl    ← CP 版本的 MLA 层
```

**算子（Op）** 只管 attention 计算本身。
**Impl** 是完整的层，包含 RoPE、KV Cache 写入、输入/输出矩阵乘法。

---

## 3. 非 CP 的流程（对照理解）

`SparseMlaImpl.forward()` 的完整流程（`flashmla_sparse_impl.py:801`）：

```
输入: q [T, H, qk_head_dim],  compressed_kv [T, kv_lora_rank],  k_pe [T, rope_dim]

Step 1: RoPE + KV Cache Write
  ├─ (融合路径) fused_qk_rope_cat_cache_mla(q, compressed_kv, k_pe, kv_cache, ...)
  │    → 一个 Triton kernel 同时做：
  │       a) 对 q 的 rope 部分做 RoPE (in-place)
  │       b) 对 k_pe 做 RoPE (in-place)
  │       c) 把 [compressed_kv || rope'd_k_pe] 写入 paged KV cache
  │
  └─ (非融合路径) 分开调用：
       a) rope_impl.forward(q_pe, k_pe, params)
       b) kv_cache_write_op.forward(compressed_kv, k_pe, kv_cache, params)

Step 2: Input BMM — 投影 Q 到 KV 的低秩空间
  q_nope, q_pe = split(q)
  q_transformed[:, :, :kv_lora_rank] = q_nope @ W_kc   (BMM)
  q_transformed[:, :, kv_lora_rank:] = q_pe             (copy)
  → 得到 [T, H, kv_lora_rank + rope_dim]

Step 3: Sparse Attention
  flash_mla_with_kvcache(q_transformed, kv_cache, topk_indices, ...)
  → 得到 [T, H, kv_lora_rank]

Step 4: Output BMM — 投影回 value 空间
  output = attn_output @ W_vc
  → 得到 [T, H, v_head_dim]
```

---

## 4. CP 流程详解 (`SparseMlaFp8CPOp`)

### 4.1 整体调用链

```
SparseMlaCpImpl.__init__()
  → super().__init__()          # 创建 rope_impl, kv_cache_write_op, weights
  → fmha_impl = SparseMlaFp8CPOp(...)
  → create_params() → prepare() → plan()   # 首次规划

SparseMlaCpImpl.forward(q, compressed_kv, k_pe, kv_cache, layer_id, topk)
  → RoPE (用 full_rope_pos_ids)
  → fmha_impl.forward(q_transformed, compressed_kv, k_pe, topk, ..., kv_cache)
      → all_gather
      → restore
      → kv_cache_write
      → attend (只算本 rank 的 Q)
      → scatter back
  → output BMM
```

### 4.2 plan() — 规划阶段

**调用时机**: 每次 forward 前，在 `prepare()` 中调用。

**输入**:
- `mla_params`: 包含 batch_indice、positions、slot_mapping 等
- `block_table`: paged KV cache 的页表 `[batch, max_pages]`
- `attn_inputs`: 包含 CP 信息的注意力输入

**核心计算**:

```python
# 1. 从 chunk_lengths 做 zig-zag 分配
chunk_lengths_list = [2, 2]  # 例: 2 个 request, 每个 chunk 2 token
q0_idx, q1_idx = generate_q_indices(chunk_lengths_list)
# q0_idx = [0, 2]  (每个 chunk 的前半)
# q1_idx = [1, 3]  (每个 chunk 的后半)

# 2. 调用 C++ 算出各种索引映射
mla_params.fill_cp_plan_params(padding_mask, kv_restore, q0_idx, q1_idx, ...)
```

**产出** (存在 `self` 上，供 forward 使用):

| 字段 | 形状 | 含义 |
|------|------|------|
| `kv_restore_unpad_indices` | `[total_kv_len]` | all_gather 后 KV 的重排索引 |
| `total_global_ids` | `[n_q]` | 本 rank 的 Q 在全局 padded 序列中的位置 |
| `total_local_ids` | `[n_q]` | 本 rank 的 Q 在 local token 中的位置 |
| `cu_kv_seqlens_global` | `[batch+1]` | 全局 KV 累积长度 |
| `total_kv_len` | `int` | 全局 KV 总长度 |
| `precomputed_req_ids` | `[n_q]` | 每个 global Q token 对应的 request id |
| `full_rope_pos_ids` | `[padded_T]` | 本 rank 每个 local token 的 RoPE 位置 |

**full_rope_pos_ids 的构建逻辑**:

```python
full_rope = zeros(padded_T)
full_rope[total_local_ids] = positions_d[total_global_ids]
```

即：只在本 rank 拥有的 token 位置填入对应的 position id，其余位置为 0（这些位置的 RoPE 结果不会被使用）。

### 4.3 forward() — 前向推理（核心）

**签名**:
```python
def forward(self, q, compressed_kv, k_pe, topk, batch_indice_d, kv_cache, layer_id)
```

注意：进入 forward 时，RoPE 已经在外层 `SparseMlaCpImpl.forward()` 中做过了。

#### Step 1: All-Gather KV

```python
gathered_ckv = all_gather(compressed_kv, group=Group.TP)   # [cp_size * local_T, kv_lora_rank]
gathered_k_pe = all_gather(k_pe, group=Group.TP)           # [cp_size * local_T, rope_dim]
```

每张卡只有自己那份 compressed_kv 和 k_pe，all_gather 把所有卡的拼在一起。

```
Rank 0 有: ckv_0 [local_T, 512]    Rank 1 有: ckv_1 [local_T, 512]
                       ↓ all_gather ↓
两张卡都拿到: [ckv_0 || ckv_1]  →  [2*local_T, 512]
```

#### Step 2: Restore to Global Order

```python
restored_ckv = gathered_ckv[kv_restore_unpad_indices]
restored_k_pe = gathered_k_pe[kv_restore_unpad_indices]
```

all_gather 的结果是 `[rank0_tokens, rank1_tokens, ...]` 的拼接顺序，但 KV cache 需要的是原始序列顺序。`kv_restore_unpad_indices` 把它重排回正确位置，并去掉 padding。

```
all_gather 顺序: [rank0_t0, rank0_t1, rank1_t2, rank1_t3]
                                  ↓ restore ↓
全局正确顺序:    [t0, t1, t2, t3]    (按原始 position 排列)
```

#### Step 3: Write KV Cache

```python
self.kv_cache_write_op.forward(restored_ckv, restored_k_pe, kv_cache, self.mla_params)
```

把 restored 后的 KV 写入 paged cache。底层调用 `concat_and_cache_mla`：
- 把 `compressed_kv` 和 RoPE 后的 `k_pe` 拼接
- 按 `slot_mapping` 写入对应的 cache page 和 slot

此时每张卡的 KV cache 都有了完整的 KV 数据。

#### Step 4: Select Local Q Tokens

```python
if total_local_ids_is_identity:
    q0 = q                              # 优化: local ids 恰好是 [0,1,...,n-1]，无需 gather
else:
    q0 = q[total_local_ids].contiguous() # 只取本 rank 负责的 Q token
```

`total_local_ids` 表示本 rank 拥有的 Q token 在 local tensor 中的索引。如果恰好是连续的 `[0, 1, ..., n-1]`，就跳过 gather 操作。

#### Step 5: Sparse Attention (两条路径)

**路径 A: `_attend_with_kvcache` (默认)**

直接在 FP8 paged cache 上做 attention：

```python
# 1. topk 索引从 request-local 转成 paged cache 全局 block 索引
global_topk = self._convert_topk_indices_to_global(topk)
# 例: topk_indices[q_i] = [5, 12, 100]  (request 内第 5/12/100 个 KV token)
#   → global_indices[q_i] = [page3_slot5, page7_slot12, page12_slot36]

# 2. 调 flash_mla_with_kvcache
attn_out, _ = flash_mla_with_kvcache(
    q=q0,
    k_cache=kv_cache_fp8,     # 直接读 FP8 cache
    block_table=block_table,
    indices=global_topk,      # 只读 topk 指定的位置
    is_fp8_kvcache=True,
    softmax_scale=scale,
)
```

**路径 B: `_attend_gather` (USE_GATHER_PATH=1)**

先把 FP8 cache gather 到 BF16 连续 buffer，再做 attention：

```python
# 1. FP8 paged cache → BF16 连续 workspace
rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_v2(
    src=kv_cache_fp8,          # FP8 paged
    dst=workspace.fused_kv,    # BF16 连续 [total_kv_len, kv_lora_rank + rope_dim]
    block_table=block_table,
    seq_lens=seq_lens,
    ...
)

# 2. topk 索引加上 per-request offset
offsets = workspace_starts[precomputed_req_ids]
global_indices = topk_2d + offsets   # request-local → workspace 全局位置
global_indices.masked_fill_(topk_2d < 0, -1)  # 保留 -1 padding，防止跨 request 污染

# 3. flash_mla_sparse_fwd 在 BF16 buffer 上做 attention
out, _, _ = flash_mla_sparse_fwd(q0, workspace.fused_kv, global_indices, scale, ...)
```

路径 B 在大 prefill 时更快（~1.7x），因为连续内存访问对 GPU 更友好。

#### Step 6: Scatter Back

```python
if use_identity_q:
    return out0                                    # 不需要 scatter
out = triton_kv_scatter(out0, total_local_ids, q.size(0))
return out
```

`out0` 只有本 rank 计算的 Q token 的结果，需要散回全局位置。`triton_kv_scatter` 把 `out0[i]` 写到 `output[total_local_ids[i]]`，其余位置为 0。

---

## 5. SparseMlaCpImpl — 外层封装

`SparseMlaCpImpl` 继承 `SparseMlaImpl`，在外层处理 CP 特有逻辑。

### 5.1 `__init__`

```python
# CP 场景下 input_lengths 要用 actual lengths，不是 chunk lengths
attn_inputs_for_init = copy.copy(attn_inputs)
attn_inputs_for_init.input_lengths = cp_info.prefill_actual_input_lengths_cpu

super().__init__(..., fmha_impl=SparseMlaFp8CPOp)

# 把 write op 传给 CP op（CP op 内部自己写 cache）
self.fmha_impl.kv_cache_write_op = self.kv_cache_write_op
self.fmha_impl.write_cache_store_impl = self.write_cache_store_impl
```

### 5.2 `forward`

```python
def forward(self, q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices):
    # 1. RoPE — 用 full_rope_pos_ids，只对本 rank 有效的位置做 RoPE
    q_pe = q[:, :, nope_head_dim:]
    self.rope_impl.forward(q_pe, k_pe, self.rope_params,
                           precomputed_pos_ids=full_rope_pos_ids)

    # 2. Input BMM: q @ W_kc
    q_transformed = self._apply_input_bmm(q, layer_id)

    # 3. CP attention (all_gather → restore → write cache → attend → scatter)
    attn_output = self.fmha_impl.forward(
        q_transformed, compressed_kv, k_pe, topk_indices, ..., kv_cache)

    # 4. Output BMM: attn_output @ W_vc
    return self._apply_output_bmm(attn_output, layer_id)
```

注意与非 CP 版本的关键区别：
- 非 CP: RoPE 和 KV cache write 在 `SparseMlaImpl.forward()` 中做
- CP: RoPE 在 `SparseMlaCpImpl.forward()` 中做（用 full_rope_pos_ids），KV cache write 在 `SparseMlaFp8CPOp.forward()` 中做（因为要先 all_gather + restore）

### 5.3 `_refresh_cp_params`

把 plan 的结果打包成 `cp_params` namespace，供上层 indexer 使用：

```python
cp_params = SimpleNamespace(
    kv_restore_unpad_indices=...,
    total_global_ids=...,
    total_local_ids=...,
    cu_kv_seqlens_global=...,
    full_rope_pos_ids=...,
    precomputed_ks=fmha_params.ks[total_global_ids],
    precomputed_ke=fmha_params.ke[total_global_ids],
    precomputed_lengths=fmha_params.expanded_seq_lens[total_global_ids],
    precomputed_topk_off=fmha_params.topk_indices_offset[total_global_ids],
    precomputed_req_ids=...,
)
```

这些 precomputed 字段是对 fmha_params 按 `total_global_ids` 索引后的子集，避免 indexer 每次都做 gather。

---

## 6. 数据流总览

```
                    Rank 0                                    Rank 1
                    ──────                                    ──────
Input:          q_0, ckv_0, kpe_0                       q_1, ckv_1, kpe_1
                    │                                         │
                    ├── RoPE(q_pe_0, kpe_0, pos_ids_0)       ├── RoPE(q_pe_1, kpe_1, pos_ids_1)
                    │                                         │
                    ├── q_transformed_0 = q_0 @ W_kc         ├── q_transformed_1 = q_1 @ W_kc
                    │                                         │
         ┌──────────┴──────────────────────────────────────────┴──────────┐
         │                        all_gather(ckv, kpe)                    │
         │              每张卡都拿到完整的 [ckv_0||ckv_1, kpe_0||kpe_1]    │
         └──────────┬──────────────────────────────────────────┬──────────┘
                    │                                         │
                    ├── restore[kv_restore_indices]           ├── restore[kv_restore_indices]
                    │   → 按全局顺序重排                        │   → 同样的重排
                    │                                         │
                    ├── write to paged KV cache               ├── write to paged KV cache
                    │   (两张卡写入相同的 KV 数据)               │   (两张卡的 cache 内容一致)
                    │                                         │
                    ├── q0 = q_transformed_0[local_ids_0]     ├── q0 = q_transformed_1[local_ids_1]
                    │   (只取本 rank 负责的 Q)                   │
                    │                                         │
                    ├── attend(q0, cache, topk)               ├── attend(q0, cache, topk)
                    │                                         │
                    ├── scatter(out, local_ids_0, T)          ├── scatter(out, local_ids_1, T)
                    │                                         │
                    └── output_0 @ W_vc                       └── output_1 @ W_vc
```

---

## 7. 关键设计决策

### 7.1 为什么每张卡都写完整的 KV cache?

因为 sparse attention 的 topk 索引可能指向序列中任意位置的 KV token，不能假设某张卡只需要部分 KV。让每张卡持有完整 cache 是最简单正确的方案。

### 7.2 为什么 KV cache write 不在外层做?

非 CP 版本中，KV cache write 在 `SparseMlaImpl.forward()` 中做（RoPE 之后立即写）。CP 版本中必须先 all_gather 再 restore 再写，所以 write 移到了 `SparseMlaFp8CPOp.forward()` 内部。

### 7.3 total_local_ids_is_identity 优化

当 zig-zag 分配的结果恰好是连续的 `[0, 1, ..., n-1]` 时（常见于单 request 场景），跳过 `q[total_local_ids]` 的 gather 和最后的 scatter，直接用/返回原始 tensor。

### 7.4 两种 attend 路径详解

两条路径进入时的状态完全一致：KV cache 已写好（all_gather → restore → write 完成），`q0` 已是本 rank 负责的 Q tokens，`topk` 是 indexer 输出的 request-local 稀疏索引 `[T, 1, topk]`。最终都返回 `[n_q, H, kv_lora_rank]` 的 attention 输出。区别在于 **KV 数据怎么喂给 flash_mla kernel**，以及由此带来的 **topk 索引语义完全不同**。

#### 7.4.1 `_attend_with_kvcache` — 直接读 FP8 Paged Cache（默认路径）

```python
def _attend_with_kvcache(self, q0, kv_cache, topk, layer_id):
    # 1. cache 按 uint8 视图，补一个 head 维度
    #    → [num_blocks, block_size, 1, 656]
    kv_cache_flat = _as_uint8(
        kv_cache.kv_cache_base.view(-1, 1, kv_cache.kv_cache_base.size(-1))
    )

    # 2. topk 索引转换: request-local → paged cache 全局 slot ID
    global_topk = self._convert_topk_indices_to_global(topk)

    # 3. 直接在 FP8 paged cache 上做 attention
    attn_out, _ = flash_mla_with_kvcache(
        q=q0.unsqueeze(0),
        k_cache=kv_cache_flat,      # FP8 paged，kernel 内部解量化
        block_table=self.block_table,
        indices=global_topk,        # 只读 topk 指定的 slot
        is_fp8_kvcache=True,
        softmax_scale=self.scale,
    )
    return attn_out.squeeze(0)
```

**topk 索引转换详解**（`_convert_topk_indices_to_global` → `triton_convert_req_index_to_global_index`）：

indexer 输出的 topk 是 **request 内的 token 位置**，需要翻译成 paged cache 的全局 slot ID。这个翻译需要查 block_table（页表）：

```
假设: page_size = 64
      request 0 的 block_table = [page3, page7, page12]

topk_indices[q_i] = [5, 12, 100]   ← request 内第 5/12/100 个 KV token

token 5:   在 page3 的 slot 5   → 全局 slot = 3 * 64 + 5   = 197
token 12:  在 page3 的 slot 12  → 全局 slot = 3 * 64 + 12  = 204
token 100: 在 page7 的 slot 36  → 全局 slot = 7 * 64 + 36  = 484
           (100 // 64 = 1 → 第 1 页 = page7, 100 % 64 = 36)

global_topk[q_i] = [197, 204, 484]   ← paged cache 的全局 slot 索引
```

`flash_mla_with_kvcache` kernel 拿到全局 slot 索引后，直接从 FP8 paged cache 对应位置读取数据，在 kernel 内部做 FP8→BF16 反量化，然后计算 attention。

#### 7.4.2 `_attend_gather` — 先 Upconvert 到 BF16 连续 Buffer

需要设置环境变量 `USE_GATHER_PATH=1` 且为 prefill 阶段才会启用。

```python
def _attend_gather(self, q0, kv_cache, topk):
    ws = self._gather  # plan() 中预分配的 _GatherWorkspace

    # 1. 把 FP8 paged cache gather + upconvert 到 BF16 连续 buffer
    #    input:  [num_blocks, block_size, 656] FP8 paged (随机页)
    #    output: [total_kv_len, kv_lora_rank + rope_dim] BF16 连续
    src = _as_uint8(kv_cache.kv_cache_base)
    rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_v2(
        src,                       # FP8 paged cache
        ws.fused_kv,               # BF16 连续输出
        self.block_table.to(torch.int32),
        ws.seq_lens,               # 每个 request 的 KV 长度
        ws.workspace_starts,       # 每个 request 在 buffer 中的起始偏移
        ws.batch_size,
        ws.total_kv_len,
    )
    # workspace 内存布局变为:
    # [req0: kv0 kv1 ... kv99 | req1: kv0 kv1 ... kv49 | req2: ...]
    #  ^ws_starts[0]=0          ^ws_starts[1]=100         ^ws_starts[2]=150

    # 2. topk 索引转换: request-local → workspace 线性偏移
    offsets = ws.workspace_starts[self.precomputed_req_ids]
    topk_2d = _topk_2d(topk)
    padding_mask = topk_2d < 0
    raw_global = topk_2d + offsets.unsqueeze(1)
    global_indices = raw_global.masked_fill(padding_mask, -1)

    # 3. 在 BF16 连续 buffer 上做 attention
    out, _, _ = flash_mla_sparse_fwd(
        q0,
        ws.fused_kv.unsqueeze(1),  # [total_kv_len, 1, 576] BF16
        global_indices.unsqueeze(1),
        self.scale,
        d_v=self.kv_lora_rank,
    )
    return out
```

**topk 索引转换详解**：

和 with_kvcache 路径不同，这里不需要查页表，只需做简单加法：

```
workspace 布局 (BF16 连续, 所有 request 的 KV 紧密排列):
  [req0: kv0 kv1 kv2 ... kv99 | req1: kv0 kv1 ... kv49 | ...]
  offset:  0                     100                      150

Case 1: q_i 属于 request 0
  topk_indices[q_i] = [5, 12, 100]
  precomputed_req_ids[q_i] = 0
  offsets = ws.workspace_starts[0] = 0
  global_indices = [0+5, 0+12, 0+100] = [5, 12, 100]

Case 2: q_j 属于 request 1
  topk_indices[q_j] = [3, 7, -1]    ← -1 是 padding (不足 topk 个)
  precomputed_req_ids[q_j] = 1
  offsets = ws.workspace_starts[1] = 100
  raw_global  = [100+3, 100+7, 100+(-1)] = [103, 107, 99]
                                                      ↑ 严重错误!
  → masked_fill 修正:
  global_indices = [103, 107, -1]    ← -1 被保护了
```

**为什么 -1 保护至关重要**：不做 `masked_fill` 时，`-1 + 100 = 99`，这个位置属于 request 0 的 KV 区域（offset 0~99）。attention kernel 会把 request 0 的 KV 数据混入 request 1 的计算中，导致**跨 request KV 污染**，产生乱码或重复输出。

#### 7.4.3 总结对比

| 维度 | `_attend_with_kvcache` | `_attend_gather` |
|------|------------------------|-------------------|
| **开启条件** | 默认路径 | `USE_GATHER_PATH=1` 且 prefill 且非 CUDA Graph |
| **KV 数据格式** | FP8 paged（原地读） | BF16 连续 buffer（预拷贝） |
| **内存访问模式** | 随机跳页（page3→page7→page12） | 连续内存（一个大 buffer） |
| **FP8 反量化** | kernel 内部逐次做 | 预先由 C++ op 批量做一次 |
| **额外显存** | 无 | `total_kv_len × 576 × 2` bytes（BF16 workspace） |
| **调用的 kernel** | `flash_mla_with_kvcache`（原生支持 paged + FP8） | `flash_mla_sparse_fwd`（需要连续 BF16 输入） |
| **topk 索引语义** | paged cache 全局 slot ID（经页表翻译） | BF16 workspace 的线性偏移（简单加法） |
| **topk 转换方式** | `triton_convert_req_index_to_global_index`（查 block_table） | `topk + ws_starts[req_id]`（算术加法） |
| **-1 padding 处理** | kernel 内部处理 | 必须手动 `masked_fill` 保护 |
| **CUDA Graph** | 支持 | 不支持（workspace 大小随 batch 变化） |
| **适用场景** | 通用（decode + prefill） | 仅 prefill |

#### 7.4.4 为什么 gather 路径在大 prefill 时更快 (~1.7x)

**with_kvcache 路径的瓶颈**：

1. **随机内存访问**：kernel 需要通过 block_table 查到 page 地址，再跳到对应 page 读 FP8 数据。不同 topk 位置散落在不同 page，GPU cache line 利用率低，大量 L2 cache miss。
2. **重复反量化**：多个 Q token 的 topk 如果指向同一个 KV position（常见于 prefill），FP8→BF16 反量化会重复做多次。

**gather 路径的优势**：

1. **一次 gather，连续访问**：`cp_gather_and_upconvert_fp8_kv_cache_v2` 按序列顺序将所有 KV 连续排列到 BF16 buffer。后续 `flash_mla_sparse_fwd` 做连续内存读取，GPU memory bandwidth 利用率高。
2. **反量化只做一次**：gather op 一次性把所有 page 的 FP8 数据解量化到 BF16，之后的 attention kernel 直接读 BF16，无需重复解量化。
3. **kernel 效率**：`flash_mla_sparse_fwd` 针对连续 BF16 输入优化，无需处理 paged 寻址逻辑，指令路径更短。

**代价**：额外显存（BF16 workspace）和一次 gather+upconvert 拷贝。但对大 prefill（`seq_len` 数万~数十万）来说，attention 计算量远大于拷贝开销，净收益为正。Decode 阶段 Q 只有一个 token，gather 的固定开销不划算，所以只在 prefill 时启用。

---

## 8. CUDA Graph 兼容性

CP 的 plan() 会在 CUDA graph capture 模式下特殊处理：
- tensor 不能重新分配（shape/dtype/device 改变会报错）
- 用 `copy_()` 原地更新，保持 tensor 地址不变
- `_fp8_kernel_metadata` 的 tile scheduler 也需要保持地址稳定

这通过 `_copy_or_replace_graph_tensor()` 工具函数实现。

---

## 9. 常见调试

- **`RTP_LLM_PD_DEBUG=1`**: 在 plan() 首次调用时打印详细的索引信息
- 关注 `total_global_ids` 和 `total_local_ids` 是否合理
- 检查 `kv_restore_unpad_indices` 是否覆盖了所有有效 token
- `total_local_ids_is_identity` 为 True 时代表走了优化路径
