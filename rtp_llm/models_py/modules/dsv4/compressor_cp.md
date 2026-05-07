# DSV4 Compressor CP 机制说明

本文说明 `rtp_llm/models_py/modules/dsv4/compressor.py` 中 compressor 在
Context Parallel（CP）场景下的执行方式。

## 核心结论

CP 场景下，compressor 只把 `wkv` / `wgate` 线性投影分摊到各个 CP rank 上。
投影完成后，代码会把每个 rank 的 `kv` / `score` all-gather 回完整的全局序列，
然后每个 CP rank 都基于完整序列执行后续的 pooling、RMSNorm、RoPE 和 `kv_cache`
写入。

换句话说：

- 本地计算：`x_local -> kv_local / score_local`
- 跨 rank 同步：`kv_local / score_local -> kv_full / score_full`
- 每个 rank 重复执行：full-seq compressor pooling 和 cache 写入

所以 compressor 不是“每个 CP rank 只压缩自己的 seq 分片”，而是“每个 CP rank
只计算自己的投影分片，但压缩逻辑仍然按 full sequence 执行”。

## 相关代码入口

- `Compressor.set_cp_ctx()` 保存当前 forward 的 `CPContext`。
- `Compressor.forward()` 在 CP 场景下使用 `cp_ctx.prefix_length` 和
  `cp_ctx.seq_len_full` 作为 state 的全局起止位置。
- `Compressor._forward_scalar_impl()` 先做本地 `wkv` / `wgate`，再在 CP 场景下
  all-gather `kv` / `score`。
- `cp_should_gather()` 只判断 `cp_ctx is not None and cp_ctx.cp_size > 1`，并且
  故意忽略 `start_pos`，所以 fresh prefill 和 continuation prefill 都会 gather。
- `cp_all_gather_full_async()` / `cp_wait_gather_full()` 负责把本地二维
  `[T_local, H]` 恢复成全局顺序的 `[T_full, H]`。

## CPContext 保存了什么

`CPContext` 是每次 prefill forward 的 CP 元信息，主要包含：

- `cp_size`：CP rank 数。
- `cp_rank`：当前 rank。
- `chunk_length`：当前 rank 收到的本地 token 数，包括 padding。
- `padded_seq_len`：全局 padding 后长度，等于 `cp_size * chunk_length`。
- `seq_len_full`：真实全局输入长度，不含 padding。
- `relative_positions`：当前 rank 的本地 token 对应的全局相对位置。
- `prefix_length`：continuation prefill 已经存在于 cache 中的前缀长度。
- `global_positions`：`prefix_length + relative_positions`。
- `local_is_real`：本地 token 是否是真实 token，padding 为 false。
- `unpad_restore`：all-gather 后把拼接结果恢复成全局顺序、并去掉 padding 的索引。

## 执行流程

### 1. Transformer / Prefill 构建并传播 CPContext

prefill 入口根据 framework 传入的 CP metadata 构建 `CPContext`，然后传播到每层：

```text
V4Transformer / prefill forward
  -> build_cp_context(...)
  -> _propagate_cp_ctx(cp_ctx)
  -> attn.set_cp_ctx(cp_ctx)
  -> attn.compressor.set_cp_ctx(cp_ctx)
  -> indexer.set_cp_ctx(cp_ctx)
  -> indexer.compressor.set_cp_ctx(cp_ctx)
```

当没有 CP，或者 decode 阶段不需要 CP prefill gather 时，`cp_ctx` 会被清成 `None`，
模块自然退化为单 rank 路径。

### 2. Compressor 先按 rank-local token 做线性投影

进入 `Compressor._forward_scalar_impl()` 后，输入 `x` 在 CP prefill 中是当前 rank
的本地分片：

```text
x: [1, T_local, dim]
```

代码先在本地执行：

```python
kv = torch.nn.functional.linear(x_bf, self.wkv).float()
score = torch.nn.functional.linear(x_bf, self.wgate).float()
```

因此这部分计算量是按 CP 分摊的，每个 rank 大约只处理 `seq_len_full / cp_size`
个 token，外加 zigzag padding。

### 3. CP 场景下 gather 投影后的 kv / score

如果 `cp_ctx.cp_size > 1`，compressor 会把本地投影后的 `kv` 和 `score` 发起
all-gather：

```python
kv_gather_handle = cp_all_gather_full_async(kv.squeeze(0), cp_ctx, stream=gather_stream)
score_gather_handle = cp_all_gather_full_async(score.squeeze(0), cp_ctx, stream=gather_stream)
```

等待 gather 完成后：

```python
kv = cp_wait_gather_full(kv_gather_handle).unsqueeze(0)
score = cp_wait_gather_full(score_gather_handle).unsqueeze(0)
bsz, seqlen = 1, cp_ctx.seq_len_full
```

此时 `kv` / `score` 已经从本地分片变成完整全局序列：

```text
kv:    [1, seq_len_full, H]
score: [1, seq_len_full, H]
```

后续 compressor 逻辑看到的是 full sequence。

### 4. full sequence 上执行 pooling

prefill 分支会按 `compress_ratio` 把 full sequence 切成窗口：

```python
remainder = seqlen % ratio
cutoff = seqlen - remainder
kv = kv.unflatten(1, (-1, ratio))
score = score.unflatten(1, (-1, ratio)) + self.ape
```

然后执行 weighted pooling：

```python
kv = v4_compressor_pool(kv_c, sc_c)
```

或者 fallback：

```python
kv = (kv * score.softmax(dim=2)).sum(dim=2)
```

这一步发生在每个 CP rank 上，输入都是完整 `seq_len_full`。

### 5. RMSNorm、RoPE、kv_cache 写入也是 full sequence 语义

pooling 输出后，compressor 继续执行：

```python
kv = self._rmsnorm(kv.to(dtype))
freqs_cis = self.freqs_cis[sp_int : sp_int + cutoff : ratio]
apply_rotary_emb(kv[..., -rd:], freqs_cis)
```

随后写入 `kv_cache`：

```python
write_start = sp_int // ratio
self.kv_cache[:bsz, write_start : write_start + cutoff // ratio] = kv
```

CP 场景下 `sp_int` 来自 `cp_ctx.prefix_length`，所以 cache slot 使用的是全局位置，
而不是当前 rank 的本地位置。

## 例子：cp_size = 2，seq_len_full = 13

假设：

```text
cp_size = 2
seq_len_full = 13
padded_seq_len = 16
chunk_length = 8
pair_size = chunk_length / 2 = 4
```

CP 的 zigzag 切分会得到：

```text
rank0 relative positions:
[0, 1, 2, 3, 12, 13, 14, 15]

rank1 relative positions:
[4, 5, 6, 7, 8, 9, 10, 11]
```

因为真实长度是 13，所以全局位置 13、14、15 是 padding：

```text
rank0 real token: [0, 1, 2, 3, 12]
rank0 padding   : [13, 14, 15]

rank1 real token: [4, 5, 6, 7, 8, 9, 10, 11]
```

### 本地投影

两个 rank 各自只对自己的本地 token 做 `wkv` / `wgate`：

```text
rank0:
  [0, 1, 2, 3, 12, pad, pad, pad]
    -> kv0 / score0, shape [8, H]

rank1:
  [4, 5, 6, 7, 8, 9, 10, 11]
    -> kv1 / score1, shape [8, H]
```

### all-gather 后恢复全局顺序

all-gather 的物理拼接结果是：

```text
[rank0 rows, rank1 rows]
= [0, 1, 2, 3, 12, pad, pad, pad, 4, 5, 6, 7, 8, 9, 10, 11]
```

这不是全局 token 顺序，所以会使用 `unpad_restore` 恢复成：

```text
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

因此 gather 完成后，每个 CP rank 都拥有相同的：

```text
kv_full:    [1, 13, H]
score_full: [1, 13, H]
```

### full sequence compressor pooling

假设 `compress_ratio = 4`：

```text
seqlen = 13
cutoff = 12
remainder = 1
```

前 12 个 token 会参与本次压缩，最后 1 个 token 留到 state 中供下一次 continuation
使用：

```text
tokens [0, 1, 2, 3]    -> compressed kv_cache slot 0
tokens [4, 5, 6, 7]    -> compressed kv_cache slot 1
tokens [8, 9, 10, 11]  -> compressed kv_cache slot 2
token  [12]            -> kv_state / score_state
```

这三个 compressed block 会在每个 CP rank 上都写入自己的 `kv_cache`。

## overlap=True 的特殊情况

当前代码中：

```python
self.overlap = compress_ratio == 4
```

也就是 `compress_ratio == 4` 时会走 CSA overlap 路径。

overlap 路径下，`_overlap_transform()` 会把：

```text
[B, num_blocks, ratio, 2D]
```

变换为：

```text
[B, num_blocks, 2 * ratio, D]
```

语义上是：

- 当前窗口贡献后半部分 `D`
- 前一个窗口贡献前半部分 `D`
- 第一个窗口在 continuation prefill 时会从上一次保存的 state 中补 prefix tail

以上面的例子为例，fresh prefill 大致可以理解为：

```text
block0 pool input:
  [empty/prefix part] + tokens 0..3 的后半 D

block1 pool input:
  tokens 0..3 的前半 D + tokens 4..7 的后半 D

block2 pool input:
  tokens 4..7 的前半 D + tokens 8..11 的后半 D
```

如果是 continuation prefill，`block0` 的 prefix part 会来自上一次写入 state pool 的
边界状态。

## state pool 和 kv_cache 的全局位置语义

CP 场景下，compressor 绑定 state 时使用：

```text
state_start_pos = cp_ctx.prefix_length
state_end_pos   = cp_ctx.prefix_length + cp_ctx.seq_len_full
```

这保证了：

- fresh prefill 从全局位置 0 开始初始化 state；
- continuation prefill 能从全局 prefix 边界恢复 overlap state；
- `kv_cache` 写入 slot 使用 `prefix_length // compress_ratio` 开始的全局 compressed
  slot；
- 每个 CP rank 最终都有完整 compressed KV cache，后续 decode 不需要再为 compressor
  cache 做 CP gather。

## 计算量拆分

以 compressor prefill 为例：

| 阶段 | CP rank 上的输入长度 | 是否按 CP 分摊 |
| --- | --- | --- |
| `wkv` linear | `T_local` | 是 |
| `wgate` linear | `T_local` | 是 |
| `kv/score` all-gather | `T_local -> seq_len_full` | 通信 |
| compressor pooling | `seq_len_full` | 否，每个 rank 重复 |
| RMSNorm | compressed full blocks | 否，每个 rank 重复 |
| RoPE | compressed full blocks | 否，每个 rank 重复 |
| `kv_cache` 写入 | compressed full blocks | 否，每个 rank 写完整 cache |

因此，如果只问“每个 CP 的计算量是不是只有自己的那份 seq”，答案是：

- 对 `wkv` / `wgate` 投影：是。
- 对 pooling 及其后续 compressor 操作：不是，是每个 rank 都对 full sequence 做。

