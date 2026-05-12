# block_table Shape and Invalid Values

> **Workspace:** c:\WorkingSpace\Code\vllm-ascend

---

## User
block_table的shape以及里面无效block的值

## Assistant
现在我来查看 v2 block_table，以便理解其

Now I have a comprehensive understanding. Let me also check the `PAD_SLOT_ID` value and how the block_table is used in the FIA/paged attention kernel:

Now I have a complete picture. Let me summarize the findings about `block_table`:

---

## `block_table` 的 Shape 与无效 Block 的值

### Shape

`block_table` 的 shape 为 **`(max_num_reqs, max_num_blocks_per_req)`**，即：

- **第 0 维**：最大请求数（`max_num_reqs`），每个请求对应一行
- **第 1 维**：每个请求最大可用的 block 数（`max_num_blocks_per_req`）

其中 `max_num_blocks_per_req` 的计算方式为：

```python
max_num_blocks_per_req = cdiv(max_model_len, block_size * total_cp_world_size)
```

参见 [block_table.py:77](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/block_table.py#L77)：
```python
self.block_table = self._make_buffer(max_num_reqs * duplicate_size, logical_table_size, dtype=torch.int32)
```

**dtype 为 `torch.int32`**。

在 hybrid block 模式下（`use_hybrid_blocks=True`），第 1 维会扩展为 `max_num_blocks_per_req * blocks_per_phys_block`，因为一个物理 block 会被拆分为多个逻辑 block。

### 实际使用时的切片

在 attention forward 中，`block_table` 会根据实际请求数进行切片：

| 场景 | 切片方式 | 代码位置 |
|---|---|---|
| PrefillCacheHit | `block_tables[:batch_size, :]` | [attention_v1.py:700](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L700) |
| DecodeOnly | `block_tables`（完整使用） | [attention_v1.py:717](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L717) |
| Int8 量化 decode | `block_tables[:num_decodes]` | [attention_v1.py:1213](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/attention_v1.py#L1213) |
| MLA | `block_table_tensor[:block_table_size]` | [mla_v1.py:442](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/attention/mla_v1.py#L442) |

### 无效 Block 的值

**无效 block 的填充值为 `0`**。

关键代码在 [model_runner_v1.py:2102](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2102)：

```python
blk_table_tensor[num_reqs:num_reqs_padded].fill_(0)
```

即对于 `num_reqs` 到 `num_reqs_padded` 之间的 padding 行，全部填 0。

同时，`BlockTable` 初始化时使用 `CpuGpuBuffer`，底层由 `torch.zeros` 创建，且 `clear()` 方法也是 `fill_(0)`（见 [block_table.py:195](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/block_table.py#L195)）。

**对比 `slot_mapping`**：slot_mapping 中无效 slot 的填充值为 **`-1`**（见 [model_runner_v1.py:2101](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2101)），而上游 vLLM 定义的 `PAD_SLOT_ID` 也是 `-1`。

### 总结

| 属性 | 值 |
|---|---|
| **Shape** | `(max_num_reqs, max_num_blocks_per_req)` |
| **dtype** | `torch.int32` |
| **无效 block 值** | `0` |
| **无效 slot 值** | `-1`（`PAD_SLOT_ID`） |

**为什么无效 block 是 0 而不是 -1？** 因为 block_table 中的值是物理 block 编号，block 0 是合法的 block（KV cache 的第 0 个 block）。无效行填 0 是安全的，因为 FIA 算子会根据 `actual_seq_lengths_kv` / `seq_lens` 来限制实际访问的 block 数量，不会越界读取 padding 行对应的 block 0 的数据。

