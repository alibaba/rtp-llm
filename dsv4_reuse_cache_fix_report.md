# DSV4 reuse_cache 问题修复报告

## 1. 背景和结论

本轮排查目标是解决 `v4_flash_native_fp4_fp8_tp1_pd_sm100` 冒烟测试中相同请求 Q0 和 Q2 输出不一致的问题。其中 Q2 命中 `reuse_cache`，预期应复用 Q0 的前缀 KV，并且在逻辑上得到与 Q0 一致的输出。

当前结论：

- reuse 链路的主问题不在单一算子，而是由多类 KV/STATE pool 读写假设叠加造成：SWA continuation prefill 读写、CSA/HCA/INDEXER STATE tail block 定位、当前 suffix compressed-K overlay、CSA decode compressed-K 写入、以及 pool 写入精度。
- 已提交 commit `f59726d50 Fix DSV4 tail cache pool indexing`，修复 STATE pool 按 tail physical blocks 读写的问题。
- 当前未提交 diff 继续修复了 SWA/CSA/HCA/INDEXER 读写、layer2 边界 state、continuation mask、compressed-K overlay、数值一致性和调试 dump。
- 最新冒烟日志中 Q0 和 Q2 的 actual 文本已经一致，Q2 `reuse_len=2560`，Q2 TTFT 明显小于 Q1，说明 reuse prefill 路径实际生效。
- 冒烟测试仍然对 golden 失败，但失败已经不是 Q0/Q2 reuse 输出分歧：Q0 本身和 golden 不一致，Q2 与 Q0 一致。

最新验证日志：

```text
Q0 reuse_len=0, TTFT=13936.671 ms
Q1 reuse_len=0, TTFT=9330.414 ms
Q2 reuse_len=2560, TTFT=3879.595 ms
Q0 actual == Q2 actual
```

日志路径：

```text
/data1/liudu.ld/.cache/bazel/_bazel_liudu.ld/84d057db8e5a90ce200fb9f7ad96ac63/execroot/rtp_llm/bazel-out/k8-opt/testlogs/internal_source/rtp_llm/test/smoke/v4_flash_native_fp4_fp8_tp1_pd_sm100/test.log
```

## 2. 定位方法

本次没有从最终文本直接猜测原因，而是按 tensor bisection 定位：

1. 先确认请求事实：
   - Q0 和 Q2 是相同逻辑请求。
   - Q2 命中 `reuse_len=2560`。
   - Q2 TTFT 明显低于非 reuse 请求，证明 reuse prefill 确实执行。
2. 开启 `MOEDBG`，先比较 prefill 顶层输出：
   - `prefill_layerXX_out`
   - `prefill_hc_reduced`
   - `prefill_final_norm`
3. 找到首个分歧层后，再打开该层 level-2 dump：
   - `LXX_decode_attn_in`
   - `LXX_decode_q`
   - `LXX_decode_kv`
   - `LXX_decode_topk_idxs`
   - `LXX_decode_swa_local/global`
   - `LXX_decode_cmp_local/global`
   - `LXX_decode_kv_packed`
   - `LXX_decode_attn_out`
4. 对 decode 按 `(start_pos, input_id)` 配对，而不是按 dump step 号机械对齐。
5. 对 paged KV 读写区分 local slot 和 global slot：
   - local slot 相同表示逻辑 topk 一致。
   - global slot 可以不同，因为 Q0/Q2 的 physical block 可能不同。
   - 如果 local 相同但 `kv_packed` 不同，问题在 pool writer 或 pool slot 映射。

这个流程已经整理为 skill：

```text
/data1/liudu.ld/.codex/skills/dsv4-reuse-cache-bisect/SKILL.md
```

## 3. 问题和修复明细

### 3.1 SWA continuation prefill 读取使用了错误的 ring 语义

涉及文件：

- `rtp_llm/models_py/modules/dsv4/attention.py`

问题原因：

continuation prefill 中，suffix 内每个 query 位置需要看到“该 query 全局位置之前”的 SWA window。但原实现把 continuation prefill 的 topk 当作最终 ring slot 来算，相当于所有 query 都从当前 suffix 写完后的 ring 状态读。这样早期 suffix token 会读到 later suffix token 覆盖后的 ring slot，Q2 的 prefill attention 上下文和 Q0 full prefill tail 不一致。

修复内容：

- `_get_window_topk_idxs` 和 `_get_window_topk_idxs_batched` 对 continuation prefill 改为返回 absolute token positions。
- 新增 `_prefill_read_swa_dense_abs_from_pool`，从 SWA pool 重建 dense absolute SWA view。
- 对当前 suffix 的 `kv_full` 做 overlay，使 dense view 表示 `[prefix tail | current suffix]`。
- continuation prefill 的 compressed offset 从原 `seqlen_full` 改为 `prefill_swa_dense_len`，保证 `topk_idxs` 指向新的 dense absolute 拼接布局。

为什么这样能解决：

Q0 full prefill 的 tail token 和 Q2 continuation suffix token 在逻辑上应读同一个全局 token window。absolute token positions 表达的是“每个 query 当时可见的历史”，不会被最终 ring 状态污染。current suffix overlay 则保证 suffix 内已产生的 KV 能被后续 query 立即读取。

### 3.2 SWA prefill 只写最终 window，导致 prefix-cache 停在中间边界时 tail blocks 不完整

涉及文件：

- `rtp_llm/models_py/modules/dsv4/attention.py`

问题原因：

`_prefill_write_swa_to_pool` 原本只写最后 `window_size` 个 token。对普通 attention 来说最终 sliding window 足够，但 prefix-cache reuse 可能停在一个早于最终 token 的 block 边界。下一次请求从该边界续算时，需要的是当前 request tail physical blocks 中对应的 SWA 行，而不是最终窗口的最后一段。

修复内容：

- `_prefill_write_swa_to_pool` 改为写入当前 prefill 中所有能映射到已分配 SWA block-table entry 的 token。
- 无效位置仍使用 `-1` sentinel，通过 `write_kv_to_pool(mask_negative=True)` 跳过。

为什么这样能解决：

reuse 的物理块粒度由 allocator 决定，不等价于最终 sliding window。写满当前 request 映射到的 tail physical blocks 后，无论 prefix-cache 在哪个已分配边界截断，后续 continuation prefill 都能读到一致的 SWA KV。

### 3.3 CSA/HCA/INDEXER STATE 读写假设 `block_table[:, 0/1]`，但 allocator 实际使用 tail physical blocks

涉及文件：

- `rtp_llm/models_py/modules/dsv4/compressor.py`
- commit `f59726d50`

问题原因：

`CSA_STATE`、`HCA_STATE`、`INDEXER_STATE` 是固定长度 request-local state，但底层由 tail-group allocator 分配。block table 在绝对 token-block 空间可能是稀疏的，当前 request 有效 state 应落在“最后几个有效 physical blocks”，而不是固定的 `block_table[:, 0]` 或 `block_table[:, 1]`。

原实现用 `_compute_pool_slots` 按 dense block index 直接查 block table，等价于认为 state pool 的第 0/1 块总在 block table 前两列。这在 reuse、PD、tail allocator 场景下会读写到错误 physical block。

修复内容：

- 新增 `_compute_tail_pool_slots`。
- `_bind_state_from_pool` 和 `_scatter_state_to_pool` 改为通过 `_compute_tail_pool_slots` 定位当前 request 的 tail physical blocks。
- `HybridKVCacheAllocator.cc` 未改动，按用户要求只在 Python 层修复读写定位。

`_compute_pool_slots` 和 `_compute_tail_pool_slots` 的核心区别：

- `_compute_pool_slots`：把 logical position 映射到 `block_table[:, logical_block_idx]`，适用于 dense、从第 0 列开始排列的 KV cache。
- `_compute_tail_pool_slots`：先找每个 request 行中最后 `tail_cap` 个有效 block，再把 state 的 dense row 映射到这些 tail blocks，适用于 tail allocator 管理的 STATE cache。

为什么这样能解决：

STATE tensor 自身是 dense 的，但它对应的 physical blocks 是 request 的尾部有效块。修复后，Python 层 bind/scatter 与 allocator 的真实分配策略一致，不再错误读取其他 block 或空洞 block。

### 3.4 layer2 CSA/INDEXER boundary state 在 block 边界续算时丢失

涉及文件：

- `rtp_llm/models_py/modules/dsv4/compressor.py`

问题原因：

CSA/INDEXER 的 overlap state 依赖前一个 compression ratio 窗口的尾部状态。Q2 从 block boundary 续算时，常见位置正好落在 `256` token 边界，STATE pool 里 request-local dense state 已经恢复，但边界处还需要上一块结束时的 overlap rows。没有额外保存和恢复这些边界 rows 时，layer2 的 CSA/INDEXER state 会在续算第一段发生偏差。

修复内容：

- 新增 `_STATE_BOUNDARY_CACHE`。
- 新增 `_remember_boundary_state`，在 prefill 覆盖完整 256-token boundary 时缓存对应 overlap rows。
- 新增 `_restore_boundary_state`，在 continuation 从 boundary 开始时恢复前一块结束状态。

为什么这样能解决：

边界续算需要的不是“当前块开头初始化 state”，而是“上一块末尾 overlap state”。显式缓存 completed boundary rows 后，Q2 进入 layer2 时能恢复与 Q0 full prefill tail 相同的 state。

### 3.5 continuation prefill 的 compressed topk 对每个 query 可见范围计算不正确

涉及文件：

- `rtp_llm/models_py/modules/dsv4/attention.py`

问题原因：

compressed topk 原逻辑在 `start_pos > 0` 时使用固定 `(start_pos + 1) // ratio`，导致 continuation suffix 中所有 query 看到相同数量的 compressed blocks。但实际第 `i` 个 query 的全局位置是 `start_pos + i`，可见 compressed blocks 应为 `(start_pos + i + 1) // ratio`。

修复内容：

- `_get_compress_topk_idxs` 和 `_get_compress_topk_idxs_batched` 改为按 query global position 计算可见 compressed block 上界。

为什么这样能解决：

这让 Q2 continuation prefill 的 compressed attention causal 约束与 Q0 full prefill 对齐。否则 suffix 后半段 query 会少看已经生成的 compressed block，产生隐藏状态偏差。

### 3.6 Indexer continuation prefill 缺少 causal mask

涉及文件：

- `rtp_llm/models_py/modules/dsv4/indexer.py`

问题原因：

`Indexer.forward` 原来只把 `start_pos == 0` 当作 prefill 并加 causal mask。continuation prefill 的 `seqlen > 1` 但 `start_pos > 0`，因此没有按 global position 屏蔽未来 compressed KV。

修复内容：

- `is_prefill` 改为 `not is_batched and seqlen > 1`。
- causal mask 的 query position 改为 `start_pos + arange(1, seqlen + 1)`。

为什么这样能解决：

无论 fresh prefill 还是 continuation prefill，只要是多 token prefill，就必须按每个 query 的全局位置做 causal mask。修复后 INDEXER 选出的 topk 与 full prefill 的因果可见范围一致。

### 3.7 当前 suffix compressed-K 没有及时写入和 overlay，导致 sparse attention 读到旧值或缺值

涉及文件：

- `rtp_llm/models_py/modules/dsv4/attention.py`
- `rtp_llm/models_py/modules/dsv4/indexer.py`
- `rtp_llm/models_py/modules/dsv4/compressor.py`

问题原因：

continuation prefill 中，当前 suffix 会产生新的 compressed-K。后续 query 和 Indexer 需要立即读到这些 compressed rows。原链路依赖 pool 中已有内容，但当前 suffix 的 compressed-K 未必已经写回对应 CSA/HCA/INDEXER pool，或者 dense `kv_cat`/`kv_cache` 未 overlay 当前 rows。

修复内容：

- `Attention._prefill_paged_write_kv_range`：把当前 prefill 产生的 `kv_compress` 写入 CSA/HCA KV pool。
- `Attention.forward`：continuation dense `kv_cat` 从 pool gather 后，把当前 `kv_compress` overlay 到对应 range。
- `Indexer._write_current_compressed_to_pool`：写当前 suffix compressed rows 到 INDEXER_KV pool。
- `Indexer._overlay_current_compressed`：bind pool 后把当前 compressed rows overlay 到 `self.kv_cache`。
- `Compressor` 增加 `_kv_write_mask`，只 scatter 本轮实际写过的 compressed rows，避免把未更新的旧行写回 pool。

为什么这样能解决：

sparse attention 和 indexer 读的是“prefix 已复用内容 + 当前 suffix 新生成内容”。写 pool 和 overlay dense view 两者都需要做：前者保证后续 paged read 正确，后者保证当前 forward 内立即可见。

### 3.8 CSA decode 只运行 Indexer，没有写主 CSA_KV

涉及文件：

- `rtp_llm/models_py/modules/dsv4/attention.py`

问题原因：

CSA decode 分支会运行 `Indexer`，但 `Indexer` 写的是 `INDEXER_KV`。CSA attention 真正读取的是 `CSA_KV`。因此在 decode 边界，topk 可以由 Indexer 产生，但 CSA_KV pool 对应 compressed row 可能没有由主 compressor 写入。

修复内容：

- 在 CSA decode 分支中，除了 `self.indexer.forward_decode_vectorized(...)`，还调用 `self.compressor.forward_decode_vectorized(x, start_pos)`。
- 非 vectorized 路径也对应调用 `forward_decode`。

为什么这样能解决：

Indexer 负责选 compressed block，主 compressor 负责给 CSA attention 的 pool 写 compressed-K。两条链路写入的 pool 不同，必须同时执行。

### 3.9 `write_kv_to_pool(mask_negative=True)` 的 delta-add 写法在 eager 路径留下 1 ULP stale 值

涉及文件：

- `rtp_llm/models_py/modules/dsv4/decode/kv_write_decode_op.py`

问题原因：

原写法为了兼容 `-1` slot 和 CUDA graph，使用 safe slot + delta-add：

```python
existing + (target - existing)
```

在 bf16 场景下，这种表达不保证得到 target 的逐 bit 值。排查中出现过 local `kv_cache` byte-equal，但 pool read 回来存在 1 ULP 差异，最终放大为 decode 分歧。

修复内容：

- 非 CUDA graph capture 路径改为对 valid rows 使用 `pool_view.index_copy_` 精确覆盖。
- CUDA graph capture 中保留原 safe-redirect/delta-add 路径，避免动态 shape 的 `nonzero` 破坏 graph capture。

为什么这样能解决：

eager 调试和常规运行中 valid pool row 现在被精确覆盖，不再经过 bf16 加减法重构。graph capture 路径仍保持原先的静态图安全行为。

### 3.10 shape-dependent GEMM/TileKernel 数值漂移导致 Q0 tail 和 Q2 suffix 不一致

涉及文件：

- `rtp_llm/models_py/modules/dsv4/block.py`
- `rtp_llm/models_py/modules/dsv4/compressor.py`
- `rtp_llm/models_py/modules/dsv4/indexer.py`
- `rtp_llm/models_py/modules/dsv4/transformer.py`

问题原因：

Q0 full prefill 和 Q2 suffix prefill 的 tensor shape 不同。即使数学上输入相同，底层 GEMM 或 TileKernel 可能因 M 维 shape 不同选择不同 kernel/tiling，产生微小数值差异。对自回归 decode 来说，这些差异可能被后续 topk、MoE 或采样放大。

修复内容：

- `Compressor._linear_abs_blocked`：按 absolute 256-token block 分段跑 projection。
- `Indexer._weights_proj_abs_blocked`：按 absolute 256-token block 分段跑 weights projection。
- `Block._hc_linear_mixes` 和 `V4Transformer._hc_head_reduce`：在有 absolute positions 时按 256-token block 对齐计算。
- 对 flat prefill case 避免使用与 Q2 suffix shape 不一致的 TileKernel 快路径。

为什么这样能解决：

Q0 tail token 和 Q2 suffix token 使用相同 absolute block 粒度执行关键 projection，减少 shape-dependent kernel 选择差异，使逐层 hidden 对齐更稳定。

### 3.11 调试 dump 能力补齐

涉及文件：

- `rtp_llm/models_py/modules/dsv4/_record_tensor.py`
- `rtp_llm/models_py/modules/dsv4/prefill/forward.py`
- `rtp_llm/models_py/modules/dsv4/decode/forward.py`
- `rtp_llm/models_py/modules/dsv4/block.py`
- `rtp_llm/models_py/modules/dsv4/attention.py`
- `rtp_llm/models_py/modules/dsv4/moe.py`
- `rtp_llm/models_py/modules/dsv4/compressor.py`
- `rtp_llm/models_py/modules/dsv4/indexer.py`

问题原因：

原有 dump 只能粗略覆盖前几层，无法稳定定位“Q0/Q2 第一个不同 hidden 出现在哪一层、哪个算子之后”。decode 多 step 和多进程还会发生 dump 文件名冲突。

修复内容：

- `_record_tensor.should_record_layer` 支持 `MOEDBG_LAYER` 和 `MOEDBG_ALL_LAYERS`。
- 支持 `MOEDBG_TAIL_TOKENS` 和 `MOEDBG_FULL_THRESHOLD`，避免大 tensor dump 失控。
- dump 文件名增加 pid。
- prefill/decode forward 记录每层输出、final norm、input ids、positions、start_pos。
- Block/Attention/Compressor/Indexer/MoE 增加 level-2 关键中间 tensor dump。

为什么这样能解决：

这不是业务逻辑修复，但让后续相同请求输出偏差可以直接按 hidden tensor 二分，不需要反复靠最终 token 猜测。

## 4. 当前变更 Review

### 4.1 高优先级风险：`_STATE_BOUNDARY_CACHE` 是 module-global，生命周期和 key 需要进一步收敛

位置：

- `rtp_llm/models_py/modules/dsv4/compressor.py`

风险：

`_STATE_BOUNDARY_CACHE` 当前是进程级全局 dict，key 包含 state pool ptr、block_id、compress_ratio、head_dim，但没有显式 layer_id/request id。pool ptr 通常能区分不同 state pool，block_id 也能定位 physical block，因此当前 Q0/Q2 场景能工作。但长期运行中，如果 pool 复用、block id 复用或 layer pool ptr 生命周期变化，存在 stale boundary state 被误用的风险。

建议：

- 如果可以拿到更稳定的 layer/pool identity，加入 key。
- 在 request/block 释放或 pool reset 时清理相关 boundary cache。
- 至少增加 cache size 或生命周期监控，防止长期服务进程内增长。

### 4.2 中优先级风险：CSA decode 额外运行 main compressor 有性能成本

位置：

- `rtp_llm/models_py/modules/dsv4/attention.py`

风险：

CSA decode 现在同时运行 Indexer 和 main compressor，语义上必要，因为两者写不同 pool。但这会增加 decode token 的计算量，尤其在非 boundary token 上如果 compressor 内部不能完全跳过无效写，可能影响延迟。

建议：

- 保持当前正确性修复。
- 后续评估是否能把 main compressor 调用限制到需要 emit compressed row 的 boundary token。
- 增加 decode benchmark，量化该改动对 TTFT/TPOT 的影响。

### 4.3 中优先级风险：absolute 256-token 分块里存在 `.item()` 和 Python loop

位置：

- `rtp_llm/models_py/modules/dsv4/block.py`
- `rtp_llm/models_py/modules/dsv4/compressor.py`
- `rtp_llm/models_py/modules/dsv4/indexer.py`
- `rtp_llm/models_py/modules/dsv4/transformer.py`

风险：

这些改动用于让 Q0 tail 和 Q2 suffix 的关键 projection 走一致 shape，正确性收益明确。但 `.item()` 和 Python loop 会引入 host sync，可能影响 prefill 性能。

建议：

- 短期保留，优先保证 reuse parity。
- 后续如果性能回归明显，可用更批量化的分段接口或在只需要 parity 的路径上加开关。

### 4.4 中优先级风险：非 capture `write_kv_to_pool` 使用动态 `nonzero`

位置：

- `rtp_llm/models_py/modules/dsv4/decode/kv_write_decode_op.py`

风险：

当前通过 `torch.cuda.is_current_stream_capturing()` 区分 graph capture 与 eager。eager 用 `nonzero + index_copy_` 是正确的，capture 保留原静态图安全写法。需要注意 CPU 或非 CUDA 环境下该分支也会走 `index_copy_`，目前逻辑可用，但最好有单元测试覆盖。

建议：

- 增加 `write_kv_to_pool(mask_negative=True)` 的 CPU/CUDA eager 单测，验证 valid row byte-equal 覆盖、invalid row 不改。
- 如果有 CUDA graph 单测，确认 capture 路径仍不触发动态 shape。

### 4.5 中优先级风险：debug dump 虽然 env-gated，但开启后成本很高

位置：

- `rtp_llm/models_py/modules/dsv4/_record_tensor.py`
- `rtp_llm/models_py/modules/dsv4/prefill/forward.py`
- `rtp_llm/models_py/modules/dsv4/decode/forward.py`
- `rtp_llm/models_py/modules/dsv4/block.py`
- `rtp_llm/models_py/modules/dsv4/attention.py`
- `rtp_llm/models_py/modules/dsv4/compressor.py`
- `rtp_llm/models_py/modules/dsv4/indexer.py`
- `rtp_llm/models_py/modules/dsv4/moe.py`

风险：

`MOEDBG=0` 时基本不生效；但开启后会 clone tensor、搬 CPU、保存 pt 文件，可能显著影响时序和显存/内存。`MOEDBG_TAIL_TOKENS` 和 `MOEDBG_FULL_THRESHOLD` 已经降低风险，但仍应只用于排障。

建议：

- 保留作为 reuse parity 排障工具。
- 在提交前确认默认环境下没有额外 dump 和 print。
- 文档中明确 dump 只用于 debug。

### 4.6 低优先级风险：`_dbg_prefix` 是临时属性，异常路径已处理但仍需保持局部性

位置：

- `rtp_llm/models_py/modules/dsv4/attention.py`
- `rtp_llm/models_py/modules/dsv4/compressor.py`
- `rtp_llm/models_py/modules/dsv4/indexer.py`

风险：

`_dbg_prefix` 用于给 Compressor/Indexer dump 加前缀。部分调用已经用 `try/finally` 清理，但这是 mutable module state，未来新增调用时容易漏清理。

建议：

- 保持 `try/finally` 模式。
- 后续可考虑 context manager，减少漏清理概率。

### 4.7 低优先级：当前 worktree 还有与本次修复无关的脏文件

当前 `git status` 中还有：

```text
M stub_source
?? .githooks/pre-push
?? "fer timeout"
?? "pensource]$ git log"
```

说明：

- `stub_source` 按用户要求不纳入 commit，不作为本次修复内容。
- 这些未跟踪文件看起来不是 DSV4 reuse 修复的一部分，建议提交前清理或确认用途。

## 5. 已执行检查

已执行：

```bash
git status --short
git diff --stat -- rtp_llm/models_py/modules/dsv4
git diff --check -- rtp_llm/models_py/modules/dsv4
rg -n "reuse_len|TTFT|actual|Q0|Q1|Q2" <latest test.log>
```

结果：

- `git diff --check` 对 DSV4 diff 未报告 whitespace/error。
- 最新 smoke log 显示 Q2 `reuse_len=2560`，TTFT 明显低于 Q0/Q1，Q0/Q2 actual 文本一致。
- smoke 仍对 golden 失败，当前应作为 golden 或非 reuse parity 问题继续单独处理。

## 6. 建议的后续动作

1. 为 `_compute_tail_pool_slots` 增加单元测试，覆盖 sparse block table、空洞、tail_cap 小于有效 block 数的情况。
2. 为 `write_kv_to_pool(mask_negative=True)` 增加 byte-equal 覆盖测试。
3. 为 continuation prefill 增加最小回归测试：
   - Q0 full prefill。
   - Q2 从 256-token boundary 续算。
   - 比较 SWA/CSA/HCA/INDEXER topk、pool gather、layer output。
4. 评估 `_STATE_BOUNDARY_CACHE` 的生命周期和清理策略。
5. 在正确性稳定后，压测 CSA decode 额外 compressor 调用和 absolute 256-token 分块对性能的影响。
