# DSV4 CP Overlap Page-RR PD 解码复读排查记录

## 续接说明

如果对话上下文被压缩，继续排查前先读本文件。把它当作这个问题的持久工作记忆，后续有新结论也继续更新到这里。

用户要求的记忆规则：每次上下文压缩或 handoff 后，先重新加载本文件，再从最新 checkpoint 继续，不要从头排查。

重要范围说明：不要把本地已有的 `main_logs/`、`prefill_logs/`、`decode_logs/` 或旧的 `tmp_perf_artifacts/` 日志作为本问题证据。那些日志来自另一个问题。除非用户提供这个复现的新日志，否则以代码 diff、静态审查和 fresh smoke 为准。

## 问题背景

- 现象：当前分支在 page-RR + PD 分离场景下，decode 输出出现复读或跑飞。
- 用户指出的触发条件：`DSV4_PREFILL_CP_OVERLAP=1`。
- 用户给出的实际部署：
  - prefill: CP8
  - decode: DP16 + CUDA graph
  - prefill 开启 `kv_cache_sharded` 和 overlap
  - 当前猜测是 overlap 实现导致，不是 decode-side graph 本身。
- 已知好提交：`fc55d82eb179b5a9d2bd04084a8409d0a96aba42`。
- 本次审查范围：`fc55d82eb179b5a9d2bd04084a8409d0a96aba42..HEAD`。
- 工作假设：这是 CP/page-RR prefill cache 路径里的正确性回归，prefill 侧污染后续 decode。

## 最新结论和修复

- 已确认复现 case 的污染源是 **current-layer fresh SWA `kv_full` CP all-gather 被误纳入 prefill overlap**，不是用户原本设计的 prefix cache SWA gather/read overlap。
- 修复方式：删除 current-layer `swa_kv_full` async overlap 代码链；`_prefill_compute_qkv()` 的当前层 `swa_kv_full` 只走同步 `cp_all_gather_full_varlen()`，返回非 `PrefillWorkspace` union-backed 的 `kv_full`。
- 保留的 overlap：compressor overlap，以及 `_attn_via_workspace()` 中读取 cached prefix tail 的 SWA prefix overlap。
- 不采用的方案：不给 Q 或 SWA `kv_full` 新增额外大块存储，不做“Q/SWA 哪个单独 buffer”的复杂生命周期修复；因为这条 current-layer fresh KV all-gather 本来就不属于 overlap 设计范围。
- 验证：focused UT 通过；CP2/DP2 DeepSeek-V4 Flash page-RR overlap reuse smoke 在 GPU `0,1,2,3` 通过。
- 代码清理：已移除 `_should_overlap_swa_kv_gather_for_prefill()`、SWA 专用 CP stream、`_CP_ROLE_SWA_KV_FULL`、`PrefillQKV` pending handle 字段，以及 `PrefillWorkspace` 中的 `cp_gather_swa/cp_restore_swa` 预留区。

## good..HEAD 中最相关的提交

高可疑提交：

- `535145505 perf(dsv4): overlap CP prefill gathers`
  - 增加 overlap + CP + `kv_cache_sharded` 下的 async workspace read 路径。
  - 涉及 compressed-K pool reader、SWA prefix gather、indexer-K gather 和 attention 编排。
- `f9d014b62 perf(dsv4): reduce CP state-read cache overhead`
  - CP state-read cache 从全 block-table gather 改成只 gather tail block。
  - 正确性依赖 prefix/reuse start 的对齐假设。
- `a5c9f2953 chore(dsv4): consolidate CP streams and tracing`
  - 把 CP gather/post streams 改成进程级共享 stream。
  - 这会更依赖正确的 stream ordering 和 workspace 生命周期。

低可疑提交：

- `8f18ceb9f perf(dsv4): avoid CP last-hidden nonzero sync`
  - 只改 `ZigzagProcessor::computeLocalLastHidden`，更可能影响 final hidden gather，不太像 cache 内容污染。
- dash-sc/tool-call/metrics/repetition-monitor 相关提交更像下游观测或响应格式，不像根因。

## 2026-06-28 Checkpoint：新的主嫌

当时的最高概率根因：overlap SWA `kv_full` CP gather 把 restore 后的 `kv_full` 放进 `PrefillWorkspace` 的 CP region，但 `_materialize_prefill_q()` 把 Q 写入同一个 union workspace 后，`qkv.kv_full` 仍然会被后续 attention 使用。

相关代码：

- `rtp_llm/models_py/modules/dsv4/fp8/attention.py`
  - `_prefill_compute_qkv()` 在 `DSV4_PREFILL_CP_OVERLAP=1` 时调用 `cp_all_gather_full_async(..., cp_role=_CP_ROLE_SWA_KV_FULL, workspace=common.workspace)`，返回 `qkv.kv_full=None` 和 pending handle。
  - `_ensure_prefill_kv_full()` 等这个 handle，然后把 `qkv.kv_full` 返回成 `workspace.cp_restore_swa(...)` 的 view；prefix-restore fast path 下也可能是 `workspace.cp_gather_swa(...)` 的 view。
  - `_forward_prefill_compressed()` 会先调用 `_materialize_prefill_q()`，再调用 `_attn_via_workspace()`。
  - `_attn_via_workspace()` 后面在 `workspace.overlay_new_k` 中继续使用 `qkv.kv_full`。
  - `_forward_prefill_swa_only()` 也会先 `_materialize_prefill_q()`，再由 `_attn_fp8_swa_via_kv_full()` 消费 `qkv.kv_full`。
- `rtp_llm/models_py/modules/dsv4/prefill_workspace.py`
  - `prefill_q()` 使用 `[0, q_bytes)`。
  - CP buffers 和 Q 共享同一个 union tensor：
    - `cp_gather_swa` 从 `2*main_bytes + 2*idx_bytes` 开始。
    - `cp_restore_swa` 从 `2*main_bytes + 2*idx_bytes + swa_bytes` 开始。
  - 旧注释认为 `_ensure_prefill_kv_full` 在 Q materialization 前“消费”了 SWA gather，这个说法太弱。它只是 wait/restore，`qkv.kv_full` 仍然是 overlay/direct attention 的 live input。

为什么这匹配复读现象：

- 只在 `DSV4_PREFILL_CP_OVERLAP=1` 激活。
- 会在 cache 写完之前污染 prefill hidden states，decode 侧 kernel 即使没错，也可能后续复读。
- 不需要 tail-only state read 出错也能解释。
- 对 CP=2 的 PD 场景非常吻合。以 43B 形状举例，`n_heads=64`、`head_dim=512`、`q_rows=4096`、`full_rows=8192`、`main_w=2048`、`idx_w≈512`、`swa_w=512`：
  - `q_bytes = 4096 * 64 * 512 * 2 = 268 MiB`
  - `cp_restore_swa` 偏移大约是 `2*64MiB + 2*16MiB + 8MiB = 168 MiB`
  - 所以 Q materialization 会在 `_attn_via_workspace()` 或 `_attn_fp8_swa_via_kv_full()` 消费 SWA `kv_full` 前覆盖其存储。
- 某些更大的 CP size 或配置下，SWA region 可能在 `q_bytes` 之后，所以这个 bug 可能对配置敏感。

最初建议的 A/B：

1. 保持 `DSV4_PREFILL_CP_OVERLAP=1`，但让 `_should_overlap_swa_kv_gather_for_prefill()` 返回 false。
   - 这样保留 compressor overlap 和 async workspace-read 实验，同时让 `kv_full` 来自 `cp_all_gather_full_varlen()` 的新存储。
   - 如果复读消失，根因就是 SWA `kv_full` workspace/Q alias 生命周期问题。
2. 后续已经确认，最终修复方向不是为 Q 或 SWA 再引入额外常驻/临时大块存储，而是收窄 overlap feature 边界：
   - 当前层 fresh SWA `kv_full` 不属于 prefix cache SWA overlap。
   - 不再让 `_prefill_compute_qkv()` 对它启动 workspace-backed async CP gather。
   - `kv_full` 继续来自同步 `cp_all_gather_full_varlen()`，Q 仍按原设计复用 `PrefillWorkspace.prefill_q()`。

这段曾被用户给出的 CP8 部署信息短暂降级：当时估算 CP8 下 SWA CP sub-region 会在 Q region 之后，所以认为 alias 可能不是 CP8 复现的第一主嫌。后续 CP2/DP2 smoke 证明这个 alias 是真实可复现问题。

## 2026-06-28 更新：CP8 部署一度收窄怀疑范围

用户澄清失败部署是 prefill CP8、decode DP16 + CUDA graph，prefill 开启 `kv_cache_sharded` 和 overlap。

这会降低 SWA `kv_full` / Q workspace alias 直接解释该 CP8 复现的概率：

- Workspace sizing 使用 capacity 值：
  - `q_bytes = q_rows * n_heads * head_dim * 2`
  - `full_rows = q_rows * cp_size`
  - `main_w = 4 * head_dim`，fp32
  - `idx_w = 4 * index_head_dim`，fp32
  - `swa_w = head_dim`，bf16
- 当前 DSV4 默认 `n_heads=64`、`head_dim=512`、`index_head_dim=128`：
  - `q_bytes / q_rows = 65536`
  - `cp_gather_swa` 每个 `q_row` 的偏移是 `cp_size * (32*head_dim + 32*index_head_dim) = cp_size * 20480`
  - `cp_restore_swa` 每个 `q_row` 的偏移是 `cp_size * (34*head_dim + 32*index_head_dim) = cp_size * 21504`
  - CP8 下 `cp_gather_swa/q_bytes = 2.5x`，`cp_restore_swa/q_bytes = 2.625x`，两者都从 Q region 之后开始。
- 因此，前面记录的 SWA `kv_full` / Q alias 对小 CP/config 仍然是真风险，但当时看起来不太像 CP8 复现的直接原因。

当时对 CP8/page-RR/overlap 的第一主嫌转为 `535145505` 的 async workspace-read 路径，因为 gate 正好是：

- `DSV4_PREFILL_CP_OVERLAP=1`
- CP active
- `kv_cache_sharded=True`

这些 read 不直接写 decode KV cache，但会影响 prefill attention 输出。如果它们 restore 了 stale/partial compressed K 或 SWA prefix K，后续层会写入被污染的 KV cache，PD decode 再消费；decode CUDA graph 可能让现象稳定，但不一定是源头。

当时建议的 CP8 A/B 顺序：

1. 保持 `DSV4_PREFILL_CP_OVERLAP=1`，只禁用 `_should_async_workspace_reads_for_prefill()`。
   - 如果复读消失，根因在 `535145505` 的 async compressed-K / SWA-prefix / indexer-K read 实现。
2. 如果确认，再隔离子路径：
   - `_attn_via_workspace()` 里的 compressed-K async read。
   - `_attn_via_workspace()` 里的 SWA-prefix async byte-sliced read。
   - `IndexerFP8._gather_prefill_k_cache()` 里的 indexer-K async gather。
3. SWA `kv_full` overlap A/B 保留为二级检查。

## 当前结论

最终 CP2/DP2 page-RR smoke 的 A/B 结果证明：本次已复现的复读根因是 SWA `kv_full` overlapped CP gather 路径，不是 async workspace reads。

原因：

- baseline page-RR reuse 通过。
- full overlap page-RR reuse 在 63k cold prefill 请求上失败。
- overlap + 禁 async workspace reads 仍失败。
- overlap + 只禁 SWA `kv_full` gather overlap 通过。
- 所以根因是 SWA `kv_full` workspace/Q alias 生命周期 bug。

`f9d014b62` 的 tail-only CP state-read 仍值得警惕：如果 prefix/reuse start 没有足够对齐，它可能错误。但 page size 256/512 本身尚不能证明该条件被破坏。

## Page Size / Fixed Pool 结论

在 `good..HEAD` 范围内，没有发现这些文件有相关 diff：

- `rtp_llm/cpp/cache/DSV4CacheConfigHelper.cc`
- `rtp_llm/models_py/bindings/OpDefs.h`
- `rtp_llm/cpp/cache/HybridKVCacheAllocator.cc`

所以 page/fixed-pool sizing 模型不是坏范围中新引入的。

当前代码里的运行时 sizing：

- `DSV4CacheConfigHelper.cc`
  - `physical_tokens_per_block = kv_cache_config.seq_size_per_block`，默认 `256`。
  - `kernel_tokens_per_block = kv_cache_config.kernel_seq_size_per_block`，默认和 physical 一样。
  - FULL paged regions：
    - `CSA_KV`: entries/block = `kernel_tokens_per_block / 4`
    - `HCA_KV`: entries/block = `kernel_tokens_per_block / 128`
    - `INDEXER_KV`: entries/block = `kernel_tokens_per_block / 4`
  - prefill CP slicing 下的 fixed/SWA regions：
    - `fixed_tokens_per_block = physical_tokens_per_block * cp_size`
    - page=256、cp=4 时，fixed/SWA row 覆盖 `1024` tokens
    - page=512、cp=4 时，fixed/SWA row 覆盖 `2048` tokens
- `OpDefs.h`
  - FULL region 的 `getLayerCache(region)` 在 kernel size 设置时暴露 `seq_size_per_block = kernel_seq_size_per_block`。
  - fixed/SWA region 暴露 `seq_size_per_block = groupSeqSizePerBlock(group_id)`，也就是 physical * cp。
- `_kv_cache_utils.require_pool_tokens_per_block()`
  - FULL regions 使用 kernel row size。
  - fixed/SWA regions 使用 group row size。

解释：

- FULL paged kernel rows 和 fixed/SWA compact rows 的 mismatch 是有意设计。
- 这个 mismatch 不是 `good..HEAD` 新引入的。
- 对 state/SWA ring blocks，关键 token 覆盖是 fixed/SWA `state_tokens_per_block`，不是 FULL kernel page size。

## CP State-Read Tail-Only 检查

`f9d014b62` 修改了：

- `rtp_llm/models_py/modules/dsv4/fp8/compressor.py` 里的 `_build_cp_full_state_read_cache()`

旧行为：

- 从 block table gather 所有有效的 CP-sliced state blocks。

新行为：

- `_select_cp_state_read_tail_blocks()` 只 gather 包含 `seq_start_per_req - 1` 的 block。
- 它明确依赖 prefix-cache match 产生的 reuse length 和 state block size 对齐。

Compressor state read window：

- CSA / indexer: `token_count = (1 + overlap) * ratio = 8`，suffix length `7`。
- HCA: `token_count = 128`，suffix length `127`。

只选 tail block 在每个 request 满足以下条件时是安全的：

- `seq_start % state_tokens_per_block == 0`，或
- `seq_start % state_tokens_per_block >= suffix_len`。

page=256 或 page=512 且 cp=4 时：

- fixed `state_tokens_per_block` 是 1024 或 2048。
- 如果 `seq_start` 对齐 physical page size，那么 fixed row 内 offset 为：
  - page=256: `0, 256, 512, 768`
  - page=512: `0, 512, 1024, 1536`
- 这些 offset 对 HCA suffix 127 和 CSA suffix 7 都安全。

因此 page=256/512 本身大概率不足以破坏 tail-only state-read。真正风险是 `seq_start_per_req` 变成不对齐，例如 P2P side-channel reuse length、partial prefix reuse，或 decode/prefill handoff 直接设置 token reuse 而不是 block-aligned reuse。

需要用 runtime logs/asserts 验证：

- `seq_start_per_req`
- `state_tokens_per_block`
- `suffix_len`
- `seq_start % state_tokens_per_block`
- 是否满足 `offset == 0 || offset >= suffix_len`

## Reuse Length / P2P Side-Channel 结论

正常 cache reuse length 路径：

- `StreamCacheResource::updateReuseLengthsFromContext()`
  - 使用 `reuseBlockTokens()`。
- `StreamCacheResource::reuseBlockTokens()`
  - CP sharding 下返回 `CPSlotMapper::virtualBlockSize()`，也就是 `seq_size_per_block * cp_size`。

所以正常 context 路径应该产生 CP-virtual-block-aligned reuse lengths。

P2P side-channel 路径：

- `applyP2PSideChannelToStream()` 直接应用：
  - `payload->total_reuse_len`
  - `payload->local_reuse_len`
  - `payload->remote_reuse_len`
  - `payload->memory_reuse_len`
- 这绕过了本地 block count 到 virtual-block tokens 的转换。
- 观察到的来源：
  - `PrefillLoadCaller::Result::updateStreamFromResponse()`
  - `P2PConnector.cc`
  - `PrefillRpcServer.cc` 把 prefill/decode aux reuse lengths 复制到 response。

未关闭的点：

- 仍需证明 page-RR PD 路径里的 payload reuse lengths 总是 CP-virtual-block aligned。
- 如果 payload length 可能只是 raw physical-page aligned 或 token-exact，就可能破坏 `f9d014b62` tail-only 假设。

## Async Workspace Read 风险细节

相关文件：

- `rtp_llm/models_py/modules/dsv4/fp8/attention.py`
- `rtp_llm/models_py/modules/dsv4/fp8/_pool_reader.py`
- `rtp_llm/models_py/modules/dsv4/fp8/_swa_dequant_triton.py`
- `rtp_llm/models_py/modules/dsv4/fp8/_indexer_cp_assembler.py`
- `rtp_llm/models_py/modules/dsv4/fp8/indexer.py`

具体高风险变化：

- `_attn_via_workspace()` 现在会先 start compressed-K 和 SWA-prefix async reads，再 overlay new K 到 `workspace`，combine topk，然后在 FlashMLA 前 prepare/wait side-stream post work。
- `flash_mla_sparse_fwd()` 在 `wait_*` 之后读 workspace，所以任何缺失 event/wait 或 buffer lifetime 问题都会静默喂入 stale/partial K。

为什么这看起来匹配症状：

- 这条路径只在 overlap + CP + page-RR 下激活。
- 它影响 prefill attention 输出，以及 decode 后续消费的 cache state。
- silent stale K/state corruption 是 decode 后续复读的合理原因。

## 已完成的 A/B 检查

1. 保持 `DSV4_PREFILL_CP_OVERLAP=1`，只禁 async workspace reads。
   - 结果：复现仍失败。
   - 结论：compressed-K async read / prefix cache SWA async read / indexer-K async gather 不是这次已复现 case 的唯一根因。

2. 保持 `DSV4_PREFILL_CP_OVERLAP=1`，只禁当前层 fresh SWA `kv_full` overlap gather。
   - 结果：原复现 smoke 通过。
   - 结论：根因是 `qkv.kv_full` / `PrefillWorkspace.prefill_q()` alias 生命周期 bug。

3. tail-only CP state-read 尚未证明参与本次复现。
   - page=256/512 与 fixed/SWA block size 的关系本身看起来安全。
   - 如果后续遇到 reuse length 非 CP-virtual-block 对齐的 case，再单独验证 `f9d014b62` 的 tail-only state-read selection。

保留的 debug-only 对齐检查建议：
   - `seq_start_per_req`
   - `state_tokens_per_block`
   - `suffix_len`
   - 安全条件：`offset == 0 || offset >= suffix_len`

保留的启动配置检查建议：
   - `DSV4 physical block=...`
   - `kernel block=...`
   - 每一条 `DSV4 pool desc`
   - prefill/decode role CP size 和 `kv_cache_sharded`

## 当前风险排序

1. SWA `kv_full` workspace/Q alias 生命周期 bug：已由 A/B 和 smoke 确认，当前修复已切断。
2. `535145505` async workspace reads：本次已复现 case 中不是唯一根因；代码仍需保持 rank-identical shape 和 wait/prepare ordering。
3. `f9d014b62` CP state-read tail-only：reuse length 不对齐时可能；page=256/512 本身看起来安全。
4. `a5c9f2953` shared CP streams / multi-stream NCCL ordering：如果 async path ordering/lifetime 不完整，可能参与。
5. Page/fixed-pool physical sizing：相关 sizing 代码在 `good..HEAD` 未变，作为新根因概率低。

## 2026-06-28 续查：静态审查和 smoke 构造

不要依赖旧本地日志。只从代码审查和新 smoke 继续。

对 `535145505` 的补充静态审查：

- 回归面一开始仍然更像 async workspace reads，因为 gate 正好是 overlap + CP + `kv_cache_sharded`。
- async 路径包括：
  - `_attn_via_workspace()` 通过 `CPShardedPoolReader.start_fill_async()` 读 compressed-K。
  - 通过 `_swa_dequant_triton.start_dequantize_and_gather_k_cache_slots_cp_byte_sliced()` 读 SWA prefix。
  - 通过 `IndexerFP8._gather_prefill_k_cache()` 和 `_indexer_cp_assembler.start_assemble_indexer_k_async()` 读 indexer-K。
- 明显的 stream-lifetime 检查大多存在：side stream 会 wait current stream 后读 producer tensors；临时输入使用 `record_stream`；FlashMLA / score 消费输出前 current stream 会 wait post-stream ready event。
- 剩余可疑类别更偏语义问题，而不仅是 allocator lifetime：
  - delayed restore 会在其他 default-stream work 后写入 attention/indexer workspace。
  - 所有 CP workspace reads 共享同一个 TP process group 和 serialized CP stream，rank-dependent conditional launch 或隐藏 shape divergence 都可能 corrupt/hang。
  - SWA-prefix async all-gather shape 依赖 compaction 的 `unique_blocks.numel()`，在 CP byte-sliced page-RR 下必须 rank-identical。
- 这仍然比 page size/fixed-pool sizing 本身更匹配“只在 overlap + page-RR/kv_cache_sharded 下触发”。

已加 concrete smoke：

- 用户澄清 CP size 不是关键条件；用 Flash CP2/DP2 page-RR smoke 复现，不需要 CP8/DP16。
- 在 `internal_source/rtp_llm/test/smoke/BUILD` 手动加 DeepSeek-V4 Flash smoke target：
  - name: `smoke_v4_flash_pd_cp2ep2_dp2ep2_mtp_page_rr_overlap_logits_sm100`
  - baseline A/B target 已有：`smoke_v4_flash_pd_cp2ep2_dp2ep2_mtp_page_rr_logits_sm100`
  - prefill: CP2/EP2，page-RR `--prefill_cp_kv_cache_sharded 1`，`DSV4_PREFILL_CP_OVERLAP=1`
  - decode: DP2/EP2，CUDA graph enabled，`--prefill_cp_size 2`
  - page size: `--seq_size_per_block 256`，`--kernel_seq_size_per_block 128`
  - fixture: 既有 logits/reuse fixture `q_r_v4_flash_pd_cp2ep2_dp2ep2_mtp_page_rr_logits_sm100_arm.json`
- 使用方式：
  - 先跑 baseline target，再跑 overlap target。
  - 如果 overlap 失败而 baseline 通过，可确认 overlap/page-RR 回归与 CP size 无关。
  - 下一步隔离 A/B 应在保持 smoke topology 不变时禁用 `_should_async_workspace_reads_for_prefill()`。

## 2026-06-28 更新：用户把复现收窄到 CP2/DP2

用户澄清：不要把问题视为 CP-size-dependent；DeepSeek-V4 Flash smoke 使用 prefill CP2 和 decode DP2 就应该足够复现。

这让 SWA `kv_full` / Q workspace alias 在 CP2 smoke 中重新成为强主嫌：

- `DSV4_PREFILL_CP_OVERLAP=1` 时，`_prefill_compute_qkv()` 会启动 SWA `kv_full` CP all-gather，目标是 `PrefillWorkspace.cp_restore_swa`。
- compressed overlap 路径随后调用 `_ensure_prefill_kv_full()` 并写 SWA cache，但 `qkv.kv_full` 会作为 workspace view 继续存活。
- `_forward_prefill_compressed()` 随后调用 `_materialize_prefill_q()`，把 Q 写入同一个 union buffer 前部的 `PrefillWorkspace.prefill_q()`。
- `_attn_via_workspace()` 后面在 `workspace.overlay_new_k` 中消费 `qkv.kv_full`。如果 `cp_restore_swa` byte range 和 Q byte range 重叠，就会把被污染的 K 复制进 prefill attention workspace。

用当前 sizing 代码做 offset 检查：

- `q_bytes = q_rows * n_heads * head_dim * 2`
- `cp_rows = q_rows * cp_size`
- `main_w = 4 * head_dim`，`idx_w = 4 * index_head_dim`，`swa_w = head_dim`
- `cp_restore_swa_offset = cp_rows * (34 * head_dim + 32 * index_head_dim)`
- 存在重叠的条件：
  `cp_size * (34 * head_dim + 32 * index_head_dim) < 2 * n_heads * head_dim`

DeepSeek-V4 风格维度下，`head_dim=512`、`index_head_dim=128`：

- 如果 `n_heads=128`，`cp_restore_swa_offset / q_bytes = cp_size * 21504 / 131072`
  - CP2: 约 0.328，明确在 Q 内。
  - CP8: 约 1.3125，起点在 Q 后面。
- 如果 `n_heads=64`，CP2 约 0.656，也在 Q 内。

所以 CP2/DP2 是这个 alias bug 的好 smoke。它不一定解释每一个 CP8 部署，但它是一条具体的 overlap-only corruption 路径，应优先在 CP2 smoke 中隔离。

验证状态：

- Bazel target query 通过：
  `//internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_mtp_page_rr_overlap_logits_sm100`
- 2026-06-28 在本机 GPU `0,1,2,3` 使用 `--config=cuda13`：
  - Baseline target 约 470s 通过：
    `//internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_mtp_page_rr_logits_sm100`
  - Overlap target 约 479s 通过：
    `//internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_mtp_page_rr_overlap_logits_sm100`
  - fresh logs 确认 DSV4 physical block `256`，kernel block `128`，CP2 下 fixed/SWA pool `tokens_per_block=512`。也就是说 fixed/SWA pool rows 使用 `physical_page * cp_size`，不是 kernel page size。
- 这个 CP2/DP2 logits smoke 没有复现复读。它没有清掉 overlap 主嫌，只说明 logits fixture 没有压到同一个长 decode / reuse / PD handoff 行为。

后续 long-generation smoke：

- 已有 baseline target：
  `//internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_mtp_page_rr_cowboy_longgen_sm100`
- 新增 overlap A/B target：
  `//internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_mtp_page_rr_overlap_cowboy_longgen_sm100`
- 使用既有 deterministic long-generation fixture：
  - `max_new_tokens=5000`
  - `temperature=0`
  - `top_k=1`
  - 返回 output ids
  - 精确 expected response
  - 比 logits fixture 更接近“decode 输出复读”的观察。
- 本机路径修复：把 longgen fixture 和 target checkpoint path 改为 `/mnt/nas1/hf/DeepSeek-V4-Flash`。
- 2026-06-28 状态：overlap longgen target 的 Bazel query 通过。下一步是 GPU `0,1,2,3` 跑 baseline 再跑 overlap。
- 2026-06-28 在本机 GPU `0,1,2,3` 使用 `--config=cuda13`：
  - Baseline longgen target 约 484s 通过：
    `//internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_mtp_page_rr_cowboy_longgen_sm100`
  - Overlap longgen target 约 493s 通过：
    `//internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_mtp_page_rr_overlap_cowboy_longgen_sm100`
- 解释：单请求 long-generation 仍不足以复现。下一个 smoke 应覆盖 prefix/reuse/multiple-request，因为最高可疑的 async workspace-read 路径会读已有 cache/state；单个 fresh prompt 主要验证 write + decode，而不是 cache-hit prefill。

后续 page-RR reuse smoke：

- 新增 baseline target：
  `//internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_page_rr_reuse_memory_cache_sm100`
- 新增 overlap target：
  `//internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_page_rr_overlap_reuse_memory_cache_sm100`
- 它们复用既有 CP2/DP2 multi-request fixture：
  `q_r_v4_flash_pd_cp2ep2_reuse_cache_sm100_arm.json`
- 同时在两侧加 `--prefill_cp_kv_cache_sharded 1`，decode 加 `--prefill_cp_size 2`。
- fixture 覆盖：
  - repeated short prompt cache hit
  - 15k memory-cache reuse
  - 64k prefill
  - small prompt batch
- 这个 target 比 logits 和单请求 longgen 更贴近 async read 主嫌，因为 `_attn_via_workspace()` 只有 `common.any_cont` 为 true 时才会 start async SWA-prefix path，compressed-K reader 也需要非零 cached compressed prefix，也就是 `wm.N > 0`。

第一次 baseline 尝试：

- GPU `0,1,2,3`，约 503s 后失败，但只是 usage accounting：
  - expected `cached_tokens=2688`
  - actual `cached_tokens=2560`
  - 生成内容完全匹配：`DSV4_PD_REUSE_CACHE_OK`
  - aux 显示 reuse 生效：`memory_reuse_len=2560`，`prefill_memory_reuse_len=2560`
- 这不是复读/正确性失败。
- 因为 page-RR 下 fixed cache-token 计数可以合法变化，而内容才是正确性信号，所以给该 cache-accounting 请求加了 `compare_config: {"skip_usage": true}`。

重新跑 usage compare patch 后：

- Baseline page-RR reuse target 约 158s 通过：
  `//internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_page_rr_reuse_memory_cache_sm100`
- Overlap page-RR reuse target 约 475s 失败：
  `//internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_page_rr_overlap_reuse_memory_cache_sm100`

fresh Bazel 输出里的失败细节：

- Query: `q_r_v4_flash_pd_cp2ep2_reuse_cache_sm100_arm.query_4.json`
- Prompt tokens: `63287`
- Expected content: `DSV4_PD_REUSE_TECH_DOC_OK`
- Actual content: `" 2 2/3 cup of water.\n 2/3 cup of"`
- Finish reason: expected `stop`，actual `length`
- Aux info: `reuse_len=0`，`memory_reuse_len=0`，`prefill_memory_reuse_len=0`

重要解释：

- 第一个真正复现是 overlap-only + page-RR。
- 失败请求是 long cold prefill，不是 memory-cache hit 请求。
- 前面的 2.8k memory-cache reuse 请求以 `reuse_len=2560` 通过。
- 这把方向重新指向 long-context overlap prefill cache/workspace ordering。

隔离 async workspace reads：

- 在 `attention.py` 加 isolation env gate：
  `DSV4_PREFILL_CP_ASYNC_WORKSPACE_READS=0`
- 它只禁用 `_should_async_workspace_reads_for_prefill()`，同时保持 `DSV4_PREFILL_CP_OVERLAP=1`。
- 新增 isolation target：
  `//internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_page_rr_overlap_no_async_reads_reuse_memory_cache_sm100`
- 在 GPU `0,1,2,3` 跑 no-async isolation target：
  - 约 478s 失败，仍是同一个 `query_4` long cold prefill。
  - Expected: `DSV4_PD_REUSE_TECH_DOC_OK`
  - Actual: `" Ịt is what it is.\n 我恨你，你這個"`
  - Aux info 仍显示 `reuse_len=0`
- 解释：这排除了“只有 async workspace reads 是根因”。

隔离 SWA `kv_full` gather overlap：

- 当时曾在 `attention.py` 临时加 no-SWA-gather isolation env gate。
- 它只用于 A/B 验证，不是最终正式开关；当前代码里没有保留这个 env，也没有保留 current-layer `swa_kv_full` async overlap 实现。
- 新增 isolation target：
  `//internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_page_rr_overlap_no_swa_gather_reuse_memory_cache_sm100`
- 在 GPU `0,1,2,3` 跑 no-SWA-gather isolation target：
  - 约 479s 通过。

当前根因结论：

- Baseline page-RR reuse 通过。
- Full overlap page-RR reuse 在 63k cold prefill 请求失败。
- 禁 async workspace reads 后仍失败。
- 只禁 SWA `kv_full` gather overlap 后通过。
- 因此根因是 SWA `kv_full` overlapped CP gather 路径，不是 async workspace read 路径。

这与前面的静态 alias 结论一致：

- CP2 下，workspace-backed SWA `kv_full` gather result 可以和后续 Q materialization 位于同一个 `PrefillWorkspace` union。
- `qkv.kv_full` 会一直 live 到 `_attn_via_workspace().overlay_new_k`。
- 所以 Q 写入会在 attention 消费前污染 gathered SWA K。

已应用的修复：

- 明确 overlap 设计边界：只覆盖 compressor overlap 和 prefix cache SWA read/gather overlap，不覆盖当前层新 KV 的 SWA `kv_full` all-gather。
- current-layer `swa_kv_full` async overlap 代码链已删除，不再依赖 env 或“永远 false”的 guard。
- `_prefill_compute_qkv()` 始终走原来的同步 `cp_all_gather_full_varlen()` 来生成当前层 `kv_full`，不会再启动 workspace-backed async `swa_kv_full` gather。
- `_attn_via_workspace()` 里的 prefix cache SWA overlap 保留：`start_dequantize_and_gather_k_cache_slots_cp_byte_sliced(...)` 仍受 `DSV4_PREFILL_CP_OVERLAP=1` 和 page-RR/`kv_cache_sharded` 条件控制。
- 更新了 `PrefillWorkspace` 注释，不再声称 SWA `kv_full` 会在 Q materialization 前被消费完；记录当前层 SWA `kv_full` 不能作为 overlap workspace-backed live tensor 使用。

修复后验证：

- 在 GPU `0,1,2,3` 重新跑原先失败的 full overlap page-RR reuse target。
- 结果：约 176s 通过。
- Target：
  `//internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_page_rr_overlap_reuse_memory_cache_sm100`

## 2026-06-28 验证：用户指定 0,1,2,3 卡 smoke

命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bazelisk --output_user_root=/data1/serina.wzq/.cache/bazel test //internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_page_rr_overlap_reuse_memory_cache_sm100 --config=cuda13 --test_output=errors --test_timeout=7200 --cache_test_results=no
```

结果：

- 通过。
- Bazel 输出：
  - `PASSED in 168.0s`
  - 总 elapsed time 约 `176.227s`
- 解释：这个 CP2/DP2 page-RR overlap reuse 主 smoke 之前能复现 long cold-prefill 复读；硬禁当前层 `swa_kv_full` overlap 后已通过，prefix cache SWA overlap 仍保留。

Focused unit tests：

- 第一次 focused test run 暴露了旧测试预期：旧测试认为 `DSV4_PREFILL_CP_OVERLAP=1` 会自动打开 SWA `kv_full` gather overlap。
- 已更新测试：当前层 SWA `kv_full` gather 不属于 overlap feature，即使 `DSV4_PREFILL_CP_OVERLAP=1` 也不能启动 async `cp_all_gather_full_async()`。
- 新增 UT：
  - `test_prefill_compute_qkv_does_not_overlap_current_swa_kv_full`：锁住 `_prefill_compute_qkv()` 不会对当前层 `swa_kv_full` 启动 async gather。
  - `test_prefill_compute_qkv_q_failure_has_no_pending_swa_gather`：Q 侧失败时没有已经启动的 `swa_kv_full` pending gather 需要清理。
- 重跑：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bazelisk --output_user_root=/data1/serina.wzq/.cache/bazel test //rtp_llm/models_py/modules/dsv4/fp8/test:test_attention_cp_prefill_paths //rtp_llm/models_py/modules/dsv4/fp8/test:test_attention_hca_overlap --config=cuda13 --test_output=errors --test_timeout=1200 --cache_test_results=no
```

- 结果：
  - `//rtp_llm/models_py/modules/dsv4/fp8/test:test_attention_cp_prefill_paths` 通过，用时 `7.9s`。
  - `//rtp_llm/models_py/modules/dsv4/fp8/test:test_attention_hca_overlap` 通过，用时 `10.0s`。

最终 smoke 验证：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bazelisk --output_user_root=/data1/serina.wzq/.cache/bazel test //internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_page_rr_overlap_reuse_memory_cache_sm100 --config=cuda13 --test_output=errors --test_timeout=7200 --cache_test_results=no
```

- 结果：通过。
- Bazel 输出：
  - `PASSED in 170.2s`
  - 总 elapsed time 约 `171.814s`
- 解释：`DSV4_PREFILL_CP_OVERLAP=1` 仍开启 compressor overlap 和 prefix cache SWA async read；当前层 `swa_kv_full` overlap 被硬禁后，复现 case 通过。

删除 current-layer `swa_kv_full` async 实现链之后再次验证：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bazelisk --output_user_root=/data1/serina.wzq/.cache/bazel test //internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_page_rr_overlap_reuse_memory_cache_sm100 --config=cuda13 --test_output=errors --test_timeout=7200 --cache_test_results=no
```

- 结果：通过。
- Bazel 输出：
  - `PASSED in 185.5s`
  - 总 elapsed time 约 `186.920s`
- 解释：删除 SWA `kv_full` async branch / workspace role 后，保留 prefix cache SWA overlap 的主复现 smoke 仍通过。

SM100_X86 上的可选额外覆盖：

1. 跑 baseline：
   `//internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_mtp_page_rr_logits_sm100`
2. 跑 overlap：
   `//internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_mtp_page_rr_overlap_logits_sm100`
3. 这些不再是主复现路径。上面的 CP2/DP2 page-RR overlap reuse smoke 是当前问题的主要 regression guard。

## 2026-06-28 追加审查：是否还有类似 overlap 生命周期风险

检查问题：在已知好提交 `fc55d82eb179b5a9d2bd04084a8409d0a96aba42` 和当前分支之间，还有没有其他 prefill-overlap 改动存在和 SWA `kv_full` workspace/Q alias 同类的 bug。

审查过的 overlap 相关面：

- `rtp_llm/models_py/modules/dsv4/cp.py`
  - `CudaAsyncCPGatherImpl.start/wait`
  - workspace-backed roles: `main`、`indexer`、`swa_kv_full`
- `rtp_llm/models_py/modules/dsv4/fp8/attention.py`
  - `_forward_prefill_csa_overlapped`
  - `_forward_prefill_hca_overlapped`
  - `_forward_prefill_compressed`
  - `_attn_via_workspace`
  - `_prefill_compute_qkv`
  - `_materialize_prefill_q`
- `rtp_llm/models_py/modules/dsv4/fp8/compressor.py`
  - split-phase `start_prefill` / `wait_prefill_gather` / `finish_prefill`
- `rtp_llm/models_py/modules/dsv4/fp8/indexer.py`
  - nested-compressor overlap 和 async indexer-K gather
- `rtp_llm/models_py/modules/dsv4/fp8/_pool_reader.py`
  - async compressed-K workspace fill
- `rtp_llm/models_py/modules/dsv4/fp8/_swa_dequant_triton.py`
  - async byte-sliced SWA prefix read
- `rtp_llm/models_py/modules/dsv4/fp8/_indexer_cp_assembler.py`
  - async indexer-K all-gather/restore
- `rtp_llm/models_py/modules/dsv4/prefill_workspace.py`
  - union layout 和 role offsets

结论：

- 在已审查的默认开启 overlap 路径里，没有发现第二个已确认的同类 alias bug。
- Main compressor CP gather (`_CP_ROLE_MAIN`) 和 nested indexer compressor CP gather (`_CP_ROLE_INDEXER`) 都会在 `_forward_prefill_compressed()` 调用 `_materialize_prefill_q()` 前被 `finish_prefill()` 消费完。它们的 workspace-backed tensor 不会在正常 compressed path 中跨 Q materialization 存活。
- Async compressed-K pool reads、async byte-sliced SWA-prefix reads、async indexer-K reads 写的是 attention/indexer workspace，不是 `PrefillWorkspace.prefill_q()`。它们在 FlashMLA 或 indexer score 消费输出前有显式 prepare/wait event。
- 原先唯一剩余同类风险是当前层 `swa_kv_full` async all-gather。现在这条路径已硬禁，不再属于 `DSV4_PREFILL_CP_OVERLAP=1` 的执行范围。
- SWA-only/direct `kv_full` consumers 也被这个修复覆盖：它们继续消费同步 gather 得到的独立 `kv_full`，不会消费 workspace-backed live view。

额外验证：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bazelisk --output_user_root=/data1/serina.wzq/.cache/bazel test //rtp_llm/models_py/modules/dsv4/fp8/test:test_compressor_start_finish_prefill //rtp_llm/models_py/modules/dsv4/fp8/test:test_indexer_forward_with_pending_nested //rtp_llm/models_py/modules/dsv4/test:test_indexer_cp_assembler //rtp_llm/models_py/modules/dsv4/test:test_pool_reader //rtp_llm/models_py/modules/dsv4/test:test_prefill_workspace --config=cuda13 --test_output=errors --test_timeout=1200 --cache_test_results=no
```

结果：

- `test_compressor_start_finish_prefill` 通过，用时 `16.4s`。
- `test_indexer_forward_with_pending_nested` 通过，用时 `8.4s`。
- `test_indexer_cp_assembler` 通过，用时 `1.7s`。
- `test_pool_reader` 通过，用时 `1.7s`。
- `test_prefill_workspace` 通过，用时 `8.8s`。

测试维护记录：

- `test_pool_reader` 里有旧的私有 helper 引用：`_compute_local_seq_lens` / `_compute_local_owned_kv_lens`。
- 已改成 `_pool_reader.py` 当前导入的 helper：`cp_padded_local_kv_lens` 和 `cp_actual_owned_kv_lens`。

## 2026-06-28 最终修复边界

用户确认：本次 overlap 设计本来只覆盖 **prefix cache SWA gather/read**，不是当前层新 KV 的 SWA `kv_full` all-gather。

因此最终修复不是给 `swa_kv_full` 或 Q 新增额外大块存储，而是把这条误扩大的 overlap 路径从代码里拿掉：

- 删除 `_should_overlap_swa_kv_gather_for_prefill()`。
- 删除 SWA 专用 CP stream：`_SWA_CP_GATHER_STREAMS` / `_get_swa_cp_gather_stream()`。
- 删除 `_CP_ROLE_SWA_KV_FULL` 以及 `PrefillWorkspace.cp_gather_swa/cp_restore_swa` 预留区。
- 删除 `PrefillQKV.kv_full_gather_handle/kv_full_trailing_shape` pending 状态。
- `_prefill_compute_qkv()` 的当前层 `swa_kv_full` 继续走同步 `cp_all_gather_full_varlen()`。
- `_attn_via_workspace()` 中读取 cached prefix tail 的 SWA prefix overlap 保留：
  `start_dequantize_and_gather_k_cache_slots_cp_byte_sliced(...)`。
- `DSV4_PREFILL_CP_ASYNC_WORKSPACE_READS` 只影响 workspace cache reads，包括 compressed-K read 和 prefix cache SWA read，不影响当前层 `swa_kv_full`。

为什么这是更干净的修复：

- 复现失败的 63k cold prefill 请求 `reuse_len=0`，不需要 prefix cache SWA read；它仍失败，说明污染源不是 prefix cache SWA overlap。
- 当前层 `swa_kv_full` 是 fresh KV all-gather，返回的 `qkv.kv_full` 会继续 live 到 SWA cache write、direct SWA attention 或 `_attn_via_workspace().overlay_new_k`。
- 如果这条 fresh KV all-gather 返回 `PrefillWorkspace.cp_gather_swa/cp_restore_swa` 的 view，后续 Q 写 `PrefillWorkspace.prefill_q()` 就可能覆盖 live `kv_full`。
- 把这条误开的 overlap 删除后，Q 仍复用 workspace，SWA prefix cache overlap 仍保留，没有额外 Q/SWA 大 buffer 开销。

显存/性能取舍：

- 相比“硬返回 false 但保留死代码”的版本，删除实现不会增加显存；反而移除了 `PrefillWorkspace` 里给 `swa_kv_full` 预留的 gather/restore region。
- 相比修复前的错误 async path，当前层 `kv_full` 回到同步 all-gather 的独立 tensor，少了那条错误路径的 workspace-backed 存储节省。但这是 good commit 附近的基线语义，不是新引入的额外语义开销。
- 这条 async path 可 overlap 的窗口主要是 `compute_qr()`，大 Q 的 `q_lora_b + RoPE` 本来就延后到 compressor drain 之后；因此收益空间小，生命周期风险大，不值得为它引入 Q/SWA 独立 buffer 或复杂拆分。

后续如果真的要优化当前层 `swa_kv_full` all-gather，需要单独设计生命周期，不能挂在 prefix cache SWA overlap 下面。下面只是未来设计备选，不是当前修复：

- 先把 `kv_full` overlay 到 attention workspace，再 materialize Q；但这只覆盖 compressed workspace path。
- 或让 `kv_full` 使用独立 storage；但会增加 SWA full buffer 开销。
- 或让 Q 使用独立 storage；但开销通常比 SWA full buffer 大得多。

## 2026-06-28 SWA `kv_full` alias bug 触发代码链

本节记录修复前把当前层 SWA `kv_full` all-gather 纳入 overlap 后，复读 bug 的准确触发代码链。当前修复点在步骤 2：不再启动这条 async gather。

### 必要运行条件

坏路径需要：

- CP prefill active：`common.cp_on == True`，`common.cp_ctx.cp_size > 1`。
- CUDA prefill，且 prefill 侧不处于 CUDA graph capture。
- 主 overlap 开启：`DSV4_PREFILL_CP_OVERLAP=1`。
- 修复前，当前层 SWA `kv_full` gather 被主 overlap 路径一起打开。
  - 当前修复后，这条 async 实现链已删除，没有单独 env 或 guard 可以重新打开。
- `PrefillWorkspace` 预留 CP regions，并让 Q 和 CP union alias：
  - `prefill_q()` 使用 `[0, q_bytes)`。
  - `cp_gather_swa` / `cp_restore_swa` 是同一个 `_union` 的 subrange。

### 坏的 compressed-layer 链条

已复现的 smoke failure 走 compressed CSA/HCA layer：

1. `AttentionFP8._forward_prefill(x, positions)`
   - 通过 `_prefill_common_setup` 构造 `common`。
   - 调用 `_prefill_compute_qkv(x, common)`。

2. `AttentionFP8._prefill_compute_qkv()`
   - 调用 `_should_overlap_swa_kv_gather_for_prefill(common)`。
   - 如果 SWA `kv_full` gather overlap 子 feature 开启，先计算 KV：`kv = compute_kv()`。
   - 把 local KV flatten 成 `local_2d`。
   - 启动 async CP gather：
     `cp_all_gather_full_async(..., workspace=common.workspace, cp_role=_CP_ROLE_SWA_KV_FULL)`。
   - 返回 `PrefillQKV`：
     - `q=None`
     - `kv_full=None`
     - `kv_full_gather_handle=<pending handle>`
     - `kv_full_trailing_shape=<shape>`
   - 关键点：Q 被刻意延后计算，以便后面使用 `common.workspace.prefill_q()`。

3. `cp.CudaAsyncCPGatherImpl.start()`
   - 对 `_CP_ROLE_SWA_KV_FULL`，从 `workspace.cp_gather_swa(...)` 分配 gather 目标。
   - 在 SWA CP stream 上 launch `torch.distributed.all_gather_into_tensor`。
   - 返回的 handle 里的 `gathered` tensor 是 `PrefillWorkspace._union` 的 view。

4. 回到 overlap orchestrator：
   - HCA：`_forward_prefill_hca_overlapped`
   - CSA：`_forward_prefill_csa_overlapped`
   - 这些路径也会 start/finish main compressor overlap，但那部分不是 bug，因为 compressor gather 结果在 Q materialization 前已经消费完。

5. orchestrator 调用 `_ensure_prefill_kv_full(qkv, common)`。
   - 它等待 pending SWA gather：
     `cp_wait_gather_full(qkv.kv_full_gather_handle)`。

6. `cp.CudaAsyncCPGatherImpl.wait()`
   - 等 SWA gather event/work。
   - 如果 `cp_ctx.unpad_restore_is_prefix` 为 false，restore 写入 `workspace.cp_restore_swa(...)`。
   - 如果 prefix restore 为 true，restore 可能直接返回 `cp_gather_swa` 的 view。
   - 两种情况下，返回的 `kv_full_flat_2d` 都可能是 `PrefillWorkspace._union` 的 view。

7. `_ensure_prefill_kv_full()`
   - 把返回 tensor view 成 `kv_full`。
   - 返回 `qkv`，其中 `qkv.kv_full=<union-backed view>`。
   - 这是关键生命周期错误：gather 已完成，但返回的 `kv_full` tensor 仍然 live，并且仍指向 union。

8. orchestrator 写 SWA cache：
   - `_prefill_write_swa_fp8_paged(common, qkv.kv_full)`。
   - 这一步消费了 `qkv.kv_full` 来写 SWA cache，但没有结束它的生命周期，因为 attention 后面仍需要 `qkv.kv_full`。

9. orchestrator 完成 main compressor work：
   - HCA：`self.compressor.finish_prefill(main_pending)`。
   - CSA：nested indexer work 后再 `self.compressor.finish_prefill(main_pending)`。
   - `cp_gather_main` / `cp_restore_main` 和 `cp_gather_idx` / `cp_restore_idx` 都在 Q materialization 前消费完。

10. `_forward_prefill_compressed(..., _skip_compressor_write=True)` 执行。
    - 跳过第二次 compressor write。
    - 然后调用 `_materialize_prefill_q(qkv, common)`。

11. 修复前的 `_materialize_prefill_q()`
    - 从同一个 union 分配 Q：
      `q_out = common.workspace.prefill_q(seqlen)`。
    - 用 `out=q_out` 跑 `wq_b` 和 RoPE。
    - 这会写 `PrefillWorkspace._union` 的 `[0, q_bytes)`。
    - 如果 SWA gather/restore subrange 和 `[0, q_bytes)` 重叠，就会覆盖仍 live 的 `qkv.kv_full` view。

12. `_forward_prefill_compressed()` 调用 `_attn_via_workspace(qkv, common, ...)`。

13. `_attn_via_workspace()`
    - 构建 attention workspace。
    - 在 `workspace.overlay_new_k` 中读取：
      `kv_bf16 = qkv.kv_full.to(torch.bfloat16).reshape(-1, D)`。
    - 因为 `qkv.kv_full` 此时可能已经包含 Q bytes 而不是 SWA K bytes，所以会把 corrupted K overlay 到 attention workspace。
    - `flash_mla_sparse_fwd()` 随后对 corrupted K 做 attention。
    - 后续 decode 消费被污染的 KV/cache/output 路径，表现为复读。

当前修复后的步骤 2：

- `_should_overlap_swa_kv_gather_for_prefill()` 已删除。
- `_prefill_compute_qkv()` 已没有 `cp_all_gather_full_async(... cp_role=_CP_ROLE_SWA_KV_FULL ...)` 分支。
- 当前层 `kv_full` 回到同步 `cp_all_gather_full_varlen()` 路径，返回独立 tensor，不是 `PrefillWorkspace` union view。
- 因此后续 `_materialize_prefill_q()` 可以继续写 `PrefillWorkspace.prefill_q()`，不会覆盖 live `qkv.kv_full`。

### Direct SWA-Only / Warmup 链条

修复前如果当前层 SWA `kv_full` gather 被纳入 overlap，direct `kv_full` consumers 也有同样生命周期问题：

1. `_prefill_compute_qkv()` 返回 pending union-backed SWA `kv_full` gather。
2. baseline `_forward_prefill()` 调用 `_ensure_prefill_kv_full()` 和 `_prefill_write_swa_fp8_paged()`。
3. `_forward_prefill_swa_only()` 立即调用 `_materialize_prefill_q()`。
4. `_attn_fp8_swa_via_kv_full()` 随后在 `flash_mla_sparse_fwd(kv=qkv.kv_full.unsqueeze(1), ...)` 中直接消费 `qkv.kv_full`。

所以完整修复不能只处理 compressed workspace path，还必须处理 direct `kv_full` users。当前最终修复是在源头禁掉当前层 `swa_kv_full` async overlap，因此 direct consumers 也不会拿到 union-backed live view。

### Alias 几何示例

以 DeepSeek-V4 风格维度：

- `q_bytes = q_rows * n_heads * head_dim * sizeof(bf16)`
- `cp_rows = q_rows * cp_size`
- `cp_restore_swa_offset = 2*main_bytes + 2*idx_bytes + swa_bytes`
- 当 `head_dim=512`、`index_head_dim=128`、`n_heads=64 or 128` 且 CP2 时，`cp_restore_swa_offset` 落在 `[0, q_bytes)` 内。

因此 CP2 是这个 bug 的强复现配置：Q materialization 可以物理覆盖 SWA `kv_full` restore region。

更大的 CP size 下，某些模型形状可能让 SWA region 起点在 Q 之后；但不能依赖这个做鲁棒修复，因为 prefix-restore path 可能返回 `cp_gather_swa`，未来 shape/layout 变化也可能让 offset 计算失效。

### 曾评估但未采用的修复位置

如果未来重新优化当前层 `swa_kv_full` all-gather，可以修这条链的位置：

1. **步骤 10/11 前：** 在 `_materialize_prefill_q()` 前把 `qkv.kv_full` copy/move 到 non-union storage。
   - 简单且安全。
   - 代价是一次大 copy/allocation。

2. **compressed workspace layers 的步骤 9 和 11 之间：** 拆分 `_attn_via_workspace()`，使它能在 Q materialization 前把 new K overlay 到 attention workspace：
   - 如有需要，先 gather/read cached compressed K 和 SWA prefix。
   - overlay `qkv.kv_full` 到 workspace。
   - dispose `qkv.kv_full`。
   - materialize Q。
   - combine indices 并运行 FlashMLA。
   - 这能保留 compressed layers 的内存收益，但不能覆盖 direct `kv_full` consumers。

3. **CP gather allocation 处：** `_CP_ROLE_SWA_KV_FULL` 不使用 `PrefillWorkspace`，改为分配独立 SWA gather/restore buffer。
   - 语义上最安全。
   - 放弃 SWA `kv_full` 的 union-buffer 内存节省。

4. **Q allocation 处：** 当 `qkv.kv_full` 是 union-backed 且仍 live 时，让 `_materialize_prefill_q()` 在 `PrefillWorkspace.prefill_q()` 之外分配 Q。
   - 对所有 direct consumers 安全。
   - 放弃该路径上的 Q 内存节省。

当前采用的修复不是上面这些复杂方案，而是在步骤 2 之前直接删除误开的路径，具体代码改动：

- 删除 `_should_overlap_swa_kv_gather_for_prefill()`。
- 删除 SWA 专用 CP stream 和 `_CP_ROLE_SWA_KV_FULL`。
- 删除 `PrefillWorkspace` 中的 SWA gather/restore 子区。
- `_prefill_compute_qkv()` 因此不再启动当前层 `swa_kv_full` async gather。
- `_materialize_prefill_q()` 保持原设计，继续使用 `common.workspace.prefill_q(seqlen)`。
- UT 覆盖 `DSV4_PREFILL_CP_OVERLAP=1` 下当前层 `swa_kv_full` 仍走同步 `cp_all_gather_full_varlen()`。

验证结果：

- Focused UT：
  `//rtp_llm/models_py/modules/dsv4/fp8/test:test_attention_cp_prefill_paths`
  和
  `//rtp_llm/models_py/modules/dsv4/fp8/test:test_attention_hca_overlap`
  通过。
- 保留 prefix cache SWA overlap 的 page-RR reuse smoke 通过：
  `//internal_source/rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_page_rr_overlap_reuse_memory_cache_sm100`
  不设置额外 `swa_kv_full` overlap env。
