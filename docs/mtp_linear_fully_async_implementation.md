# LINEAR 模型 MTP Fully Async 实现说明

## 1. 改动目标

本次改动让 LINEAR KV cache 模型在 MTP decode 中不再因为上一轮 worker 尚未完成 host 侧
`swapLinearBlocks()` 而阻塞下一轮 target verify。

实现保留了 host swap，因为它仍负责更新 cache resource 和 allocator ownership；改变的是下一轮
模型输入的依赖关系：下一轮直接在 device block table 上应用上一轮保存的最终置换结果，不再等待
host swap 完成。

这里的 Fully Async 范围是 **LINEAR block-table repair 到 target verify 的依赖链**，不是删除
decode 后半段的所有 worker 同步。存在 spec logits processor 时，以及普通 rejection sampler
需要读取 worker 更新的 host state 时，原有窄同步仍然保留。

最终时序如下：

```text
Round N target/rejection
  -> 在主 CUDA stream 上从权威 block table 构造 final-value patch
  -> 发布每个 stream 的 patch、共享的 batch ready event 和独立 epoch
  -> worker 异步执行原有 host specUpdate/swap

Round N+1 gather
  -> 在同一把 stream 锁下快照 host block table 和 completed epoch
  -> 若 host swap 未完成，等待 CUDA event 后在 device 上应用 patch
  -> 若 host swap 已完成，直接使用快照，不重复修改
  -> TP rank 0 修复后广播给其他 rank
  -> target verify 直接执行
```

## 2. 参考实现与移植边界

本地没有可直接使用的 `feat/mtp_async_schedule` ref，因此参考了可用的等价远端分支
`origin/feat-drop-async` 最近三个提交：

| Commit | 参考内容 |
|---|---|
| `6377e68ab` | 消除 target verify prepare sync，并用 CUDA stream wait 串联 MTP bookkeeping。 |
| `4cb9a26bd` | 删除 MTP async prepare 和 decode prepare 阶段同步。 |
| `475d482a2` | FP8 CUDA graph decode prepare 跳过不必要的 host work。 |

当前基础分支是 `feat/k3_dev_mtp`，已经包含 Kimi K3、GLM5/DSV4、Sparse MLA、FlashInfer 和
FP8 graph 等差异，所以没有直接 cherry-pick。这里只抽取 async 调度原则，再单独实现 LINEAR
block table 的一致性协议，避免带入参考分支中的 merge 代码。

## 3. 为什么不用下一轮再次 swap

盲目 swap 不是幂等操作：

- worker 未完成时，下一轮确实需要执行一次置换；
- worker 已完成时，再 swap 一次会把 block id 换回旧顺序；
- 两个 swap 可能共享位置，组合结果实际是 3-cycle，不能安全地拆成两个互不相关的赋值；
- 多个位置可能同时为 `NULL_BLOCK_IDX=-1`，不能通过 block id 的值反推出来源；
- allocator 可能在两轮之间 append/backfill 新 block id，直接覆盖上一轮旧值会丢失新分配。

因此实现采用 **final-value patch + explicit permutation**：

1. Round N 保存受影响位置的 `before_values`、`after_values` 和 `source_slots`。
2. Round N+1 当前值等于 `before_values` 时，直接赋值为 `after_values`。
3. 当前值已经等于 `after_values` 时不操作，保证重复 apply 幂等。
4. 当前值两者都不等时，说明 allocator 修改了 tuple；此时对最新值应用 `source_slots`，保留新 id。

`source_slots` 的语义是：

```text
after[dst_slot] = before[source_slots[dst_slot]]
```

例如两个有序 swap 为 `(1, 0)`、`(2, 1)`：

```text
positions     = [1, 0, 2, -1]
source_slots  = [2, 0, 1, -1]
before_values = [1, 0, 2, -1]
after_values  = [2, 1, 0, -1]
```

这里保存的是完整 3-cycle。即使多个 `before_values` 都为 `-1`，置换关系仍然没有歧义。

## 4. 文件级改动

| 文件 | 改动摘要 |
|---|---|
| `rtp_llm/cpp/engine_base/stream/GenerateStream.{h,cc}` | 增加 per-stream patch、epoch、完成态和一致 KV 快照。 |
| `rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.{h,cc}` | 透传跳过 LINEAR host row 的开关，并暴露专用 gather。 |
| `rtp_llm/cpp/normal_engine/NormalModelInputGatherer.{h,cc}` | 在锁内快照所有 group 的 kernel block table，拼接 pending patch，并只对 LINEAR group 做 repair。 |
| `rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.{h,cc}` | 将每个 stream 的 epoch 带入 worker `specUpdate()`，修复 hidden-state Tensor 生命周期。 |
| `rtp_llm/cpp/normal_engine/speculative/MtpExecutor.{h,cc}` | 接入下一轮 device repair、patch build、TP 广播和同步回退。 |
| `rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.{h,cu}` | 新增 final-value patch build/apply CUDA kernel。 |
| `rtp_llm/models_py/bindings/cuda/kernels/BUILD` | 增加 `cache_group_type` 依赖。 |
| `rtp_llm/models_py/model_desc/kimi_k3.py` | Kimi K3 decode fallback 改用已修复的 device block map。 |
| `rtp_llm/cpp/normal_engine/speculative/test/MtpExecutorTest.cc` | 增加 patch 正确性、幂等性、allocator 改值和 epoch 快照测试。 |

## 5. 新增变量说明

下文统一使用：

- `G`：KV cache group 数；
- `B`：model batch size；
- `S`：`allStreams().size()`；当前 fast path 要求每条 stream 恰好一个 model batch row，因此 `S == B`；
- `W`：padding 后每个 group/batch block row 的宽度；
- `P`：patch 固定宽度，当前为 `4`。

### 5.1 `GenerateStream` 持久状态

#### `StreamSpecUpdateInfo`

| 变量 | 类型 | 含义 |
|---|---|---|
| `mtp_async_epoch` | `uint64_t` | 本次 worker update 对应的 device state 版本。必须随 task 显式传递，不能在 worker 延迟执行时读取可能已被下一轮覆盖的 current epoch。`0` 表示没有关联异步 MTP epoch。 |

#### `MtpAsyncDeviceState`

| 变量 | 类型/Shape | 含义 |
|---|---|---|
| `prev_seq_len_gpu` | CUDA `int32 [1]`，可 undefined | 本轮接受 token 前的 sequence length。新同步和异步发布路径保存该值；旧、首轮或不完整 state 可以 undefined。当前批量 patch build 直接使用 `prev_seq_len_all`；该 per-stream 字段用于状态发布、兼容入口和后续读取，不是当前 build kernel 的直接输入来源。 |

#### `MtpLinearBlockPatchState`

下表 Shape 是有效 fast-path patch 的 contract。默认 patch 的 `epoch=0`，Tensor/Event 可以为空且
不会被校验；只有已经发布非零 pending epoch、host update 尚未完成，而 patch 内容为空或非法时，
下一轮才会令 `device_patch_ready=false`，同步 worker 后走 legacy gather。

| 变量 | 类型/Shape | 含义 |
|---|---|---|
| `epoch` | `uint64_t` | patch 所属 stream epoch，用于和 host swap 完成进度比较。 |
| `positions_gpu` | CUDA contiguous `int32 [1,P]` | 两个 swap 涉及的去重 block 位置，未使用槽位为 `-1`。 |
| `source_slots_gpu` | CUDA contiguous `int32 [1,P]` | 最终置换的显式来源 slot；不依赖 block id 是否唯一。 |
| `before_values_gpu` | CUDA contiguous `int32 [1,G,P]` | Round N 构造 patch 时受影响位置的原值。 |
| `after_values_gpu` | CUDA contiguous `int32 [1,G,P]` | 按 host 相同顺序执行两个 swap 后的最终值。 |
| `valid_gpu` | CUDA contiguous `int32 [1,G]` | 各 group patch 是否有效；仅 LINEAR 且所有位置合法时为 `1`。 |
| `ready_event` | `shared_ptr<void>` | 实际持有 `torch::Event`。event 已在 patch build kernel 之后 record；GPU 工作不要求在发布时完成，下一轮通过 stream wait 保证消费顺序。类型擦除避免 stream header 暴露 CUDA Event 类型，共享所有权保证事件跨线程存活。 |

#### `KVCacheBlockSnapshot`

| 变量 | 类型/Shape | 含义 |
|---|---|---|
| `kernel_blocks` | `[stream_batch][group][block]` | 在 stream 锁内深拷贝的 host kernel block table。 |
| `batch_size` | `int` | 与 block table 在同一临界区读取的 stream batch size。 |
| `linear_patch` | `MtpLinearBlockPatchState` | 与 block table 同时取得的 patch 句柄。Tensor/Event 按值复制句柄后仍保持底层资源存活。 |
| `needs_mtp_linear_patch` | `bool` | `patch.epoch != 0 && completed_epoch < patch.epoch`。为真表示快照仍需 device repair。 |

#### 新增/相关成员变量

| 变量 | 含义 | 保护方式 |
|---|---|---|
| `mtp_linear_patch_state_` | 当前 stream 最后发布的 LINEAR final-value patch。 | `mtp_async_state_mutex_`。 |
| `mtp_async_epoch_counter_` | 每个 stream 独立的单调递增 device-state 版本计数器；原字段继续作为 patch epoch 来源。 | `mtp_async_state_mutex_`。 |
| `mtp_async_state_mutex_` | `shared_ptr<std::mutex>`，保护通用 MTP device state、patch 和 epoch counter。 | 需要同时拿两把锁时固定后拿此锁。 |
| `mtp_linear_swap_completed_epoch_` | worker 已完成对应 `specUpdate()` 及其可能 LINEAR host permutation 的最大 epoch；即使本轮无需实际 swap，也会推进完成态。使用 `max` 防止旧 task 让进度倒退。 | 仅在 stream `mutex_` 下读写，并与 host update 在同一临界区更新。 |

#### 新接口参数和局部变量

| 变量 | 所在位置 | 含义 |
|---|---|---|
| `state` | `setMtpAsyncDeviceState()` | 本轮要发布的通用 MTP CUDA state。 |
| `has_pending_linear_swap` | 同上 | 是否按 pending bookkeeping 路径发布，并启用 stream 锁和 epoch 完成协议。当前 async dispatch 保守地固定传 `true`，即使 `accept_len<=1`、patch 为空或最终无需实际 swap。 |
| `linear_patch` | 同上 | 本轮生成、供下一轮使用的 per-stream patch；pending 发布会把其 `epoch` 强制设为新分配的 `state.epoch`。 |
| `stream_lock` | 同上 | pending 路径先持有 stream `mutex_`，使 patch 发布和 block snapshot 有统一序关系。 |
| `state_lock` | state 发布/快照 | 保护 `mtp_async_state_` 和 `mtp_linear_patch_state_`。 |
| `lock` | `snapshotKVCacheBlocks()` | 持有 stream `mutex_`，覆盖 block table、batch size 和 completed epoch 的一致读取。 |
| `snapshot` | 同上 | 正在构造的快照返回值。 |
| `kv_cache` | 同上 | 当前 stream 的 live `BatchKVCacheResource` 引用，只在锁内读取。 |
| `batch` | 同上 | 锁内读取的 batch 数。 |
| `groups` | 同上 | 锁内读取的 cache group 数。 |
| `batch_id` | 同上 | 快照中的 stream-local batch 下标。 |
| `group_id` | 同上 | 快照中的 cache group 下标。 |
| `lock` | 各 MTP getter/clear | 读取或清理通用 state 时持有 `mtp_async_state_mutex_` 的局部 guard。 |

所有 MTP Tensor getter 现在按值返回，并在读取时持有 `mtp_async_state_mutex_`。这会复制
Tensor handle 而不是 GPU 数据，从而避免返回一个在解锁后被新 state 覆盖的引用。

patch 与通用 MTP state 有意分离：`has_pending_linear_swap=false` 的发布不会覆盖
`mtp_linear_patch_state_`，`clearMtpAsyncDeviceState()` / `clearSpecDecodeDeviceState()` 也只清通用
state。这样 ingress/gRPC 等普通 state 发布不会擦掉一个仍等待 host `specUpdate()` 的 patch；旧
patch 可以留存，但 completed epoch 会使它不再生效。

### 5.2 LINEAR 专用 gather

#### `MtpLinearKvCacheGatherResult`

| 变量 | 类型/Shape | 含义 |
|---|---|---|
| `block_ids` | CUDA `int32 [G,B,W]` | 从锁内 host snapshot 拼出的 kernel block table；不是 physical/store block table。 |
| `group_types` | CUDA `int32 [G]` | 每个 group 的 `CacheGroupType`，缺失配置按 `FULL` 填充。 |
| `valid_block_counts` | CUDA `int32 [G,B]` | 每个 `kernelBlocks()` vector 的可索引长度，避免 patch 访问 padding；长度范围内允许包含 `NULL_BLOCK_IDX=-1`，它不是非空 id 的计数。 |
| `patch_positions` | CUDA `int32 [B,P]` | 每行上一轮 patch 的位置；无 pending/无效行填 `-1`。 |
| `patch_source_slots` | CUDA `int32 [B,P]` | 每行上一轮 patch 的显式置换；无 pending/无效行填 `-1`。 |
| `patch_before_values` | CUDA `int32 [B,G,P]` | 每行、每组的原 tuple；无 pending/无效行填 `-1`。 |
| `patch_after_values` | CUDA `int32 [B,G,P]` | 每行、每组的最终 tuple；无 pending/无效行填 `-1`。 |
| `patch_valid` | CUDA `int32 [B,G]` | 每行、每组的 patch 有效标记；无 pending/无效行填 `0`。 |
| `pending_patches` | CUDA `int32 [B]` | 对应 host `specUpdate()`/bookkeeping epoch 尚未完成且 patch 完整时，对应 batch 行取值为 `1`；本轮可能实际无需 swap。 |
| `device_patch_ready` | host `bool` | 所有 pending row 是否属于单 batch stream，且 Tensor shape/device/dtype/contiguous/event 均有效；失败时触发同步 worker 的兼容回退。 |

#### 新接口参数

| 变量 | 含义 |
|---|---|
| `skip_linear_cache_groups` | 普通 gather 是否跳过 LINEAR 的 kernel/physical host row。完全异步 decode 为 `true`，普通 decode/prefill 默认 `false`。 |
| `group_types` | `copyKvCacheBlocksToModelInput()` 判断哪些 group 是 LINEAR 的配置。 |
| `stream_groups` | 专用 gather 的 stream 集合，同时定义最终 batch row 顺序。 |
| `host_holder` | 持有 pinned host staging Tensor，确保 non-blocking H2D 完成前内存不释放。 |

#### `gatherMtpLinearKvCacheKernelBlockId()` 局部变量

| 变量 | 类型/Shape | 含义 |
|---|---|---|
| `result` | `MtpLinearKvCacheGatherResult` | 专用 gather 的返回对象。 |
| `total_batch_size` | `size_t` | 全局 `B`。 |
| `max_blocks_num` | `size_t` | 当前 batch 最大逻辑 block 数。 |
| `pinned_i32` | `TensorOptions` | pinned host `int32` staging 配置。 |
| `cuda_i32` | `TensorOptions` | CUDA `int32` patch/dummy 配置。 |
| `host_block_ids` | pinned `int32 [G,B,W]` | block snapshot 的 H2D staging table，初始化为零。 |
| `valid_block_counts_host` | pinned `int32 [G,B]` | 各 `kernelBlocks()` vector 可索引长度的 host staging，允许范围内存在 `-1`。 |
| `pending_patches_host` | pinned `int32 [B]` | pending 标记的 host staging。 |
| `group_types_host` | pinned `int32 [G]` | group type 的 host staging。 |
| `group_type` | `CacheGroupType` | 当前 `group_id` 的实际类型。 |
| `patch_width` | `int64_t = 4` | 两个 swap 最多涉及四个不同位置。 |
| `group_num` | `int64_t` | `G` 的 Tensor shape 表示。 |
| `row_width` | `size_t` | `W`，用于地址计算和越界检查。 |
| `dst_base` | `int32_t*` | `host_block_ids` 首地址。 |
| `patch_position_slices` | `vector<Tensor>` | 按 batch 收集 `[1,P]` position slice。 |
| `patch_source_slot_slices` | `vector<Tensor>` | 按 batch 收集 `[1,P]` permutation slice。 |
| `patch_before_slices` | `vector<Tensor>` | 按 batch 收集 `[1,G,P]` before slice。 |
| `patch_after_slices` | `vector<Tensor>` | 按 batch 收集 `[1,G,P]` after slice。 |
| `patch_valid_slices` | `vector<Tensor>` | 按 batch 收集 `[1,G]` valid slice。 |
| `dummy_positions` | CUDA `int32 [1,P]` | 无 patch 行的 `-1` position 占位。 |
| `dummy_source_slots` | CUDA `int32 [1,P]` | 无 patch 行的 `-1` permutation 占位。 |
| `dummy_values` | CUDA `int32 [1,G,P]` | 无 patch 行的 `-1` before/after 占位。 |
| `dummy_valid` | CUDA `int32 [1,G]` | 无 patch 行的全零 valid 占位。 |
| `append_dummy_patch` | lambda | 为一个 batch row 同时追加所有 dummy slice，保持最终 `cat` 的 batch 对齐。 |
| `fill_one_stream` | lambda | 对一个 stream 获取一致快照并填充所有 batch row。 |
| `stream` | `const GenerateStreamPtr&` | `fill_one_stream` 当前处理的 stream。 |
| `snapshot` | `KVCacheBlockSnapshot` | 当前 stream 的锁内一致快照。 |
| `stream_batch_idx` | `int` | 当前 stream 内部 batch 下标。 |
| `kernel_blocks` | `const vector<int32_t>&` | 当前 batch/group 的 block row。 |
| `dst` | `int32_t*` | host staging 中当前行的写地址。 |
| `patch` | `const MtpLinearBlockPatchState&` | snapshot 携带的上一轮 patch。 |
| `valid_device_state` | `bool` | 当前 pending patch 的完整校验结果。 |
| `ready_event` | `shared_ptr<torch::Event>` | 从 type-erased event 恢复的 CUDA event；调用 `block(current_stream)` 只插入 stream wait，不做 CPU synchronize。 |
| `batch_idx` | `int` | decode streams 在前、context streams 在后的全局 batch 游标。 |
| `group_id` | `size_t` | 初始化 group type 或复制 snapshot block row 时的 group 下标。 |

普通 gather 在 `skip_linear_cache_groups=true` 时也跳过 `kv_cache.debugString()`，因为它同样会遍历
worker 正在修改的 host block table。

### 5.3 `MtpBatchStreamProcessor`

| 变量 | 类型 | 含义 |
|---|---|---|
| `mtp_async_epochs` | `const vector<uint64_t>&` | 与 `spec_update_infos` 一一对应的 per-stream epoch，按 `allStreams()` 顺序传给 worker。 |
| `i` | `size_t` | 将第 `i` 个 epoch 写入第 `i` 个 `StreamSpecUpdateInfo`。 |
| `skip_linear_cache_groups` | `bool` | 原样透传给普通 model-input gather。 |
| `pick_hidden_states` | lambda | 优先读取主线程已发布的 `last_hidden_states_gpu`，旧/首轮 state 缺失时退回 `sp_output_buffer`。 |
| `dev` | `torch::Tensor` | getter 按值返回的 device hidden-state handle。 |
| `hidden_states` | `torch::Tensor` | 当前 stream 的 hidden-state handle；按值持有以延长 storage 生命周期。 |

`pick_hidden_states` 从返回引用改为返回 Tensor handle，避免 getter 解锁后引用函数内临时对象。

### 5.4 `MtpExecutor`

#### 新接口和跨阶段变量

| 变量 | 类型/Shape | 含义 |
|---|---|---|
| `linear_block_ids` | CUDA `int32 [G,B,W]` | 本轮已经修复、实际供模型消费的权威 device block table；Round N 结束时用它构造 Round N+1 patch。 |
| `linear_group_types` | CUDA `int32 [G]` | 各 group 类型，CUDA kernel 只处理 LINEAR。 |
| `linear_valid_block_counts` | CUDA `int32 [G,B]` | 各 `kernelBlocks()` row 的可索引长度，范围内允许存在 `-1`。 |
| `async_linear_block_swap` | `bool` | 当前 batch 是否启用 LINEAR fully async。名称保留旧 swap 语义，实际下一轮执行 final-value repair。 |
| `linear_block_status` | `StatusOr<MtpLinearKvCacheGatherResult>` | 专用锁内 gather 的状态。 |
| `linear_block_input` | `MtpLinearKvCacheGatherResult` | 专用 gather 返回的 table、patch 和有效性元数据。 |
| `block_id_status` | `StatusOr<Tensor>` | patch 不完整时，等待 worker 后 legacy re-gather 的结果。 |

#### 同步发布补齐变量

| 变量 | 含义 |
|---|---|
| `prev_seq_len_owned` | `next_seq_len_owned - accept_len_all`，得到本轮接受 token 前的 batch sequence length，并为每个 stream 发布 view。 |

#### `dispatchDecodeAsync()` 新增变量

| 变量 | 类型/Shape | 含义 |
|---|---|---|
| `accept_len_i32` | CUDA `int32 [S]` | 当前 rejection 接受长度，供 sequence-state prepare 和 patch build 共用；支持的 fast path 上 `S==B`。 |
| `prev_seq_len_all` | CUDA `int32 [S]` | 本轮更新前的 per-stream sequence length。已有 device state 时取上一轮 `next_seq_len_gpu`，新 stream 回退到 host `seqLength()`。 |
| `next_seq_len_all` | CUDA `int32 [S]` | 应用本轮 accept length 后的 sequence length。 |
| `hidden_idx_all` | CUDA `int64 [S]` | 每条 stream 被接受的最后一个 hidden-state 下标。 |
| `prev_slices` | `vector<Tensor>` | 拼接 `prev_seq_len_all` 的 per-stream slice。 |
| `gpu_val` | `torch::Tensor` | 当前 stream 上一轮发布的 `next_seq_len_gpu` handle。 |
| `linear_patch_positions` | CUDA `int32 [S,P]` | 当前轮构造的下一轮 patch 位置；kernel contract 要求 fast path 上 `S==B`。 |
| `linear_patch_source_slots` | CUDA `int32 [S,P]` | 当前轮构造的下一轮显式置换。 |
| `linear_patch_before_values` | CUDA `int32 [S,G,P]` | 当前轮权威 table 上的原 tuple。 |
| `linear_patch_after_values` | CUDA `int32 [S,G,P]` | 两个 host 等价 swap 的最终 tuple。 |
| `linear_patch_valid` | CUDA `int32 [S,G]` | 各 stream/group patch 是否有效。 |
| `linear_patch_ready_event` | `shared_ptr<void>` | batch patch build 完成事件的 type-erased handle。 |
| `group_num` | `int64_t` | 从 `linear_block_ids.size(0)` 取得的 `G`。 |
| `ready_event` | `shared_ptr<torch::Event>` | 在 build kernel 后记录的具体 CUDA event。 |
| `mtp_async_epochs` | `vector<uint64_t>` | 按 stream 顺序保存每次 state/patch 发布返回的 epoch，随后 move-capture 到 worker。 |
| `linear_patch` | `MtpLinearBlockPatchState` | 当前 stream 对 batch patch Tensor 的 `[1,...]` view。 |

只有 `useAsyncLinearBlockSwap()` 为真、`S>0`，且 `linear_block_ids`、`linear_group_types`、
`linear_valid_block_counts`、`prev_seq_len_all`、`accept_len_i32` 全部 defined 时才启动 patch build。
否则发布的 patch Tensor/Event 为空；有后续轮次时，会通过 patch 完整性检查进入同步回退。

#### Event 变量

| 变量 | 生产者/消费者 | 含义 |
|---|---|---|
| `rejection_event` | 主 stream 记录，worker 等待 | 保证 accept length/tokens 在 worker D2H/spec update 前可用。 |
| `draft_event` | 主 stream 记录，worker 等待 | 保证 draft token/probability 在 worker 读取前可用。 |
| `linear_patch_ready_event` | patch build 后记录，下一轮 gather 等待 | 保证下一轮 `cat/apply` 在 patch Tensor 完成后执行。 |

`torch::Event::block(stream)` 是 CUDA stream-to-stream wait，不是 host 或 device 全局同步。
`rejection_event` 和 `draft_event` 被 worker lambda 按值捕获，至少存活到 worker 完成；batch 只创建
一个 `linear_patch_ready_event`，每个 stream 的 patch state 共享持有它，直到 patch 被后续 epoch
覆盖且所有 snapshot handle 释放。

### 5.5 CUDA patch 接口和变量

#### 常量和 Tensor 参数

| 变量 | 类型/Shape | 含义 |
|---|---|---|
| `MTP_LINEAR_BLOCK_PATCH_WIDTH` | `int64_t = 4` | 一轮最多两个 swap，每个最多引入两个不同位置。 |
| `kLinearPatchWidth` | `int32_t = 4` | 设备代码使用的 patch width 别名。 |
| `block_ids` | CUDA `int32 [G,B,W]` | Build 只读、Apply 读写的 kernel block table。 |
| `group_types` | CUDA `int32 [G]` | cache group 类型。 |
| `valid_block_counts` | CUDA `int32 [G,B]` | 每个 row 的可索引长度，范围内允许存在 `-1`。 |
| `prev_seq_len` | CUDA `int32 [B]` | 本轮更新前 sequence length。 |
| `accept_len` | CUDA `int32 [B]` | 本轮接受 token 数；`<=1` 时不产生重排。 |
| `positions` | CUDA `int32 [B,P]` | Build 输出、Apply 输入的受影响位置。 |
| `source_slots` | CUDA `int32 [B,P]` | Build 输出、Apply 输入的显式置换。 |
| `before_values` | CUDA `int32 [B,G,P]` | Build 输出、Apply 输入的原 tuple。 |
| `after_values` | CUDA `int32 [B,G,P]` | Build 输出、Apply 输入的最终 tuple。 |
| `patch_valid` | CUDA `int32 [B,G]` | 各 patch 有效标记。 |
| `pending_patches` | CUDA `int32 [B]` | 各 batch row 是否仍需要 repair。 |
| `seq_size_per_block` | `int32_t` | 一个 KV page 容纳的 token 数。 |
| `stream` | `cudaStream_t` | kernel 执行流。 |
| `group_num` | `int32_t/int64_t` | `G`。wrapper 使用 `int64_t` 校验后传给 kernel。 |
| `batch_size` | `int32_t/int64_t` | `B`。wrapper 使用 `int64_t` 校验后传给 kernel。 |
| `row_width` | `int32_t/int64_t` | `W`。wrapper 使用 `int64_t` 校验后传给 kernel。 |

#### Build kernel 局部变量

| 变量 | 含义 |
|---|---|
| `batch_id` | 当前线程处理的 batch row。 |
| `patch_positions[P]` | 当前 row 两个 swap 涉及的位置集合，初始为 `-1`。 |
| `patch_count` | 已收集的不同位置数。 |
| `has_cached_swap` | 是否需要第一个 cached-token swap；它复现 host `getCachedTokenBlockSwapIdx()` 的边界判定，表示首尾 token 之外存在需要搬移的 cached token，单纯跨 page 不一定为真。 |
| `cached_src` / `cached_dst` | 第一个 swap 的源/目标 block 位置。 |
| `final_src` / `final_dst` | final-token swap 的源/目标 block 位置。 |
| `accepted` | 当前 row 的 accept length。 |
| `cur_cached_len` | `prev_seq_len - 1`，与 host LINEAR 逻辑使用的当前 cached length 一致。 |
| `nxt_cached_len` | `cur_cached_len + accepted`。 |
| `base_block_idx` | 当前 cached token 所在 page 的基准 block 下标。 |
| `position_offset` | 当前 row 在 `[B,P]` 输出中的扁平偏移。 |
| `patch_source_slots[P]` | 从 identity 开始，按两个 swap 顺序折叠出的最终置换。 |
| `group_id` | 当前 cache group。 |
| `value_offset` | 当前 `(batch,group)` 在 `[B,G,P]` 中的扁平偏移。 |
| `valid_block_count` | 当前 block row 的有效长度。 |
| `indices_valid` | 所有 patch position 是否同时位于有效范围和 `row_width` 内。 |
| `row` | 当前 `(group,batch)` block row 首指针。 |
| `values[P]` | 从权威 table 读取的当前 tuple。 |
| `slot` | patch tuple 中的通用 slot 下标。 |

#### Apply kernel 局部变量

| 变量 | 含义 |
|---|---|
| `idx` | 扁平 `(group,batch)` 工作项下标。 |
| `group_id` / `batch_id` | 从 `idx` 恢复的 group/batch 下标。 |
| `patch_positions` | 当前 batch position 数组首指针。 |
| `patch_source_slots` | 当前 batch permutation 数组首指针。 |
| `patch_count` | 扫描到首个 `-1` 得到的有效 slot 数。 |
| `valid_block_count` | 新快照中当前 row 的有效长度。 |
| `row` | 可写的 Round N+1 block row。 |
| `value_offset` | 当前 `(batch,group)` patch 值偏移。 |
| `before` / `after` | 当前 patch 的原值/最终值首指针。 |
| `current[P]` | 从新快照读取的当前 tuple。 |
| `matches_before` | 当前 tuple 是否仍是 Round N 原值。 |
| `matches_after` | host worker 是否已提交，使当前 tuple 已是最终值。 |
| `permuted[P]` | allocator 改值场景下，对最新 tuple 应用置换后的结果。 |
| `source_used[P]` | 校验 `source_slots` 无重复且构成合法置换。 |
| `dst_slot` / `src_slot` | 当前目的 slot 及其显式来源 slot。 |

#### CUDA helper/wrapper 局部变量

| 变量 | 含义 |
|---|---|
| `positions` / `count` | `find/addPatchPosition` 的位置数组和当前有效长度。 |
| `target` | `findPatchPosition()` 要查找的物理 block 位置。 |
| `position` | `addPatchPosition()` 要追加的物理 block 位置。 |
| `i` | `findPatchPosition()` 扫描已有 patch slot 的循环下标。 |
| `values` / `src` / `dst` | `swapPatchValues` 中被置换的 slot 数组及物理源/目标位置。 |
| `src_slot` / `dst_slot` / `tmp` | 物理位置映射到 patch slot 后的下标和交换临时值。 |
| `tensor` / `name` | shape 校验 helper 的目标 Tensor 和错误消息名称。 |
| `rows` / `columns` | 二维 Tensor 的期望 shape。 |
| `block_size` | `256`，CUDA thread block 大小。 |
| `grid_size` | 对工作项数量向上取整后的 grid 数。 |
| `work_items` | Apply 的总工作项 `G * B`。 |

所有接口都检查 Tensor 已定义、位于 CUDA、dtype 为 `int32` 且 contiguous。二维、三维 patch
Tensor 执行精确 shape 检查；`group_types`、`prev_seq_len`、`accept_len` 和 `pending_patches`
通过通用 vector helper 检查 `numel()` 至少达到期望值，不要求严格一维或元素数完全相等。
`G/B/W` 还会检查不超过 `INT32_MAX`。

### 5.6 Kimi K3 变量

| 变量 | 含义 |
|---|---|
| `device_block_map` | `kv_cache_kernel_block_id_device`。它是 device kernel block mapping；在 fully async decode 且上一轮 patch pending 时，计算流会先完成 LINEAR repair。 |
| `uses_device_block_map` | 普通 decode 或 `is_target_verify=true` 时为真。target verify 携带 prefill-shaped metadata，不能只通过 `is_prefill` 判断。 |
| `block_map` | `_block_map()` 最终选中的二维映射。decode 和 target verify 优先使用 `device_block_map`；普通 prefill 继续使用原 host metadata 逻辑。 |

这保证 Kimi K3 batched decode/target-verify kernel 和 Python/reference fallback 消费同一份已修复
映射。普通 prefill 不启用这条特殊逻辑，因此 host metadata 行为保持不变。

### 5.7 测试变量

#### `testLinearKvCacheBlockPatchKernel`

| 变量 | 值/Shape | 含义 |
|---|---|---|
| `group_num` | `4` | 同时覆盖两个 LINEAR、一个 FULL、一个 SWA group。 |
| `batch_size` | `6` | 覆盖不同 accept length、pending 和越界情况。 |
| `row_width` | `16` | 模拟 global FULL-group BPK 大于 1 时的 padded row。 |
| `patch_width` | `4` | 与生产常量一致。 |
| `page_size` | `4` | 构造跨 page 和重叠 swap。 |
| `block_ids_cpu` | `[4,6,16]` | CPU 初始 block table；部分位置设为重复 `-1`。 |
| `expected` | `[4,6,16]` | CPU 按生产 host 顺序执行两个 swap 的 oracle。 |
| `group_types_cpu` | `[4]` | `{LINEAR,FULL,SWA,LINEAR}`。 |
| `valid_counts_cpu` | `[4,6]` | 各 row 有效长度，包含刻意 under-allocated row。 |
| `prev_seq_len_cpu` | `[6]` | 各 case 更新前 sequence length。 |
| `accept_len_cpu` | `[6]` | 覆盖 3-cycle、多 token 和 `<=1`。 |
| `pending_cpu` | `[6]` | 包含不应 apply 的非 pending row。 |
| `group_id` / `batch_id` | loop index | CPU oracle 当前 group/batch。 |
| `accepted` | `int32_t` | CPU oracle 当前 accept length。 |
| `cur_cached_len` / `nxt_cached_len` | `int32_t` | CPU oracle 的更新前/后 cached length。 |
| `cached_src` / `cached_dst` | `int32_t` | CPU oracle 第一个 swap 的位置。 |
| `final_src` / `final_dst` | `int32_t` | CPU oracle 第二个 swap 的位置。 |
| `valid_count` | `int32_t` | CPU oracle 当前 row 有效长度。 |
| `row` | `int32_t*` | CPU oracle 当前 block row 的首地址。 |
| `block_ids` | CUDA `[4,6,16]` | Build/Apply 的实际输入。 |
| `group_types` | CUDA `[4]` | 实际 group type 输入。 |
| `valid_block_counts` | CUDA `[4,6]` | 实际可索引长度输入。 |
| `prev_seq_len` / `accept_len` / `pending` | CUDA `[6]` | 实际 batch 输入。 |
| `cuda_i32` | `TensorOptions` | patch 输出 Tensor 配置。 |
| `positions` / `source_slots` | CUDA `[6,4]` | Build 输出的位置和置换。 |
| `before_values` / `after_values` | CUDA `[6,4,4]` | Build 输出的前后 tuple。 |
| `patch_valid` | CUDA `[6,4]` | Build 输出的有效标记。 |
| `already_committed` | CUDA `[4,6,16]` | 输入已等于 `after`，验证 Apply 为 no-op。 |
| `allocator_edited_cpu` | CPU `[4,6,16]` | 将受影响位置改成新 id `9999`。 |
| `allocator_edited_expected` | CPU `[4,6,16]` | 对新 tuple 应用保存置换后的期望值。 |
| `allocator_edited_row` | `int32_t*` | CPU 侧 allocator-edit oracle 目标 row 的首地址。 |
| `allocator_edited` | CUDA `[4,6,16]` | 验证新 allocator id 被保留并正确移动。 |

#### `testLinearKvCacheSnapshotEpochPreventsDoubleSwap`

| 变量 | 含义 |
|---|---|
| `cache_config` | page size 为 4 的 hybrid cache 配置。 |
| `cache_manager` | 测试 KV cache allocator。 |
| `model_config` / `runtime_config` / `resource_context` | 构造测试 `GenerateStream` 所需配置和资源上下文。 |
| `stream` | 单 batch 测试 stream。 |
| `kv_cache` | 两组初始 block row。 |
| `sp_buffer` | speculative token buffer，使 stream 可执行 `specUpdate()`。 |
| `state` | 待发布的 `MtpAsyncDeviceState`。 |
| `state.accept_len_gpu` | `{3}`，触发两个有序 LINEAR swap。 |
| `state.prev_seq_len_gpu` | `{3}`，记录更新前长度。 |
| `epoch` | `setMtpAsyncDeviceState()` 返回的版本号。 |
| `before` | worker 完成前 snapshot；应标记 `needs_mtp_linear_patch=true`。 |
| `update_info` | 携带同一 epoch 的 host `specUpdate()` 输入。 |
| `after` | worker 完成后 snapshot；应标记不再需要 patch，且 LINEAR row 已完成一次置换。 |

## 6. 并发与正确性不变量

1. 锁顺序固定为 `GenerateStream::mutex_ -> mtp_async_state_mutex_`。
2. host `swapLinearBlocks()` 与 `mtp_linear_swap_completed_epoch_` 在同一 `mutex_` 临界区更新。
3. stream 锁直接保证 host table 与 completed epoch 的一致读取；嵌套 state 锁保证 patch handle 的原子复制。
4. worker 使用 task 捕获的 epoch，不读取可能已经被下一轮覆盖的 current state。
5. patch Tensor 使用共享 event 建立 producer/consumer CUDA stream 依赖，不做 CPU synchronize。
6. TP 只由 rank 0 快照和修复，之后通过 `tpSyncModelInputs()` 广播一致的 device table。
7. fully async decode 的普通 gather 不读取 LINEAR kernel/physical host row，避免与 worker 修改同一 `BlockIds` 发生数据竞争。
8. host swap 继续存在并负责资源归属；device patch 只解除模型执行对其完成时刻的依赖。

在当前执行架构中，同一 stream 由 executor 主线程串行推进，且 patch 在对应
`AsyncRunner::launch()` 前发布，runner 内 task 保持顺序执行。在这些前提下，snapshot 才会落入
“host update 未完成，需要该 epoch patch”或“host update 已完成，不需 patch”两种有效状态。这个
结论不是两把锁脱离调度约束后可独立提供的通用事务保证。

## 7. 开启条件与回退

`useAsyncLinearBlockSwap()` 同时满足以下条件才返回 `true`：

- CUDA build；
- target cache config 的 `linear_group_num > 0`；
- `RTP_LLM_STREAM_ASYNC=1`；
- `RTP_LLM_DROP_BROAD_SYNC=1`。

以下情况不启用 LINEAR fully-async repair，或在需要时进入对应兼容路径：

- 非 CUDA build，或 target cache config 没有 LINEAR group；非 LINEAR 模型仍可独立使用通用 async bookkeeping；
- 任一环境开关未开启；
- batch 中存在 `forceDisableSpRun()` 的 target-only/KDA host fallback；
- patch build 所需的 block table、group type、valid count、sequence length 或 accept length 未定义；
- pending patch 缺少 Tensor、shape/dtype/device/contiguous 校验失败、event 缺失，或该 stream 的 snapshot batch size 不为 `1`。

后两类 patch 缺失或无效情况会在下一轮先同步 bookkeeping worker，再使用 legacy host gather，
优先保证正确性。

## 8. 当前模型边界

本改动在 fully async decode 中同时跳过 LINEAR 的 kernel 和 physical host row。当前 Qwen/Kimi
LINEAR decode 消费已修复的 device kernel table，Kimi reference fallback 也已显式切换到同一来源。

后续新增 LINEAR 模型时，如果 decode 仍直接消费 physical host block table，就不能无条件复用该模式；
需要先改为消费 device mapping，或为该模型增加 capability gate。prefill 路径不受影响。

当前 patch allocation 使用 `S=allStreams().size()`，CUDA build wrapper 则以 block table 的
`B=totalModelBatchSize()` 校验输入 shape。因此 fast path 要求每条 stream 只有一个 model batch
row，即 `S==B`；beam/multi-row stream 不在当前支持范围内。该条件不满足时可能在 patch build
shape 校验阶段直接报错，不应视为上述下一轮缺失 patch 的自动同步回退。

## 9. 验证结果

### 定向单测

```bash
bazelisk test //rtp_llm/cpp/normal_engine/speculative/test:mtp_executor_test \
  --config=cuda13 \
  --config=daily_aone_bazel_cache \
  --remote_header=x-aone-bazel-api-key=ai-infra-cicd \
  --run_under=//rtp_llm/test/utils:gpu_lock \
  --test_timeout=300 \
  --test_arg=--gtest_filter=MtpExecutorTest.testLinearKvCacheBlockPatchKernel:MtpExecutorTest.testLinearKvCacheSnapshotEpochPreventsDoubleSwap
```

结果：两个新增 case 通过。

覆盖内容：

- 两个有序 swap 与 host 逻辑等价；
- 重叠 swap 的 3-cycle；
- 重复 `NULL_BLOCK_IDX=-1`；
- 重复 Apply 幂等；
- host 已完成时 no-op；
- allocator 在两轮间修改 block id；
- FULL/SWA、非 pending、`accept_len<=1` 和 block row 可索引长度不足时不误修改；
- host swap 完成 epoch 能阻止下一轮二次应用。

### 编译

在内部仓库根目录执行：

```bash
sh build.sh
```

结果：成功完成 `//rtp_llm/test:server_test`、`//:rtp_compute_ops`、`//:th_transformer`
三个目标的 CUDA 13 编译。

另外已通过：

```bash
git diff --check
python3 -m py_compile rtp_llm/models_py/model_desc/kimi_k3.py
```

当前尚未执行真实多 rank TP 和完整 LINEAR MTP async 端到端压测；这是剩余验证项，不影响现有
定向 kernel/epoch 单测和全量 `build.sh` 编译结论。
