# Qwen 3.6 / Qwen3-Next LINEAR→FULL KV Cache 串味导致乱码 — 完整排查与修复

## TL;DR

最近 5 个 commit (`48a878df3` → `87e9870d2`) 把 LINEAR-attention（fp32 SSM/conv）
块归还 BlockPool 之前的清零路径全部补齐，**线上仍然乱码**。剩下的最后一个漏洞：

> `BlockPool` 的物理 buffer 是 `torch::empty` 分配的 **未初始化 GPU 显存**；
> 而所有 page_table 的 padding 值都是 `0`（`torch::zeros` / `fill_(0)`），等价
> 于"指向 block 0"。block 0 永远不会被 malloc，所以也永远走不到 LINEAR 的清零
> 路径，bf16 解释这段未初始化字节有概率出 NaN/Inf → XQA softmax 中毒 → 乱码。

修复一行：在 `BlockPool::initializeCacheBuffer` 末尾加一次性清零 + 同步。

```cpp
// rtp_llm/cpp/cache/BlockPool.cc
cache_aligned_buffer_.zero_();
if (allocation_type_ == AllocationType::DEVICE) {
    c10::cuda::getCurrentCUDAStream().synchronize();
}
```

下面把整个排查过程、所有路径覆盖核对、以及为什么这一个修复 **必要且充分** 写清楚。

---

## 1. 模型结构回顾

`rtp_llm/models/qwen3_next/qwen3_next.py` 把 Qwen3-Next / qwen35_moe / qwen35_dense
都按 `full_attention_interval` 切：

```python
for i in range(num_layers):
    if (i + 1) % attention_step == 0:
        types.append(HybridAttentionType.NONE)   # FULL attention
    else:
        types.append(HybridAttentionType.LINEAR) # SSM/conv
```

`HybridConfigCreator` (`rtp_llm/cpp/cache/HybridConfigCreator.cc`) 关键决定：

* `group_layer_num = max(linear_count_per_group, full_count_per_group)` —— LINEAR
  group 和 FULL group 共用同一组物理 layer 槽位（每个槽位的 row 既存 LINEAR 也
  存 FULL，看该 block 当前归谁）。
* `kv_block_stride_bytes = max(full, linear) = full_block_size_bytes`，
  LINEAR 的 fp32 写入只占 row 的前缀，row 的尾部对 LINEAR 来说不可见。

也就是说：物理 `slot[i, block_id, :]` 这块 row，**LINEAR 的 fp32 写入和 FULL 的
bf16 写入会先后落在同一片字节上**，必须靠"释放时清零"防止字节互窜。

## 2. 请求生命周期 + 所有 free 路径

```
HTTP 请求 → frontend → backend RPC → NormalEngine → Scheduler/Executor
  └─ StreamCacheResource::initKVBlock / incrKVBlock
       → KVCacheManager::malloc → HybridTypeKVCacheAllocator::initMalloc / incrMalloc
           ├─ initMallocForCommonLen (复用 + reference)
           └─ incrMalloc (per-group->malloc + 可选 removeSkippedBlocks)
  └─ Forward (per layer)
       ├─ select_block_map_for_layer (按 gid 切换 page_table)
       ├─ LINEAR: causal_conv1d / fused_recurrent / load_initial_state / store_ssm_state
       │   + asyncWriteByLayer (P2P prefill 端推送)
       └─ FULL  : RoPE → 写 KV → XQAImpl/XQADecodeImpl 读 page_table
  └─ tryReleaseKVBlock (流结束)
       ├─ insertIntoCache (BlockCache, blockCacheRef++)
       ├─ storeCacheAsync (memory/remote 二级缓存, connectorRef++)
       ├─ evictDeviceCacheToMemory (可选：popBlocksFromCache → blockCacheFree)
       └─ cache_manager->free → HybridType::free → per-group::free
            ├─ LinearKVCacheGroup::free → ✓ zeroLinearWriteRegion → requestFree
            └─ FullKVCacheGroup::free → requestFree (无清零，FULL bf16 不会自身造 NaN)
```

### 全部能让 block 回到 BlockPool free-list 的路径核对

| 路径 | 文件:行 | LINEAR 清零状态 |
|---|---|---|
| `LinearKVCacheGroup::free` | `LinearKVCacheGroup.cc:212` | ✓ `zeroLinearWriteRegion` |
| `LinearKVCacheGroup::removeSkippedBlocks` | `LinearKVCacheGroup.cc:187` | ✓ |
| `HybridTypeKVCacheAllocator::incrMalloc` 回滚 | `HybridTypeKVCacheAllocator.cc:250` | ✓ |
| `HybridTypeKVCacheAllocator::decrKVCacheRef`（request & connector） | `HybridTypeKVCacheAllocator.cc:519` | ✓ |
| `KVCacheAllocator::blockCacheFree(BatchKVCacheResourcePtr)` | `KVCacheAllocator.cc:264` | ✓ via `cleanBlocksBeforeBlockCacheFree` 钩子 |
| `KVCacheGroup::ensureFreeBlocks` 驱逐 | `KVCacheGroup.cc:60` | ✓ via `block_cache_free_hook_` |
| `FullKVCacheGroup::free` | `FullKVCacheGroup.cc:67` | 无（FULL→bf16 不会自身造 NaN） |
| `KVCacheMemoryConnector::freeBlocks` | `KVCacheMemoryConnector.cc:737/740` | N/A（独立 host BlockPool） |

→ **稳态运行（每个 block 至少跑过一次 LINEAR 释放）以后，每个 block 字节都是
"已知干净"的 0 / fp32（自家 LINEAR 还在写）/ bf16（自家 FULL 还在写）状态。**

但是仍然乱码 → 一定有一个起点根本没经过任何清零路径。

## 3. 真正的漏洞：BlockPool buffer 起手 **从未被清零**

### 3.1 现状

`rtp_llm/cpp/cache/BlockPool.cc:39` 的 `initializeCacheBuffer`：

```cpp
cache_aligned_buffer_ = torch::empty({total_size_bytes},
                                     options().dtype(kUInt8).device(kCUDA));
```

`torch::empty` **不会**清零，拿到的是 GPU 上的 **未初始化字节**（取决于显存分配
器上一次的内容）。`initFreeBlocks` 把 block `1..N-1` 全部丢进 `free_block_ids_`，
**block 0 留作"reserved sentinel"**。Triton 侧 commit `44e13a46a` 已经把所有
**写**路径加严：`> 0` 才允许写、对块 0 写入直接日志拦截。但 **读** 路径并没有禁止
读 block 0；这里的隐含假设是"block 0 始终是 0"——可惜不成立。

### 3.2 page_table padding 全是 0 → 等价于"指向 block 0"

非 cuda graph 路径：`rtp_llm/cpp/normal_engine/NormalModelInputGatherer.cc:184`

```cpp
model_input.kv_cache_kernel_block_id =
    torch::zeros({group_num, total_batch, max_blocks_num * kbpkb}, pinned_i32);
```

cuda graph 路径：`rtp_llm/cpp/cuda_graph/cuda_graph_runner.cc:122 / 191-192`

```cpp
// clear kv_cache_kernel_block_id_device, otherwise it will cause the cache block pollution
py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device.fill_(0);
... per-group ...
py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device_by_group[g].fill_(0);
py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host_by_group[g].fill_(0);
```

两条路径 padding 都是 0。`copyKvCacheBlocksToModelInput` 用 `memcpy` 把每个 batch
的真实 block id 写到 row 的前缀，row 尾部（该 batch `curBlocksNum` 小于全 batch
`max_blocks_num` 的部分）保持 0。

也就是说，**page_table 的 padding 字节就是 block id = 0**。

### 3.3 触发链

1. 服务启动，`cache_aligned_buffer_` 是未初始化的 GPU 显存（`torch::empty`）。
   block 0 的物理 row（每个 layer 槽位都有一份）继承到这堆未初始化字节。
2. 一个 batch 内有长短不一的 sequence，max_blocks_num 取最大值。短的那个
   sequence 的 page_table row 尾部是 0 → block 0。
3. XQA / flashinfer 在做 batch 对齐 / warp 前瞻 / cuda graph 静态形状下读
   page_table 时，理论上靠 `seq_lens` 做 mask；但 **IEEE 754 下 `0 * NaN = NaN`
   且 `NaN + 任何 = NaN`**，只要任意一个 K/V load 命中 NaN 比特，softmax
   就全部塌掉。
4. block 0 的物理 row 是未初始化字节，bf16 解释概率性带 NaN/Inf。
5. → XQA 输出整 token NaN → sampler 取乱字符 → 乱码。

### 3.4 为什么 LINEAR 那条链已经修了，乱码还在

* 把 LINEAR 全部 free 路径都堵掉之后，"循环再利用"过一次的 block 都是干净的。
* 但是 **block 0 永远不会被 malloc → 永远不会被 LINEAR free → 永远不被 zero**。
* 而 page_table 的 padding 又确实是 0，等价于"指向 block 0"。
* 所以在长 batch / 短 seq 混跑、cuda graph 静态形状大于实际 seq、PD 分离 decode
  端 batch 对齐等场景下，XQA 仍然能从 block 0 读到未初始化字节。

`xqa.py` 之前的 `torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)` 正是
拦截这种"个别 token NaN"的兜底。把它去掉以后，bug 就回归暴露了。

### 3.5 为什么修 block 0 一次性 zero 就够，不需要再加 FULL free 清零 / async 流同步

#### FULL→FULL 链路

* block X 起手 = 0（启动清零之后的不变量）
* FULL malloc → 写 bf16 K/V → free（无清零，bytes 仍是 bf16 K/V）
* FULL malloc → 读尚未覆盖位置 = bf16 K/V（不是 NaN）
* XQA 由 `seq_lens` 限制读取范围，未覆盖位置即使被读到，也只是 bf16 数值（不是 NaN）

→ FULL→FULL 不会造 NaN，所以 `FullKVCacheGroup::free` 不需要补 zero。

#### LINEAR→FULL 链路

* LINEAR free 已经把整 row 清零 → FULL malloc 起点 = 0 → FULL 写 bf16 → 读 bf16 ✓

#### FULL→LINEAR 链路

* FULL free 留 bf16 → LINEAR malloc → LINEAR 只写自己的 LINEAR-stride 前缀，
  只读自己的 LINEAR-stride 前缀（causal_conv1d / fused_recurrent / load_initial_state
  全是 LINEAR-spec offset），尾部 bf16 对 LINEAR 不可见 ✓

#### 异步流序竞态

worker 线程释放 + engine 线程 malloc 的最坏情况是：worker 线程的 `tensor.zero_()`
排在 engine 线程的 FULL 写**之后**执行，把刚写好的 K/V 又抹成 0。这是 **数据被
零化**，在 IEEE 754 下 0 的 K/V 走 softmax 不会产生 NaN，只会产生"输出磁场偏小
但合法"的结果（不是乱码，且会被后续 normalization 摊平）。

所以这条竞态**不会**触发用户观察到的 NaN→乱码模式，本次不做 stream 同步改动；
等修了 block 0 之后再观察是否还有"输出偏弱"问题，需要再决定是否加事件等待
或把 zero 移到 malloc 时。

#### MTP 子模型

`BlockPoolConfigHelper::createConfig` 里 `mtp_layout` 紧接在 main_layout 后面同
buffer，共用同一个 `cache_aligned_buffer_`。本次 buffer 整体 zero 自动覆盖。

#### KVCacheMemoryConnector 的 host pool

是独立的 BlockPool 实例（`AllocationType::HOST`），但走的也是同一份
`initializeCacheBuffer` → 也会被新加的 `zero_()` 覆盖（host 路径不需要 cuda
stream 同步，分支已处理）。

#### PD 分离 decode 端首次接收

* decode 端 malloc → 起点 = 0
* P2P RDMA 写入 layer 数据（LINEAR layer 写 LINEAR-stride，FULL layer 写
  full-stride）→ 各自的读路径只读自己的 stride，尾部 = 0 ✓

## 4. 修复

`rtp_llm/cpp/cache/BlockPool.cc` 在 `initializeCacheBuffer` 末尾追加：

```cpp
cache_aligned_buffer_.zero_();
if (allocation_type_ == AllocationType::DEVICE) {
    c10::cuda::getCurrentCUDAStream().synchronize();
}
```

* `zero_()` 一次性把整个 BlockPool（main + 所有 MTP layout）以及 host pool
  全部清零，让 block 0 真正成为安全 sentinel，并让所有未被任何 owner 写过的
  字节都是 0。
* `getCurrentCUDAStream().synchronize()` 确保 memset 在第一次 malloc 把 block
  交给前向 kernel 之前**真的落到了 GPU**。一次性启动开销，几 GB 在 H20 上
  也就百毫秒级别，对在线无影响。
* HOST 分支不需要 stream 同步（CPU memset 是同步的）。

需要的额外 include：`#include <c10/cuda/CUDAStream.h>`。

## 5. 为什么这一处改动 **必要且充分**

| 问 | 答 |
|---|---|
| 不修这一处会怎样？ | block 0 永远是未初始化字节，XQA 读到 NaN → 乱码持续。 |
| 只修这一处够吗？ | 够。所有其他 LINEAR→FULL 串味路径已被前 5 个 commit 堵死；FULL→FULL / LINEAR→LINEAR 自身不会造 NaN；page_table padding 指向 block 0 这条最后一条路被 block 0 = 0 关掉。 |
| 还需要修 FULL free 清零吗？ | 不需要。FULL bf16 字节不会被任何路径读出 NaN；加了反而拖慢释放路径。 |
| 还需要修异步 zero 流序吗？ | 不需要修这次的 NaN 乱码。竞态最坏只产生"输出偏弱"，不产生 NaN。 |
| cuda graph 场景安全吗？ | 安全。`cuda_graph_runner.cc:122 / 191-192` 在每次 replay 前都把 page_table `fill_(0)`，padding 指向 block 0；block 0 现在 = 0 → 读出 0.0 bf16，softmax 正常 mask。 |
| MTP 安全吗？ | 安全。MTP layout 共用同一份 `cache_aligned_buffer_`，一并被清零。 |
| PD 分离安全吗？ | 安全。decode 端 malloc 起点 = 0，RDMA 覆盖各自 stride 前缀，尾部 = 0 不会出 NaN。 |

## 6. 验证步骤

1. apply 修复，重启服务（注意启动时多出来的 `zero_()` + `synchronize()` 会让
   `BlockPool::init` 多花百毫秒级别，属于正常）。
2. 保持 `xqa.py` 的 `torch.nan_to_num` 兜底 **去掉** 状态（即当前 master）。
3. 复现脚本：commit `44e13a46a` 给的 4×A_title + 4×A_raw, c=16, 5 轮（之前
   baseline 6/40 乱码，commit `87e9870d2` 已经 0/40 通过；本次主要看在线长跑）。
4. 在线长跑 1h，统计乱码率。预期 0。
5. 如果还有乱码：
   * 优先怀疑模型计算端 NaN（LINEAR 算子数值溢出 / 上游 layer NaN 传播）。
   * 临时手段：在 `xqa.py` 加 `assert_no_nan(K)`、`assert_no_nan(V)`、
     `assert_no_nan(Q)` 三处，定位 NaN 入口。

## 7. 不在本修复范围、但值得后续观察

1. **LINEAR 二级缓存功能正确性**：commit `48a878df3` / `b43276aa9` 的注释已经
   明说，LINEAR free 会把 BlockCache 里同一 block 的 SSM state 也一并抹零，
   导致 LINEAR 前缀复用退化成"无前缀"。这个是正确性 vs. 复用率的取舍，
   当前选了正确性，本次不动。
2. **异步 zero 流序竞态**：worker 线程 zero + engine 线程 malloc 在不同 stream
   下可能让 zero 跑在 FULL 写之后，结果是 KV 被零化（不是 NaN）。如果未来发现
   FULL ATTN 输出"幅值偏弱"再排查，方案有：
   * 改为 free 后 record event、malloc 时 wait event；或
   * 把 zero 推迟到 malloc 时执行（在新 owner 自己的 stream 上）。
3. **`LinearKVCacheGroup::malloc` partial-failure 时的 block 泄漏**（partial
   `new_ids` 没被 `block_ids.add` 接管 → request_ref > 0 但无 owner）。这是
   独立的内存泄漏 bug，不会引起乱码，与本次无关。
