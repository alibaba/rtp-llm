# Kimi K3 Page-RR Prefill TP 到 Decode DP Cache 传输 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用最小改动正确支持 Kimi K3 Page-RR Prefill TP8/TP16 向 Decode TP1/DP16/KTP16 owner 传输 Target MLA、KDA、Eagle3 或 MTP cache，并同时支持 BF16 与 plain E4M3 MLA cache。

**Architecture:** 保留现有 ALLOCATE/LOAD/GENERATE 协议、CacheStore、Connector、Transfer Backend、等待发布和取消释放状态机。Decode owner 继续拉取数据，只在现有 load planner 中按 cache group 选择来源：FULL MLA 按 global page owner，LINEAR KDA 从全部 Prefill peers 写入不同 head partition，Eagle3 SWA 从一个确定副本读取。Page-RR prefix 的 FP8 bytes 仍由 adapter 原样恢复，随后在 attention wrapper 中按 fixed KV scale 解量化为 BF16。

**Tech Stack:** C++17、Python 3、PyTorch/CUDA、gRPC、Bazel、GoogleTest、unittest、Bash。

**Spec:** [2026-09-11-k3-page-rr-prefill-tp-decode-dp-cache-transfer-design.md](../specs/2026-09-11-k3-page-rr-prefill-tp-decode-dp-cache-transfer-design.md)

## Global Constraints

- 不新增或修改 protobuf 字段；ALLOCATE 中现有的有序 `peer_addrs`、`prefill_cp_size`、page geometry、dtype 和 FP8 fixed scales 已足够。
- 不修改 CacheStore、Connector、Transfer Backend、LOAD-before-publish、`wait_tasks_`、CUDA-event 发布门控、超时/取消或 block 释放状态机。
- 不引入 Prefill rank0 cache 聚合、Decode KTP cache 广播、逐 page 重试或换源。
- KTP 只影响投影计算，绝不能参与 cache source policy 或 destination layout 的选择。
- 保留已有 replicated Prefill→Projection-KTP、equal-attention-TP 和通用 DSV4 Page-RR 路径；新分支只处理 K3 Page-RR source→replicated Decode owner。
- 所有 source 映射使用请求携带的 shard 数 `N`，不得写死8；正式支持 `N ∈ {2,4,8,16}`，生产验收重点为8和16。
- padding block 不得进入 LOAD 计划；以有效 `cache_keys` 数而不是 allocation block-table 长度确定 K3 有效范围。
- 每个任务遵循 Red→Green→Refactor；只提交该任务列出的文件，不加入工作树中已有的未跟踪文件。

---

### Task 1: 提取并测试 K3 group-level source policy

**Files:**

- Modify: `rtp_llm/cpp/cache/KVCacheTransferPlanner.h`
- Modify: `rtp_llm/cpp/cache/KVCacheTransferPlanner.cc`
- Modify: `rtp_llm/cpp/cache/test/KVCacheTransferPlannerTest.cc`

- [ ] **Step 1: 先写 source policy 的失败测试**

在 `KVCacheTransferPlannerTest.cc` 增加参数化/循环测试，锁定三种唯一策略：

```cpp
TEST(KVCacheTransferPlannerTest, K3PageOwnerSelectsExactlyOnePeer) {
    for (int peers : {8, 16}) {
        for (size_t page = 0; page < static_cast<size_t>(peers * 2 + 3); ++page) {
            int selected = 0;
            for (int peer = 0; peer < peers; ++peer) {
                const auto plan = planK3CacheLoadSource(
                    K3CacheLoadSourcePolicy::PAGE_OWNER, page, peer, peers, 0);
                selected += plan.selected;
                EXPECT_EQ(plan.selected, peer == static_cast<int>(page % peers));
                EXPECT_EQ(plan.partition_count, 1);
                EXPECT_EQ(plan.partition_id, 0);
            }
            EXPECT_EQ(selected, 1);
        }
    }
}

TEST(KVCacheTransferPlannerTest, K3LinearFanInCoversEveryHeadPartition) {
    for (int peers : {8, 16}) {
        for (int peer = 0; peer < peers; ++peer) {
            const auto plan = planK3CacheLoadSource(
                K3CacheLoadSourcePolicy::ALL_PEER_PARTITION, 0, peer, peers, 0);
            EXPECT_TRUE(plan.selected);
            EXPECT_EQ(plan.partition_count, peers);
            EXPECT_EQ(plan.partition_id, peer);
        }
    }
}

TEST(KVCacheTransferPlannerTest, K3ReplicaSourceWrapsDecodeDpRank) {
    for (int dp_rank = 0; dp_rank < 16; ++dp_rank) {
        for (int peer = 0; peer < 8; ++peer) {
            const auto plan = planK3CacheLoadSource(
                K3CacheLoadSourcePolicy::SINGLE_REPLICA, 0, peer, 8, dp_rank);
            EXPECT_EQ(plan.selected, peer == dp_rank % 8);
            EXPECT_EQ(plan.partition_count, 1);
            EXPECT_EQ(plan.partition_id, 0);
        }
    }
}
```

再覆盖 `peer_count <= 0`、越界 `peer_index`、负 `decode_dp_rank`，要求抛出 `std::invalid_argument`。增加拓扑谓词测试，要求只有 `source_shards > 1 && source_shards == prefill_attention_tp && peer_count == source_shards && decode_attention_tp == 1` 被识别为 Page-RR→Decode-owner；该接口刻意不接收 KTP。

- [ ] **Step 2: 运行测试并确认因接口不存在而失败**

Run:

```bash
bazel test //rtp_llm/cpp/cache/test:kv_cache_transfer_planner_test --test_output=errors
```

Expected: 编译失败，提示 `K3CacheLoadSourcePolicy`、`planK3CacheLoadSource` 和 `isK3PageRRToReplicatedDecode` 尚未定义。

- [ ] **Step 3: 实现最小纯函数接口**

在 header 中增加：

```cpp
enum class K3CacheLoadSourcePolicy {
    PAGE_OWNER,
    ALL_PEER_PARTITION,
    SINGLE_REPLICA,
};

struct K3CacheLoadSourcePlan {
    bool selected        = false;
    int  partition_count = 1;
    int  partition_id    = 0;
};

bool isK3PageRRToReplicatedDecode(int prefill_attention_tp,
                                  int decode_attention_tp,
                                  int source_shards,
                                  int peer_count,
                                  int configured_upstream_shards);

K3CacheLoadSourcePlan planK3CacheLoadSource(K3CacheLoadSourcePolicy policy,
                                             size_t block_position,
                                             int peer_index,
                                             int peer_count,
                                             int decode_dp_rank);
```

在 `.cc` 中仅实现以下语义：

```cpp
PAGE_OWNER        => selected = block_position % peer_count == peer_index, partition 1/0
ALL_PEER_PARTITION=> selected = true, partition peer_count/peer_index
SINGLE_REPLICA    => selected = peer_index == decode_dp_rank % peer_count, partition 1/0
```

所有索引先做边界检查。不要把 cache group、KTP 或传输后端类型放进这个纯函数。

- [ ] **Step 4: 运行 planner 测试**

Run:

```bash
bazel test //rtp_llm/cpp/cache/test:kv_cache_transfer_planner_test --test_output=errors
```

Expected: PASS。

- [ ] **Step 5: 提交 source policy**

```bash
git add rtp_llm/cpp/cache/KVCacheTransferPlanner.h \
        rtp_llm/cpp/cache/KVCacheTransferPlanner.cc \
        rtp_llm/cpp/cache/test/KVCacheTransferPlannerTest.cc
git commit -m "feat(k3): model page-rr decode cache sources"
```

---

### Task 2: 锁定 TP8/TP16 KDA destination partition 与 local/upstream geometry

**Files:**

- Modify: `rtp_llm/cpp/cache/test/LinearKVCacheGroupPartitionTest.cc`
- Modify: `rtp_llm/cpp/cache/test/K3CacheGeometryManagerTest.cc`

- [ ] **Step 1: 把 KDA 物理分区测试扩展为真实96-head布局**

将现有只测8-head/8-partition的 case 改为对 `{8,16}` 循环，令 `local_num_k_heads = local_num_v_heads = local_head_num_kv = 96`。对每个 partition 检查：

- P8 每段12 heads，P16 每段6 heads；
- SSM 区间连续、互不重叠，合计完整覆盖 SSM bytes；
- 两份 convolution history 的 Q/K/V 子区间分别连续、互不重叠，合计覆盖完整96 heads；
- 每个 partition 仍返回 `1 + 2 × 3 = 7` 个 `BlockInfo`。

- [ ] **Step 2: 增加 Decode TP1/upstream8或16的 manager geometry 测试**

在 `AllocatedMainAndMtpLayoutsRetainLocalAndUpstreamGeometry` 中覆盖：

```text
Prefill: local TP=N, local_shards=N, upstream_shards=N
Decode equal-TP baseline: local TP=N, local_shards=1, upstream_shards=N
Decode DP-owner target: local TP=1, local_shards=1, upstream_shards=N
N = 2, 4, 8, 16
```

Decode DP-owner 的 KDA physical stride 必须按96个 local heads计算，而不是 `96/N`；FULL/MTP MLA 的 physical page stride保持 `page_tokens × 576 × dtype_bytes`。所有角色的 LINEAR checkpoint span 都是 `page_tokens × N`。

- [ ] **Step 3: 运行 characterization tests**

Run:

```bash
bazel test \
  //rtp_llm/cpp/cache/test:linear_kv_cache_group_partition_test \
  //rtp_llm/cpp/cache/test:k3_cache_geometry_manager_test \
  --test_output=errors
```

Expected: PASS；如果 TP16 暴露 divisibility 或 stride 错误，只在现有 `MemoryLayoutStrategy::createLinearPartitionedBlockInfo()` / cache spec geometry 中修正真实不变量，不增加另一套 layout。

- [ ] **Step 4: 提交 geometry characterization**

```bash
git add rtp_llm/cpp/cache/test/LinearKVCacheGroupPartitionTest.cc \
        rtp_llm/cpp/cache/test/K3CacheGeometryManagerTest.cc
git commit -m "test(k3): cover tp16 page-rr cache geometry"
```

---

### Task 3: 将 Page-RR source→replicated Decode owner 接入现有 LOAD planner

**Files:**

- Modify: `rtp_llm/cpp/model_rpc/DecodeRpcServer.cc`
- Modify: `rtp_llm/cpp/engine_base/stream/test/PdSepKVCacheReleaseTest.cc`

- [ ] **Step 1: 先写真实 Decode load-plan 的失败测试**

在 `PdSepKVCacheReleaseTest.cc` 增加 `testK3PageRRSourceFansIntoReplicatedDecodeOwner`。沿用 `K3TransferFixture` 和 `MemoryBackedCacheStore`，但让 tiny K3 config 可显式使用96个 KDA heads。覆盖：

```text
P8  -> Decode TP1/DP16/KTP16，decode_dp_rank=11
P16 -> Decode TP1/DP16/KTP16，decode_dp_rank=15
```

每个 case 构造 `N` 个有序 Prefill peers，并断言：

- Target FULL 和 MTP FULL 的每个有效 global page key 只出现在 `peer[g % N]`；
- LINEAR terminal key 的每个 segment key 在所有 `N` 个 peers 各出现一次；
- 每个 LINEAR destination address 精确等于 `convertIndexToBuffer(..., N, peer_index)` 返回的地址；
- P8 每个 destination partition覆盖12 heads，P16覆盖6 heads，所有区间无重叠且完整覆盖96 heads；
- Eagle3 SWA 只出现在 `peer[decode_dp_rank % N]`，因此 P8/DP rank11选择 peer3；
- 在有效 `cache_keys` 后附加的 destination padding block 保持哨兵值，且没有相应 load key。

沿用本文件已有的 deferred callback/failure case，增加新 topology 变体，确认后续 peer 计划失败时仍等待已投递 callback 完成，再返回整体 LOAD 失败。

Run:

```bash
bazel test //rtp_llm/cpp/engine_base/stream/test:pd_sep_kv_cache_release_test \
  --test_filter=PdSepKVCacheReleaseTest.testK3PageRRSourceFansIntoReplicatedDecodeOwner \
  --test_output=errors
```

Expected: 当前实现因 K3 Page-RR rank-affine 检查拒绝 Decode TP1，或只规划 peer0 的 KDA，因此失败。

- [ ] **Step 2: 在 ALLOCATE 前识别并校验新组合拓扑**

在 `prepareGenerateContext()` 的 K3 segmented-linear 分支中分别计算：

```cpp
const int  source_shards = decode_context.prefill_cp_size;
const bool page_rr_to_replicated_decode = isK3PageRRToReplicatedDecode(
    prefill_attention_tp,
    decode_attention_tp,
    source_shards,
    static_cast<int>(decode_context.peer_addrs.size()),
    static_cast<int>(parallelism_config.upstream_kv_page_rr_shard_count()));
const bool projection_ktp = source_shards == 1 && decode_ktp > 1
                            && decode_attention_tp == 1
                            && prefill_attention_tp == 8;
const bool equal_attention_tp = decode_attention_tp == prefill_attention_tp;
```

允许条件改为 `equal_attention_tp || projection_ktp || page_rr_to_replicated_decode`。对 Page-RR 新分支额外检查：

- `source_shards ∈ {2,4,8,16}`；
- `peer_addrs.size() == prefill_attention_tp == source_shards`；
- 全局 KDA K/V head 数均可被 `source_shards` 整除；
- Decode local LINEAR spec 的 head 数等于 `global_heads / decode_attention_tp`，所以 Decode TP1 必须拥有完整96 heads；
- page、kernel page、cache dtype、state dtype、FP8 format/scales 沿用现有一致性检查。

请求字段不合法时用现有 `GRPC_RET_IF_ERROR(..., INVALID_ARGUMENT, ...)` 返回请求错误，确保在 `allocateResource()` 前停止。

- [ ] **Step 3: 保持 RemoteLoad 请求使用全部有序 peers**

检查 `constructRemoteLoadRequestForMla()` 和 `constructRemoteLoadRequest()` 的 `prefill_cp_size > 1` 分支仍按 ALLOCATE 中的顺序复制全部 peers。只修正 K3 合法拓扑条件和注释，不设置请求级 `partition_count=N`；group 的 partition 必须留给 Decode planner 决定。

- [ ] **Step 4: 在 `loadCache()` 增加一个正交的新 mode**

在已有 `is_page_level_rr`、`projection_ktp` 旁增加：

```cpp
const bool page_rr_to_replicated_decode =
    k3_hybrid_cache && is_page_level_rr && decode_attention_tp == 1;
```

合法性必须成为三选一：

```text
existing equal/rank-affine load
existing replicated-Prefill Projection-KTP fan-in
new Page-RR-Prefill replicated-Decode fan-in
```

新 mode 要求 Decode local Page-RR 关闭、`source_shards == upstream_kv_page_rr_shard_count()`、`peer_count == source_shards`。KTP 不出现在新 mode 的判断中。

- [ ] **Step 5: 按 group 调用 Task 1 的 source policy**

在 `appendModelLoadBuffers()` 中仅对 `page_rr_to_replicated_decode` 使用下面映射，旧分支保持原状：

```text
Target/MTP FULL MLA -> PAGE_OWNER
segmented LINEAR KDA -> ALL_PEER_PARTITION
K3 Eagle3 SWA -> SINGLE_REPLICA
```

实现顺序：

1. 根据 `group_type`、`segmented_linear_group`、`is_k3_eagle_swa` 选 policy；
2. 在每个有效 `block_pos` 上调用 `planK3CacheLoadSource()`；
3. `selected == false` 时跳过该 peer/block；
4. 将返回的 `partition_count/partition_id` 原样传给现有 `cache_manager->convertIndexToBuffer()`；
5. LINEAR 继续用 terminal cache key 和 `makeLinearCacheSegmentKey()`；
6. FULL 继续用 `block_pos % source_shards` owner；
7. Eagle3 由 helper 计算 `decode_dp_rank % source_shards`；
8. MTP 继续先经 `resolvePhysicalGroupId()` 找到目标 physical group。

K3 canonical block 数继续按下面公式截断 allocation padding：

```cpp
FULL:   cache_keys.size()
LINEAR: ceil(cache_keys.size() / source_shards)
```

不得遍历 `block_ids.size()` 形成额外传输。

- [ ] **Step 6: 增加请求级摘要日志**

累计三个 logical load 计数：page-owner pages、KDA partitions、replica blocks；仅在所有 load 成功后输出一条 `[K3_PD_PAGE_RR_FAN_IN]` INFO 日志，包含 request ID、source shards、Decode DP rank、三个计数、FP8 开关和 `status=ok`。失败时改为输出 `[K3_PD_PAGE_RR_FAN_IN_FAILED]` WARNING。不增加逐 page 日志。

- [ ] **Step 7: 编译并运行相关 C++ 测试**

Run:

```bash
bazel test \
  //rtp_llm/cpp/cache/test:kv_cache_transfer_planner_test \
  //rtp_llm/cpp/cache/test:linear_kv_cache_group_partition_test \
  //rtp_llm/cpp/cache/test:k3_cache_geometry_manager_test \
  //rtp_llm/cpp/engine_base/stream/test:pd_sep_kv_cache_release_test \
  --test_output=errors
bazel build //rtp_llm/cpp/model_rpc:model_rpc_server
```

Expected: 全部 PASS/编译成功。再检查旧的 equal-TP 和 replicated Projection-KTP 条件仍存在：

```bash
rg -n "equal_attention_tp|projection_ktp|page_rr_to_replicated_decode" \
  rtp_llm/cpp/model_rpc/DecodeRpcServer.cc
```

- [ ] **Step 8: 提交 Decode LOAD 接入**

```bash
git add rtp_llm/cpp/model_rpc/DecodeRpcServer.cc \
        rtp_llm/cpp/engine_base/stream/test/PdSepKVCacheReleaseTest.cc
git commit -m "feat(k3): fan in page-rr cache to decode owners"
```

---

### Task 4: 放行 K3 TP16、Decode TP1/upstream geometry 与 plain FP8 cache

**Files:**

- Modify: `rtp_llm/models_py/modules/kimi_k3/cache_geometry.py`
- Modify: `rtp_llm/models_py/modules/hybrid/test/kimi_k3_cache_geometry_test.py`

- [ ] **Step 1: 先重构测试 fixture，使 local 与 upstream topology 可独立表达**

测试 helper 显式接收：

```python
tp_size: int
ep_size: int
upstream_shards: int
local_page_rr: bool
is_decode_role: bool
cache_dtype: KvCacheDataType
mla_fp8_compute: bool
```

`prefill_cp_config` 同时提供 `prefill_cp_size` 和 `is_enabled=lambda: False`，避免把 Query CP 与 Page-RR placement 混为一谈。

- [ ] **Step 2: 写失败测试覆盖 topology 与 precision 矩阵**

至少覆盖：

- Prefill TP8/EP8 与 TP16/EP16，local Page-RR 开启；
- Decode equal TP8 baseline，local Page-RR 关闭、upstream8；
- Decode TP1/DP-owner，local Page-RR 关闭、upstream8和16，EP可为8或16；
- `checkpoint_tokens == page_tokens × upstream_shards`，而不是 Decode local TP；
- BF16 compute + BASE cache + `mla_fp8_compute=false` 合法；
- BF16 compute + FP8 cache + `mla_fp8_compute=true` 合法；
- BASE/FP8 与 `mla_fp8_compute` 不匹配、INT8、FP32 compute 均失败；
- shard size 3、page/kernel-page 非法、`linear_step != 1` 均失败；
- query budget 对齐使用 upstream shard 数。

- [ ] **Step 3: 运行测试并确认当前 TP16/DP-owner/FP8 case 失败**

Run:

```bash
bazel test //rtp_llm/models_py/modules/hybrid/test:kimi_k3_cache_geometry_test \
  --test_output=errors
```

Expected: 新增的 TP16、Decode TP1/upstream 和 FP8 case 至少各有一个失败。

- [ ] **Step 4: 实现 role-aware geometry 校验**

在 `validate_kimi_k3_page_rr_target()` 中计算：

```python
upstream_shards = (
    int(parallelism.prefill_cp_config.prefill_cp_size)
    if is_decode_role
    else int(parallelism.tp_size)
)
```

然后应用：

```text
Prefill: TP == EP, local Page-RR=true, Query CP disabled
Decode:  local Page-RR=false；允许 equal-attention-TP baseline 或 TP1 replicated owner
Both:    checkpoint_tokens == page_tokens × upstream_shards
         upstream_shards in (2,4,8,16)
```

precision 判定写成两个明确合法分支，不使用宽泛的“非 INT8 即可”：

```python
bf16_cache = kv_cache_dtype == BASE and not mla_fp8_compute
fp8_cache  = kv_cache_dtype == FP8 and mla_fp8_compute
compute_dtype is torch.bfloat16 and (bf16_cache or fp8_cache)
```

预算对齐改用 `upstream_shards`。保留 `(128,128)`、`(256,256)`，Decode 额外保留 `(256,128)`。

- [ ] **Step 5: 运行 geometry 测试**

Run:

```bash
bazel test \
  //rtp_llm/models_py/modules/hybrid/test:kimi_k3_cache_geometry_test \
  //rtp_llm/cpp/cache/test:k3_cache_geometry_manager_test \
  --test_output=errors
```

Expected: PASS。

- [ ] **Step 6: 提交 geometry 校验**

```bash
git add rtp_llm/models_py/modules/kimi_k3/cache_geometry.py \
        rtp_llm/models_py/modules/hybrid/test/kimi_k3_cache_geometry_test.py
git commit -m "feat(k3): validate tp16 page-rr decode geometry"
```

---

### Task 5: 在 attention wrapper 中解释 Page-RR FP8 prefix

**Files:**

- Modify: `rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/flashinfer_mla_wrapper.py`
- Modify: `rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test/flashmla_dense_prefill_params_test.py`
- Modify: `rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test/flashmla_dense_prefill_packed_kv_test.py`
- Modify: `rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test/mla_page_rr_cache_test.py`

- [ ] **Step 1: 写 wrapper 层 FP8 失败测试**

在 CUDA test class 中用 `object.__new__(MlaFlashMLAPrefillImpl)` 构造最小 wrapper，mock：

- `page_rr_cache_adapter.read_prefix()` 返回 CUDA `torch.float8_e4m3fn` canonical rows；
- `fmha_impl.forward()` 捕获 `canonical_prefix_kv`；
- `attn_configs.kv_lora_rank=512`、`rope_head_dim=64`、`mla_fp8_compute=True`、`mla_fp8_kv_scale=0.5`。

断言传给 forward 的 prefix 是 BF16，并精确等于：

```python
raw_fp8.to(torch.bfloat16) * 0.5
```

另加 BF16 case，断言不做 dtype/scale 转换并保持原 tensor identity。再测试 FP8 config 收到 BF16 raw cache、BF16 config 收到 FP8 raw cache时 fail fast。

- [ ] **Step 2: 运行 wrapper 测试并确认当前 FP8 raw cache 被拒绝**

Run:

```bash
bazel test \
  //rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test:flashmla_dense_prefill_params_test \
  --test_output=errors
```

Expected: FP8 case 失败，错误包含当前的 `requires BF16 raw cache`。

- [ ] **Step 3: 实现 dtype-aware prefix 读取**

在 `compute_prefill_context()` 中将固定 BF16 校验改为：

```python
expected_cache_dtype = (
    torch.float8_e4m3fn
    if self.attn_configs.mla_fp8_compute
    else torch.bfloat16
)
```

shape/device 检查保持不变。`adapter.read_prefix()` 后仅在 FP8 模式执行：

```python
canonical_prefix_kv = canonical_prefix_kv.to(torch.bfloat16)
canonical_prefix_kv.mul_(self.attn_configs.mla_fp8_kv_scale)
```

不要把 scale 放进 `MlaPageRRCacheAdapter`，不要为 plain E4M3 创建 scale block，也不要修改 PD copy bytes。

- [ ] **Step 4: 扩展 TP16 pack/restore 边界测试**

在 CPU slot-mapping test 中将 shard-size case 参数化到8和16，验证每个 global page 恰有一个 owner。

在 synthetic-allgather 的 packed-KV test 中增加：

```text
TP8:  prefix 1023 / 1024 / 1025
TP16: prefix 2047 / 2048 / 2049
```

用一个 mixed batch 覆盖三种长度，继续检查恢复后的 attention 输出与 replicated canonical reference 一致。该测试不要求16张 GPU；它沿用现有 mock collective 合成16份 rank payload。

- [ ] **Step 5: 运行 Page-RR prefix 测试**

Run:

```bash
bazel test \
  //rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test:mla_page_rr_cache_test \
  //rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test:flashmla_dense_prefill_params_test \
  //rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test:flashmla_dense_prefill_packed_kv_test \
  --test_output=errors
```

Expected: CPU test PASS；有匹配 GPU 的环境中两个 GPU tests PASS。若当前机器不满足 SM103a，记录 Bazel 的 skip 证据，不把 skip 写成通过结论，并在最终生产验收前补跑。

- [ ] **Step 6: 提交 FP8 prefix 支持**

```bash
git add rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/flashinfer_mla_wrapper.py \
        rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test/flashmla_dense_prefill_params_test.py \
        rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test/flashmla_dense_prefill_packed_kv_test.py \
        rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test/mla_page_rr_cache_test.py
git commit -m "feat(k3): reuse fp8 page-rr mla prefixes"
```

---

### Task 6: 给现有全模型 smoke 增加 Page-RR 回归 profile

**Files:**

- Modify: `example/k3/kimi_k3_full_model_two_host_pd_smoke.sh`
- Modify: `example/k3/kimi_k3_full_model_two_host_pd_smoke_driver.py`
- Modify: `example/k3/kimi_k3_full_model_two_host_pd_smoke_driver_test.py`

- [ ] **Step 1: 先测试 driver 会把 Page-RR 选择传给两个角色**

在 `ForwardedOptionalEnvironmentTest` 增加：

```python
def test_forwards_page_rr_profile_to_both_roles(self) -> None:
    with mock.patch.dict(os.environ, {"SMOKE_PAGE_RR": "1"}, clear=True):
        for role in ("prefill", "decode"):
            self.assertEqual(
                driver.forwarded_optional_environment(role)["SMOKE_PAGE_RR"], "1"
            )
```

- [ ] **Step 2: 运行 driver test 并确认失败**

Run:

```bash
bazel test //example/k3:kimi_k3_full_model_two_host_pd_smoke_driver_test \
  --test_output=errors
```

Expected: `SMOKE_PAGE_RR` 未转发导致失败。

- [ ] **Step 3: 增加最小 Page-RR profile**

在 driver 的 allowlist 中加入 `SMOKE_PAGE_RR`。在 smoke shell 中：

- `SMOKE_PAGE_RR` 默认0，只允许0/1，保持现有 replicated smoke 默认行为；
- `SMOKE_PAGE_RR=1` 时，未显式设置 page size则默认 physical/kernel page均为128；
- Page-RR profile 要求 `REUSE_CACHE=1`、Prefill TP为8、physical/kernel geometry属于已验证组合；
- Prefill role 导出 `PREFILL_CP_KV_CACHE_SHARDED=1`；
- Decode role保持 local Page-RR关闭，并导出 `PREFILL_CP_SIZE=${smoke_tp_size}`；
- runtime environment verifier 同时验证以上 role-specific 值；
- 日志明确打印 `source_cache=page-rr/${smoke_tp_size}` 或 `source_cache=replicated`。

不要把这个 two-host controller 泛化为四节点编排器。它的正式自动回归仍是可运行的 P8→DP8/KTP8；P8/P16→DP16/KTP16 使用现有 `start_kimi_k3_pd.sh` 的 multi-node `GANG_CONFIG_STRING` 路径验收。

- [ ] **Step 4: 修正文档中已过期的 MTP 说明**

把“KTP with MTP is rejected”改为当前真实契约：`SP_TYPE=mtp` 可用于 Projection-KTP target，MTP draft 固定 KTP1；Eagle3 与 MTP 仍是两次独立启动，不能在同一进程同时启用。

- [ ] **Step 5: 运行静态和 driver 测试**

Run:

```bash
bash -n example/k3/kimi_k3_full_model_two_host_pd_smoke.sh
bazel test //example/k3:kimi_k3_full_model_two_host_pd_smoke_driver_test \
  --test_output=errors
```

Expected: PASS。

- [ ] **Step 6: 提交 smoke profile**

```bash
git add example/k3/kimi_k3_full_model_two_host_pd_smoke.sh \
        example/k3/kimi_k3_full_model_two_host_pd_smoke_driver.py \
        example/k3/kimi_k3_full_model_two_host_pd_smoke_driver_test.py
git commit -m "test(k3): add page-rr pd smoke profile"
```

---

### Task 7: 回归、审查与生产拓扑验收

**Files:**

- Verify only; fixes go into the owning task's files and are committed separately.

- [ ] **Step 1: 跑完整的定向回归集**

Run:

```bash
bazel test \
  //rtp_llm/cpp/cache/test:kv_cache_transfer_planner_test \
  //rtp_llm/cpp/cache/test:linear_kv_cache_group_partition_test \
  //rtp_llm/cpp/cache/test:k3_cache_geometry_manager_test \
  //rtp_llm/cpp/engine_base/stream/test:pd_sep_kv_cache_release_test \
  //rtp_llm/models_py/modules/hybrid/test:kimi_k3_cache_geometry_test \
  //rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test:mla_page_rr_cache_test \
  //rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test:flashmla_dense_prefill_params_test \
  //rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test:flashmla_dense_prefill_packed_kv_test \
  //example/k3:kimi_k3_full_model_two_host_pd_smoke_driver_test \
  //example/k3:start_kimi_k3_pd_test \
  --test_output=errors
bazel build //rtp_llm/cpp/model_rpc:model_rpc_server
```

Expected: 可运行目标全部 PASS；硬件 gate 导致的 skip 单独列出。

- [ ] **Step 2: 做约束扫描和 diff 检查**

Run:

```bash
git diff --check
git status --short
git diff --stat origin/feat/k3_dev...HEAD
rg -n "prefill_attention_tp == 8|KTP with MTP is rejected|requires BF16 raw cache" \
  rtp_llm/cpp/model_rpc/DecodeRpcServer.cc \
  rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/flashinfer_mla_wrapper.py \
  example/k3/kimi_k3_full_model_two_host_pd_smoke.sh
git diff origin/feat/k3_dev...HEAD -- \
  rtp_llm/cpp/model_rpc/proto \
  rtp_llm/cpp/disaggregate/cache_store \
  rtp_llm/cpp/cache/connector
```

Expected:

- `git diff --check` 无输出；
- 第一条 `rg` 只允许保留有意支持的 replicated-Prefill P8 Projection-KTP 条件，不再出现过期 smoke/固定 BF16 prefix 文案；
- protobuf、CacheStore、Connector diff 为空；
- 工作树原有未跟踪文件仍未加入提交。

- [ ] **Step 3: 跑 P8→DP8 Page-RR 全模型低成本回归**

分别启动两次现有 two-host smoke：

```text
SMOKE_PAGE_RR=1, SP_TYPE=eagle3, KIMI_K3_MLA_FP8=1, SMOKE_SUITE=all
SMOKE_PAGE_RR=1, SP_TYPE=mtp,    KIMI_K3_MLA_FP8=1, SMOKE_SUITE=all
```

再把 `KIMI_K3_MLA_FP8=0` 各跑一次。验收日志必须出现 Page-RR source shard 数8、KDA 8 partitions、Eagle3 单副本或 MTP FULL page-owner 传输，并完成 cold/prefix-hit/long-prefix/concurrent cases。该拓扑是传输回归，不替代 DP16 生产验收。

- [ ] **Step 4: 跑生产矩阵**

用现有 multi-node launcher 分别启动：

```text
Prefill TP8/EP8  Page-RR -> Decode TP1/DP16/KTP16/EP16
Prefill TP16/EP16 Page-RR -> Decode TP1/DP16/KTP16/EP16
```

每个拓扑依次验证无推测、Eagle3、MTP，并对 BF16 与 `KIMI_K3_MLA_FP8=1` 各运行一次。每个配置执行 cold request、prefix seed/hit、跨至少两个 Page-RR stripe 的长 prefix，以及路由到不同 Decode owners 的并发请求。

验收必须同时满足：

- output 与对应非 PD/reference 路径在既有容差内一致；
- reuse 统计和命中长度正确；
- Target/MTP FULL page 来源符合 `g % N`；
- KDA 8×12-head 或16×6-head 恰好重建96 heads，无空洞/重叠；
- Eagle3 来源符合 `decode_dp_rank % N`；
- FP8 网络传输保持 raw-byte identical，prefix attention 使用 non-unit scale 后仍与 reference 一致；
- padding 未传输，LOAD 完成前不进入 Decode；
- timeout/cancel 后无 hang、后台写入或 block 提前复用。

- [ ] **Step 5: 最终提交审查**

Run:

```bash
git log --oneline origin/feat/k3_dev..HEAD
git status --short
```

确认每个功能提交范围单一、无 `fixup!`、无用户未跟踪文件后，再进入 code review/合并流程。
