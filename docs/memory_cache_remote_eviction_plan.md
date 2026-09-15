# RTP-LLM 单 Group Memory Cache 淘汰到 Remote 实施计划

## 1. 目标

为 RTP-LLM 增加以下两级异步淘汰流水线：

```text
Task A：evictMemoryCacheToRemote
                    ↓ 依赖
Task B：evictDeviceCacheToMemory
```

第一版只覆盖 `cache_config.groupNums() == 1` 的模型，重点适配 MiniMax-M3。要求：

- 预计本次 D2H 后 Memory cache 会超过高水位时，先选择已有 Memory victim 并写入 Remote。
- 当水位计算需要 Task A 时，Task A 完成后才能启动完整的 `evictDeviceCacheToMemory()`；无需 Remote spill 时直接启动 Task B。
- Task B 内再次发现 Memory 空间不足时，直接淘汰 Memory LRU，不再递归写 Remote。
- Remote 写失败不能阻断 Device cache 回收；失败只降低缓存命中率。
- Remote 写期间，Memory victim 不得被重新匹配、释放或覆盖。
- 开启 `M3_IDX_PAGED=1` 时，main K/V 与 idx_K 必须作为同一个逻辑 block 完整写出。

## 2. 范围与非目标

### 2.1 第一版支持范围

```text
cache group 数：1
group id：0
group type：FULL
Memory backing：HOST
Remote buffer：MemoryType::CPU
```

MiniMax-M3 当前通过 `SingleConfigCreator` 创建一个逻辑 group。启用 `M3_IDX_PAGED=1` 后，idx_K 被放在同一物理 block 的 scale region 中，并不会形成第二个 group。

### 2.2 第一版暂不支持

- Hybrid attention 多 group。
- DSV4 多 region/group。
- CP 特殊 key/block 映射。
- Disk backing 到 Remote。
- CPU/GPU IOV 混合写。
- Per-layer Remote write。
- MTP 子模型独立 cache group。

## 3. 当前代码链路

### 3.1 Device 到 Memory

```text
上层触发 Device cache 淘汰
    ↓
KVCacheMemoryConnector::asyncWrite()
    ↓
buildCopyPlanForWrite()
    ↓
allocateBackingsForWrite()
    ↓
allocateOneBacking()
    ↓
Memory 无空间
    ↓
popOldestEvictable()
    ↓
releaseCacheBacking()
    ↓
D2H 并提交 Memory cache 索引
```

### 3.2 Device 到 Remote

```text
RemoteConnector::asyncWrite(KVCacheResource, Meta)
    ↓
asyncWriteTask()
    ↓
getWriteLocation()
    ↓
genWriteRequest()
    ↓
TP broadcast
    ↓
RemoteConnector::Write()
    ↓
GroupPolicy::genBlockBuffers()
    ↓
GPU IOV
    ↓
saveKvCaches()
    ↓
finishWrite()
```

新增路径需要跳过基于 device block ID 的 `genBlockBuffers()`，直接向 KVCM 提交 CPU `BlockBuffers`。

## 4. 目标时序

```text
请求成为串行队列队首，计算预计写入量 N
    ↓
读取 Memory 当前使用量 U 和高水位容量 H
    ↓
计算 R = max(0, U + N - H)
    ├── R == 0：直接启动 Task B
    └── R > 0：选择并 detach R 个 Memory victims
                     ↓
              Task A：Memory → Remote
                     ├── getWriteLocation
                     ├── 根据 block mask 过滤
                     ├── saveKvCaches(CPU BlockBuffers)
                     └── finishWrite
                     ↓
              释放 victim host backing
                     ↓
              Task B：evictDeviceCacheToMemory
                     ├── 分配 Memory backing
                     ├── 若仍不足，直接淘汰 Memory LRU
                     ├── D2H
                     ├── commit Memory cache 索引
                     └── 释放 Device cache blocks
```

两个任务对调用线程都是异步的，但 Task B 在逻辑上严格依赖 Task A。

## 4.1 基于百分比水位的淘汰策略

第一版不再使用固定的 `device_cache_min_free_blocks` 决定 Device 淘汰量。Device cache 和 Memory cache 都使用容量百分比水位，默认高水位为 95%。

建议新增配置：

```text
DEVICE_CACHE_HIGH_WATERMARK_RATIO=95
MEMORY_CACHE_HIGH_WATERMARK_RATIO=95
```

配置语义统一为“允许使用的最大容量百分比”：

```text
95%水位
= 最多使用95%的可分配blocks
= 至少保留5%的blocks
```

合法范围建议为 `[1, 100]`。`100` 表示仅在真正没有空间时淘汰；`0` 不作为关闭开关，功能是否启用仍由 `enable_tiered_memory_cache` 等现有开关控制。

### Device cache 水位

沿用当前 `evictDeviceCacheToMemory()` 使用 `notInUseBlocksNum()` 的语义，把正在异步写 Memory、随后会变为空闲的 Device blocks 视为可回收容量，避免重复淘汰。

```cpp
const size_t total = cache_manager->totalBlocksNum();
const size_t effective_free = cache_manager->notInUseBlocksNum();
const size_t effective_used = total - std::min(total, effective_free);
const size_t max_used = total * device_high_watermark_ratio / 100;

if (effective_used <= max_used) {
    return;
}

const size_t need_evict = effective_used - max_used;
```

为避免整数截断导致超过水位，也可以等价地计算需要保留的最小空闲数：

```cpp
const size_t min_effective_free =
    (total * (100 - ratio) + 99) / 100;  // ceil
const size_t need_evict =
    min_effective_free > effective_free
        ? min_effective_free - effective_free
        : 0;
```

例如 Device pool 共 1000 blocks、阈值 95%、当前 `notInUse=20`：

```text
要求至少保留50 blocks
当前只有20 blocks可回收/空闲
本轮Device→Memory淘汰30 blocks
```

当前 `ResourceContext::initCacheConfig()` 自动根据 `max_prefill_tokens` 生成 `device_cache_min_free_blocks` 的逻辑需要被水位配置替代。旧字段可以先保留一段兼容期，但当水位字段显式设置时必须以水位为准，并打印最终生效策略。

### Memory cache 水位

Memory 水位必须同时考虑当前占用和本次 Device 淘汰预计产生的 D2H block 数，而不是只看当前使用率。这里 Task A 使用的是容量预估 `estimated_d2h_block_num`；Task B 启动时会重新构造自己的执行 plan，并得到最终的 `actual_d2h_block_num`：

```cpp
const size_t total = memory_pool->totalBlocksNum();
const size_t free = memory_pool->freeBlocksNum();
const size_t used = total - std::min(total, free);
const size_t max_used = total * memory_high_watermark_ratio / 100;
const size_t incoming = eviction_estimate.estimated_d2h_block_num;

const size_t projected_used = used + incoming;
const size_t need_remote_evict =
    projected_used > max_used
        ? projected_used - max_used
        : 0;
```

Task A 的预腾空间目标为：

```text
current_used
- memory_to_remote_evicted
+ estimated_d2h_block_num
<= memory_high_watermark_blocks
```

Task B 构造最终 plan 后，必须按照 `actual_d2h_block_num` 再次检查，并通过直接淘汰 Memory LRU 保证最终硬性不变量：

```text
current_used_after_task_a
- task_b_immediate_evicted
+ actual_d2h_block_num
<= memory_high_watermark_blocks
```

所以 Task A 的远端淘汰量定义为：

```cpp
memory_to_remote_evict_blocks = std::max<int64_t>(
    0,
    current_used
        + eviction_estimate.estimated_d2h_block_num
        - memory_high_watermark_blocks);
```

这一个公式同时包含两个条件：

1. 当前 Memory cache 的占用是否已经接近或超过水位。
2. Task A 估计的本次 D2H block 数会带来多少新增占用。

不能只写成当前超水位量：

```cpp
// 错误：没有给本次D2H预留水位内空间。
max(0, current_used - max_used);
```

也不能只写成本次 D2H 相对当前 free blocks 的缺口：

```cpp
// 错误：只能避免OOM，不能保证D2H完成后仍低于95%水位。
max(0, estimated_d2h_block_num - current_free);
```

Task A 的 victim 数量来自队首请求的 `DeviceToMemoryEvictionEstimate::estimated_d2h_block_num`。该 estimate 只用于容量预留，不是 Task B 的可执行 plan。Task A 完成并释放 victims 后，Task B 必须基于当时的实时状态重新形成自己的 plan。

例如 Memory pool 共 1000 blocks、当前已经使用 920、Task B 将写入 50 blocks：

```text
max_used = 950
projected_used = 920 + 50 = 970
Task A需要淘汰到Remote：970 - 950 = 20 blocks
```

这与“只有 Memory 没有空闲空间才淘汰”不同：即使当前还有 80 个 free blocks，本轮仍然提前淘汰 20 个，使 Task B 完成后的 Memory 使用率不超过 95%。

Memory 使用率必须使用实际占用 backing 的 `freeBlocksNum()` 计算，不使用可能把 cache-held block 视作 available 的 `availableBlocksNum()`。如果存在已经 detach、正在写 Remote 的 victim，串行事务下它们属于当前 Task A，不会与另一个 plan 重复计算。

### Task B 也必须执行 Memory 水位约束

Task A 按预测值腾出空间；Task B 实际分配时仍需再次检查 Memory 水位。若实际状态超过 95%，Task B 直接淘汰额外 Memory LRU，不写 Remote：

```cpp
if (current_used + remaining_incoming > max_used) {
    const size_t emergency_evict =
        current_used + remaining_incoming - max_used;
    auto victims = popForImmediateEviction(emergency_evict);
    releaseCacheBacking(victims);
}
```

因此两处都满足同一个 Memory 高水位：

```text
Task A：超过Memory水位的预测部分，写Remote后释放
Task B：执行时仍超过Memory水位的部分，直接淘汰后释放
```

### 极端情况

如果 `incoming > max_used`，单次 Device 淘汰量本身已经超过 Memory 的目标容量。Memory cache 是可丢弃缓存，不应为了满足水位阻塞推理。第一版应限制本轮实际写入 Memory 的 block 数，或允许 Task B 写入时持续淘汰旧块，使最终 retained blocks 不超过 `max_used`；不能形成无法满足的等待循环。

推荐 Task B 在形成最终 plan 时优先保留 Device eviction candidates 中较新的连续后缀，同时保证 prefix-tree 依赖和 complete-tail 约束。Task A 的 estimate 不承担这些执行正确性约束；估计偏差由 Task B 的实时水位复查兜底。

## 5. 数据结构

### 5.1 Memory victim

```cpp
struct MemoryRemoteEvictionItem {
    CacheKeyType cache_key{0};
    BlockIdxType memory_block_id{NULL_BLOCK_IDX};
    size_t       block_size{0};
    bool         is_complete{true};
    uint64_t     generation{0};
};
```

`generation` 用于避免相同 cache key 删除、重新插入后产生 ABA 问题。

### 5.2 Memory eviction plan

```cpp
struct MemoryRemoteEvictionPlan {
    std::vector<MemoryRemoteEvictionItem> items;
    // 请求级追踪字段可保留，但token_ids必须允许为空。
    std::shared_ptr<Meta>                 remote_meta;
};
```

### 5.3 Device→Memory 容量预估

Task A 不构造 Task B 的执行 plan，只保存计算 Memory 水位所需的轻量预估和请求引用：

```cpp
struct DeviceToMemoryEvictionEstimate {
    std::shared_ptr<KVCacheResource> resource;
    std::shared_ptr<Meta>            meta;
    size_t                           estimated_d2h_block_num{0};
};
```

这里不保存 `CopyInfoPerKey`、Device block ID、Memory backing 或 D2H layout。`resource` 引用只保证请求在 Task A 期间仍可由 Task B 重新分析；最终复制范围和 `actual_d2h_block_num` 由 Task B 启动时独立生成。

`estimated_d2h_block_num` 默认取 Device 高水位计算出的本轮 `need_evict`，作为尚未执行 Memory match 前的保守估计。允许实现使用只读统计进一步收紧该值，但不得因此冻结具体 Device blocks 或 CopyInfo。

`actual_d2h_block_num` 的统一语义是：Task B 完成实时 match/裁剪后，需要新分配 Memory backing 并执行 D2H 的逻辑 block 数。已经命中 Memory、被裁掉或不再需要淘汰的 blocks 不计入该值。

### 5.4 Memory eviction lease

```cpp
struct MemoryEvictionLease {
    std::vector<MemoryRemoteEvictionItem> items;
    // 持有Memory pool/backing所需的强引用。
};
```

该 lease 只负责保证 Remote 异步读取期间 host backing 有效，不自行执行 cache 状态迁移或释放。`finishRemoteEviction()` 是唯一完成 `REMOTE_EVICTING → FREE` 和归还 backing 的入口，必须保证只调用一次。

## 6. MemoryBlockCache 修改

涉及文件：

```text
rtp_llm/cpp/cache/connector/memory/MemoryBlockCache.h
rtp_llm/cpp/cache/connector/memory/MemoryBlockCache.cc
```

### 6.1 Remote spill 淘汰接口

当前 `pop(int n)` 只返回 block ID，无法构造 Remote key。新增：

```cpp
std::vector<CacheItem> detachForRemoteEviction(int n);
```

行为：

1. 在 LRU 锁内只选择 `!is_resident && is_complete` 的 victim；第一版不把 incomplete tail 写入 Remote。
2. 从可匹配索引中删除。
3. 返回完整 `CacheItem`。
4. 不执行 `blockCacheFree()`。
5. 不在 LRU 锁内执行 RPC 或 KVCM 写入。

### 6.2 Task B 的直接淘汰接口

保留或新增语义清晰的接口：

```cpp
std::vector<CacheItem> popForImmediateEviction(int n);
```

该接口仅供 `evictDeviceCacheToMemory()` 内空间不足时使用，返回后直接释放 backing，绝不触发 Remote。

### 6.3 Evicting 状态

推荐维护：

```cpp
std::unordered_map<CacheKeyType, CacheItem> remote_evicting_items_;
```

生命周期：

```text
ACTIVE
  ↓ detachForRemoteEviction
REMOTE_EVICTING
  ↓ finishRemoteEviction
FREE
```

## 7. KVCacheMemoryConnector 修改

涉及文件：

```text
rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h
rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.cc
```

### 7.1 空间查询

```cpp
size_t totalMemoryBlocks() const;
size_t freeMemoryBlocks() const;
```

两个值必须使用同一个 Memory backing pool 的口径，并与 `MemoryBlockCache::freeBlocksNum()` 一致。第一版只处理单 group，因此不额外传 group/kind；未来支持多 pool 时再显式增加 pool ID。

### 7.2 准备 victim

```cpp
std::vector<MemoryRemoteEvictionItem>
prepareRemoteEviction(size_t block_num);
```

该接口最多返回 `block_num` 个 victims，负责 detach 并 pin，但不释放 host backing。由于 resident、in-flight 或 incomplete blocks 不可选，返回数量允许小于请求数量；未预腾出的差额由 Task B 根据 `actual_d2h_block_num` 直接淘汰兜底。

### 7.3 构造 CPU BlockBuffers

```cpp
bool buildHostBlockBuffers(
    const std::vector<MemoryRemoteEvictionItem>& items,
    const std::vector<size_t>& selected_indices,
    kv_cache_manager::BlockBuffers& buffers) const;
```

规则：

- 只为 `selected_indices` 指向的远端缺失 keys 构造 buffers。
- 一个被选中的 cache key 对应一个 `BlockBuffer`。
- 一个 `BlockBuffer` 包含该 block 所有层的 IOV。
- 所有 IOV 的类型均为 `kv_cache_manager::MemoryType::CPU`。
- IOV 顺序与当前 Device Remote 写入顺序完全一致。
- MiniMax `M3_IDX_PAGED=1` 时包含每层 main KV 和 idx_K。

示意代码：

```cpp
for (const auto& item : items) {
    kv_cache_manager::BlockBuffer block_buffer;
    auto infos = convertMemoryIndexToBuffer(/*group_id=*/0,
                                            item.memory_block_id);
    for (const auto& info : infos) {
        block_buffer.iovs.push_back({
            kv_cache_manager::MemoryType::CPU,
            info.addr,
            info.size_bytes,
            false,
        });
    }
    buffers.push_back(std::move(block_buffer));
}
```

应复用 Memory connector 的 layout/address 转换能力，不要在 RemoteConnector 中重复计算 pitch、layer offset 或 idx_K offset。

### 7.4 完成淘汰

```cpp
void finishRemoteEviction(
    const std::vector<MemoryRemoteEvictionItem>& items,
    bool remote_success);
```

Task A 完成后，无论 Remote 成功还是失败，都允许释放 host backing：

```text
Remote 成功：Memory cache 降级到 Remote。
Remote 失败：Memory cache 被丢弃，后续重新计算。
```

## 8. RemoteConnector 修改

涉及文件：

```text
rtp_llm/cpp/cache/connector/remote_connector/RemoteConnector.h
rtp_llm/cpp/cache/connector/remote_connector/RemoteConnector.cc
```

### 8.1 新增 Memory block 写入口

```cpp
std::shared_ptr<AsyncContext> asyncWriteMemory(
    const std::vector<MemoryRemoteEvictionItem>& victims,
    const std::shared_ptr<Meta>& remote_meta,
    const std::shared_ptr<MemoryEvictionLease>& lease);
```

`lease` 代表 victim backing 生命周期，必须一直保留到异步任务结束。该入口不预先接收完整 `BlockBuffers`；它先用 `cache_keys` 查询 Remote，仅对 `block_mask` 标记为缺失的 blocks 延迟构造 CPU buffers。

### 8.2 统一 Device/Memory 写入入口

Device cache 和 Memory cache 的差异只应存在于 `BlockBuffers` 的生成阶段：

```text
Device cache
    ↓ group_id + device block_id
生成 MemoryType::GPU BlockBuffers
    ┐
    ├── 进入统一 Remote write pipeline
    │
Memory cache
    ↓ memory block id + host layout
生成 MemoryType::CPU BlockBuffers
    ┘
```

得到 `BlockBuffers` 后，两条路径必须复用相同的：

```text
getNeedWriteGroups
    ↓
getWriteLocation
    ↓
解析 block_mask
    ↓
构造/筛选待写 BlockBuffers
    ↓
saveKvCaches
    ↓
收集 actual URI
    ↓
finishWrite
    ↓
更新 AsyncContext 和 metrics
```

不应为 Memory cache 复制一套独立的 KVCM 控制面状态机。

建议使用统一输入和延迟构造 buffer 的 provider：

```cpp
struct RemoteWriteInput {
    CacheKeysType         cache_keys;
    std::shared_ptr<Meta> meta;

    // Memory→Remote允许meta->tokens()为空；block身份由cache_keys确定。
    // 公共控制面不得依赖tokens重新计算或校验cache key。

    // 保证底层 Device/Host buffer 在整个 Remote 写期间有效。
    std::shared_ptr<void> buffer_lease;

    // 输入为 block_mask 解析后的原始 cache-key 下标，只构造缺失 block。
    std::function<bool(
        const std::vector<size_t>&,
        kv_cache_manager::BlockBuffers&)> build_buffers;
};
```

Device 入口保留现有公共接口：

```cpp
std::shared_ptr<AsyncContext> RemoteConnector::asyncWrite(
    const std::shared_ptr<KVCacheResource>& resource,
    const std::shared_ptr<Meta>& meta) {
    RemoteWriteInput input;
    input.cache_keys  = resource->cacheKeys();
    input.meta        = meta;
    input.buffer_lease = resource;
    input.build_buffers = makeDeviceBufferProvider(resource);
    return asyncWriteCommon(std::move(input));
}
```

Memory 入口：

```cpp
std::shared_ptr<AsyncContext> RemoteConnector::asyncWriteMemory(
    const std::vector<MemoryRemoteEvictionItem>& victims,
    const std::shared_ptr<Meta>& remote_meta,
    const std::shared_ptr<MemoryEvictionLease>& lease) {
    RemoteWriteInput input;
    input.cache_keys   = extractCacheKeys(victims);
    input.meta         = remote_meta;
    input.buffer_lease = lease;
    input.build_buffers = makeMemoryBufferProvider(victims, lease);
    return asyncWriteCommon(std::move(input));
}
```

统一核心接口：

```cpp
std::shared_ptr<AsyncContext>
RemoteConnector::asyncWriteCommon(RemoteWriteInput input);
```

使用 provider 而不是提前构造全部 buffers 的原因：

- `getWriteLocation()` 会发现部分 block 已经存在于 Remote。
- 先解析 `block_mask`，只为缺失 block 构造 IOV。
- 避免为已存在的 block 构造无用 buffer。
- `locations[j]` 与 `buffers[j]` 更容易保持严格对齐。
- Device 和 Memory 入口都只负责地址生成，不复制 Remote 写入协议逻辑。

Memory→Remote 调用的额外接口约束：

- `Meta::tokens()` / `token_ids` 允许为空。
- `cache_keys` 是 block 身份、顺序以及 `block_mask` 对齐的唯一依据。
- `getWriteLocation()`、`genWriteRequest()` 和公共写入流程不得从 token IDs 重新生成 cache keys。
- token IDs 为空不代表没有 blocks；是否为空写由 `cache_keys.empty()` 判断。
- 不得把触发本次 Device→Memory 淘汰请求的 token IDs 填到历史 Memory victims 上。
- 空 token `remote_meta` 由上层每次调用构造一次；RemoteConnector 不再复制或二次改写 Meta。

### 8.3 抽取公共控制面

将 Remote 写入拆为：

```text
asyncWriteDeviceTask()   // 现有GPU路径
asyncWriteMemoryTask()   // 新增CPU路径
writeBlockBuffers()      // 公共KVCM写流程
```

两条路径共用：

- `getNeedWriteGroups()`。
- `getWriteLocation()`。
- block mask 处理。
- `saveKvCaches()`。
- `finishWrite()`。
- AsyncContext 状态和 metrics。

区别仅在数据来源：

```text
Device：group_id + device block_id → 按block_mask延迟构造GPU IOV
Memory：memory block id + host layout → 按block_mask延迟构造CPU IOV
```

### 8.4 Block mask 对齐

`getWriteLocation()` 可能指出部分 key 已存在。必须同步过滤：

```text
cache_keys[i]
locations[i]
block_buffers[i]
```

公共控制面先把 mask 转成原始 key 下标，再交给对应 provider：

```cpp
std::vector<size_t> selectNeedWriteIndices(
    size_t key_count,
    const kv_cache_manager::BlockMask& mask);

auto selected_indices = selectNeedWriteIndices(cache_keys.size(), mask);
build_buffers(selected_indices, selected_buffers);
```

`selected_buffers[j]` 必须对应 `cache_keys[selected_indices[j]]`；禁止先构造全部 buffers 再过滤，否则会抵消 provider 的延迟构造收益。

如果 `selected_indices.empty()`，表示所有 victims 已经存在于 Remote。公共流程应将 Task A 作为成功的 no-op write 完成，不调用空的 `saveKvCaches()`；随后仍执行 `finishRemoteEviction()` 释放这些本地 victims，并启动 Task B。

需要区分两个“空”条件：

```text
cache_keys.empty()        → 没有Memory victims，本次不创建Task A
selected_indices.empty()  → victims均已在Remote，Task A成功完成
token_ids.empty()         → 合法metadata，不影响上述判断
```

### 8.5 KVCM CPU buffer 前提

当前 SDK 定义并实现了：

```cpp
enum class MemoryType : uint8_t {
    CPU = 0,
    GPU = 1,
};
```

SDK 动态库中存在 `cudaHostRegister`、`cudaHostUnregister` 和相关错误处理。第一版可以使用 CPU IOV，由 SDK 按需注册。每个 `BlockBuffer` 内禁止混用 CPU/GPU IOV，避免触发 `ER_UNCONSISTENT_MEMORY_TYPE`。

### 8.6 MR 注册策略

第一版不在 RTP-LLM 中自行调用 `ibv_reg_mr()`，也不额外维护 RDMA MR。RTP-LLM 只负责：

```text
MemoryType::CPU
+ host virtual address
+ size
+ backing lease
```

然后将 CPU IOV 交给 KVCM SDK。当前 SDK 动态库已经包含 host-register、RDMA backend 和错误处理逻辑，应由 SDK 根据实际 backend 完成所需的 pin/register/MR 管理。

第一版选择该方式的原因：

- RTP-LLM 看不到 KVCM backend 的 protection domain、device 和 QP 生命周期。
- RTP-LLM 自建 MR 很难安全地交给 SDK 复用。
- 重复注册同一地址可能引入冲突、重复 pin 和复杂的退出清理。
- 先验证 CPU IOV 的功能正确性，再根据 profiler 判断注册是否为瓶颈。

需要注意：`cudaHostRegister()` 与 RDMA `ibv_reg_mr()` 不是同一个操作。前者负责 CUDA 对 host memory 的 pin/mapping，后者负责将内存注册到 RDMA NIC 的 protection domain。实际传输是否需要两者、由谁执行，应由 KVCM SDK backend 决定。

性能优化阶段再评估“整个 Memory cache pool 一次性注册”：

```text
Memory pool 初始化
    ↓
KVCM SDK 注册整段连续 host pool
    ↓
每个 victim 使用 base + block offset
    ↓
服务退出时由 SDK 统一注销
```

这要求 KVCM SDK 提供正式的多 span 或额外 host-pool 注册接口，并保证注册句柄与 transfer client 生命周期一致。当前公开 `InitParams` 只有一个 `RegistSpan*`，而 RTP-LLM 已将它用于 Device KV pool，因此在 SDK 接口扩展前，不应擅自把该 span 改成 Memory pool，也不应在 RTP-LLM 内部自行注册但不告知 SDK。

MR 决策分阶段如下：

| 阶段 | RTP-LLM 手工注册 MR | 策略 |
|---|---:|---|
| 功能原型 | 否 | CPU IOV 直接交给 KVCM SDK |
| 正确性 Smoke | 否 | 验证 pageable/pinned host buffer 均可工作 |
| 性能测试 | 否 | 观察 SDK register、写带宽和延迟 |
| 注册确认成为瓶颈 | 仍不直接注册 | 推动 KVCM SDK 增加 host pool 注册 API |
| SDK 支持整池注册后 | 通过 SDK 注册 | 初始化一次、退出统一注销 |

## 9. 两个异步任务的依赖

新增链式 context，例如：

```text
rtp_llm/cpp/cache/connector/ChainedAsyncContext.h
rtp_llm/cpp/cache/connector/ChainedAsyncContext.cc
```

接口示意：

```cpp
class ChainedAsyncContext : public AsyncContext {
public:
    using NextTask =
        std::function<std::shared_ptr<AsyncContext>(bool first_success)>;

    ChainedAsyncContext(std::shared_ptr<AsyncContext> first,
                        NextTask next);

    bool done() const override;
    bool success() const override;
    void waitDone() override;
};
```

状态机：

```text
WAIT_FIRST
    ↓
START_SECOND
    ↓
WAIT_SECOND
    ↓
DONE
```

第一阶段失败时仍然启动 Task B。该依赖包含 CPU 线程池、RPC 和网络 I/O，不能只用 CUDA event 表达。

当前 `RemoteConnectorAsyncContext::waitDone()` 是空实现，不能依赖它完成阻塞等待。应通过 coordinator 周期性 `update()`、completion callback 或 continuation worker 在 `done()` 后启动 Task B。

## 10. 上层编排

编排应放在 Device cache 淘汰的拥有者中，不应让 Memory connector 直接依赖 RemoteConnector。

建议新增：

```cpp
std::shared_ptr<AsyncContext> asyncTieredEvictDeviceCache(
    const DeviceToMemoryEvictionEstimate& estimate);
```

伪代码：

```cpp
std::shared_ptr<AsyncContext>
TieredCacheManager::asyncTieredEvictDeviceCache(
    const DeviceToMemoryEvictionEstimate& estimate) {
    RTP_LLM_CHECK_WITH_INFO(cache_config_.groupNums() == 1,
                            "only one cache group is supported");

    const size_t total = memory_connector_->totalMemoryBlocks();
    const size_t free = memory_connector_->freeMemoryBlocks();
    const size_t used = total - std::min(total, free);
    const size_t high = total * memory_high_watermark_ratio_ / 100;
    const size_t incoming = estimate.estimated_d2h_block_num;
    const size_t remote_evict_num =
        used + incoming > high ? used + incoming - high : 0;

    if (remote_evict_num == 0) {
        return evictDeviceCacheToMemory(estimate.resource,
                                        estimate.meta);
    }

    auto victims = memory_connector_->prepareRemoteEviction(
        remote_evict_num);
    if (victims.empty()) {
        return evictDeviceCacheToMemory(estimate.resource,
                                        estimate.meta);
    }

    auto lease = makeMemoryEvictionLease(victims);
    auto remote_meta = makeRemoteWriteMeta(
        estimate.meta, /* token_ids = */ {});
    auto remote_ctx = remote_connector_->asyncWriteMemory(
        victims, remote_meta, lease);

    if (!remote_ctx) {
        memory_connector_->finishRemoteEviction(victims, false);
        return evictDeviceCacheToMemory(estimate.resource,
                                        estimate.meta);
    }

    return makeChainedAsyncContext(
        remote_ctx,
        [this, victims, estimate](bool remote_success) {
            memory_connector_->finishRemoteEviction(
                victims, remote_success);
            // Task B调用完整入口并在此时独立构造最终D2H plan。
            return evictDeviceCacheToMemory(estimate.resource,
                                            estimate.meta);
        });
}
```

## 11. Task B 的语义

Task B 必须调用完整的：

```cpp
evictDeviceCacheToMemory(resource, meta)
```

不能直接把它替换为 `memory_connector_->asyncWrite()`，因为完整 Device 淘汰还可能负责：

- Detach/pin Device cache 节点。
- 管理 Device cache tree/LRU。
- 等待 D2H 完成。
- 提交 Memory cache 索引。
- 释放 Device cache 引用和 block。
- 处理失败回滚。

Task B 在形成最终 plan、得到 `actual_d2h_block_num` 后，必须先按 Memory 高水位计算实际缺口。直接淘汰可以复用现有逐 block 释放机制，但触发数量改为水位公式，而不是仅在 allocation 失败时被动释放：

```cpp
const size_t immediate_evict_num = std::max<int64_t>(
    0,
    current_used + actual_d2h_block_num - memory_high_watermark_blocks);
auto victims = block_cache_->popForImmediateEviction(immediate_evict_num);
releaseCacheBackings(victims);  // 直接释放，不写Remote
```

建议通过不同接口明确区分：

```text
detachForRemoteEviction()  // Task A：写Remote
popForImmediateEviction()  // Task B：直接释放
```

## 12. 并发与生命周期

### 12.1 第一版并发策略：完整事务串行化

第一版不允许多个两级淘汰事务并行执行。串行化粒度为同一个 Memory pool 上的完整事务：

```text
Q1：计算estimate
    → Task A：Memory→Remote
    → Task B：实时构造plan并evictDeviceCacheToMemory
    → Q1完成
    ↓
Q2：重新计算estimate
    → Task A：Memory→Remote
    → Task B：实时构造plan并evictDeviceCacheToMemory
    → Q2完成
```

禁止以下并行方式：

```text
Q1：Task A → Task B
Q2：Task A → Task B
```

原因是 Q1、Q2 如果同时读取 `memory_pool_->freeBlocksNum()`，会把同一批空闲 block 重复计算到各自 plan 中；仅仅 detach 不同 victim 不能解决 free-capacity 重复消费问题。

建议在负责 Device cache 淘汰的上层增加单飞队列：

```cpp
class TieredCacheEvictionQueue {
public:
    std::shared_ptr<AsyncContext> enqueue(
        DeviceEvictionRequest request);

private:
    void tryStartNext();
    void onCurrentFinished(bool success);

private:
    std::mutex                        mutex_;
    std::deque<DeviceEvictionRequest> pending_;
    bool                              running_{false};
};
```

入队逻辑：

```cpp
std::shared_ptr<AsyncContext>
TieredCacheEvictionQueue::enqueue(DeviceEvictionRequest request) {
    auto result = request.result_context;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.push_back(std::move(request));
    }
    tryStartNext();
    return result;
}
```

单飞规则：

1. `running_ == false` 时取出队首请求，并设置 `running_ = true`。
2. 真正轮到请求时只生成 Task A 所需的 `DeviceToMemoryEvictionEstimate`。
3. 完成 Task A 后启动 Task B，由 Task B 基于实时状态独立生成执行 plan。
4. Task B 完成或失败后才设置 `running_ = false`。
5. 然后重新读取 Memory pool 状态并启动下一请求。

### 12.1.1 Task B 独立构造实时执行 plan

当某个请求成为队首后，Task A 之前只计算容量预估；Task A 完成后，Task B 重新读取实时状态并独立构造最终执行 plan：

```text
请求成为队首
    ↓
estimateDeviceCacheToMemoryBlocks()
    ↓
得到轻量 DeviceToMemoryEvictionEstimate
    └── estimated_d2h_block_num
    ↓
根据 estimate 计算 Memory 水位缺口和 Task A victims
    ↓
Task A：Memory→Remote
    ↓
Task B：evictDeviceCacheToMemory(resource, meta)
    ├── 重新读取实时 cache 状态
    ├── 独立完成 match/裁剪
    ├── 生成最终 CopyInfoPerKey/CopyPlan
    ├── 得到 actual_d2h_block_num
    ├── 按实际数量再次检查 Memory 水位
    └── 分配 backing、D2H、commit
```

这里的关键约束是：

```text
Task A 的 estimate 只决定预先写 Remote 的数量；
Task B 自己形成的实时 plan 才决定真正复制哪些 blocks。
```

Task B 启动时必须重新执行以下分析：

- 读取 `resource->cacheKeys()` 并确定当前有效 key 范围。
- 计算 Memory matched prefix。
- 判断最后一个 complete block。
- 裁剪 incomplete tail。
- 生成最终 `CopyInfoPerKey` 和 CopyPlan。

Task A 可能按 6 个 D2H blocks 预腾空间，而 Task B 实际形成 8 个或 4 个：

- 实际为 8：Task B 对额外 2 个 block 所需空间直接淘汰 Memory LRU，不再写 Remote。
- 实际为 4：Task A 多腾出的 2 个 block 保持为空闲空间，不影响正确性。

estimate 不保存 D2H 数据和布局快照，也不提前分配 Memory backing：

```text
Task A之前保存：请求/resource引用、estimated_d2h_block_num
Task B开始后执行：重新分析、构造plan、Memory backing分配、必要的直接淘汰、D2H、commit
```

预估对象统一使用第 5.3 节定义的 `DeviceToMemoryEvictionEstimate`，不得在编排层再定义另一种 plan 类型。

Task A 完成并释放 Memory victims 后，Task B 调用原始完整入口：

```cpp
evictDeviceCacheToMemory(resource, meta);
```

最终执行 plan 只能在以下时机生成：

```text
Q1：Q1的Task A完成后，由Q1的Task B生成
Q2：等待Q1的Task B结束，且Q2的Task A完成后，由Q2的Task B生成
```

因此 Task B 不会消费在异步等待期间过期的数据快照，也避免 Q2 在排队期间基于过期状态提前构造执行 plan。

Q2 入队时只保存生成 plan 所需的最小请求描述，不读取或缓存：

```text
freeBlocksNum
Memory shortage
Memory victim列表
```

这些容量相关信息必须等 Q2 真正成为队首后重新计算。

请求入队和 Task A 阶段都不得冻结具体 Device victims、Device block IDs 或 CopyInfo。Task B 启动时重新计算 Device 水位并选择 victims；如果压力已经消失，Task B 可以形成空 plan 并成功 no-op。estimate 与实际数量的差异只影响 Task A 的 Remote spill 收益，不影响 Task B 正确性。

串行化边界必须覆盖 Task A 和 Task B，而不是只串行 Task A：

```text
错误：Q1 Task A完成后立刻允许Q2 Task A，Q1 Task B仍在运行
正确：Q1 Task B也结束后才允许Q2 prepare/Task A
```

这样第一版不需要实现 `MemoryCapacityReservation`。Task B 内仍保留 Memory 不足时的直接淘汰兜底，用于处理不可淘汰 resident/in-flight block、Remote 失败和实际分配差异。

队列应按 Memory pool 隔离。第一版单 group 只有一个相关 pool，可以使用一个全局单飞队列；未来支持多 group/独立 pool 后，再按 pool ID 建立独立队列。

### 12.2 生命周期要求

链式 context 必须持有：

```text
DeviceToMemoryEvictionEstimate
├── Device KV resource
├── Meta
└── MemoryEvictionLease
    └── victim host blocks
```

Task A 完成前：

- Memory victim 不可匹配。
- Memory victim 不可被第二次淘汰。
- Host backing 不可释放或重新分配。
- 请求级 `resource`/`meta` 引用必须有效；此时尚未选定具体 Device victims，因此不持有 Device block lease。

Task A 完成后的顺序必须是：

```text
确认KVCM不再读取CPU IOV
    ↓
释放Memory victim backing
    ↓
启动evictDeviceCacheToMemory
```

第一版保守地以 `finishWrite()` 和整个 Remote context 完成为 host backing 的释放点。

### 12.3 取消与退出

- 排队但尚未开始的请求被取消时，从队列删除并释放其 Device 引用。
- 正在执行 Task A 的请求取消时，不得提前释放 Memory victim；等待 KVCM 不再访问 CPU IOV后再清理。
- Task A 无论成功、失败或超时，都必须继续完成 Memory victim 清理，再决定是否执行 Task B；若 Device 淘汰仍是资源回收所必需，Task B 必须继续。
- 服务退出时停止接受新请求，依次 drain 当前任务和队列；超过退出超时后执行明确的失败清理，不能遗留 host/device block 引用。

## 13. Remote metadata 问题

Memory→Remote 调用 RemoteConnector 时，token IDs 允许为空。控制面需要的字段为：

```text
unique_id
trace_id
cache keys
location spec group names
```

`unique_id`、`trace_id` 等请求级字段只用于调用追踪；block 的内容身份和远端存在性判断由显式 `cache_keys` 表达。`Meta::tokens()` 不是 Memory→Remote 写入的必要字段，可以传空数组。

RemoteConnector 必须提供按 cache key 写入的语义：

```cpp
getWriteLocationByCacheKeys(...)
```

如果复用现有 `getWriteLocation()` 接口，则需保证其接收显式 `cache_keys`，并允许：

```cpp
meta->tokens().empty() == true;
```

Memory cache item 不需要额外保存 `block_tokens`。尤其不得直接用当前请求的 token metadata 填充历史 victim，否则会制造 cache key 与 tokens 不一致的伪 metadata。

## 14. 配置开关

建议新增：

```text
ENABLE_MEMORY_CACHE_REMOTE_EVICTION=0
DEVICE_CACHE_HIGH_WATERMARK_RATIO=95
MEMORY_CACHE_HIGH_WATERMARK_RATIO=95
MEMORY_CACHE_REMOTE_EVICTION_TIMEOUT_MS=2000
MEMORY_CACHE_REMOTE_EVICTION_MAX_BLOCKS=32
MEMORY_CACHE_REMOTE_EVICTION_FAILURE_POLICY=DROP
```

启用条件：

```text
enable_memory_cache
&& enable_remote_cache
&& enable_memory_cache_remote_eviction
&& cache_config.groupNums() == 1
```

默认关闭，避免改变现有线上行为。

`device_cache_min_free_blocks` 标记为兼容字段。推荐优先级：

```text
显式设置DEVICE_CACHE_HIGH_WATERMARK_RATIO
    → 使用百分比水位
否则显式设置device_cache_min_free_blocks
    → 暂时使用旧逻辑并打印deprecated warning
否则
    → 默认使用95%水位
```

最终移除旧字段前，需要迁移相应 smoke、部署参数和 `ResourceContext` 单测。

## 15. 失败策略

| 场景 | 行为 |
|---|---|
| 无可淘汰 Memory victim | 直接启动 Task B |
| CPU BlockBuffer 构造失败 | 丢弃已 detach victim，启动 Task B |
| Remote task 提交失败 | 释放 victim，启动 Task B |
| getWriteLocation 失败 | 释放 victim，启动 Task B |
| saveKvCaches 失败 | 释放 victim，启动 Task B |
| finishWrite 失败 | 释放 victim，启动 Task B |
| Task B Memory 仍不足 | 直接淘汰其他 Memory LRU，不写 Remote |
| Task B D2H 失败 | 按原 Device 淘汰逻辑回滚或失败 |

原则：Remote 故障不得阻止 Device cache 回收。

## 16. 指标和日志

建议新增：

```text
memory_remote_evict_qps
memory_remote_evict_block_count
memory_remote_evict_success_block_count
memory_remote_evict_failed_block_count
memory_remote_evict_latency_us
memory_remote_evict_bytes
memory_remote_evict_inflight_blocks
memory_emergency_evict_block_count
device_to_memory_after_remote_latency_us
```

日志至少包含：

```text
trace_id
unique_id
cache_key
memory_block_id
block_size
IOV数量与总bytes
write_session_id
Remote写结果
Task B启动时间
Task B直接淘汰block数
```

## 17. 测试计划

### 17.1 MemoryBlockCache 单测

- `detachForRemoteEviction()` 返回完整 item。
- Resident block 不会被选择。
- Detach 后不能再 match。
- Detach 后不能被再次选中。
- Remote 完成前 backing 不释放。
- Generation 不匹配时不能释放新条目。

### 17.2 CPU BlockBuffer 单测

使用两层模拟数据：

```text
layer0 main KV
layer0 idx_K
layer1 main KV
layer1 idx_K
```

验证：

- 一个 victim 对应一个 BlockBuffer。
- IOV 顺序与 Device 路径一致。
- 所有 IOV 均为 `MemoryType::CPU`。
- IOV 总大小等于逻辑 block 大小。
- provider 只为 `selected_indices` 对应的 victims 构造 BlockBuffers。
- 多个被选中 victim 能批量生成多个 BlockBuffer。

### 17.3 Block mask 单测

输入：

```text
keys = [A, B, C]
mask = [已有A, 需要B, 需要C]
```

预期：

```text
uris = [B_uri, C_uri]
buffers = [B_buffer, C_buffer]
```

并验证 A 对应的 host IOV 没有被构造。

增加空 token IDs 用例：

```text
token_ids = []
keys = [A, B]
```

预期仍根据 keys 调用 `getWriteLocation()`；不得因 `token_ids.empty()` 提前返回空写。

增加全部已存在用例：mask 表示 A、B 均已在 Remote 时，不构造 CPU buffers、不调用空 `saveKvCaches()`，Task A 成功完成并正常启动 Task B。

### 17.4 Task 依赖单测

- Task A 未完成时，Task B 启动次数为 0。
- Task A 成功后，先释放 Memory victim，再启动 Task B。
- Task A 失败后，Task B 仍然启动。
- Task B 只启动一次。
- Shutdown 时未完成 task 能安全 drain/cancel。

### 17.5 水位策略单测

- Device total=1000、not-in-use=20、ratio=95，预期淘汰30 blocks。
- Device effective used 恰好95%，预期不淘汰。
- Memory total=1000、used=920、incoming=50、ratio=95，Task A预期选择20个 victims。
- Memory projected used 恰好95%，Task A预期不淘汰。
- Task A后发生实际空间偏差时，Task B通过直接淘汰恢复到95%以内，并且不新增Remote调用。
- Task A需要20个但只有12个可Remote淘汰victims时，先写出12个；Task B对最终实际缺口直接淘汰。
- Task A估计有D2H、但Task B重算后无需Device淘汰时，Task B成功no-op。
- Task B的最终plan不复用Task A阶段的Device block IDs或CopyInfo。
- 验证百分比向上/向下取整不会导致最终使用率高于配置水位。
- `incoming > max_used` 时不死循环，最终保留量不超过目标容量。
- Q2必须等待Q1完整事务结束后再读取当前used/free并计算水位。

### 17.6 Task B 直接淘汰单测

构造 Task A 只释放一个 block、Task B 需要两个 block 的场景，验证：

- 第二个缺口直接淘汰 Memory LRU。
- Remote 调用次数没有增加。
- 不产生递归 Remote spill。

### 17.7 KVCM CPU buffer 独立测试

步骤：

1. 申请 host memory 并写固定 pattern。
2. 构造 `MemoryType::CPU` IOV。
3. 调用 `saveKvCaches()`。
4. 清空本地目标 buffer。
5. 调用 `loadKvCaches()` 到另一个 CPU buffer。
6. 逐字节比较。

覆盖：

- Pageable host memory。
- `cudaMallocHost` pinned memory。
- 多 IOV。
- 多 BlockBuffer。
- 非连续地址。
- 超时和失败期间的 buffer 生命周期。
- 多 TP rank 同时写入。

### 17.8 Smoke 测试

新增 case：

```text
remote_cache_memory_eviction_basic
```

设置较小的 Memory cache，构造：

```text
Q1：前缀A进入Memory
Q2：前缀B进入Memory
Q3：触发A从Memory淘汰并写Remote，然后Device→Memory
Q4：再次请求前缀A
```

Q4 预期：

```text
memory_reuse_len = 0
remote_reuse_len > 0
输出与无cache基线一致
```

增加故障 case：

```text
remote_cache_memory_eviction_failure
```

验证 Remote 写失败后：

- Task B 仍完成。
- 服务不挂死。
- Cache miss 后重新计算结果正确。
- Remote failure metric 增加。

## 18. 分阶段实施

### 阶段一：验证 KVCM CPU IOV

- 编写独立 Save/Load 测试。
- 验证 pageable 和 pinned memory。
- 确认 `saveKvCaches()` 返回后的 host 内存安全边界。

### 阶段二：同步原型

- 只支持单 group。
- Memory victim 整体写 Remote。
- Task A 内同步完成 KVCM 写入。
- 然后调用完整 `evictDeviceCacheToMemory()`。
- 跑单测和 smoke，优先确认正确性。

### 阶段三：异步依赖

- 引入 `ChainedAsyncContext` 或 continuation。
- 调度线程不等待网络。
- Task A 到 Task B 保持严格依赖。
- 完善 shutdown/drain。

### 阶段四：并发与性能

- 限制 inflight block 数。
- 批量调用 `saveKvCaches()`。
- 增加 timeout、背压和失败降级。
- 评估整块 Memory pool 预注册，避免逐次 `cudaHostRegister()`。

### 阶段五：扩展范围

- 多 group。
- Hybrid attention。
- CP。
- Disk 到 Remote。
- DSV4 typed regions。

## 19. 验收标准

### 19.1 功能

- Memory victim 可以写入 Remote 并被后续请求命中。
- 多层 main KV 和 idx_K 内容正确。
- Task B 不会早于 Task A 完成启动。
- Task B 空间不足时不会再次调用 Remote。

### 19.2 生命周期

- 无 host block 提前释放或覆盖。
- 无 Device block 提前释放。
- 无 use-after-free 和 ABA 删除。
- 进程退出时没有无法回收的异步任务。

### 19.3 故障安全

- KVCM 超时或失败不会阻塞 Device cache 回收。
- Remote 失败只导致 cache miss，不影响推理正确性。
- 不产生调度活锁。

### 19.4 性能与可观测性

- 调度线程不阻塞于网络 I/O。
- 多个 victim 支持批量写入。
- Inflight 数量有上限。
- Remote spill 和 emergency eviction 均有独立指标。

## 20. 关键风险

1. **空 token IDs 被误判为空写**：Memory→Remote 允许 `Meta::tokens()` 为空，代码必须以显式 `cache_keys` 判断是否存在待写 blocks。
2. **CPU buffer 生命周期**：KVCM 真正结束读取前不能释放或覆盖 host block。
3. **IOV 布局不一致**：Memory 和 Device 路径的层顺序、idx_K 顺序和大小必须完全相同。
4. **并发空间竞争**：Task A 释放的空间可能被其他写任务抢走，因此 Task B 必须保留直接淘汰兜底。
5. **空实现的 waitDone**：当前 Remote async context 不能用 `waitDone()` 表达依赖，需要真实 completion 驱动。
6. **Host register 成本**：按请求注册 pageable memory 可能影响性能，后续需评估 pinned pool 或整池 MR 注册。

## 21. 最终设计原则

```text
Memory→Remote 是 Task A 的显式降级动作。
Device→Memory 是 Task B 的完整 Device 淘汰事务。
Task B 依赖 Task A，但 Task A 失败不能阻止 Task B。
Task B 内的 Memory 空间不足只做直接淘汰，不递归写 Remote。
Remote 写完成以前，Memory victim 的 host backing 必须保持有效。
```
