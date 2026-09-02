# BlockTree 与 P2P 分支合并关键问题

## 目标流程

```text
allocator 匹配 BlockTree
    → 分配本地回迁和待计算范围所需的全部 Device block
    → BlockTree 与 P2P 并行 load
    → 两路 load 均完成
    → Stream 进入运行状态
```

P2P 只补读本地未覆盖区间，不重复加载 Device 命中及 Host/Disk 待回迁部分。

---

## 1. Reuse 范围接口已对齐

### 结论

P2P 直接读取：

```cpp
connector_context.treeCoveredBlockNum();
```

`StreamCacheResource::asyncLoadCache()` 在启动 P2P 前同步生成该值：存在 `LoadAsyncContext` 时读取 Tree 总匹配范围；不存在时回退到 `resource.deviceReuseBlockNum()`，覆盖纯 Device 命中但无需异步 load 的情况。数值按值传入 `StreamConnectorContext`，不与后续 `finalizeAllocatorLoad()` reset Context 竞争。

allocator 完成后，P2P 传输区间为：

```text
[connector_context.treeCoveredBlockNum(), p2pMatchedBlockNum)
```

区间长度按逻辑 cache-key block 计数。

### 命名建议

| 接口 | 含义 |
|---|---|
| `deviceReuseBlockNum()` | 已经位于 Device、无需 Tree/P2P load 的 block 数量 |
| `LoadAsyncContext::matchedBlocks()` | Device 命中加 Host/Disk 待回迁的 Tree 计划覆盖数量 |
| `treeCoveredBlockNum()` | 传给 P2P 的稳定快照，作为 P2P 起始 block |

当前分支尚使用旧接口名 `logicalMatchedBlocks()`；同步 `ef36df62` 后机械替换为 `matchedBlocks()`。

---

## 2. P2P 目标计划与地址转换

### 分配边界

allocator 成功返回后，请求范围需要的 Device block 已全部分配。P2P：

- 不再执行 malloc；
- 不直接操作 block pool；
- 只向请求已分配的目标 block 写入数据。

P2P 写入完成后，仍由 allocator 既有的 `insertIntoCache()` 路径将数据发布到 BlockTree。

### `LayerBlockConverter`

保留 `LayerBlockConverter`，用于隔离 P2P 与 cache 底层实现，并继续支持轻量 Mock：

- 将 `KVCacheRegionName` 接口替换为稳定 tag 接口；
- 内部继续调用 allocator 的 `convertIndexToBufferByTag()`；
- 完成 `layer/tag/block_id → BlockInfo` 地址转换。

`getAllBuffers()` 的职责保持不变：通过 `allLayerCacheBase()`，按 tag 和 layer 展开 KV/scale 底层内存，供 RDMA 初始化注册；它不参与请求级目标计划。

### 请求级目标计划

`LayerCacheBufferUtil` 改用 `KVCacheResource + CacheTopology`：

1. 枚举 layer 和 tag；
2. 生成 `cache_key → block_id` 目标计划；
3. 将目标计划交给 `LayerBlockConverter` 转换为传输地址。

FULL、SWA、LINEAR 以及 CP 场景下，“逻辑 cache-key 位置 → 物理 block-table 位置”的映射由 allocator 提供公共接口，不能继续使用旧 region 路径的同下标切片。

### 跨节点对齐

两端使用以下稳定信息对齐：

```text
tag + layer_id + cache_key + partition_id
```

- 不传递进程内 `group_id`；
- 建连时校验 tag 集合、layer 数量、block 布局和数据类型。

---

## 3. BlockTree 与 P2P 异步 Context 融合

### Stream 持有方式

Stream 不创建额外的 combine context，而是分别持有：

```cpp
tree_load_context_;
p2p_load_context_;
```

P2P 生产路径直接返回 `P2PConnectorAsyncReadContext`。`FusedAsyncReadContext` 只剩保留的 Coordinator 源码使用。

### 启动顺序与并发

1. allocator 在 malloc 内同步确定 `localCoveredBlockNum()`；
2. allocator 分配全部目标 block，绑定并 commit Tree context；
3. malloc 返回后，Tree 复制在后台运行；
4. `asyncLoadCache()` 立即启动 P2P，并显式传入本地覆盖快照；
5. P2P 与 Tree 并行写入互不重叠的区间。

### 完成条件

`loadCacheDone()` 分别检查两路 Context。只有两路均满足“Context 不存在或已经完成”，Stream 才能离开 `LOADING_CACHE`。

| Tree 状态 | P2P 状态 | Stream 行为 |
|---|---|---|
| 未完成 | 任意 | 继续等待 |
| 已完成 | 未完成 | 继续等待 |
| 成功 | 成功或无 Context | 离开 `LOADING_CACHE` |
| 失败 | 任意 | 立即发起 P2P cancel；保持 `LOADING_CACHE`，待 P2P 静默后结束请求 |

### 失败与重试

- Tree 失败：立即通过 `cancelP2PLoad()` 将取消信号传给 P2P；不转为 P2P 兜底重算，也不在取消完成前提交 Stream 终态；
- P2P 传输失败：与 Tree 保持一致，直接结束请求，不在 Stream 侧重试；
- Prefill 的请求级资源会在首次 `handleRead()` 时从 store 中取走，不能使用同一个 `unique_key` 再次发起传输；
- 无远端增量命中时按未命中处理，不上报传输错误。

P2P 只有成功后才能更新总 reuse、remote reuse、MTP 和 side-channel。失败的目标 block 不得发布到 BlockTree。

### 取消与释放

任一路失败或 Stream 释放时：

1. Tree 调用 `cancelLoad()`；
2. P2P 调用新增的 `cancelP2PLoad()`；
3. P2P 取消同时覆盖 match 和 read 阶段；`cancelP2PLoad()` 只负责发出异步取消信号，返回不代表在途传输已经停止；
4. Stream 保留 `p2p_load_context_` 并维持 `LOADING_CACHE`，由 ContextChecker 推进取消和资源租约查询，Stream 侧轮询不会阻塞调度线程；
5. P2P Context 满足 `done() == true` 后才进入释放判断；若是 `P2PConnectorAsyncReadContext`，还必须满足 `resourceHoldPending() == false`，确认 Connector 不再持有目标 Device block；
6. P2P 静默后清理 Context、提交 Tree load 错误，最后释放请求的 Device block 和对应引用。

等待期间继续由 Connector 引用和资源租约保护目标 block，避免 Tree load 失败后 Stream 先释放内存，而 P2P 线程仍在访问同一批地址。

引用生命周期语义不变：

- 原 P2P 的 `requestReference()/requestFree()` 适配为 DSV4 统一的 `BlockRefType::REQUEST`；
- 来源端继续由 lease 和 `BlockRefType::CONNECTOR` 保护；
- 成功、失败、重试耗尽、取消和 Stream 释放路径都必须正确释放对应引用。

### Coordinator 与 RPC 接线

P2P 生产链路不再经过 `KVCacheConnectorCoordinator`，由 `KVCacheManager` 直接完成：

- 初始化和持有 `P2PConnector`；
- `asyncMatch()` 与 `asyncRead()` 串接；
- Connector 类型的 block 引用保护；
- `handleRead()`、`executeFunction()`、side channel 和 cancel 路由；
- shutdown。

P2P 异步任务由自有线程池和 ContextChecker 推进，不再使用 Coordinator 轮询线程。

---

## 4. Prefill 逐层写入适配

### 调用链

Python attention 在每层 KV 计算结束后继续调用 `WriteCacheStoreOp`。该 Op 只负责生成层级写入任务，实际工作转移到现有异步 writer 和 P2P Prefill worker：

```text
WriteCacheStoreOp
    → CacheStoreAsyncWriter
    → execWriteCacheStore
    → KVCacheManager::writeP2PLayer
    → P2PConnector::writeByLayerTag
    → P2PConnectorWorkerPrefill::writeByLayerTag
    → StoreWaitContextChecker
```

生产链路不经过 Coordinator。`CacheStoreAsyncWriter` 承担 CUDA event 等待和 host block-table 读取，engine 线程只创建 event 并提交任务；`StoreWaitContextChecker` 在 event 就绪后将 layer/tag buffer 发布给发送线程。

### Region、CP 与 block 引用

- 每次调用携带明确的 `layer_id + tag`，同一层的多个 tag 分开存放，buffer key 为 `layer_id:tag`；
- P2P 与普通 CacheStore 共用 `buildCacheStorePlan()` 的逻辑，先完成 FULL/SWA/LINEAR 与 CP rank 投影，再把对齐后的 `cache_key + block_id` 交给 Manager；
- P2P worker 不再重复执行 tag 或 CP 映射；
- Manager 将模型内 layer id 转为全局 layer id，并校验该层是否拥有对应 tag；
- 通过 allocator 的 `BlockRefType::CONNECTOR` 引用保护本次传输涉及的 block，引用由 `LayerCacheBuffer` 生命周期自动释放；
- 请求 deadline 随逐层任务传入，并受 `p2p_prefill_resource_hold_ms` 上限约束。

### 与普通 CacheStore 的关系

两条写入可以并存：普通 CacheStore 为空时，P2P 逐层写入仍会执行；普通 CacheStore 启用时，二者共用同一个 layer event 和映射元数据。这样无需恢复 Coordinator，也不会要求 P2P Connector 依赖 CacheStore 实现。

---

## 5. Coordinator 生产依赖清理（已完成）

### 现状

当前 `KVCacheManager -> P2PConnector` 生产路径不依赖 Coordinator。P2P 自有线程池和 ContextChecker
负责异步任务推进，因此已删除 Coordinator 的接口、实现、构建目标、测试和 Mock。

### 已完成的清理

1. 将共享接口拆成 `connector_base` 和 `connector_context`：生产目标只依赖 `Meta`、`KVCacheConnector`、LayerContext 和 ReadWriteContext。
2. P2P `components`、P2P `connector` 及其 Mock 只依赖共享接口目标，不再依赖 Coordinator 构建目标。
3. `KVCacheManager` 和 Stream 直接依赖 `connector_context`；models_py 中未使用的 Coordinator header 依赖已经移除。
4. 删除仅服务旧多 Connector 编排的 Coordinator 轮询线程、融合上下文和专属测试。
4. Coordinator 源码、接口、轮询/fused 逻辑及专属测试保持不变，避免把本次清理扩大为源码删除。

---

## 6. CP/CSA 场景适配

### 当前状态

CP-aware rank 调度已在 `feat(p2p): add CP-aware rank scheduling` 中实现初版。当前选择仍由 rank 0 运行 Decode scheduler；由于 allocator 的 block id 在各 rank 间同步，rank 0 可以根据 `worker_rank % cp_size` 为每个 worker 分别构造 CP-local 目标计划，再通过逐 rank RPC 分发。无需在每个 worker 重复运行 scheduler。

### 当前设计

1. **CP 信息交换**：Prefill 的 `GetPeerInfo` 返回有效 KV-cache `cp_size`；Decode 将其保存到 Stream，并通过 `Meta::P2PRoutingContext::prefill_cp_size` 传给 P2P scheduler。旧节点未返回该字段时按 `cp_size=1` 处理。
2. **布局校验**：发生真实传输时要求 Prefill 与 Decode 的 `cp_size` 一致，同时要求 worker 数能被 `cp_size` 整除；不满足时在创建传输前失败。`no_transfer` 只走控制面，不依赖数据布局。
3. **逐 rank 目标计划**：rank 0 遍历全部 Decode worker，以 `worker_rank % cp_size` 计算其 CP rank，并调用 `LayerCacheBufferUtil::convert()` 和 `CPSlotMapper` 生成各自的 cache key、block id 与地址计划。
4. **逐 rank 分发**：`P2PBroadcastClient::broadcastPerRank()` 按 worker 地址发送对应计划，不再把 rank 0 的计划复制给所有 worker。
5. **合法空投影**：某个 Decode rank 没有目标 block 时，请求携带 `allow_empty_projection=true`。该标记只允许在 `cp_size>1` 且请求不含 layer block 时使用，普通空 READ 或“空标记与非空计划并存”仍判为错误。
6. **Prefill 投影**：Prefill worker 使用自身 `tp_rank % cp_size` 构造逐层发送计划，使来源 key 与对应 Decode rank 的期待集合对齐。

### 已有测试

单测已覆盖 CP round-robin key 分发、各 worker 请求隔离、合法空投影、Prefill/Decode `cp_size` 不一致以及 `CPSlotMapper` 的 CP 映射。100% Tree 命中的 `no_transfer` 路径不创建数据计划，仍执行 StartLoad/side-channel 控制面收敛。

### 剩余验证与改造

1. Prefill 逐层生产尚未提供独立的 `ready + empty` 状态；CSA/HCA/SWA 下某 rank 的某个 `(layer, tag)` 合法为空时，仍需避免生产端误判失败或等待到 deadline。
2. Connector 引用目前仍由 `LayerCacheBuffer` 持有请求资源；需要验证其实际保护范围与 CP 投影后的发送 block 完全一致，并覆盖成功、失败、超时、取消及晚到回调。
3. 需要执行 CP × CSA/HCA/SWA 多节点回归，至少覆盖 block 数少于 `cp_size`、不均匀尾部、部分 Tree 命中、100% Tree 命中、传输失败和取消，并检查所有 rank 的完成通知和引用归零。
