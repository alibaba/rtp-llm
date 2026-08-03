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
resource.deviceReuseBlockNum();
```

将其作为原实现中的 `already_reuse_num`。该值实际保存的是 `logical_reuse_blocks`，包括 Device 命中以及 Host/Disk 待回迁部分，因此已经能够表示 P2P 所需的本地覆盖范围。

allocator 完成后，P2P 传输区间为：

```text
[resource.deviceReuseBlockNum(), p2pMatchedBlockNum)
```

区间长度按逻辑 cache-key block 计数。

### 命名建议

| 接口 | 含义 |
|---|---|
| 当前 `deviceReuseBlockNum()` | 功能上可直接复用，但名称与实际语义不一致 |
| 建议 `localCoveredBlockNum()` | 统一表达 Device 命中加 Host/Disk 待回迁的本地覆盖数量 |
| 可选 `deviceReadyBlockNum()` | 仅表示已经位于 Device、可以直接使用的 block 数量 |

改名是语义清理，不是 P2P 接入的前置条件。

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
| 失败 | 任意 | 结束请求并收敛两路任务 |

### 失败与重试

- Tree 失败：结束请求，不转为 P2P 兜底重算；
- P2P 传输失败：与 Tree 保持一致，直接结束请求，不在 Stream 侧重试；
- Prefill 的请求级资源会在首次 `handleRead()` 时从 store 中取走，不能使用同一个 `unique_key` 再次发起传输；
- 无远端增量命中时按未命中处理，不上报传输错误。

P2P 只有成功后才能更新总 reuse、remote reuse、MTP 和 side-channel。失败的目标 block 不得发布到 BlockTree。

### 取消与释放

任一路失败或 Stream 释放时：

1. Tree 调用 `cancelLoad()`；
2. P2P 调用新增的 `cancelP2PLoad()`；
3. P2P 取消同时覆盖 match 和 read 阶段；
4. 等待两路任务及在途传输终止；
5. 最后释放 Device block 和对应引用。

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

为保留原 P2P Coordinator 源码，合并时对以下内容做了 DSV4 兼容适配：

- `IKVCacheConnectorCoordinator.h`；
- `KVCacheConnectorCoordinator.h/.cc`；
- `connector/BUILD`；
- Coordinator 相关测试和 Mock。

适配包括 `AsyncContext` 路径迁移、group/tag 资源结构、allocator 引用保护，以及移除已不存在的 Memory/Remote Connector 依赖。

当前 `KVCacheManager -> P2PConnector` 生产路径已经不依赖 Coordinator。Coordinator 的接口、实现和测试继续保留，但 `//rtp_llm/cpp/cache/connector:connector` 只服务 Coordinator 自身及其测试，不再进入 P2P 生产或 P2P 测试目标的依赖闭包。

### 已完成的清理

1. 将共享接口拆成 `connector_base` 和 `connector_context`：生产目标只依赖 `Meta`、`KVCacheConnector`、LayerContext 和 ReadWriteContext。
2. P2P `components`、P2P `connector` 及其 Mock 只依赖共享接口目标，不再依赖 Coordinator 构建目标。
3. `KVCacheManager` 和 Stream 直接依赖 `connector_context`；models_py 中未使用的 Coordinator header 依赖已经移除。
4. Coordinator 源码、接口、轮询/fused 逻辑及专属测试保持不变，避免把本次清理扩大为源码删除。

---

## 6. 遗留：CP/CSA 场景统一适配

### 状态与处理边界

本节问题已经确认，但本轮暂不修改实现。非 CP 路径继续独立检查；后续应将 CP、CSA、HCA、SWA 的 rank 投影、完成通知和覆盖校验作为一组改动统一处理，避免只修复某个局部等待条件。

### 已确认问题

1. **Decode 目标计划只按 rank 0 构建。** 当前 Decode scheduler 使用自身的 `cp_rank/cp_size` 生成一次接收计划，再把同一计划广播给所有 Decode worker。CP 下每个 rank 应按自己的 block 所有权构建本地计划，否则非 rank 0 worker 会收到错误的 cache key 和 block 地址。
2. **Prefill 与 Decode 的 rank 投影可能不相交。** Prefill 逐层写入已经按本 rank 的 `buildCacheStorePlan()` 投影，而 Decode worker 收到的仍可能是 rank 0 计划，导致来源端提供的 key 与目标端期待的 key 不一致。
3. **缺少“当前 rank 无数据但该 layer/tag 已生产完成”的表示。** CSA/HCA 的 FULL 类 group 使用 block round-robin。短请求或不均匀尾部下，某些 rank 的合法计划为空；`writeCacheToP2P()` 不会为其生成 buffer，而 Prefill `sendKVCache()` 仍按 topology 等待全部 `(layer, tag)`，最终只能依赖 deadline 退出。
4. **CP 的 RDMA expected 集合仍需按 rank 正确构建。** RDMA 数据面已经校验所有 Decode expected key、子 block 和长度均由 Prefill 覆盖，并允许 Prefill 提供额外 key；CP 剩余问题是 expected 集合本身必须由各 Decode rank 本地生成，并通过 `ready + empty` 表达合法空集合。
5. **Connector block 引用需要使用 CP 投影后的资源。** 逐层写入所保护的 block 必须与本 rank 实际发送计划一致，不能直接对未投影的请求资源增加 Connector 引用。

100% Tree 命中的 `no_transfer` 控制路径不进入普通数据发送等待，因此不依赖上述空 rank 数据计划；它仍需纳入 CP 回归测试，验证所有 rank 都能完成控制面收敛。

### 统一修复方案

1. Decode scheduler 只广播请求级元数据；每个 Decode worker 使用自己的 `cp_rank/cp_size` 构建本地接收计划和地址映射。
2. 为每个 `(layer, tag)` 增加明确的 ready 状态，允许 `ready + empty`。Prefill 等待生产完成状态，只对非空 buffer 创建 RDMA 传输。
3. 在 RDMA 建立任务前双向校验 expected/offered key 集合；除明确的 `ready + empty` 外，缺失或多余 key 都返回错误。
4. Connector 引用基于 `buildCacheStorePlan()` 投影后的 block 集合获取和释放，并覆盖成功、失败、超时、取消及晚到回调。
5. 增加 CP × CSA/HCA/SWA 测试矩阵，至少覆盖：block 数少于 `cp_size`、不均匀尾部、部分 Tree 命中、100% Tree 命中、传输失败和取消；检查所有 rank 的完成通知与引用归零。
