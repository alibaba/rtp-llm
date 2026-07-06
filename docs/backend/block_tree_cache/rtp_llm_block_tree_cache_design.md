# RTP-LLM BlockTreeCache — KV Cache 统一树形管理器设计

> `BlockTreeCache` 是 RTP-LLM 的统一树形 KV Cache 管理器（无分裂机制，纯 block 粒度树），替换当前碎片化的缓存管理组件（`SharedBlockCache` / `PrefixTreeMemoryBlockCache` / `BlockCache` / `KVCacheMemoryConnector`），实现 Device/Host/Disk/Remote 四层统一树形管理。

---

📚 **BlockTreeCache 设计文档系列**
| **设计文档（主）** | [C++ 代码参考](rtp_llm_block_tree_cache_code_reference.md) | [示例与详细图示](rtp_llm_block_tree_cache_examples.md) |

---

## 目录

1. [整体介绍](#1-整体介绍)
   - 1.1 [背景与动机](#11-背景与动机)
   - 1.2 [当前架构分析](#12-当前架构分析)
   - 1.3 [设计目标与非目标](#13-设计目标与非目标)
   - 1.4 [架构概览](#14-架构概览)
   - 1.5 [核心数据结构](#15-核心数据结构)
   - 1.6 [数据结构关系图](#16-数据结构关系图)
2. [Tree Cache 核心机制](#2-tree-cache-核心机制)
   - 2.1 [Match 机制（前缀匹配）](#21-match-机制前缀匹配)
   - 2.2 [Insert 机制（插入）](#22-insert-机制插入)
   - 2.3 [淘汰机制](#23-淘汰机制)
   - 2.4 [锁机制（引用计数）](#24-锁机制引用计数)
   - 2.5 [多层存储与数据迁移](#25-多层存储与数据迁移)
   - 2.6 [EvictionHeap（淘汰堆）](#26-evictionheap淘汰堆)
   - 2.7 [BlockTreeCache](#27-blocktreecache)
3. [Component 系统](#3-component-系统)
   - 3.1 [Component 与 Tree Cache 的对接](#31-component-与-tree-cache-的对接)
   - 3.2 [FullComponentGroup](#32-fullcomponentgroup)
   - 3.3 [SWAComponentGroup](#33-swacomponentgroup)
   - 3.4 [LinearComponentGroup](#34-linearcomponentgroup)
   - 3.5 [BlockTreeCache 与 BlockPool 的协同关系](#35-blocktreecache-与-blockpool-的协同关系)
4. [与现有系统的集成](#4-与现有系统的集成)
   - 4.1 [对 KVCacheAllocator 的影响](#41-对-kvcacheallocator-的影响)
   - 4.2 [删除 KVCacheMemoryConnector，整合到 BlockTreeCache](#42-删除-kvcachememoryconnector整合到-blocktreecache)
   - 4.3 [移除 Namespace 机制与独立驱逐组适配](#43-移除-namespace-机制与独立驱逐组适配)
   - 4.4 [替换策略与迁移路径](#44-替换策略与迁移路径)
5. [补充信息](#5-补充信息)
   - 5.1 [完整配置](#51-完整配置)
   - 5.2 [典型请求链路](#52-典型请求链路)

**配套参考文件**：
- [C++ 代码参考](rtp_llm_block_tree_cache_code_reference.md) — 所有 C++ 接口定义和数据结构代码
- [示例与详细图示](rtp_llm_block_tree_cache_examples.md) — DSV4 模型示例、流程图和目录结构

---

## 1. 整体介绍

### 1.1 背景与动机

#### 当前问题

RTP-LLM 的 KV Cache 管理经过多轮迭代，形成了**三层独立管理**的碎片化架构：

| 层级 | 当前组件 | 问题 |
|---|---|---|
| **Device (GPU)** | `SharedBlockCache` + `BlockCache`（legacy） | 树形 `PrefixTreeNode` + 扁平 LRU 并存；Namespace 机制复杂；独立驱逐依赖额外的 leaf_lru 集合 |
| **Host (CPU)** | `KVCacheMemoryConnector` + `PrefixTreeMemoryBlockCache` | 3500+ 行实现；独立的树结构；与 Device 侧完全不共享元数据 |
| **Disk** | `MemoryDiskBlockCache` + `DiskBlockPool` | 嵌在 MemoryConnector 内部；无独立淘汰策略 |
| **Remote** | `RemoteConnector` | 树外独立管理；与 Device/Host/Disk 的驱逐决策完全脱节 |

**核心痛点**：

1. **无统一视图** — Device、Host、Remote 各自维护独立的元数据，无法跨层做全局最优驱逐决策
2. **load_back 不可见** — Host→GPU 的数据恢复由 Connector 异步驱动，树不参与调度
3. **驱逐不协调** — Device 驱逐到 Host 后，Host 驱逐时无法感知 Device 的状态变化；Remote 写入与本地驱逐完全脱节
4. **SWA/LINEAR 特殊化散落** — 混合模型的 SWA 窗口锁、chain eviction 等逻辑分散在多处
5. **MemoryConnector 过重** — 3500+ 行代码混合了缓存管理、数据搬运、TP 广播等多重职责
6. **Remote 树外管理** — RemoteConnector 独立于缓存树，无法感知本地 Disk 状态，无法做 EAGER_BACKUP 等优化

#### 目标

构建 `BlockTreeCache` —— 一个**统一的树形 KV Cache 管理器**：

1. **统一树结构** — Device/Host/Disk/Remote 四层数据共用一棵树的元数据
2. **逐层淘汰（Tier-to-Tier Demotion）** — DeviceLeaf → HostLeaf → DiskLeaf → Remote 四级逐层，全局最优决策
3. **Component 化** — FULL/SWA/LINEAR 通过 Component 接口统一管理
4. **替换碎片化组件** — 替代 `SharedBlockCache`、`PrefixTreeMemoryBlockCache`、`BlockCache`
5. **保持兼容** — 不改变 `KVCacheManager`、`KVCacheAllocator` 的公开接口（`KVCacheConnectorCoordinator` 长期将删除，当前仅因 P2P Connector 保留方式未确定而暂留）

---

### 1.2 当前架构分析

#### 当前组件关系图

```
KVCacheManager
├── KVCacheAllocator (GPU block 分配 + 缓存管理)
│   ├── BlockPool (GPU block 物理池)
│   ├── SharedBlockCache (Device 前缀缓存)
│   │   ├── LRUCache (扁平缓存: CacheKeyType → block indices)
│   │   ├── PrefixTreeNode 树 (依赖追踪 + chain eviction)
│   │   ├── leaf_lru_ (set<LeafKey>，淘汰候选)
│   │   └── Namespace 机制 (default / gpu_logical / cp_canonical)
│   └── BlockRefCounter (引用计数)
│
├── KVCacheConnectorCoordinator (offload 管理)
│   ├── KVCacheMemoryConnector (CPU 缓存层，3500 行)
│   │   ├── PrefixTreeMemoryBlockCache (Host 侧树形缓存)
│   │   │   ├── Node 树 (children / parent / kinds[2])
│   │   │   ├── leaf_lru_[kKindCount] (按 kind 的淘汰候选)
│   │   │   └── retired_items (in-flight 引用)
│   │   ├── MemoryDiskBlockCache + DiskBlockPool (Disk 层)
│   │   ├── BlockPool (CPU 物理内存池: complete/incomplete/compressed/state_swa)
│   │   └── BroadcastManager (TP 广播)
│   ├── P2PConnector (Prefill-Decode 分离)
│   └── RemoteConnector (远端缓存，可选)
```

---

### 1.3 设计目标与非目标

#### 目标

| 目标 | 说明 |
|---|---|
| **统一树结构** | Device/Host/Disk/Remote 四层数据在一棵树中管理，共享拓扑和元数据 |
| **No-Split Invariant** | 固定 block 粒度，不分裂/合并节点（RTP-LLM 已满足：`CacheKeyType` 按 block 对齐） |
| **Component 化** | FULL/SWA/LINEAR 各自定义验证器、淘汰规则（引用保护由 BlockPool 统一管理） |
| **逐层淘汰** | DeviceLeaf → HostLeaf → DiskLeaf → Remote 四级逐层淘汰，全局最优 |
| **Remote 纳入树编排** | 移除旧 `RemoteConnector`，Remote 作为 L4 通过 `StorageBackend` 接口由树统一编排（树负责 prefetch/backup 调度，但不在节点中持有 remote_key，Remote 不影响节点生命周期） |
| **保持接口兼容** | `KVCacheManager` / `KVCacheAllocator` 的公开接口不变（`KVCacheConnectorCoordinator` 长期将删除，当前仅因 P2P Connector 保留方式未确定而暂留） |
| **C++17 实现** | 无 Python 依赖，纯 C++ header + implementation |
| **线程安全** | 单锁或分区锁保护（替代当前多组件各自持锁） |
| **增量迁移** | 可分阶段替换，不要求一次性替换所有组件 |

#### 非目标

| 非目标 | 说明 |
|---|---|
| **不改变 P2P 连接器** | P2PConnector（PD 分离）保持独立，不参与树管理 |
| **不改变 TP 广播机制** | BroadcastManager 保持不变 |
| **不支持 GDS 直通** | Disk↔GPU 不支持 GDS 直通；Remote↔GPU 支持 RDMA 直通（加载和备份均允许绕过 CPU） |
| **不改变 CacheKey 计算方式** | 继续使用现有的 `initCacheKeys()` / `updateCacheKeys()` |

---

### 1.4 架构概览

#### 新架构组件关系

```
KVCacheManager
├── KVCacheAllocator (GPU block 分配，接口基本不变)
│   └── BlockPool (GPU block 物理池，多独立 Device Pool，DSV4: 7 个)
│
├── BlockTreeCache ★ 新组件，淘汰流程协调者（替换 SharedBlockCache + KVCacheMemoryConnector 的协调职责）
│   ├── BlockTree ★ 纯树数据结构（统一拓扑，每个节点持有 GroupSlot[]）
│   │   └── TreeNode 树 (cache_key → children map，每个节点持有 GroupSlot[])
│   │
│   ├── ComponentGroup[] ★ 主动管理实体（从 CacheConfig 推导，持有行为接口 + 三层淘汰堆）
│   │   ├── FullComponentGroup (group_type=FULL)
│   │   │   ├── device_heap (EvictionHeap, DeviceLeaf 淘汰堆，仅 Leaf 节点入堆)
│   │   │   ├── host_heap   (EvictionHeap, HostLeaf 淘汰堆，仅 Leaf 节点入堆)
│   │   │   └── disk_heap   (EvictionHeap, DiskLeaf 淘汰堆，仅 Leaf 节点入堆)
│   │   ├── SWAComponentGroup (group_type=SWA)
│   │   │   ├── device_heap (EvictionHeap, 任意节点可入堆，优先淘汰中间节点)
│   │   │   ├── host_heap   (EvictionHeap, 任意节点可入堆，优先淘汰中间节点)
│   │   │   └── disk_heap   (EvictionHeap, 任意节点可入堆，优先淘汰中间节点)
│   │   └── LinearComponentGroup (group_type=LINEAR)
│   │       ├── device_heap (EvictionHeap, 任意节点可入堆，优先淘汰中间节点)
│   │       ├── host_heap   (EvictionHeap, 任意节点可入堆，优先淘汰中间节点)
│   │       └── disk_heap   (EvictionHeap, 任意节点可入堆，优先淘汰中间节点)
│   │
│   ├── Component[] (纯描述性对象：MemoryBlockLayerTagSlot 布局 + 所属 group + type)
│   │
│   ├── ThreadPool ★ 异步淘汰任务线程池（执行 EvictionTask：数据搬运 + write-through）
│   ├── host_pool_ (BlockPool, AllocationType::HOST，CPU 物理内存池，同 group 多 device block 打包为 1 个 host block)
│   ├── DiskBlockPool (磁盘池，镜像 host_pool_)
│   ├── CopyEngine (GPU↔CPU↔Disk 数据搬运，依赖 MemoryBlockLayerTagSlot 计算 offset)
│   ├── BroadcastManager (TP 广播，从 KVCacheMemoryConnector 移入)
│   └── StorageBackend ★ 新接口（可插拔远端后端）
│       └── RemoteBackend — 替代旧 RemoteConnector
│
└── KVCacheConnectorCoordinator (精简，长期将删除)
    └── P2PConnector (保留，暂不做测试；P2P Connector 保留方式确定后 Coordinator 将被移除)

(删除) KVCacheMemoryConnector → 逻辑移入 BlockTreeCache 目录
(删除) RemoteConnector → 由 StorageBackend 替代

```

#### 职责划分

| 组件 | 职责 | 不做什么 |
|---|---|---|
| **BlockTreeCache** | 淘汰工作流协调、异步任务管理、跨 group 级联、节点删除、线程安全 | 不直接操作树或堆的底层细节 |
| **BlockTree** | 纯树拓扑操作：查找、插入、删除节点 | 不做淘汰决策，不感知 heap |
| **ComponentGroup** | 选择候选（driveEviction）、执行淘汰（evictFromTier）、定义匹配/插入语义、构建 TransferDescriptor、持有三层淘汰堆 | 不协调跨 group，不删除节点 |
| **EvictionHeap** | 纯堆操作：push/pop/invalidate/onAccess，策略适配（LRU/LFU/FIFO/PRIORITY） | 不感知树结构，不做协调 |
| **Component** | 纯描述：MemoryBlockLayerTagSlot 布局、所属 group、type、device pool 映射 | 不持有行为接口 |
| **BlockPool** | GPU/Host/Disk block 物理分配/释放 | 不做缓存管理决策 |
| **StorageBackend** | 可插拔远端存储接口（batch_read/write/exists/delete） | 不感知树拓扑 |
| **KVCacheConnectorCoordinator** | 编排 P2P Connector 的异步操作 | 不再管理 Memory/Remote Connector（长期将删除） |

#### 多层存储拓扑

BlockTree 采用四层星型拓扑，CPU 为中心节点：

- **GPU（L1 热层）**：容量最小、速度最快，存放正在使用的 KV 数据
- **CPU（L2 暖层）**：默认中继层，GPU 淘汰数据的第一站
- **Disk（L3 本地冷层）**：本地持久化，所有数据经 CPU 中转
- **Remote（L4 持久化层）**：由 `StorageBackend` 接口抽象（替代旧 `RemoteConnector`），树负责编排 prefetch/backup 调度，但不在 GroupSlot 中持有 remote_key，Remote 不影响节点生命周期

**关键决策**：

- Disk 必须经 CPU 中转：Disk↔GPU 不支持 GDS 直通，淘汰和加载均经 CPU 内存中转
- Remote 支持 RDMA 直通：Remote↔GPU 允许绕过 CPU 直接传输（加载和备份均可），也可选择经 CPU 中转
- Remote 数据自管：树不持有 remote_key，只通过节点 token 路径的确定性 hash 查询/写入远端（树负责决策何时 prefetch/backup，但不在节点中跟踪 Remote 状态）
- **树节点数据范围**：L1 + L2 + L3（GroupSlot 持有 device/host/disk 数据）；Remote 由树编排 prefetch/backup 调度但不在 GroupSlot 中跟踪
- **Remote 数据不影响节点生命周期**：Remote 由后端自行管理 GC/TTL。节点被本地删除后，远端数据仍可能存在，后续同一前缀再次出现时可通过 prefetch 拉回

---

### 1.5 核心数据结构

#### ComponentGroup 机制

对于 DSV4 等复杂模型，GPU 上有多个独立 BlockPool（如 7 个），但由于 CUDA kernel 实现的限制，这些 pool 无法合并。而在 CPU Memory / Disk 层，同类型的多个 device pool 可以**打包合并**为一个 host block（一个 host block 内部按 `MemoryBlockLayerTagSlot` 顺序拼接所有层的数据）。

> **注意**：某些 group（如 DSV4 的 `hca_state`，`NON_REUSABLE`）是临时 GPU 缓冲区，不参与前缀复用。这些 group **不入 BlockTreeCache**——它们由上层 allocator 通过 BlockPool 直接管理（分配 → 使用 → 直接还给 BlockPool）。BlockTreeCache 只包含可复用（REUSABLE）的 group。

**ComponentGroup** 是跨层的分组单位，也是**行为的主体**：
- 同 group 的多个 device block 在 Host/Disk 层**共享**一个物理 block
- 同 group 的 block **生命周期一致**，以 group 为最小单位淘汰
- ComponentGroup 持有所有行为接口（match 验证、insert 提交、驱逐决策、Transfer 构建）和三层淘汰堆
- `component_group_id` 从 `CacheConfig` 的 group tag 推导（BlockTreeCache 构造时建立映射，由 BlockTree 持有树拓扑）
- `group_type`（FULL / SWA / LINEAR）决定该 group 的淘汰候选规则和匹配语义

**DSV4 示例**：

| component_group_id | group_type | 包含的 Component | Host Block 内容 |
|---|---|---|---|
| 0 (compressed) | FULL | csa_kv, hca_kv, indexer_kv | 所有 COMPRESSED_KV 层数据打包 |
| 1 (state_swa) | SWA | indexer_state, csa_state, swa_kv | 所有 FIXED_STATE 层数据打包 |

> `hca_state`（NON_REUSABLE）不入 BlockTreeCache，由上层 allocator 通过 BlockPool 直接管理。

普通模型（LLaMA/Qwen 等）只有一个 ComponentGroup（id=0, group_type=FULL），所有行为与单 pool 一致。

#### MemoryBlockLayerTagSlot（Memory Block 布局描述符）

`MemoryBlockLayerTagSlot` **（Memory/Disk 层独有）** 描述 memory block 内部一个 (layer, group) 槽位的位置和大小，包含 `layer_id`、`tag`（group tag）和 `stride_bytes` 三个字段。与现有 `KVCacheMemoryConnector::layerTagSlots()` 语义一致。Device 层不需要此概念——每个 device block 直接对应一个独立的 Device BlockPool。

> 📎 [完整 MemoryBlockLayerTagSlot 定义](rtp_llm_block_tree_cache_code_reference.md#memoryblocklayertagslot)

`MemoryBlockLayerTagSlot` 列表由 Component 持有，用于指导 D2H/H2D 拷贝时 memory block 内的 byte offset 计算（与现有 `prepareCopyBuffers` 逻辑一致）。Tree 本身不感知 `MemoryBlockLayerTagSlot` 细节。

#### ComponentGroup（主动管理实体）

BlockTreeCache 构造时从 `CacheConfig` 推导所有 ComponentGroup。ComponentGroup 是行为的主体，持有 **7 个行为钩子**（Match 2 个、Insert 2 个、Evict 2 个、Transfer 1 个）和三层淘汰堆（device/host/disk）。基类提供默认实现（virtual，非纯虚），子类（`FullComponentGroup` / `SWAComponentGroup` / `LinearComponentGroup`）按需覆盖。

> 📎 [完整 ComponentGroup + MatchValidator 定义](rtp_llm_block_tree_cache_code_reference.md#componentgroup主动管理实体)

`host_block_size` 的计算方式：对该 group 内所有 `MemoryBlockLayerTagSlot` 的 `stride_bytes` 求和（与现有 `memoryCacheBlockSizeBytes()` 逻辑一致）。计算完成后**向上对齐到 4KB**（或可配置的 alignment），以支持 O_DIRECT（Host↔Disk 绕过 page cache）和 SIMD 对齐的 memcpy 优化。多出的 padding 不影响正确性，仅略多占用内存。

**ComponentGroup 与 MemoryBlockLayerTagSlot 的关系**：`ComponentGroup` 通过 `component_indices` 引用纯描述性的 `Component[]`，每个 Component 持有自己的 `MemoryBlockLayerTagSlot[]`。关系链：`ComponentGroup → component_indices → Component[] → memoryBlockLayerTagSlots → MemoryBlockLayerTagSlot[]`。`component_indices` 用于：(1) 计算 `host_block_size`（遍历 group 内所有 Component 的 MemoryBlockLayerTagSlot 求和 stride_bytes）；(2) D2H/H2D 打包时收集 MemoryBlockLayerTagSlot 按 layer_id 排序计算 byte offset。

#### GroupSlot（多层数据位置）

每个 `GroupSlot` 对应一个 ComponentGroup 在一个 TreeNode 上的数据位置。包含三层数据位置（`device_blocks`、`host_block`、`disk_slot`）、三层堆状态标志（`in_device_heap` 等）和状态查询方法（`has_device_value()` 等）。`device_blocks` 中每个元素来自一个独立 Device BlockPool，`host_block` 是打包后的单个 block。

> 📎 [完整 GroupSlot 定义](rtp_llm_block_tree_cache_code_reference.md#groupslot多层数据位置)

**关键语义**：
- `device_blocks` 中每个元素对应一个独立 Device BlockPool 的 block index（如 DSV4 compressed group 有 csa_kv、hca_kv、indexer_kv 三个 device block）
- `host_block` 是一个**打包 block**——内部按 `MemoryBlockLayerTagSlot` 顺序拼接了该 group 所有层的数据

#### TreeNode

TreeNode 包含树结构（`cache_key`、`children` map、`parent`）和多层数据位置（`group_slots`，按 `component_group_id` 索引）。

> 📎 [完整 TreeNode 定义](rtp_llm_block_tree_cache_code_reference.md#treenode)

> **Root 节点**：BlockTree 内部持有 `TreeNode* root_`。root 是一个特殊的空节点：`cache_key = 0`（无实际数据），`group_slots` 全为空（不持有任何 block），仅作为树的起点。所有实际数据节点都是 root 的子孙。root 不参与淘汰。

> **淘汰元信息**：`last_access_time`、`hit_count`、`is_resident` 不在 TreeNode 中管理，而是由各 ComponentGroup 按需维护。理由：不同 group_type 可能使用不同的淘汰策略（Full 用 LRU，SWA 可能不需要，LINEAR 可能用 LFU），且 `is_resident` 的粒度是 ComponentGroup 级别。`last_access_time` 使用 `steady_clock::time_point`（或 `uint64_t` 纳秒时间戳），支持 lifetime 统计。

> **驱逐状态**：`in_device_heap` / `in_host_heap` / `in_disk_heap` 是 **per-group** 的（因为淘汰以 group 为单位），放在 GroupSlot 上而非 TreeNode 上。


#### 数据位置状态表（以 GroupSlot 为单位）

**REUSABLE group**（write-back 策略下，L1-L3 之间同一时刻同一 group 的数据只可能在一个层级有效）：

| device_blocks | host_block | disk_slot | 状态 |
|---|---|---|---|
| 有 valid | invalid | invalid | 仅在 GPU（刚插入或 load_back 后） |
| 全 invalid | valid | invalid | GPU 已淘汰，仅在 Host |
| 全 invalid | invalid | valid | Host 已淘汰，仅在 Disk |
| 全 invalid | invalid | invalid | **该 group 为空** |

**节点删除条件**：节点所有 group 的 `is_empty()` 均为 true（即 `device_blocks` 全 invalid 且 `host_block` 无效 且 `disk_slot` 无效）时，节点从树中移除。单 group 数据缺失（如 SWA group 被独立淘汰）不会导致节点删除，只要其他 group 在任意层级仍有数据。

#### MatchResult

`BlockTreeMatchResult` 包含最佳匹配节点、匹配 block 数、block 索引序列、异步加载上下文和分层加载统计（host/disk/remote load_back 数）。

> 📎 [完整 BlockTreeMatchResult 定义](rtp_llm_block_tree_cache_code_reference.md#blocktreematchresult)

### 1.6 数据结构关系图

本节通过图示说明各数据结构之间的静态定义关系和运行时协作关系，包含 DSV4 静态元数据、运行时 TreeNode 数据、D2H 降级字节偏移计算、结构引用关系总图和普通模型 vs DSV4 退化关系。

> 📎 [完整示例与图示](rtp_llm_block_tree_cache_examples.md)

---

## 2. Tree Cache 核心机制

### 2.1 Match 机制（前缀匹配）

采用 No-Split Match 逻辑，适配 RTP-LLM 的 `CacheKeyType`。`match()` 接受按 block 计算的 `cache_keys` 序列，返回 `BlockTreeMatchResult`。

> 📎 [match() 签名](rtp_llm_block_tree_cache_code_reference.md#blocktreecache-对外接口)

#### 匹配流程

从 root 开始，逐个取出 `cache_keys[i]`，在当前节点的 `children` 中查找。命中则移动到子节点，运行**所有 Component 的验证器**；所有验证器通过才记为有效匹配边界。未命中或 `cache_keys` 序列耗尽时终止。由于 children map 以 `CacheKeyType` 为 key，不存在"部分匹配"——hash 命中或未命中，没有中间状态。`cache_keys` 序列由 `initCacheKeys()` 按 `seq_size_per_block` 对齐生成，每个 key 对应一个完整 block，尾部不完整 block 不产生 key。

#### No-Split 带来的简化

- children map 以 `CacheKeyType` 为 key，不存在"部分匹配"或"key 冲突"
- 不需要 `_split_node` / `redistribute_on_node_split`
- 验证器不需要处理分裂后的状态重分配
- 匹配是纯读操作，没有副作用

RTP-LLM 天然是 block-aligned 的（`CacheKeyType` 按 `seq_size_per_block` 对齐），不需要额外适配。`seq_size_per_block` 通常为 64/128/256，损失可忽略。

#### 验证器共识

所有组件验证器必须同时通过。验证器使用列表求值（非短路），确保所有验证器**无论结果如何都会被调用**——SWA 验证器是有状态的（累计窗口长度），不能因其他验证器短路而跳过状态更新。

| 组件 | 验证逻辑 |
|---|---|
| Full | 路径上**每个节点**在任意层级有数据（device/host/disk 任一有效）。要求从 root 到当前节点的整条路径都有 Full 数据 |
| SWA | 累计连续窗口 ≥ `sliding_window_size`。验证器内部维护 `connected_to_root` 和累计长度 |
| LINEAR | 当前节点有 LINEAR 状态数据 |

#### 多层存储下的匹配

验证器允许 Host/Disk 数据参与匹配——只要节点在任意层级有数据即可命中。match 内部自动处理 load_back：匹配完成后，所有命中的 block 均已位于 GPU（或正在加载中，通过 `async_context` 等待）。调用方不需要区分"纯 device 匹配"和"需要加载的匹配"。

#### 与当前的对接

`KVCacheAllocator::initMalloc()` 中现有的 `SharedBlockCache::match()` 调用替换为 `BlockTreeCache::match()`。

---

### 2.2 Insert 机制（插入）

`insert()` 从 `parent` 节点（match 结果节点，nullptr=root）开始逐 block 向下遍历，接受 `cache_keys` 序列和每个节点的 `GroupSlot`（`slots.size() == cache_keys.size()`）。

> 📎 [insert() 签名](rtp_llm_block_tree_cache_code_reference.md#blocktreecache-对外接口)

从 `parent` 节点开始（而非始终从 root）逐 block 向下遍历：

- **命中**：children 中存在该 `CacheKeyType` 对应的子节点，移动到该子节点继续。命中时不创建新节点，但会更新热度（LRU 时间戳或 hit_count），Full/SWA/LINEAR 均按此逻辑处理。
- **未命中**：创建新子节点作为当前节点的分叉，附加 `slots[i]` 作为该节点的 `GroupSlot` 数据

剩余 token 不足构成一个完整 block key 时也停止。具体来说，RTP-LLM 的 `initCacheKeys()` 在生成 cache_key 序列时已按 `seq_size_per_block` 对齐计算——只有完整 block 才会产生一个 cache_key，尾部不完整 block 不生成 key，也不参与 match/insert。这与 rtp-llm 内部逻辑一致。

#### 与当前的对接

`KVCacheAllocator::insertIntoCache()` 中现有的 `SharedBlockCache::put()` 调用替换为 `BlockTreeCache::insert()`。

---

### 2.3 淘汰机制

#### 淘汰层级总览

淘汰以 **ComponentGroup 为最小单位**，同 group 的所有 device block 一起操作，不允许 group 内部分淘汰。淘汰采用**异步三阶段**（详见 [2.7 BlockTreeCache](#27-blocktreecache)）：

| 操作 | 方向 | 触发条件 | 结果 |
|---|---|---|---|
| Device Demotion (REUSABLE) | GPU → CPU | Device 使用率超过水位（`watermark_ratio`） | Phase 1: 从 device_heap 选候选；Phase 2: CopyEngine 将同 group 所有 `device_blocks` 打包写入 1 个 `host_block`（异步）；Phase 3: 释放 GPU slots，更新状态 |
| Host Demotion | CPU → Disk/Remote/∅ | Host 使用率超过水位 | 释放 `host_block`，降级到 Disk（异步）+ write-through 写入 Remote |
| Disk Eviction | Disk → ∅ | Disk 使用率超过水位 | 直接释放 disk slot（Remote 已通过 write-through 写入，无额外 I/O） |
| 级联淘汰 | 同节点同介质 | 高优先级 group 淘汰后自动触发 | FULL 淘汰 → 级联 SWA + LINEAR；SWA 淘汰 → 级联 LINEAR |
| Node Deletion | All → ∅ | 所有 group 为空 | 从树中移除节点 + 祖先链清理 |

#### 淘汰候选管理（每个 ComponentGroup 持有三层堆）

每个 ComponentGroup 持有独立的三层 **heap（堆）**（Device/Host/Disk 各一个）管理淘汰候选。不同 `group_type` 对“哪些节点有资格进入 heap”的规则不同。Leaf 判定以 **GroupSlot** 为单位（而非单个 block）。

**淘汰堆元素（EvictionEntry）**：为适配多种淘汰策略（LRU/LFU/FIFO/Priority），堆中每个元素维护统一的状态字段（`last_access_time`、`hit_count`、`priority`、`insert_seq`），不同策略使用不同子集。

> 📎 [完整 EvictionEntry 定义](rtp_llm_block_tree_cache_code_reference.md#evictionentry淘汰堆元素)

**策略适配**：堆的实现封装在 `EvictionHeap` 类中（详见 [2.6 EvictionHeap](#26-evictionheap淘汰堆)），底层为 `std::priority_queue` + 失效标记机制，排序 key 由当前配置的淘汰策略决定：

| 策略 | 排序 key | 更新时机 |
|---|---|---|
| **LRU** | `last_access_time` | match/insert 时更新为当前时间 |
| **LFU** | `hit_count` | match 命中时递增 |
| **FIFO** | `insert_seq` | 插入时设置，之后不更新 |
| **Priority** | `priority` | 业务逻辑设置 |

淘汰时先弹出堆顶元素，再检查 `isEvictable()`（引用计数等），不满足则跳过继续弹下一个。

| ComponentGroup (group_type) | Device Heap 候选 | Host Heap 候选 | Disk Heap 候选 | 淘汰后果 |
|---|---|---|---|---|
| **Full** | DeviceLeaf（无子节点在该 group 有 device value） | HostLeaf（无子节点有 host value） | DiskLeaf（无子节点有 disk value） | 删除整个节点（级联清空所有 group） |
| **SWA** | 任意有 device value 的节点，优先中间节点 | 任意有 host value 的节点，优先中间节点 | 任意有 disk value 的节点，优先中间节点 | 释放 SWA group 数据 + 级联 LINEAR |
| **LINEAR** | 任意有 device value 的节点，优先中间节点 | 任意有 host value 的节点，优先中间节点 | 任意有 disk value 的节点，优先中间节点 | 仅释放 LINEAR 状态 |

#### Full Device 淘汰（DeviceLeaf）

**DeviceLeaf 入堆条件**（以 GroupSlot 为单位）：非 root 节点、该 group 的 `device_blocks` 有 valid、所有 device block 的 BlockPool 引用计数为 1（仅 BlockTreeCache 自身引用，无 request/CopyEngine 使用）、无子节点在该 group 的 Device 上有数据。

> **DeviceLeaf ≠ 树叶子节点**。DeviceLeaf 不要求 `children.empty()`，只要求没有子节点在该 group 有 device value。当中间节点的所有子节点被淘汰后，该节点**自动变为 DeviceLeaf**。

> **"夹心饼干"场景**：如果一个节点的子节点全部被淘汰（device_blocks 全 invalid），该节点自动变为 DeviceLeaf，可以入堆。这个级联是隐式的，不需要显式检查。

**DeviceLeaf 级联示例**：淘汰 C → parent B 自动变为 DeviceLeaf → 淘汰 B → parent A 自动变为 DeviceLeaf，级联是隐式的。

> 📎 [完整 DeviceLeaf 级联示例](rtp_llm_block_tree_cache_examples.md#5-淘汰级联示例)

Device 淘汰流程：从 Full heap 弹出最冷节点 → 若 Host 有容量则执行 device_to_host 降级（同 group 所有 `device_blocks` 通过 Component 的 `MemoryBlockLayerTagSlot` 打包写入 1 个 `host_block`，释放 GPU slots） → 触发双层级联：parent 可能变为新 DeviceLeaf（同层），被降级的节点可能变为 HostLeaf（跨层）。

#### Full Host 淘汰（HostLeaf）

**HostLeaf 入堆条件**（以 GroupSlot 为单位，仅 REUSABLE group）：非 root 节点、该 group 已驱逐（`device_blocks` 全 invalid）、有 host 数据（`host_block` 有效）、host block 的引用计数为 1（仅 BlockTreeCache 自身引用）、无子节点在该 group 拥有 `host_value`。

HostLeaf 有两个来源：

1. **DeviceLeaf 降级产生**：DeviceLeaf 被淘汰时，若 Host 有容量，执行 device_to_host 降级（同 group `device_blocks` 全 invalid，打包写入的 `host_block` 保留）。降级后若该节点无子节点在该 group 拥有 `host_value`，则成为 HostLeaf
2. **子节点 HostLeaf 被淘汰后父节点自动补位**：当一个 HostLeaf 被淘汰（释放 `host_block`）后，其 parent 若也是 evicted 状态且有 `host_block`，且现在所有子节点在该 group 都没有 `host_value`，则 parent 自动变为新的 HostLeaf

**淘汰流程**：从 Host heap 弹出最冷 HostLeaf → 若 Disk 有容量则 host_to_disk 降级，同时 write-through 异步写入 Remote，否则若有 Remote 则异步写入 Remote，否则仅释放。释放后检查 parent 是否满足 HostLeaf 条件（来源 2）。

#### Full Disk 淘汰（DiskLeaf）

**DiskLeaf 入堆条件**（以 GroupSlot 为单位，仅 REUSABLE group）：非 root 节点、该 group 的 `device_blocks` 和 `host_block` 均无效、有 disk 数据（`disk_slot` 有效）、disk slot 的引用计数为 1（仅 BlockTreeCache 自身引用）、无子节点在该 group 拥有 `disk_value`。

DiskLeaf 有两个来源：

1. **HostLeaf 降级到 Disk 产生**：HostLeaf 被淘汰时，若 Disk 有容量，执行 host_to_disk 降级（`host_block=invalid`，`disk_slot` 保留）。降级后若该节点无子节点在该 group 拥有 `disk_slot`，则成为 DiskLeaf
2. **子节点 DiskLeaf 被淘汰后父节点自动补位**：当一个 DiskLeaf 被淘汰（释放 `disk_slot`）后，其 parent 若也是 evicted 状态且有 `disk_slot`，且现在所有子节点在该 group 都没有 `disk_value`，则 parent 自动变为新的 DiskLeaf

**淘汰流程**：从 Disk heap 弹出最冷 DiskLeaf → 直接删除（Remote 数据已通过 write-through 写入，无需检查）。释放后检查 parent 是否满足 DiskLeaf 条件（来源 2）。

#### 淘汰策略

每层可配置独立的淘汰策略（默认：Device=LRU、Host=LRU、Disk=FIFO，Remote 由 StorageBackend 自管 GC/TTL）。各层的可配置范围和设计理由详见 [2.6 节 EvictionHeap](#26-evictionheap淘汰堆)。

#### 级联驱逐（同节点同介质跨 ComponentGroup）

当高优先级 ComponentGroup 在某层淘汰时，级联淘汰**同节点同介质**的低优先级 group 数据。这是同节点内不同 ComponentGroup 之间的驱逐级联，区别于 Device→Host→Disk 的**逐层淘汰**。级联由 BlockTreeCache 在 ComponentGroup 完成自身淘汰后统一触发。

**优先级排序**：FULL > SWA > LINEAR

| 触发者 (group_type) | 触发层级 | 级联行为（同节点同介质） |
|---|---|---|
| **Full** evict at tier T | Device / Host / Disk | 级联：同节点 SWA group 在 tier T 的数据也被淘汰 → 再级联 LINEAR group 在 tier T |
| **SWA** evict at tier T | Device / Host / Disk | 级联：同节点 LINEAR group 在 tier T 的数据也被淘汰 |
| **LINEAR** evict at tier T | Device / Host / Disk | 不级联 |

**级联示例**：包含三个场景——A: Full Device 淘汰级联 SWA、B: Full Host 淘汰级联 SWA、C: SWA 独立淘汰不影响 Full。

> 📎 [完整级联示例](rtp_llm_block_tree_cache_examples.md#5-淘汰级联示例)

**级联规则补充**：
- 级联淘汰仅影响**同一节点**的**同一层级**，不会跨节点也不会跨层级
- 被级联淘汰的 group 节点会从对应 heap 中移除（已不在该层），并可能进入下一层 heap

#### 驱逐调度流程（BlockTreeCache → ComponentGroup → EvictionTask）

完整的驱逐调度链路分为异步三阶段：触发与定位 → Phase 1 选择候选（同步持锁）→ Phase 2 异步执行（无锁）→ Phase 3 完成回调（同步持锁）→ 堆维护 → 循环。

> 📎 [完整驱逐调度流程](rtp_llm_block_tree_cache_examples.md#6-驱逐调度流程)

**独立驱动 vs 级联**的关系：
- **独立驱动**：每个 ComponentGroup 的 `driveEviction` 都可以被独立调用。当 SWA group 的 device pool 空间不足时，直接调用 SWA group 的 `driveEviction`，不需要经过 Full group
- **级联**：无论是独立驱动还是被级联触发，淘汰后都会检查是否需要向下级联（Full → SWA → LINEAR）

#### 祖先链清理

节点被删除后，从 parent 向上检查是否变为空节点（无子节点且所有 **REUSABLE** group 无数据）。清理逻辑：从 parent 向上遍历，若当前节点有子节点则停止，若任一 **REUSABLE** group 有 device/host/disk 数据则更新 heap 状态后停止，否则从 `parent.children` 中移除并继续向上。

---

### 2.4 锁机制（引用计数）

不使用线程锁，通过 `BlockPool` 的引用计数防止活跃 block 被淘汰。TreeNode 本身不持有引用计数，所有引用状态由 BlockPool 统一管理。

#### 引用计数由 BlockPool 统一管理

TreeNode 本身不持有引用计数。所有 block 级别的引用计数由 `BlockPool` 统一管理（对应现有的 `BlockRefCounter`）。

**引用计数基线**：BlockTreeCache 将 block 存入缓存时，调用 `blockCacheReference()` 增加一次引用（对应现有 `SharedBlockCache::put()` 中的 `group_pools_[gid]->blockCacheReference()`）。因此：

| 引用计数 | 含义 | 可驱逐？ |
|---|---|---|
| **1** | 仅 BlockTreeCache 自身引用 | ✅ 可驱逐 |
| **>1** | 还有 request 或 CopyEngine 在使用 | ❌ 不可驱逐 |

BlockTreeCache 通过 `isEvictable()` 检查 GroupSlot 上各层 block 的 BlockPool 引用计数：`device_blocks`、`host_block`、`disk_slot` 任一引用计数 > 1 则不可驱逐。

> 📎 [完整 isEvictable() 实现](rtp_llm_block_tree_cache_code_reference.md#isevictable引用计数检查)

#### 各 ComponentGroup 引用保护策略

| ComponentGroup (group_type) | 策略 | 范围 |
|---|---|---|
| Full | 路径锁 | match 节点 → root，通过 BlockPool 引用计数保护路径上的 block |
| SWA | 窗口锁 | match 节点向上累计 ≥ `sliding_window_size` 时停止。释放时沿同路径回退 |
| LINEAR | 单节点锁 | 仅 match 节点 |

**SWA 窗口锁**：从 match 节点向上，累计 token 数达到 `>= sliding_window_size` 时停止。释放时沿同路径回退相同节点数。支持提前释放（`releaseWindowLock`）——当 decode 位置已超过窗口范围时，SWA 锁可提前释放，同时级联释放 LINEAR 锁。

**No-Split 下的 SWA 窗口锁**：粒度从 token 精度退化为 **block 粒度**，可能锁定略多于 `sliding_window_size` 的 token（最多多出 `seq_size_per_block - 1` 个 token）。额外锁定的节点不会被驱逐，但也不影响正确性。

---

### 2.5 多层存储与数据迁移

#### 淘汰/写入路径（Eviction / Write）

本地层采用 **write-back** 策略（数据仅在淘汰时才写入下层），Remote 采用 **write-through** 策略（淘汰时同步写入 Remote）：

**本地层降级（write-back）**：

| 路径 | 触发者 |
|---|---|
| GPU → CPU | Device 淘汰（水位触发，DeviceLeaf → Host 降级） |
| CPU → Disk | Host 淘汰（水位触发，HostLeaf → Disk 降级） |
| GPU → ∅ | Host 未开启时，Device 淘汰直接释放 |
| CPU → ∅ | Disk/Remote 均未开启时，Host 淘汰直接释放 |
| Disk → ∅ | Disk 淘汰直接删除 |

**Remote 备份（write-through）**：

| 路径 | 触发者 |
|---|---|
| GPU → Remote | Device 淘汰时异步写入 Remote（支持 RDMA 直通） |
| CPU → Remote | Host 淘汰时异步写入 Remote |

**约束**：
- 本地层降级严格逐层（GPU→CPU→Disk），不允许 GPU→Disk 跨层直写
- Remote 写入允许 GPU→Remote RDMA 直通（绕过 CPU）
- **不存在 Disk→Remote 路径**：Remote 数据在 Device 或 Host 淘汰时已写入（write-through），Disk 淘汰时直接删除即可

#### 读取/加载路径（Load / Read）

match 命中低层数据后的加载路径允许跨层，以减少中间拷贝、提升性能：

| 路径 | 触发者 | 说明 |
|---|---|---|
| CPU → GPU | load_back | match 命中 Host 数据，直接 DMA 到 GPU |
| Disk → GPU | load_back | match 命中 Disk 数据，跨层直接加载到 GPU（经 CPU 中转内存，但不在 Host 层驻留） |
| Remote → GPU | prefetch | 从远端拉取到 GPU，支持 RDMA 直通（绕过 CPU）或经 CPU 中转 |
| Remote → CPU | prefetch | 从远端拉取到 Host，等待后续 load_back |

**跨层加载的含义**：
- **Disk → GPU**：经 CPU 内存中转（不支持 GDS 直通），但**不在 Host 层创建缓存条目**——CPU 内存仅作 DMA 中转缓冲区。理由：(1) load_back 场景下数据从 Disk 加载到 GPU 后立即被使用，不需要在 Host 保留副本；(2) Host 是稀缺资源（GB 级），不应被一次性传输占满；(3) 后续若 GPU 淘汰该数据，会正常走 Device→Host 降级路径，此时才创建 Host 副本
- **Remote → GPU**：支持 RDMA 直通（绕过 CPU 直接写入显存），也支持经 CPU 中转。无论哪种方式均不在 Host 层创建缓存条目

**选择策略**：load_back 时优先从最近的**已开启且有数据**的层级加载。若 Host 开启且有数据则 Host→GPU；若 Host 无数据但 Disk 开启且有数据则 Disk→GPU（跨层，经 CPU 中转）；若本地无数据但 Remote 开启则 Remote→GPU（跨层，支持 RDMA 直通）。关闭的层不参与加载。

#### 各层容量与淘汰触发

各层容量计算方式如下：

| 层级 | 容量来源 | 计算方式 |
|---|---|---|
| L1 Device | `CacheConfig::block_num` | GPU block 总数（已有） |
| L2 Host | `KVCacheConfig::memory_cache_size_mb` | `memory_cache_size_mb * 1MB / block_size_bytes` |
| L3 Disk | `KVCacheConfig::memory_cache_disk_size_mb` | `memory_cache_disk_size_mb * 1MB / block_size_bytes` |
| L4 Remote | — | 由 StorageBackend 自管容量 |

其中 `block_size_bytes` 可从 `CacheConfig` 获取（所有层存储相同数据，block 大小一致）。Host/Disk 容量为 0 时表示不启用该层。

#### 层级开关机制

各层可独立开关，通过 `KVCacheConfig` 中的 bool 字段控制：

| 层级 | 开关字段 | 默认值 | 依赖 |
|---|---|---|---|
| L1 Device | `enable_device_cache` | true | 无 |
| L2 Host | `enable_memory_cache` | false | 无 |
| L3 Disk | `enable_disk_cache` | false | **必须 L2 Host 开启** |
| L4 Remote | `enable_remote_cache` | false | 无 |

**依赖规则**：`enable_disk_cache` 必须 `enable_memory_cache = true` 才能开启，因为 Disk I/O 必须经 CPU 内存中转。若 `enable_memory_cache = false` 而 `enable_disk_cache = true`，构造时抛出异常。

**开关对树行为的影响**：

| 操作 | 层关闭时的行为 |
|---|---|
| **match** | 跳过关闭层的数据，不在该层查找匹配 |
| **insert** | 不向关闭层写入数据，不分配该层 block |
| **evict** | 不触发关闭层的淘汰；上层降级时跳过关闭层（如 Host 关闭，Device 淘汰时不降级到 Host，直接释放） |
| **load_back** | 不从关闭层加载数据 |
| **backup** | 不向关闭层备份（如 Remote 关闭，淘汰时不做 write-through） |

**Device 关闭的特殊语义**：当 `enable_device_cache = false` 时，树仅作为元数据 + 低层缓存管理器运行（适用于缓存/存储节点场景）。此时：
- `match` 返回的 `block_indices` 仅包含 Host/Disk 层的索引，无 GPU block
- `insert` 不分配 GPU block，仅写入 Host/Disk/Remote
- 不触发 `evict(DEVICE)`，也无 load_back 到 GPU 的操作
- 树的拓扑结构和前缀匹配逻辑不受影响，仅跳过 L1 层的数据分配和查找

**典型配置组合**：

Device 开启（推理节点）：

| 场景 | Device | Host | Disk | Remote | 说明 |
|---|---|---|---|---|---|
| 纯 GPU | ✅ | ❌ | ❌ | ❌ | 最简单，无缓存复用 |
| GPU + Host | ✅ | ✅ | ❌ | ❌ | 常用配置，Device 淘汰降级到 Host |
| GPU + Host + Disk | ✅ | ✅ | ✅ | ❌ | 大容量本地缓存 |
| GPU + Remote | ✅ | ❌ | ❌ | ✅ | 跨机器缓存复用，无本地 Host/Disk |
| GPU + Host + Remote | ✅ | ✅ | ❌ | ✅ | Host 缓存 + 远端持久化 |
| 全层 | ✅ | ✅ | ✅ | ✅ | 最大缓存覆盖 |

Device 关闭（缓存/存储节点，不参与推理）：

| 场景 | Device | Host | Disk | Remote | 说明 |
|---|---|---|---|---|---|
| 纯 Host | ❌ | ✅ | ❌ | ❌ | 纯内存缓存节点 |
| Host + Disk | ❌ | ✅ | ✅ | ❌ | 内存 + 磁盘缓存节点 |
| Host + Remote | ❌ | ✅ | ❌ | ✅ | 内存缓存 + 远端持久化 |
| Host + Disk + Remote | ❌ | ✅ | ✅ | ✅ | 全层缓存节点（无 GPU） |
| 纯 Remote | ❌ | ❌ | ❌ | ✅ | 纯远端存储节点 |

**淘汰触发**（仅对开启的层生效，采用水位机制而非满才淘汰）：

- **Device 使用率超过水位** → `evict(DEVICE)` → 选 DeviceLeaf → 降级到 Host（同时 write-through 写入 Remote）；若 Host 未开启则跳过 Host 降级（但若 Remote 开启仍执行 write-through 写入 Remote，随后释放 Device）；若 Host 和 Remote 均未开启则仅释放 Device
- **Host 使用率超过水位** → `evict(HOST)` → 选 HostLeaf → 降级到 Disk（同时 write-through 写入 Remote），若 Disk/Remote 均未开启则仅释放
- **Disk 空间不足** → `evict(DISK)` → 直接删除（Remote 数据已在 Device/Host 淘汰时写入，无需再检查）

**水位机制**：各层配置 `watermark_ratio`（如 0.9 = 90%），当使用率超过水位时触发淘汰，提前腾出空间，避免分配时阻塞。水位低于阈值时淘汰停止。

#### load_back 机制

`match()` 内部自动处理 load_back：发现数据在 Host/Disk/Remote 时，内部发起异步加载。调用方通过 `match_result.async_context->waitDone()` 等待加载完成后 GPU block 即就绪。

**load_back 路径选择**（允许跨层加载，优先从最近层级）：

- 若 `host_block` 有效：Host → GPU（直接 DMA）
- 若 `host_block` 无效但 `disk_slot` 有效：Disk → GPU（跨层加载，经 CPU 中转但不在 Host 驻留）
- 若本地无数据：通过 `loadFromRemote()` 拉取到 GPU（跨层加载）

#### 数据迁移与 Connector 的协作

BlockTreeCache 负责决策 + 内部 I/O 调度（match/evict 触发 CopyEngine 和 StorageBackend），CopyEngine 执行 L1-L3 I/O，StorageBackend 执行 L4 I/O。

> 📎 [完整协作图](rtp_llm_block_tree_cache_examples.md#7-数据迁移协作图)

#### TransferDescriptor

`TransferDescriptor` 以 component group 为操作单位，包含 source/target tier、component_group_id、节点列表、源/目标 block 索引和 storage keys。

> 📎 [完整 TransferDescriptor 定义](rtp_llm_block_tree_cache_code_reference.md#transferdescriptor传输描述符)

**ComponentGroup 维度**：`TransferDescriptor` 以 component group 为操作单位。D2H 降级时，收集同 group 的所有 `device_blocks`，通过 Component 的 `MemoryBlockLayerTagSlot` 信息计算 byte offset，打包写入一个 `host_block`。H2D 加载时反向解包。

组件和 CopyEngine/StorageBackend 之间的通信协议：`buildTransfer()` 只返回 TransferDescriptor，不做 I/O。CopyEngine 拿着描述符执行 L1-L3 I/O（依赖 Component 的 `MemoryBlockLayerTagSlot` 计算偏移），StorageBackend 执行 L4 I/O。

#### StorageBackend 接口

所有远端存储通过统一的 `StorageBackend` 接口抽象（替代旧 `RemoteConnector`），提供 `batchRead`、`batchWrite`、`batchExists`、`batchDelete` 四个异步批量操作。

> 📎 [完整 StorageBackend 接口](rtp_llm_block_tree_cache_code_reference.md#storagebackend远端存储接口)

**预置后端**：

| 后端 | 适用场景 | 说明 |
|---|---|---|
| `RemoteConnectorBackend` | 替代现有 RemoteConnector | 复用现有远端传输实现，包装为 StorageBackend 接口 |

#### Prefetch 与 Backup

**prefetchFromRemote** 已重命名为 **`loadFromRemote`**（因为是同步加载而非预取）。

**loadFromRemote**（match 内部触发）：计算请求路径上各节点 token 路径的 hash → 调用 `batch_exists` 查询远端是否有数据 → 对存在数据的节点，通过 RDMA 直通加载到 GPU 或经 CPU 中转到 GPU。同一次 match 内完成，不需要两次 match。

**backupToRemote**（write-through，淘汰时触发）：Device 或 Host 淘汰时，同时启动 `batch_write` 异步写入 Remote。Device→Remote 支持 RDMA 直通；Host→Remote 经 CPU 中转。写入与降级并行进行，不阻塞淘汰流程。不存在 Disk→Remote 路径。

---

### 2.6 EvictionHeap（淘汰堆）

EvictionHeap 是纯数据结构，负责管理淘汰候选节点。每个 ComponentGroup 持有 3 个 EvictionHeap（Device/Host/Disk 各一个），每个堆可独立配置淘汰策略。

#### 淘汰策略枚举

`EvictionPolicy` 支持四种策略：LRU、LFU、FIFO、PRIORITY。

> 📎 [完整 EvictionPolicy 枚举](rtp_llm_block_tree_cache_code_reference.md#evictionpolicy淘汰策略枚举)

各层默认策略与可配置范围：

| 层级 | 默认策略 | 可配置范围 | 理由 |
|---|---|---|---|
| Device | LRU | LRU / LFU / PRIORITY | Device 访问时间局部性强，LRU 收益高 |
| Host | LRU | LRU / LFU / FIFO | Host 容量较大，LRU 仍有效 |
| Disk | FIFO | FIFO / LRU | Disk 访问无时间局部性，FIFO 产生顺序写对 SSD/HDD 友好 |

#### 数据结构

EvictionHeap 由底层容器（`std::priority_queue`）+ 辅助索引（`entry_map_`）组成，提供 `push`、`pop`、`invalidate`、`onAccess` 等操作，采用失效标记策略（Lazy Deletion）。

> 📎 [完整 EvictionHeap 定义](rtp_llm_block_tree_cache_code_reference.md#evictionheap淘汰堆)

#### 失效标记机制（Lazy Deletion）

EvictionHeap 不维护严格的堆删除，而是采用**失效标记**策略：

- `invalidate()` 仅从 `entry_map_` 中移除条目，堆中条目保留
- `pop()` 时检查堆顶是否在 `entry_map_` 中存在，不存在则跳过继续弹下一个
- 优点：避免昂贵的堆重建操作，简化实现
- 代价：堆中可能有少量“尸体”条目，占用少量内存，不影响正确性

#### 策略适配

比较器根据 `EvictionPolicy` 选择排序字段：

| 策略 | 排序字段 | onAccess 行为 |
|---|---|---|
| LRU | `last_access_time` | 更新时间戳为当前时间，重新入堆 |
| LFU | `hit_count` | 命中计数递增 |
| FIFO | `insert_seq` | 无操作（序号在 push 时设定，之后不变） |
| PRIORITY | `priority` | 无操作（由业务逻辑设置） |

---

### 2.7 BlockTreeCache

BlockTreeCache 是淘汰流程的协调者，负责将 BlockTree（树）、ComponentGroup（行为主体 + 堆容器）、CopyEngine（数据搬运）、StorageBackend（远端存储）串联起来，实现异步任务驱动的淘汰流程，并统一管理线程安全。

#### 设计动机

淘汰流程涉及多个数据结构的协同修改：

- **BlockTree**：树拓扑（添加/删除节点）
- **EvictionHeap**：每个 ComponentGroup × 每个 Tier = 多个堆
- **GroupSlot**：节点上的多层数据状态
- **BlockPool**：引用计数管理

这些数据结构在多线程环境下需要协调一致。BlockTreeCache 将协调职责集中化，让 BlockTree 和 EvictionHeap 各自只提供纯操作接口。

#### 三层分离架构

BlockTreeCache（协调者）拥有 BlockTree 和 ComponentGroups，ComponentGroup 拥有 EvictionHeap。三层各司其职。

> 📎 [三层分离架构图](rtp_llm_block_tree_cache_examples.md#8-三层分离架构图)

#### 对外接口

BlockTreeCache 对外暴露三个核心接口（线程安全）：`match` / `insert` / `evict`。完整接口定义、配置参数和构造时校验见 [5.1 节](#51-完整配置)。

#### 异步淘汰三阶段流程

淘汰采用异步任务驱动设计，分为三个阶段：

**Phase 1: 选择候选（同步，持有锁）**

ComponentGroup 的 `driveEviction()` 只负责从 heap 中选出候选节点，产生 `EvictionResult`（包含 node、component_group_id、source/target tier、TransferDescriptor、blocks_to_release、target_block），不执行实际的数据搬运。

> 📎 [完整 EvictionResult 定义](rtp_llm_block_tree_cache_code_reference.md#evictionresult淘汰结果)

选择候选时同时增加引用计数（保护 block 不被再次选中），并从 heap 中临时移除。

**Phase 2: 创建 EvictionTask 并提交到线程池（异步，无锁）**

`EvictionTask` 是纯头文件的状态机结构体（`EvictionTask.h`），包含四种状态（PENDING/RUNNING/COMPLETED/FAILED）和状态转换校验方法。任务执行逻辑在 `BlockTreeCache` 中。

> 📎 [完整 EvictionTask 定义](rtp_llm_block_tree_cache_code_reference.md#evictiontask异步淘汰任务状态机)

任务执行期间，TreeNode 和 GroupSlot 的旧状态保持不变（引用计数已增加，不会被重复选中）。

**Phase 3: 任务完成回调（同步，持有锁）**

任务完成后，协调者执行状态提交：

1. **更新 GroupSlot**：源层级 → 目标层级（如 `device_blocks` 置 invalid，`host_block` 设新值）
2. **释放源层级 block**：引用计数减 1，归零则归还 BlockPool
3. **更新 Tree**：检查节点是否应删除（所有 REUSABLE group 为空），执行祖先链清理
4. **更新 Heap**：parent 可能变为 Leaf 候选入堆，被降级节点可能进入下一层 heap
5. **触发级联淘汰**：高优先级 group 淘汰 → 同节点同介质低优先级 group 级联

#### EvictionTask 的生命周期

PENDING → RUNNING → COMPLETED（Phase 3 状态提交）；或 RUNNING → FAILED（回滚/可重试）。

> 📎 [EvictionTask 生命周期图](rtp_llm_block_tree_cache_examples.md#9-evictiontask-生命周期)

#### 线程安全策略

| 操作 | 锁策略 | 说明 |
|---|---|---|
| `match()` | 持有 mutex | 读 tree，更新 heap 状态 |
| `insert()` | 持有 mutex | 修改 tree，更新 heap |
| Phase 1: 选择候选 | 持有 mutex | 修改 heap，增加引用计数 |
| Phase 2: Task 执行 | 无锁 | 纯数据搬运，不访问 tree/heap |
| Phase 3: 完成回调 | 持有 mutex | 修改 tree、heap、slot、引用计数 |

**锁选择**：单 `std::mutex` 保护 tree 和所有 heaps。理由：match/insert/evict 都涉及树+堆联合操作，细粒度锁复杂度高且收益有限；淘汰操作不频繁（水位触发），不是性能瓶颈。

> 各组件职责边界详见 [1.4 节职责划分表](#14-架构概览)。

---

## 3. Component 系统

### 3.1 ComponentGroup 与 Tree Cache 的对接

#### 分工原则

BlockTreeCache、BlockTree、ComponentGroup、Component 之间是**四层分离**的分工（职责边界详见 [1.4 节职责划分表](#14-架构概览)）。核心约束：树**不知道**各 group 的具体语义，组**不操作**树的拓扑结构，描述符**不持有**任何行为接口，协调者**不直接操作**树或堆的底层细节。

#### ComponentGroup 行为接口

ComponentGroup 共定义 **7 个**行为钩子，全部放在 ComponentGroup 上（而非单个 Component）。完整接口定义参见 [1.5 节 ComponentGroup](#15-核心数据结构)。

**Component 描述符**（纯数据，无行为）：包含 `component_id`、`component_group_id`、`type`（FULL/SWA/LINEAR）、`memory_block_layer_tag_slots`（布局）和 `device_pool_index`。

> 📎 [完整 Component 定义](rtp_llm_block_tree_cache_code_reference.md#component纯描述符)

**`MemoryBlockLayerTagSlot` 的角色**：Component 描述符持有的 `MemoryBlockLayerTagSlot` 列表与现有 `KVCacheMemoryConnector::layerTagSlots()` 语义一致。CopyEngine 在执行 D2H/H2D 拷贝时，通过 ComponentGroup 的 `componentIndices()` 收集所有 Component 的 `memory_block_layer_tag_slots`，计算 memory block 内的 byte offset。Tree 和 ComponentGroup 本身不直接使用此列表。

#### ComponentGroup 类型对比

| | FullComponentGroup | SWAComponentGroup | LinearComponentGroup |
|---|---|---|---|
| **group_type** | FULL | SWA | LINEAR |
| **数据** | 完整 KV Pool 索引 | SWA KV Pool 索引 | 线性注意力状态向量 |
| **引用保护** | 路径锁（路径上所有 block） | 窗口锁（窗口内 block） | 单节点锁 |
| **驱逐优先级** | 最高 | 中 | 最低 |
| **Device Heap 候选** | DeviceLeaf (仅 Leaf 节点) | 任意节点，优先中间 | 任意节点，优先中间 |
| **Host/Disk Heap 候选** | HostLeaf / DiskLeaf | 任意节点，优先中间 | 任意节点，优先中间 |
| **Match 验证** | device/host/disk 任一有效 | 累计连续窗口 ≥ window_size | 当前节点有 LINEAR 数据 |

---

### 3.2 FullComponentGroup

FullComponentGroup 是基础组件组（`CacheGroupType::FULL`），管理完整 KV Cache 索引。**任何 BlockTreeCache 实例都必须包含 FullComponentGroup**。

#### 路径锁

FullComponentGroup 采用路径锁：从匹配节点到 root 的路径上，通过 BlockPool 引用计数保护所有非 evicted 节点的 block。保证请求正在使用的前缀路径不会被驱逐。

#### 淘汰候选与级联

FullComponentGroup 管理三层独立驱动的 heap：**Device heap**（DeviceLeaf 候选）、**Host heap**（HostLeaf 候选）、**Disk heap**（DiskLeaf 候选）。每层堆可独立触发淘汰，Leaf 判定以 GroupSlot 为单位：

- **DeviceLeaf**：没有子节点在该 group 拥有 device value 的节点。当子节点的 device value 被淘汰后，中间节点自动变为 DeviceLeaf
- **HostLeaf**：已驱逐（该 group `device_blocks` 全 invalid）且有 host 数据的节点中，没有子节点在该 group 拥有 host_value 的节点
- **DiskLeaf**：已驱逐（`device_blocks` 全 invalid 且 `host_block` 无效）且有 disk 数据的节点中，没有子节点在该 group 拥有 disk_value 的节点

淘汰时触发级联：DeviceLeaf 淘汰后 parent 可能变为新 DeviceLeaf（同层），被降级的节点可能变为 HostLeaf（跨层）。详见 2.3 节。

**淘汰执行**：`evictFromTier(node, slot, DEVICE)` 释放同 group 所有 `device_blocks`，对 REUSABLE group 执行 D2H 降级；`evictFromTier(node, slot, HOST)` 释放共享的 `host_block`。`buildTransfer(D2H)` 通过 `componentIndices()` 收集所有 Component 的 `MemoryBlockLayerTagSlot` 生成正确的打包拷贝描述。

#### 资源隔离

FullComponentGroup 和 SWAComponentGroup 使用各自独立的 KV Pool 分配器（对应不同的 `BlockPool`），`evictFromTier()` 中各自只释放自己 group 内 Component 管理的 KV，互不干扰。

#### finalizeMatchResult

FullComponentGroup 的 `finalizeMatchResult` 计算 `load_back_blocks`：统计 match 过程中需要从 Host/Disk/Remote 加载到 GPU 的 block 数。match 内部自动发起异步加载，调用方通过 `async_context` 等待完成。

#### transfers（LOAD_BACK）

FullComponentGroup 在 `buildTransfer(LOAD_BACK)` 阶段：从 `best_match_node` 向上收集所有 evicted 节点的 `host_block`/`disk_slot`，拼接为一个 `TransferDescriptor`。CopyEngine 据此执行数据搬运。

---

### 3.3 SWAComponentGroup

SWAComponentGroup（`CacheGroupType::SWA`）管理滑动窗口注意力的专用 KV Pool。同一节点上 FullComponentGroup 和 SWAComponentGroup 的数据位于不同的显存 pool。

#### 窗口验证器

SWAComponentGroup 的验证器使用 `connected_to_root` 布尔标志：从 root 到当前节点的路径上所有节点均有 SWA 数据时，无需逐 block 累计；一旦遇到缺失，归零重算。

- **初始状态**：`connected_to_root = true`，`len = 0`
- 遇到有 SWA 数据的节点：若 `connected_to_root` 为 true，保持；若为 false，`len += block_tokens`
- 遇到 SWA 数据缺失的节点（被淘汰）：`connected_to_root = false`，`len = 0`（归零重算）
- 多层存储模式下，若节点 `host_block`/`disk_slot` 存在，验证器**不归零**，匹配可穿过该节点继续累计

#### SWA 独立淘汰

SWAComponentGroup 的三层堆均可独立驱动淘汰（`driveEviction(n, tier)`）。当 SWA group 的 device pool 空间不足时，直接调用 SWAComponentGroup 的 `driveEviction(n, DEVICE)`，不需要经过 FullComponentGroup。淘汰后节点 SWA group 的 `device_blocks` 全 invalid 但 Full group 的 `device_blocks` 仍在。Match 时 SWA 验证器遇到该节点自然归零重算；Insert 经过时若在窗口内则回填 SWA 数据。不影响 Full 数据的正常使用。

> **SWA 淘汰不会导致节点从树中删除**——节点仍在树中，Full group 数据不受影响。只有当所有 REUSABLE group 的数据都被淘汰时，节点才会从树中移除。**优化**：如果 SWA 淘汰的是叶子节点（`children.empty()`）且该节点没有其他 REUSABLE group 的数据，可以直接删除节点，减少树的大小。

#### 淘汰候选规则（Any-node）

SWAComponentGroup 的堆候选与 FullComponentGroup 不同——任意有数据的节点均可入堆（不要求是 Leaf），优先淘汰中间节点。这是因为 SWA 数据只需窗口范围内连续即可，中间节点数据淘汰后，窗口内仅需补算。

#### 窗口锁

SWA 锁从 match 节点向上累计 ≥ `sliding_window_size` 时停止。释放时沿同路径回退相同节点数。支持提前释放（`releaseWindowLock`）：decode 位置超过窗口范围后，SWA 锁可提前释放，同时级联释放 LINEAR 锁。

#### LRU 刷新策略

| 事件 | 刷新行为 |
|---|---|
| **WALKDOWN**（match 遍历经过） | 不刷新（避免窗口外祖先被提升为 MRU） |
| **MATCH_END** | 只刷新窗口内祖先（向上累计不超过 `sliding_window_size + block_size`） |
| **INSERT_END** | 同 MATCH_END |

---

### 3.4 LinearComponentGroup

LinearComponentGroup（`CacheGroupType::LINEAR`）管理线性注意力/SSM 模型的隐藏状态。与 Full/SWA 不同，LINEAR 状态是**点状态**（只在特定节点保存快照），不需要 Copy-on-Write——match 命中时直接使用对应节点的状态继续推理。

#### LINEAR 状态生命周期

| 阶段 | 行为 |
|---|---|
| Insert | 上层指定哪些节点需要分配 LINEAR 状态 block，缓存层只负责登记 |
| Match | 命中有 LINEAR 状态的节点时，请求直接使用该状态继续推理（无需 CoW） |
| 淘汰 | 级联淘汰：Full 淘汰 → 级联 SWA + LINEAR；SWA 淘汰 → 级联 LINEAR；LINEAR 也可独立驱动淘汰 |
| No-Split | 不存在 Split，LINEAR 数据一旦创建永远不会被清空 |

#### 淘汰候选规则（Any-node）

LinearComponentGroup 的堆候选与 SWAComponentGroup 相同——任意有数据的节点均可入堆，优先淘汰中间节点。三层堆均可独立驱动（`driveEviction(n, tier)`）。

---

### 3.5 BlockTreeCache 与 BlockPool 的协同关系

BlockTreeCache **直接感知并持有** `BlockPool` 引用（通过 ComponentGroup 的 component_indices 引用 Component 描述符获取 pool 映射），但对 BlockPool 的使用是**单向且受限的**——只做 `free` 和查询，**不调用 `alloc`**（LinearComponentGroup 除外）。BlockTreeCache 不引入额外的 Allocator 层，直接用 `BlockPool` 作为分配器。

BlockTreeCache 通过 ComponentGroup 的 `componentIndices()` 知道 group 拓扑。`GroupSlot` 的 `device_blocks` 来自各独立 Device BlockPool，Host block 分配使用合并后的 `host_block_size`（同 group 所有 `MemoryBlockLayerTagSlot` 的 `stride_bytes` 之和）。

#### 依赖关系

Scheduler 通过 BlockPool 分配 KV slot，通过 BlockTreeCache match/insert；BlockTreeCache 只做 free 和查询，不做 alloc；ComponentGroup 在 evictFromTier 时释放对应资源。

> 📎 [完整 BlockPool 依赖关系图](rtp_llm_block_tree_cache_examples.md#10-blockpool-依赖关系图)

#### 为什么 BlockTreeCache 不调用 alloc

这是**关注点分离**的结果：

- **alloc 是“请求需要新资源”** — 属于调度层（batch 大小、chunked prefill 分块等约束）
- **free 是“资源生命周期结束”** — 属于缓存管理（重复数据释放、驱逐归还）

#### 完整职责分工

| 角色 | alloc | free | 查询 |
|---|---|---|---|
| **上层引擎** | 通过 BlockPool 分配新 slot | — | 准入控制 |
| **BlockTreeCache** | — | 释放重复/未对齐 slot | load_back 空间判断 |
| **ComponentGroup** | — | evictFromTier 时释放 KV/SWA/LINEAR | — |
| **BlockPool** | 执行分配 | 执行释放 | 返回空闲量 |

---

## 4. 与现有系统的集成

### 4.1 对 KVCacheAllocator 的影响

`KVCacheAllocator` 内部替换 `SharedBlockCachePtr` 为 `BlockTreeCachePtr`，公开接口不变。match/insert/evict 的调用点分别替换为 `block_tree_cache_->match()`/`insert()`/`evict()`。

> 📎 [完整 KVCacheAllocator 修改](rtp_llm_block_tree_cache_code_reference.md#kvcacheallocator-修改)

---

### 4.2 删除 KVCacheMemoryConnector，整合到 BlockTreeCache

`KVCacheMemoryConnector`（3500+ 行）将被**完全删除**，其职责拆分并移入 `BlockTreeCache` 目录下。

> 📎 [完整目录结构和删除前后对比](rtp_llm_block_tree_cache_examples.md#11-文件目录结构)

**对 Connector 体系的影响**：

- **KVCacheMemoryConnector**：删除，逻辑移入 BlockTreeCache 目录
- **RemoteConnector**：删除，由 `StorageBackend` 替代
- **P2PConnector**：保留，暂不做测试
- **KVCacheConnectorCoordinator**：精简，只管理 P2PConnector

---

### 4.3 移除 Namespace 机制与独立驱逐组适配

#### 移除 Namespace 机制

当前 `SharedBlockCache` 的 Namespace 机制（`kDefaultNamespace` / `kGpuLogicalNamespace` / `kGpuCpCanonicalNamespace`）是为 Context Parallelism (CP) 分片场景设计的：同一个 `cache_key` 在树中维护两种不同的 parent 依赖关系（rank-local 视图 vs CP canonical 视图），驱逐时据此决定释放哪个视角的 block。

**BlockTreeCache 不需要 Namespace**，原因如下：

1. **CP 分片由 `CPSlotMapper` 在树外部处理**：调用方先通过 `CPSlotMapper` 算出 canonical cache_keys，再传给树。树只看到一组 key，不需要维护两套 parent 关系
2. **树中每个 `cache_key` 只有一条路径**：不存在"同一个 key 在两个 namespace 下有不同 parent"的情况，树的 children map 保持简单的 `CacheKeyType → TreeNode*` 映射
3. **驱逐时的 CP 归属通过 `CPSlotMapper` 反查**：不再需要 `evicted_namespaces` 信息，`CPSlotMapper` 可根据 block index 反查 CP 归属

这一简化消除了 `NamespacedKey`、`aliases_by_cache_key_`、`touchTreeAliasesLocked()`、`refreshAllTreeAliasesLocked()` 等一系列复杂逻辑，树结构更清晰。

#### 独立驱逐组适配

当前 `SharedBlockCache` 支持 `independent_group_eviction`，允许按 group_id 独立驱逐。在 BlockTreeCache 中：

- 每个 `CacheGroupType` 对应一个 ComponentGroup
- 每个 ComponentGroup 持有独立的三层淘汰堆（Device/Host/Disk）
- `CacheEvictPolicy::INDEPENDENT` 的 group 使用独立的 `driveEviction()` 入口

---

### 4.4 替换策略与迁移路径

#### 分阶段迁移

| 阶段 | 目标 | 替换的组件 | 风险 |
|---|---|---|---|
| **Phase 1** | 实现 BlockTreeCache + BlockTreeCache + FullComponentGroup + EvictionHeap | `SharedBlockCache` | 中 — 需要完整回归所有 cache 相关测试 |
| **Phase 2** | 添加 SWAComponent + LinearComponent | SWA/LINEAR 的特殊处理逻辑 | 中 — SWA window lock 需要充分测试 |
| **Phase 3** | 删除 KVCacheMemoryConnector，移入 BlockTreeCache 目录 | `KVCacheMemoryConnector` + `PrefixTreeMemoryBlockCache` + `MemoryDiskBlockCache` | 高 — 涉及 3500+ 行代码重构 |
| **Phase 4** | 添加 load_back 调度 + TransferDescriptor | 当前的隐式 Host→GPU 加载 | 中 — 需要与异步 I/O 框架集成 |
| **Phase 5** | 集成 StorageBackend，删除旧 RemoteConnector | `RemoteConnector` | 中 — 包装现有远端传输为 StorageBackend 接口 |

#### Phase 1 详细计划

**目标**：用 `BlockTreeCache` + `BlockTreeCache` + `FullComponentGroup` + `EvictionHeap` 替换 `SharedBlockCache`

> 📎 [Phase 1 详细文件清单](rtp_llm_block_tree_cache_examples.md#11-文件目录结构)

#### 兼容性保证

- Phase 1 期间，`SharedBlockCache` 保留但不再被 `KVCacheAllocator` 直接使用
- `BlockTreeCache` 实现 `SharedBlockCache` 的关键接口子集（`match` / `put` / `selectAndEvict`），通过适配器模式接入
- 所有现有单元测试应继续通过

---

## 5. 补充信息

### 5.1 完整配置

BlockTreeCache 引入 `BlockTreeCacheConfig` 配置结构体（从 `KVCacheConfig` 提取 L2/L3 层相关字段）和工厂函数 `createBlockTreeCache()`。BlockTreeCache 构造函数接受已构建的组件，工厂函数负责从 `CacheConfig` + `KVCacheConfig` 推导并组装。

> 📎 [完整配置结构体、构造函数和工厂函数](rtp_llm_block_tree_cache_code_reference.md#6-配置与-api)

**从 CacheConfig 读取**：

| 配置项 | 字段 | 说明 |
|---|---|---|
| block token 数 | `seq_size_per_block` | 每 block 的 token 数 |
| block 字节大小 | `block_size_bytes` | 用于计算 Host/Disk block 数 |
| 组件类型 | `groups[].policy.group_type` | FULL / SWA / LINEAR |
| GPU block 数 | `block_num` / `groups[gid].block_num` | Device 容量 |
| 滑动窗口 | `specForGroup(swa_gid)->sliding_window_size` | SWA 窗口大小 |
| 独立驱逐组 | `groups[].policy.evict_policy` | INDEPENDENT 标记 |
| **复用策略** | `groups[].policy.reuse_policy` | **NON_REUSABLE group 不入 BlockTreeCache，由上层 allocator 直接管理** |
| **Group Tag** | `tagForGroup(gid)` | **推导 component_group_id 映射（同 tag 的 layer 属于同一 component group）** |
| **Layer → Group 映射** | `layerGroupIdsSnapshot()` | **遍历所有 layer 的 group_ids，生成 `MemoryBlockLayerTagSlot` 列表** |

**从 KVCacheConfig 读取**：

| 配置项 | 字段 | 说明 |
|---|---|---|
| Device 开关 | `enable_device_cache` | L1 层开关（默认 true），关闭时树仅作为低层缓存管理器 |
| Host 开关 | `enable_memory_cache` | L2 层开关（默认 false） |
| Host 缓存大小 | `memory_cache_size_mb` | 用于计算 Host block 数：`size_mb * 1MB / block_size_bytes` |
| Disk 开关 | `enable_disk_cache` | L3 层开关（默认 false），**必须 Host 开关开启** |
| Disk 缓存大小 | `memory_cache_disk_size_mb` | 用于计算 Disk block 数：`size_mb * 1MB / block_size_bytes` |
| Remote 开关 | `enable_remote_cache` | L4 层开关（默认 false） |
| 水位比例 | `watermark_ratio` | 各层淘汰触发水位（如 0.9 = 使用率超过 90% 时触发淘汰） |
| Device 淘汰策略 | `device_eviction_policy` | EvictionPolicy 枚举，默认 LRU |
| Host 淘汰策略 | `host_eviction_policy` | EvictionPolicy 枚举，默认 LRU |
| Disk 淘汰策略 | `disk_eviction_policy` | EvictionPolicy 枚举，默认 FIFO |
| 淘汰线程数 | `eviction_thread_pool_size` | BlockTreeCache 线程池大小，默认 2 |

**通过工厂函数参数传入**：

| 参数 | 类型 | 说明 |
|---|---|---|
| `allocator` | `shared_ptr<KVCacheAllocator>` | 现有分配器（用于推导 device pool 信息） |
| `swa_configs` | `SWAGroupConfig`（`unordered_map<int, int>`） | 每个 SWA group 的滑动窗口大小（token 数） |
| `storage_backend` | `shared_ptr<StorageBackend>` | 远端存储后端（null = 不启用） |
| `broadcast_manager` | `shared_ptr<BroadcastManager>` | TP 广播管理器（可选） |

#### 对外接口

BlockTreeCache 对外暴露 `match`、`insert`、`evict` 三个核心接口（线程安全），外加 `isEvictable`、`getStats`、`waitForPendingTasks` 查询和控制接口。

> 📎 [完整对外接口定义](rtp_llm_block_tree_cache_code_reference.md#blocktreecache-对外接口)

**构造时校验**：

- 若 `enable_disk_cache = true` 且 `enable_memory_cache = false`，抛出异常（Disk 依赖 Host 中转）
- 若 `enable_memory_cache = true` 且 `memory_cache_size_mb = 0`，抛出异常（Host 开启但未配置容量）
- 若 `enable_disk_cache = true` 且 `memory_cache_disk_size_mb = 0`，抛出异常（Disk 开启但未配置容量）

**接口设计说明**：

- **`match`** 内部自动处理 load_back 和 Remote prefetch：发现数据在 Host/Disk 时，内部构建 TransferDescriptor 并发起异步加载；发现本地无数据但 Remote 有时，内部发起 prefetch。仅查找开启的层。调用方通过 `BlockTreeMatchResult` 中的异步上下文等待加载完成
- **`evict`** 内部自动处理 write-through 写入 Remote：Device/Host 淘汰时同时异步写入 Remote。仅对开启的层触发淘汰。调用方不需要感知 Remote 交互
- **`buildLoadBackTransfer` / `loadFromRemote` / `backupToRemote`** 均为内部实现，不透出对外接口

---

### 5.2 典型请求链路

一个完整请求在 BlockTreeCache 中的生命周期：请求到达 → 前缀匹配 + 数据加载 → 推理 → 插入 → 释放 → 驱逐（按需，异步三阶段）。

> 📎 [完整请求链路](rtp_llm_block_tree_cache_examples.md#12-典型请求链路)
