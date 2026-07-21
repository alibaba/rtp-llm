# BlockTreeCache 示例与详细图示

> 本文件是 [BlockTreeCache 设计文档](rtp_llm_block_tree_cache_design.md) 的配套参考。
> 包含 DSV4 模型具体示例、详细流程图、级联淘汰步骤和目录结构，供深入理解设计细节时查阅。

---

📚 **BlockTreeCache 设计文档系列**
| [设计文档（主）](rtp_llm_block_tree_cache_design.md) | [C++ 代码参考](rtp_llm_block_tree_cache_code_reference.md) | **示例与详细图示** |

---

## 目录

1. [DSV4 数据结构示例](#1-dsv4-数据结构示例)
2. [数据流示例](#2-数据流示例)
3. [结构引用关系总图](#3-结构引用关系总图)
4. [普通模型 vs DSV4 退化关系](#4-普通模型-vs-dsv4-退化关系)
5. [淘汰级联示例](#5-淘汰级联示例)
6. [驱逐调度流程](#6-驱逐调度流程)
7. [数据迁移协作图](#7-数据迁移协作图)
8. [三层分离架构图](#8-三层分离架构图)
9. [EvictionTask 生命周期](#9-evictiontask-生命周期)
10. [BlockPool 依赖关系图](#10-blockpool-依赖关系图)
11. [文件目录结构](#11-文件目录结构)
12. [典型请求链路](#12-典型请求链路)

---

## 1. DSV4 数据结构示例

> **注意**：以下示例以 DSV4 模型为例（包含 csa_kv / hca_kv / indexer_kv 等具体 group tag），这些 group tag 是 DSV4 特有的。对于普通模型（LLaMA / Qwen 等纯 MHA/MLA），通常只有一个 component group（id=0），包含一个 Full component，GroupSlot 退化为 1:1:1 语义。

### 1.1 静态元数据（构造时建立，运行期不变）

BlockTreeCache 构造时从 CacheConfig 推导出两组静态元数据：`component_groups_`（ComponentGroup 主动实体，持有行为接口 + 三层堆）和 `components_`（纯描述性对象，提供 layout 信息）。多个 Component 可以属于同一个 ComponentGroup。

```
BlockTree
|
+-- component_groups_: vector<ComponentGroup*>   (主动实体，持有行为接口 + 三层淘汰堆)
|     |
|     +-- [0] FullComponentGroup { id=0, group_type=FULL, REUSABLE,
|     |         component_indices=[0,1,2],
|     |         host_block_size = sum(91 slots * stride),
|     |         device_heap (Leaf-based),
|     |         host_heap   (Leaf-based),
|     |         disk_heap   (Leaf-based) }
|     |
|     +-- [1] SWAComponentGroup { id=1, group_type=SWA,
|               component_indices=[3,4,5],
|               host_block_size = sum(121 slots * stride),
|               device_heap (Any-node, prefer middle),
|               host_heap   (Any-node, prefer middle),
|               disk_heap   (Any-node, prefer middle) }
|
+-- components_: vector<Component>   (纯描述，无行为，供 CopyEngine 使用)
      |
      +-- [0] { csa_kv,  type=FULL,    group_id=0, pool_idx=0, slots=[L1:csa, L3:csa, ...] }
      +-- [1] { hca_kv,  type=FULL,    group_id=0, pool_idx=1, slots=[L2:hca, L4:hca, ...] }
      +-- [2] { idx_kv,  type=FULL,    group_id=0, pool_idx=2, slots=[L1:idx, L3:idx, ...] }
      |
      +-- [3] { idx_st,  type=SWA,     group_id=1, pool_idx=3, slots=[L1:idx_st, ...] }
      +-- [4] { csa_st,  type=SWA,     group_id=1, pool_idx=4, slots=[L1:csa_st, ...] }
      +-- [5] { swa_kv,  type=SWA,     group_id=1, pool_idx=5, slots=[L0:swa, ...] }
```

`ComponentGroup` 拥有行为接口（match/insert/evict/transfer）+ 三层淘汰堆。`Component` 只提供 `MemoryBlockLayerTagSlot` layout 信息供 `CopyEngine` 计算偏移。

### 1.2 运行时树节点数据

每个 TreeNode 持有 `group_slots[]`，长度等于 `component_groups_.size()`。每个 GroupSlot 对应一个 component group 在该节点上的数据位置：

```
TreeNode  (cache_key = 0xABCD1234)
|
+-- group_slots: vector<GroupSlot>    // len = component_groups_.size()
      |
      +-- [0] compressed group (REUSABLE)
      |     |
      |     +-- device_blocks = [42, 107, 203]
      |     |     |                |    |    +-- idx_kv pool, block 203
      |     |     |                |    +------- hca_kv pool, block 107
      |     |     |                +------------ csa_kv pool, block 42
      |     |     +-- len = number of components in this group (= independent Device BlockPools)
      |     |         each device_block from a different Device BlockPool
      |     |
      |     +-- host_block = 15         // 1 packed block
      |     |     internal layout (dictated by MemoryBlockLayerTagSlots):
      |     |     [L1:csa|L1:idx|L2:hca|L3:csa|L3:idx|L4:hca|...]
      |     |      <-str-> <-str-> <-str->
      |     |
      |     +-- disk_slot = 8
      |
      +-- [1] state_swa group
      |     +-- device_blocks = [5, 12, 78]
      |     +-- host_block = 23
      |     +-- disk_slot = -1  (invalid)
```

**关键语义**：
- `device_blocks` 中每个元素来自一个**独立的 Device BlockPool**（如 DSV4 有 7 个独立 pool）
- `host_block` 是一个**打包 block**——内部按 `MemoryBlockLayerTagSlot` 顺序拼接了该 group 所有层的数据

---

## 2. 数据流示例

### Device-to-Host 降级数据流（DSV4）

淘汰触发时，CopyEngine 依赖 Component 的 `MemoryBlockLayerTagSlot` 计算 byte offset，将多个分散的 device block 打包写入一个 host block：

```
evict(DEVICE) on node N, target group = 0 (compressed, REUSABLE)

  N.group_slots[0].device_blocks = [42, 107, 203]
  (3 GPU blocks from 3 independent pools)

  Collect MemoryBlockLayerTagSlots from all components in group 0:

  Comp[0] (csa_kv):       Comp[1] (hca_kv):       Comp[2] (idx_kv):
  +-----------------+     +-----------------+     +-----------------+
  | L1:csa  str=584 |     | L2:hca  str=584 |     | L1:idx  str=132 |
  | L3:csa  str=584 |     | L4:hca  str=584 |     | L3:idx  str=132 |
  | ...             |     | ...             |     | ...             |
  +-----------------+     +-----------------+     +-----------------+

  Merge & sort by layer_id:

  +-----------------------------------------------------------+
  | Host Block 15 (packed, 91 slots total)                    |
  |                                                           |
  | off=0:    L1:csa  (584B) <-- device_block 42              |
  | off=584:  L1:idx  (132B) <-- device_block 203             |
  | off=716:  L2:hca  (584B) <-- device_block 107             |
  | off=1300: L3:csa  (584B) <-- device_block 42              |
  | off=1884: L3:idx  (132B) <-- device_block 203             |
  | off=2016: L4:hca  (584B) <-- device_block 107             |
  | ...                                                       |
  | total = sum(91 * stride) = host_block_size                |
  +-----------------------------------------------------------+

  After demotion:
    N.group_slots[0].device_blocks = [NULL, NULL, NULL]
    N.group_slots[0].host_block    = 15
```

这与现有 `KVCacheMemoryConnector::prepareCopyBuffers()` 的逻辑一致：遍历 slots，累加 `stride_bytes` 作为 memory block 内的偏移，从各 GPU buffer 拷贝到对应位置。

---

## 3. 结构引用关系总图

```
+=========================================================================+
|                          BlockTreeCache                                |
|                                                                         |
|  component_groups_[] (主动实体，持有行为+堆)                            |
|  +-------------------------------------------------------------------+  |
|  | [0] FullComponentGroup    indices=[0,1,2]                          |  |
|  |     .createMatchValidator()  .driveEviction()                     |  |
|  |     .device_heap (Leaf)  .host_heap (Leaf)  .disk_heap (Leaf)     |  |
|  |                                                                   |  |
|  | [1] SWAComponentGroup     indices=[3,4,5]                         |  |
|  |     .createMatchValidator()  .driveEviction()                     |  |
|  |     .device_heap (Any)   .host_heap (Any)   .disk_heap (Any)      |  |
|  +-------------------------------------------------------------------+  |
|         |                                                               |
|         | component_indices 引用                                         |
|         v                                                               |
|  components_[] (纯描述，无行为，供 CopyEngine 使用)                     |
|  +-------------------------------------------------------------------+  |
|  | [0] csa_kv  type=FULL    group=0  pool=0  slots=[L1:csa, L3:csa]  |  |
|  | [1] hca_kv  type=FULL    group=0  pool=1  slots=[L2:hca, L4:hca]  |  |
|  | [2] idx_kv  type=FULL    group=0  pool=2  slots=[L1:idx, L3:idx]  |  |
|  | [3] idx_st  type=SWA     group=1  pool=3  slots=[L1:idx_st]       |  |
|  | [4] csa_st  type=SWA     group=1  pool=4  slots=[L1:csa_st]       |  |
|  | [5] swa_kv  type=SWA     group=1  pool=5  slots=[L0:swa]          |  |
|  +-------------------------------------------------------------------+  |
|         |                                                               |
|         | GroupSlot 引用                                                 |
|         v                                                               |
|  TreeNode Tree                                                          |
|  +-------------------------------------------------------------------+  |
|  |  root                                                             |  |
|  |   +-- A .group_slots[0..2]                                        |  |
|  |   |    +-- B .group_slots[0..2]                                   |  |
|  |   |    |    +-- C .group_slots[0..2]   (tree leaf)                |  |
|  |   |    +-- D .group_slots[0..2]                                   |  |
|  |   +-- E .group_slots[0..2]                                        |  |
|  |                                                                   |  |
|  |  GroupSlot[g]:                                                    |  |
|  |    .device_blocks[] --> independent Device BlockPools             |  |
|  |    .host_block ------> host_pool_ (BlockPool-Host, merged packed block) |  |
|  |    .disk_slot -------> DiskBlockPool                              |  |
|  +-------------------------------------------------------------------+  |
|         |                                                               |
|         | evict(DEVICE) triggered by pool pressure                       |
|         v                                                               |
|  驱逐调度流程 (以 Full group Device 淘汰为例)                           |
|  +-------------------------------------------------------------------+  |
|  |  1. BlockTreeCache 确定目标 group:                            |  |
|  |     pool pressure --> component_groups_[0] (Full)                 |  |
|  |                                                                   |  |
|  |  2. Full group 选择候选:                                          |  |
|  |     component_groups_[0]->driveEviction(DEVICE)                |  |
|  |       --> pop from device_heap --> EvictionResult                |  |
|  |                                                                   |  |
|  |  3. BlockTreeCache 创建 EvictionTask 提交线程池:          |  |
|  |     CopyEngine 执行 D2H demotion (异步)                       |  |
|  |                                                                   |  |
|  |  4. 任务完成回调，BlockTreeCache 触发级联 + 堆维护:  |  |
|  |     parent 可能变为新 DeviceLeaf --> 入堆                        |  |
|  |     被降级节点可能进入 host_heap                                 |  |
|  +-------------------------------------------------------------------+  |
+=========================================================================+
```

---

## 4. 普通模型 vs DSV4 退化关系

普通模型（LLaMA / Qwen 等纯 MHA/MLA）只有一个 component group，GroupSlot 退化为简化的 1:1:1 语义：

```
Normal model (LLaMA/Qwen):

  component_groups_ = [ { id=0, REUSABLE,
                       components=[FullComp], host_block_size=block_size } ]

  TreeNode.group_slots = [
    [0] GroupSlot {
      device_blocks = [42]       // only 1 device block (1 pool)
      host_block    = 15         // 1:1 mapping (no packing)
      disk_slot     = 8
    }
  ]

  --> GroupSlot degenerates to the original ComponentSlot semantics


DSV4 (hybrid attention):

  component_groups_ = [ compressed(3 comps), state_swa(3 comps) ]

  TreeNode.group_slots = [
    [0] GroupSlot { device_blocks=[42,107,203], host_block=15, disk=8  }  // 3->1 pack
    [1] GroupSlot { device_blocks=[5,12,78],    host_block=23, disk=-1 }  // 3->1 pack
  ]

  --> multiple device blocks merge into 1 host block
```

---

## 5. 淘汰级联示例

### DeviceLeaf 级联示例

以某个 group 为例，说明 DeviceLeaf 淘汰后 parent 自动补位为新 DeviceLeaf 的过程：

```
树: root → A → B → C (C 是树叶子)

初始: A.group_slots[g].device_blocks=✅ B.group_slots[g].device_blocks=✅ C.group_slots[g].device_blocks=✅
Full heap = {C}  ← 只有 C 没有带 device value 的子节点

淘汰 C (降级到 Host):
  C.group_slots[g].device_blocks=全 invalid
  → 同 group 的 device_blocks 打包写入 1 个 host_block
  → 检查 parent B
  B 没有该 group 带 device value 的子节点 → B 变为 DeviceLeaf
Full heap = {B}

淘汰 B:
  B.group_slots[g].device_blocks=全 invalid → 打包写入 host_block
  → 检查 parent A
  A 变为 DeviceLeaf
Full heap = {A}
```

### 跨 ComponentGroup 级联驱逐示例

节点 N 初始状态:
```
group_slots[0] (Full, REUSABLE): device=✅ host=✅ disk=✅
group_slots[1] (SWA,  REUSABLE): device=✅ host=✅ disk=❌
```

**场景 A: Full group Device 淘汰** (DeviceLeaf evict at DEVICE)
```
  1. Full group:   N.group_slots[0].device_blocks 降级到 Host
  2. 级联 SWA[1]: N.group_slots[1].device_blocks 降级到 Host (同节点同介质)

  结果:
  group_slots[0]: device=❌ host=✅(新) disk=✅
  group_slots[1]: device=❌ host=✅(新) disk=❌
```

**场景 B: Full group Host 淘汰** (HostLeaf evict at HOST)
```
  1. Full group:   N.group_slots[0].host_block 降级到 Disk
  2. 级联 SWA[1]: N.group_slots[1].host_block 降级到 Disk (同节点同介质)

  结果:
  group_slots[0]: device=❌ host=❌ disk=✅(新)
  group_slots[1]: device=❌ host=❌ disk=✅(新)
```

**场景 C: SWA[1] group 独立 Device 淘汰** (中间节点)
```
  1. SWA[1] group: N.group_slots[1].device_blocks 降级到 Host
  2. Full group:   不受影响 (低优先级不向上级联)

  结果:
  group_slots[0]: device=✅ host=✅ disk=✅  (不变)
  group_slots[1]: device=❌ host=✅(新) disk=❌
```

---

## 6. 驱逐调度流程

完整的驱逐调度链路（异步三阶段）：

```
1. 触发: BlockPool 报告某层空间不足 (如 Device pool[0] 使用率 > watermark)

2. 定位: BlockTreeCache 根据 pool → ComponentGroup 映射确定目标 group
   (如 pool[0] 属于 Full group → target = component_groups_[0])

3. Phase 1 选择候选 (同步，持有锁):
   BlockTreeCache 调用 target->driveEviction(tier)
   ComponentGroup 从自己的 tier-heap 中弹出最冷候选节点
   候选节点必须满足引用计数 == 1（仅 BlockTreeCache 自身引用）
   增加引用计数（保护 block 不被重复选中）
   返回 EvictionResult（包含 TransferDescriptor）

4. Phase 2 异步执行 (无锁):
   BlockTreeCache 创建 EvictionTask，提交到线程池
   - CopyEngine 执行 D2H 降级 / H2Disk 降级
   - 同时 StorageBackend write-through 写入 Remote（如开启）

5. Phase 3 完成回调 (同步，持有锁):
   BlockTreeCache 更新 GroupSlot 状态
   释放源层级 block（引用计数减 1）
   对同节点的低优先级 group 触发同介质级联
   for each lower_group in priority_order_below(target):
       if node->group_slots[lower_group.id] has data at tier:
           lower_group->evictFromTier(node, slot, tier)

6. 堆维护:
   - 同层: parent 可能变为新 Leaf (Full group) 或新候选 (SWA/LINEAR group) → 入堆
   - 跨层: 被降级节点可能进入下一层 heap
   - 被级联淘汰的节点从当前层 heap 移除，可能进入下一层 heap
   - 节点删除: 所有 REUSABLE group 为空 → 从树中移除 + 祖先链清理

7. 重复: 若 num_blocks 未满足，继续从步骤 3 开始循环
```

---

## 7. 数据迁移协作图

```
BlockTreeCache (决策 + 内部 I/O 调度)
  │
  ├── match() → 查找匹配 + 内部 load_back/prefetch
  │   ├── 命中 Host/Disk 数据 → 内部生成 TransferDescriptor 并发起加载
  │   └── 本地无数据但 Remote 有 → 内部发起 loadFromRemote
  │
  ├── evict(DEVICE) → write-back 降级到 Host + write-through 写入 Remote
  │   ├── CopyEngine 通过 MemoryBlockLayerTagSlot 将 device_blocks 打包写入 host_block
  │   └── StorageBackend: device_blocks → Remote（write-through，可 RDMA 直通）
  │
  ├── evict(HOST) → write-back 降级到 Disk + write-through 写入 Remote
  │   ├── CopyEngine: host_block → disk_slot（降级）
  │   └── StorageBackend: host_block → Remote（write-through）
  │
  └── evict(DISK) → 直接删除
      └── Remote 数据已通过 write-through 写入，无需再检查

CopyEngine (执行 L1-L3 I/O，从 KVCacheMemoryConnector 移入)
  │
  └── asyncExecuteTransfer(TransferDescriptor)
      ├── D2H: 依赖 Component 的 MemoryBlockLayerTagSlot 计算 memory block 内 byte offset
      │   └── 遍历 slots，累加 stride_bytes，将多个 device block 打包写入 1 个 host block
      ├── H2D: 从 1 个 host block 解包到多个 device block
      └── H2D/D2D Disk: memcpy / disk I/O

StorageBackend (执行 L4 I/O)
  │
  ├── batch_read(keys) → 从 Remote 拉取数据
  ├── batch_write(items) → 写入 Remote
  ├── batch_exists(keys) → 查询远端是否存在数据
  └── batch_delete(keys) → 删除远端数据
```

---

## 8. 三层分离架构图

```
┌───────────────────────────────────────────────────────────────┐
│                    BlockTreeCache（协调者）                     │
│  - 拥有 BlockTree 和 ComponentGroups                           │
│  - 实现淘汰工作流和跨组件协调                                      │
│  - 管理线程安全（持有锁）                                         │
│  - 对外暴露 match / insert / evict 高层接口                      │
└───────────────────────────────────────────────────────────────┘
        │ owns                                    │ owns
        ▼                                          ▼
┌────────────────────────────┐    ┌──────────────────────────────┐
│  BlockTree                 │    │  ComponentGroup[]            │
│  （纯树数据结构）             │    │  （行为主体 + Heap 容器）      │
│  - findNode()              │    │  - driveEviction()           │
│  - insertNode()            │    │  - evictFromTier()           │
│  - removeNode()            │    │  - heapForTier()             │
│  - removeEmptyAncestors()  │    │    → EvictionHeap            │
└────────────────────────────┘    └──────────────────────────────┘
                                              │ owns
                                              ▼
                                ┌──────────────────────────────┐
                                │  EvictionHeap                │
                                │  （纯堆数据结构）               │
                                │  - push() / pop()            │
                                │  - invalidate()              │
                                │  - onAccess()                │
                                └──────────────────────────────┘
```

---

## 9. EvictionTask 生命周期

```
PENDING ────────── RUNNING ────────── COMPLETED
  │                │                     │
  │ 线程池拾取       │ 数据搬运完成          │ 协调者回调
  │                │                     └─→ Phase 3: 状态提交
  │                │
  └────────────────┴───────────── FAILED
                                        │
                                        ├─→ 回滚：释放引用计数，重新入堆
                                        └─→ 可重试：FAILED → PENDING（状态机允许）
```

---

## 10. BlockPool 依赖关系图

```
Scheduler / 上层引擎 (调用方)
  │
  ├─ block_pool[gid].alloc(n)        → 为新请求分配 KV slot（按 group 独立 pool）
  ├─ block_tree_cache.match()        → 查缓存（内部自动处理 load_back/prefetch）
  └─ block_tree_cache.insert()       → 登记到树

BlockTreeCache
  │
  ├─ block_pool.free(indices)        → insert 时释放重复/未对齐 slot
  ├─ block_pool.available()          → load_back 时检查空间
  ├─ 不调用 block_pool.alloc()！
  └─ 通过 ComponentGroup.componentIndices() 了解 group 拓扑

ComponentGroup (行为主体)
  │
  ├─ FullComponentGroup:    evictFromTier() 释放同 group 所有 device_blocks / host_block
  ├─ SWAComponentGroup:     evictFromTier() 释放 SWA group 的 KV slot
  └─ LinearComponentGroup:  evictFromTier() 释放 LINEAR 状态 slot
```

---

## 11. 文件目录结构

### KVCacheMemoryConnector 删除前后对比

```
删除前 (KVCacheMemoryConnector, 3500+ 行):
  ├── PrefixTreeMemoryBlockCache (Host 侧树)          → 合并到 BlockTreeCache 树
  ├── MemoryDiskBlockCache (Disk 层)              → 合并到 BlockTreeCache 树
  ├── BlockPool (CPU 物理池)                        → 移入 BlockTreeCache 目录
  ├── CopyPlan / CopyInfoPerKey (数据搬运)          → 移入 BlockTreeCache 目录，重命名为 CopyEngine
  ├── BroadcastManager (TP 广播)                    → 移入 BlockTreeCache 目录
  └── async match/read/write (异步接口)              → 移入 BlockTreeCache 目录
```

### block_tree_cache/ 目录结构

```
rtp_llm/cpp/cache/block_tree_cache/
├── BlockTreeCache.h/.cc           — 淘汰流程协调者（异步任务管理、级联、线程安全）
├── BlockTreeCacheFactory.h/.cc    — 工厂函数（从 CacheConfig + KVCacheConfig 构造 BlockTreeCache）
├── EvictionTask.h                 — 异步淘汰任务状态机（纯头文件，PENDING/RUNNING/COMPLETED/FAILED）
├── BlockTree.h/.cc                — 纯树数据结构
├── TreeNode.h                     — 节点数据结构 + Tier 枚举 + GroupSlot + MemoryBlockLayerTagSlot
├── ComponentGroup.h/.cc           — ComponentGroup 基类 + Component 描述符 + MatchValidator + TransferType
├── FullComponentGroup.h/.cc       — Full 组件组实现
├── SWAComponentGroup.h/.cc        — SWA 组件组实现
├── LinearComponentGroup.h/.cc     — LINEAR 组件组实现
├── EvictionHeap.h/.cc             — 纯堆数据结构（可配置淘汰策略）
├── TransferDescriptor.h           — 传输描述符
├── StorageBackend.h               — 远端存储接口
├── copy_engine/                   — 数据搬运引擎（从 KVCacheMemoryConnector 移入）
│   ├── CopyEngine.h/.cc           — GPU↔CPU↔Disk 数据搬运（无状态工具）
│   ├── DiskBlockPool.h/.cc        — 磁盘池（slot 管理 + 双引用计数）
│   └── DiskBlockIO.h/.cc          — 磁盘 I/O 抽象（IDiskBlockIO + PosixDiskBlockIO 实现）
└── test/
    ├── BlockTreeCacheTest.cc          — 协调者测试
    ├── BlockTreeTest.cc               — 树测试
    ├── FullComponentGroupTest.cc      — Full 组件组测试
    ├── SWAComponentGroupTest.cc       — SWA 组件组测试
    ├── LinearComponentGroupTest.cc    — LINEAR 组件组测试
    ├── EvictionHeapTest.cc            — 淘汰堆测试
    ├── CopyEngineTest.cc              — CopyEngine 测试
    └── block_tree_cache_eviction_test/ — 端到端淘汰流程测试
        ├── FullEvictionTest.cc
        ├── FullLinearEvictionTest.cc
        ├── FullSWAEvictionTest.cc
        └── FullSWALinearEvictionTest.cc
```

### Phase 1 修改文件

```
rtp_llm/cpp/cache/
├── KVCacheAllocator.h/.cc         — 替换 SharedBlockCachePtr → BlockTreeCachePtr
├── KVCacheManager.h/.cc           — 初始化 BlockTreeCache
└── BUILD                          — 新增 block_tree_cache/ 依赖
```

### Phase 1 不变文件

```
SharedBlockCache.h/.cc             — Phase 1 保留，Phase 2 后移除
BlockCache.h/.cc                   — 保留（legacy 兼容）
KVCacheConnectorCoordinator.h/.cc  — 不变
KVCacheMemoryConnector.h/.cc       — Phase 3 删除，逻辑移入 block_tree_cache/
P2PConnector.h/.cc                 — 保留，暂不做测试
```

---

## 12. 典型请求链路

一个完整请求在 BlockTreeCache 中的生命周期：

```
1. 请求到达
   Scheduler 调用 KVCacheAllocator::malloc(MallocInfo)

2. 前缀匹配 + 数据加载
   KVCacheAllocator 从 MallocInfo 中提取 cache_keys
   调用 block_tree_cache_->match(cache_keys)
   match 内部自动触发 load_back/prefetch（如需要，包含 Remote 预取）
   调用方等待 match 结果中的异步上下文完成

3. 推理
   NormalEngine 执行 prefill / decode
   使用匹配到的 GPU block

4. 插入
   推理完成后:
     block_tree_cache_->insert(match_result.matched_node, new_cache_keys, new_slots)
     树创建新的 block 节点（每个节点拥有独立的 GroupSlot）

5. 释放
   请求完成，BlockPool 释放 block 引用计数
   节点可能重新加入淘汰候选

6. 驱逐（按需，异步三阶段）
   当 GPU 空间不足时:
     KVCacheAllocator 调用 block_tree_cache_->evict(num_blocks, DEVICE)
     Phase 1: ComponentGroup 从 heap 选择候选 → EvictionResult（同步）
     Phase 2: 创建 EvictionTask 提交到线程池（异步，无锁）
     Phase 3: 任务完成后回调，提交状态更新（同步）
       → 更新 GroupSlot（device → host）
       → 释放源层级 block
       → 更新 tree / heap
       → 触发级联淘汰
```

