# P2P KV Cache 传输编排层设计：统一 TP 不对称 / RR CP / 非 RR CP

> 状态：设计稿（v2）
> 范围：`rtp_llm/cpp/cache/connector/p2p/`（decode_entrance 反转 PD 链路）
> 编排层位置：**`P2PConnectorSchedulerDecode` + `P2PConnectorSchedulerPrefill`**（两侧 rank0）
> 跨端协议（`GetPeerInfo` / `StartLoad`）：**零改动**

---

## 1. 问题定义

一次 P2P KV cache 传输，本质是在四个正交维度上做「源分片坐标 → 目的分片坐标」的映射：

| 维度 | 分片依据 | 语义 |
|---|---|---|
| D1 layer × group_tag | 拓扑 | 两侧同构（same-build 契约） |
| D2 逻辑 block 位置（序列方向） | CP `cp_mapping` | RR / COMPACT_LAST_RANK / NONE |
| D3 block 内字节区间（head 方向） | TP `partition_count/id`；CP `cp_slice` | 谁持有哪段 head / 哪段字节 |
| D4 端点配对 | prefill_rank × decode_rank | 谁发给谁 |

**当前实现把四个维度分散在四处独立决策，靠「两侧各自算出的数字恰好相同」对齐。**

| 维度 | Prefill 侧决策点 | Decode 侧决策点 |
|---|---|---|
| D2 CP | `P2PConnectorWorkerPrefill::writeByLayer`（`cp_rank = tp_rank % cp_size`） | `P2PConnectorSchedulerDecode::asyncRead`（`worker_rank % cp_size`） |
| D3 TP | `AsymmetricTpUtil::handleAsymmetricTP` | `P2PConnectorWorkerDecode::calculateRecvPartitionCount` |
| D3 CP slice | **未实现** | **未实现** |
| D4 配对 | `AsymmetricTpUtil` 自选 `decode_transfer_servers[...]` | scheduler 逐 worker 下发 buffer |

而传输层的匹配契约是**严格**的（`TcpTaskContext::executeCopy`）：

1. **键集包含**：decode 注册的每个 `cache_key` 必须出现在 prefill 请求中，否则 `cache_key missing in request`；
2. **子块形状逐位相等**：第 i 个 `BlockInfo.size_bytes` 必须等于对端第 i 个 `proto_block.len()`，否则 `BUFFER_MISMATCH`。

两侧独立推导 + 严格契约 ⇒ 任一维度不对称就在传输层炸掉（或走到错误的字节区间）。

---

## 2. 现状缺陷

### 2.1 CP size 不一致直接拒绝

`P2PConnectorSchedulerDecode.cc:125` 硬拒 `prefill_cp_size != config_.cp_size`。因此 **prefill CP=4 + decode CP=1** 在 P2P 链路完全不可用，而它在 legacy `DecodeRpcServer::loadCache` 里是支持的。

### 2.2 CP 字节切分（`cp_slice`）在 P2P 链路缺失

`CPSlotMapper::sliceBlockForPeer` / `CpBlockSliceMode::{EQUAL_BYTES, PAYLOAD_BYTES}` 只被 legacy 路径（`sliceCpDestinationForPeer`）使用。P2P 的 `LayerCacheBufferUtil::convertLayerTag` 只调了 `physicalBlockPosition`（D2），没有任何 D3-CP 处理。fixed-region / SWA 这类「prefill 把一个逻辑 block 按 CP 切成 N 段字节、decode 持整块」的 spec 在 P2P 下无法工作。

### 2.3 非 MLA NP1D（prefill_tp > decode_tp）字节数不匹配

prefill_tp=4、decode_tp=2、非 MLA，设全局 block 字节数 `B`：

- prefill 本地 block = `B/4`，decode 本地 block = `B/2`
- `handleNP1D`：`local_partition_count = 4/2 = 2`，prefill rank0 发 `{2,0}` = **自身 block 的一半 = `B/8`**
- `calculateRecvPartitionCount(4) = 2`，decode rank0 注册 `{2,0}` = **自身 block 的一半 = `B/4`**

`B/8 ≠ B/4` → size mismatch。正确语义是 legacy 的 `src={1,0}`（整块）、`dst={peer_cnt,i}`。`AsymmetricTpUtil` 注释称对齐 legacy 的 `peer_cnt/i`，但 legacy 的 `peer_cnt/i` 施加在**目的端**，平移到源端本地 block 上多切了一次。

> MLA / hybrid 不受影响：`MemoryLayoutStrategy.cc:193` 对 `is_mla || enable_hybrid_attention` **忽略** partition 参数、总返回整块。ND1P 恰好自洽。现有 `AsymmetricTpUtilTest` 只测映射算术、无字节数断言，故未覆盖。

### 2.4 隐式约定不可审计

- MLA「只有 partition 0 发送」在 prefill worker（`P2PConnectorWorkerPrefill.cc:625`），对应的 decode 侧 `recv_partition_count=1` 在 `calculateRecvPartitionCount` 的 `is_mla` 分支。改一处就静默错配。
- 传输 key `makePartitionLayerKey(..., partition_id)` 的 `partition_id`：源端填 `remote_partition_id`，目的端填循环下标——两个不同公式算出同一个命名空间，是全链路最脆的耦合点。

---

## 3. 设计决策（v2 相对 v1 的三处修正）

### 3.1 编排层 = 两侧 rank0 scheduler，worker 降级为纯执行器

- **`P2PConnectorSchedulerDecode`**（decode rank0）：计算 plan，按 `forDecodeRank(d)` 给每个 decode worker 下发 route 列表（复用已有 `broadcastPerRank`）。
- **`P2PConnectorSchedulerPrefill`**（prefill rank0）：用**同一个 planner 函数**计算镜像 plan，按 `forPrefillRank(p)` 给每个 prefill worker 下发 route 列表（`broadcast` → `broadcastPerRank`，机制已存在）。
- **worker 不再推导任何映射**：删除 `AsymmetricTpUtil`、`calculateRecvPartitionCount`、MLA 主 rank 跳过分支、`writeByLayer` 里 `cp_rank = tp_rank % cp_size` 的硬编码投影。

### 3.2 对端 layout 在本端推导，跨端协议零改动

**结论：`GetPeerInfo` / `StartLoad` 都不需要新字段。** 依据：

1. **stride / head 数是配置的纯函数**。`localKvHeadNumForSpec(type, model_config, parallelism_config)`（`SingleConfigCreator.cc:72`、`HybridConfigCreator.cc:39`）与 `MHAKVCacheSpec::create` 都只经由 `parallelism_config.get_attn_tp_size()` 读取并行度。因此本端只要**拷贝一份 `ParallelismConfig`、把 `tp_size` / `prefill_cp_config` 替换成对端的值**，再调同一个函数，就能得到对端的 `local_kv_head_num` 与 `kv_block_stride_bytes`。无需上报字节数。

2. **对端的 `(tp_size, cp_size)` 已在手**：
   - decode 侧：`Meta::P2PRoutingContext::{prefill_tp_size, prefill_cp_size}`，由 `DecodeRpcServerNew2` 从 `GetPeerInfo` 写入（已有）。
   - prefill 侧：`decode_tp_size = decode_transfer_servers.size()`（已有）；decode 的 `cp_size = kv_cache_sharded ? decode_tp_size : 1`，其中 `kv_cache_sharded` 是**部署级同配**开关。

3. **`prefill_cp_config` 本就是部署级同配项**。`OpaqueKVCacheSpec::fixedRegionCpSize` 存在 `RoleType::DECODE` 分支，直接读本端 `prefill_cp_config.prefill_cp_size` 来构建 decode 自己的 spec（`OpaqueKVCacheSpec.h:136-144`），并且 `CHECK` 它 > 1。也就是说**「decode 被配置了 prefill 的 CP 布局」已经是现有部署的硬前提**，本设计只是复用它，没有新增假设。

4. **漂移检测（替代 v1 的 digest 字段）**：`asyncRead` 入口断言
   `routing->prefill_cp_size == (local prefill_cp_config.kv_cache_sharded ? routing->prefill_tp_size : 1)`
   且与本端配置的 `prefill_cp_config.prefill_cp_size` 一致（当其 > 0）。不一致 → 立即失败并打印双方并行度。这把「配置漂移 / 版本不一致」从传输期的疑难杂症变成首个请求上的确定性报错，零协议成本。

### 3.3 修正 v1 的 rank 坐标公式（重要）

v1 假设 `head_shard_count = tp_size / cp_size`，即 CP 与 head 分片是**嵌套**的。查证后**不成立**：

```cpp
// ConfigModules.h:81
int64_t get_attn_tp_size() const { return prefill_cp_config.is_enabled() ? 1 : tp_size; }
// 注意 is_enabled() 排除了 PREFILL_CP：
bool is_enabled()         const { return method != DISABLED && method != UNKNOWN && method != PREFILL_CP; }
bool is_prefill_enabled() const { return method == PREFILL_CP; }
// KVCacheConnectorCoordinator.cc:448
cp_size = prefill_cp_config.kv_cache_sharded ? tp_size : 1;
```

三种组合：

| `method` | `kv_cache_sharded` | `get_attn_tp_size()` | connector `cp_size` | head 分片 | 序列分片 |
|---|---|---|---|---|---|
| ALL_GATHER / ALLTOALL … | true | **1** | tp_size | 无（复制） | 有 |
| PREFILL_CP | true | **tp_size** | tp_size | **有** | 有 |
| 任意 | false | tp_size | 1 | 有 | 无 |

第二行下 head 分片与序列分片**同时**沿同一个 rank 轴，`head_shard_count × cp_size = tp_size²  ≠ tp_size`。v1 的嵌套公式在这一行是错的。

**修正**：不再自造分解，直接用 `ParallelismConfig` 已有的访问器作为两个独立的坐标函数：

```cpp
int cpSize()          const { return pc.prefill_cp_config.kv_cache_sharded ? pc.tp_size : 1; }
int cpRank(int r)     const { return cpSize() > 1 ? r % cpSize() : 0; }
int headShardCount()  const { return pc.get_attn_tp_size(); }          // 1 或 tp_size
int headShard(int r)  const { return headShardCount() > 1 ? r % headShardCount() : 0; }
int rankCount()       const { return pc.tp_size; }
```

`head_shard_count` 与 `cp_size` 相互独立，各自可为 1 或 `tp_size`。plan 在 rank 全集上枚举 `(cpRank(r), headShard(r))`，不做任何除法分解。

---

## 4. 核心抽象

新增 `rtp_llm/cpp/cache/connector/p2p/plan/`（纯头 + 一个 cc，无 IO、无 GPU 依赖）。

```cpp
// plan/ShardLayout.h ——「一侧的分片布局」，全部可从 ParallelismConfig 本端推导
struct ShardLayout {
    ParallelismConfig pc;         // 本端或「替换了 tp_size/prefill_cp_config 的对端副本」
    bool              is_mla = false;
    bool              is_hybrid = false;

    int cpSize() const; int cpRank(int rank) const;
    int rankCount() const { return static_cast<int>(pc.tp_size); }

    // 每个 tag 的 policy + 字节尺寸：本端从 CacheTopology 取，对端用
    // localKvHeadNumForSpec(type, model_config, pc) 以替换后的 pc 重算。
    struct GroupLayout {
        CacheGroupPolicy policy;
        KVCacheSpecType  spec_type;
        size_t kv_block_stride_bytes = 0, kv_scale_stride_bytes = 0;
        size_t k_block_payload_bytes = 0, seq_size_per_block = 0;
        bool   pre_sliced = false;       // 见 Step 4 的 cp_slice 规则

        // head 分片数是 **spec 类型的属性**，不是这一侧的属性：只有会除
        // get_attn_tp_size() 的 spec 才被 head 切分。
        //   MHAKVCacheSpec / LinearKVCacheSpec → get_attn_tp_size()
        //   MLAKVCacheSpec  / OpaqueKVCacheSpec → 1（latent 复制 / 走 cp_slice）
        int head_shard_count = 1;
    };
    std::unordered_map<std::string, GroupLayout> groups;

    // per-group，不是 per-side。旧版把它写成 pc.get_attn_tp_size() 的全局标量，
    // 对 MLA 会错误地返回 tp_size，只是恰好被 Step 4 的 is_mla 特判挡住。
    int headShardCount(const std::string& tag) const { return groups.at(tag).head_shard_count; }
    int headShard(int rank, const std::string& tag) const {
        const int n = headShardCount(tag);
        return n > 1 ? rank % n : 0;
    }

    // 唯一的对端构造入口
    static ShardLayout forPeer(const ShardLayout& local_self,
                               const ModelConfig& model_config,
                               int peer_tp_size, int peer_cp_size);
};
```

```cpp
// plan/TransferPlan.h
struct PartitionSpec { int count = 1; int id = 0; };                                   // → LayerBlockConverter
struct SliceSpec     { CpBlockSliceMode mode = CpBlockSliceMode::NONE; int count = 1; int index = 0; };
// 键选择规则：一个「模 src.cpSize() 的剩余类」，外加 COMPACT 的尾键例外。
// 用 (modulus, residue) 而非 (mapping, cp_size, cp_rank)：mapping 已被 planner 折叠进
// (modulus, residue)，resolveKeys 因此只有 4 行、执行期不必再判 mapping 枚举。
struct KeyShardSpec {
    int  modulus           = 1;      // = src.cpSize()（Step 1 的 CP 白名单保证）
    int  residue           = 0;      // 本 route 覆盖的剩余类
    bool include_final_key = false;  // COMPACT 组：序列末位无论落在哪个剩余类都要取
    int  tail_count        = 0;      // = policy.active_tail_blocks；只保留末尾若干项，0 为不限
};

struct TransferRoute {
    int         route_id = 0;   // 编排层签发，进入传输 key
    int         src_rank = 0;   // prefill worker index
    int         dst_rank = 0;   // decode worker index
    std::string cache_tag;      // route 按 tag 分裂（不同 group 的 policy 可能不同）
    KeyShardSpec  src_keys;     // 本 route 上应发送的逻辑键集规则
    PartitionSpec src_partition, dst_partition;
    SliceSpec     src_slice,     dst_slice;
};

struct TransferPlan {
    std::vector<TransferRoute> routes;
    std::vector<const TransferRoute*> forDecodeRank(int d) const;
    std::vector<const TransferRoute*> forPrefillRank(int p) const;
    uint64_t digest() const;     // 仅用于日志 / metric，不上协议
};
```

### 4.1 route 的粒度（关键定义）

**一条 route = 一个 `(src_rank, dst_rank, cache_tag)` 组合**，语义是「prefill 第 `p` 号 rank 要把该 group 的哪一部分、以什么切分方式、发给 decode 第 `d` 号 rank」。

**route 与 layer 无关**——同一条 route 的描述对该 tag 覆盖的所有 layer 都成立。三个量必须分清：

| 量 | 定义 | 规模 |
|---|---|---|
| **route 数** | 有数据要搬的 `(src_rank, dst_rank, tag)` 组合数 | **与层数无关**，通常是 rank 数量级（个位到数十） |
| **传输单元数** | route 数 × 该 tag 覆盖的层数；每个单元一个 recv task / 一次 RDMA op，key 为 `<uk>_<layer>_<tag>_r<route_id>` | 61 层模型下是 route 数的 61 倍 |
| **键数** | 单个传输单元内由 `resolveKeys(spec, count)` 展开出的 cache_key 数 | 取决于序列长度与 prefix 命中 |

举例（§9 用例，除 A8 外均为单 tag）：

| 用例 | route 数 | 说明 |
|---|---|---|
| A1 / A2（对称） | 8 | 8 组一一配对的 rank 对；decode 每 rank 每层 1 个 recv task ⇒ 即今天的行为 |
| A3（prefill CP=4 → decode 不分片） | 4 | 全部指向同一个 decode rank。**decode rank0 每层注册 4 个 recv task**——它需要的逻辑块分散在 4 个 prefill rank 上，谁都不持有全部 |
| A6（MLA 8→4） | 4 | 8 个 prefill rank 中只有 4 个被选为发送方，落选者 `forPrefillRank()` 为空 |
| A8（hybrid） | 8 | FULL group 4 条对角线 + SWA group 4 条（同源）。同一对 rank 在不同 tag 上是两条独立 route，因 policy 不同 |

两个直接后果：

1. **协议载荷小**（§6 载荷预算的依据）：route 数不随层数增长，下发 route 列表通常 < 2KB。反之若把 route 按层展开、或把 `cache_keys` 塞进 route，就会到 100KB+/请求。
2. **运行时压力看传输单元数**：A3 的 4 条 route × 61 层 = 244 个 recv task（对称时 61 个）。这也是 §Step 3b 副本均分必须带阈值与开关的原因——均分把 route 数乘上副本数，再乘层数就很可观。

### 4.2 route 不是新概念：与现有结构的对应

route 是现有 `AsymmetricTPContext` 的推广——把它从「每个 worker 自己算」提升为「编排层签发、双端共享」：

```cpp
struct AsymmetricTPContext {                           // 今天的 prefill 侧
    std::string decode_ip;  uint32_t decode_port;      // → route.dst_rank
    int local_partition_count, local_partition_id;     // → route.src_partition
    int remote_partition_count, remote_partition_id;   // → route.dst_partition（今天靠 key 命名约定隐式传递）
};
```

route 在其之上补了三样今天缺失的维度：`cache_tag`（今天所有 tag 共用一份 ctx）、`KeyShardSpec`（CP 键维度，今天硬编码在 `writeByLayer` 的 `tp_rank % cp_size` 里）、`SliceSpec`（CP 字节维度，今天完全没有）。decode 侧对应的隐式 route 就是 `buildRecvTasks` 里的 `partition_id` 循环。

### 4.3 route 如何落到实际传输

**传输层代码完全不动**：`IKVCacheSender` / `IKVCacheReceiver` / `TransferTask` / `TransferTaskStore` 一行不改。route 只决定「谁在什么时候、用什么参数调 `send()` / `recv()`」。

一条 route 在一个 `(layer, tag)` 上产生**恰好一个** `transfer::SendRequest` 与一个 `transfer::RecvRequest`：

| route 字段 | prefill 侧落点 | decode 侧落点 |
|---|---|---|
| `route_id` | `send_req.unique_key` 后缀 | `recv_req.unique_key` 后缀 |
| `src_rank` | 决定哪个 prefill worker 收到这条 route | — |
| `dst_rank` | `send_req.ip / port`（经 `peer_workers` 索引表解析） | 决定哪个 decode worker 收到这条 route |
| `cache_tag` | 选用哪个 `LayerCacheBuffer` | 同 |
| `src_keys` | `resolveKeys` 筛出本 route 要发的 cache_key | —（dst 用自己的 needed） |
| `src_partition` | `buildKeyBlockInfos(converter, buf, count, id)` | — |
| `dst_partition` | — | `buildKeyBlockInfos(...)` |
| `src_slice` / `dst_slice` | `CPSlotMapper::sliceBlockForPeer` | 同 |

```
prefill: sendKVCache → [今天: asymmetric_tp_util_->handleAsymmetricTP() → tp_partition_ctxs]
                       [改后: forPrefillRank(my_rank) → routes                             ]
         → dispatchPendingLayerTransfers → sendLayerToPartitions
         → 每条 route 构造 1 个 transfer::SendRequest → sender_->send(req, done_cb)

decode:  read → [今天: calculateRecvPartitionCount() → partition_id 循环]
                [改后: 广播下发的 routes                                ]
         → buildRecvTasks → 每条 route 1 个 transfer::RecvRequest
         → receiver_->recv(req) → TransferTaskStore::addTask(key, block_infos, deadline)
```

**两侧的汇合点是 `unique_key` 字符串，必须逐字节相同**。传输层靠它在 `TransferTaskStore::task_map_` 做 rendezvous：

1. prefill 的 send 到达 decode，`TcpTransferService::transfer` 把它挂进 `wait_tasks_`
2. `waitCheckProc` 用 `task_store_->getTask(ctx->getUniqueKey())` 查找 decode 预先登记的 recv task
3. 命中 → `transferViaTcp` → `executeCopy`：**以 recv task 的 `block_infos` 为权威**，逐 cache_key、逐子块校验长度后拷贝
4. 未命中 → 等到 `isTimeout()`，报 `no matching recv task within deadline`

因此 route 的全部价值是：**让这个字符串、以及它两侧的 `KeyBlockInfoMap`，出自同一个决策**。今天是 `AsymmetricTpUtil` 与 `calculateRecvPartitionCount` 两个独立公式各算出一个 `partition_id` 然后碰巧相等——§2.1、§2.3 都是这个「碰巧」不成立的后果。

> `1 route × 1 (layer, tag) = 1 SendRequest = 1 TransferTask`，但底层不一定是 1 次 RDMA WR：RDMA 后端会按 `rdma_max_block_pairs_per_connection` 再拆，那一层对 route 透明。

**传输 key 由 route_id 命名**（本设计收益最高的单点改动）：

```cpp
// P2PKeyUtil 新增；makePartitionLayerKey 保留一个版本周期
static std::string makeRouteLayerKey(const std::string& base_key, int layer_id,
                                     const std::string& cache_tag, int route_id);
// => base_key + "_" + layer_id + "_" + cache_tag + "_r" + route_id
```

key 不再由「两个独立公式碰巧算出同一个 partition_id」决定，而由编排层签发的 `route_id` 决定。

> 可选加固：key 末尾追加 `plan.digest()` 的短哈希。两侧 plan 若因配置漂移而分歧 ⇒ key 不匹配 ⇒ 退化为 `no matching recv task within deadline`（TIMEOUT）。相比今天「两边都算出 `_0`、于是拷错字节」，失败模式严格变好，且零协议成本。

---

## 5. 编排算法

```cpp
// 布局级：与请求无关，可按 (src.pc, dst.pc, topology_digest) 缓存。
// 同一部署下每个 tag 只算一次，asyncRead 热路径上只是一次 map 查找。
static Expected<TransferPlan> plan(const ShardLayout& src, const ShardLayout& dst,
                                   const std::vector<std::string>& tags);

// 请求级：把 route 上的规则展开成具体逻辑位置。两侧在执行期各调一次。
static std::vector<size_t> resolveKeys(const KeyShardSpec& spec, size_t logical_count);
```

**为什么必须拆成两层**：`plan()` 不能拿 `logical_block_count`，否则就退化成每请求重算、无法缓存。而 `COMPACT_LAST_RANK` 的 `is_final_key` 判定依赖 count。解法是 `plan()` 只在 `pos ∈ [0, src.cpSize())` 的**剩余类代表元**上做符号推理（该区间对 RR/NONE/COMPACT 的周期性部分完备，规模 ≤ 8），并对 COMPACT 组**保守产出** route；真正的键集由两侧在执行期用同一个 `resolveKeys(spec, count)` 展开。某条 route 在特定 count 下解析为空集是无害的——两侧规则相同故一致判空，decode 不注册、prefill 不发送。

`plan()` 是**纯函数**：不含 `cache_keys`、不含 block id、不含时间/随机源。

### Step 1 — 前置校验（fail-fast）

```
require src.is_mla == dst.is_mla && src.is_hybrid == dst.is_hybrid
for each tag:
    require src.groups[tag].policy.group_type == dst.groups[tag].policy.group_type
    require src.groups[tag].spec_type         == dst.groups[tag].spec_type

    // (a) 字节维度：还原出的全局 block 大小必须一致
    require effective_global(src, tag) == effective_global(dst, tag)
            // effective_global = kv_block_stride_bytes
            //                  * headShardCount(tag)
            //                  * (pre_sliced ? cpSize() : 1)

    // (b) 键维度：单侧自完备性。head 分片与 block RR 分片不能同时作用于同一 group，
    //     否则 rank r 只持有 (head r) × (block ≡ r mod N)，(head h, block b) 仅当
    //     b ≡ h mod N 才有人持有 ⇒ 缺 (N-1)/N 的数据。
    for side in {src, dst}:
        require NOT (side.headShardCount(tag) > 1
                     && side.cpSize() > 1
                     && side.groups[tag].policy.cp_mapping == BLOCK_ROUND_ROBIN)
```

(a) 把 §2.3 从传输层的 `BUFFER_MISMATCH` 提前成编排期的确定性错误。

**(b) 是独立必需的，(a) 抓不到它**：MHA + `PREFILL_CP` + `kv_cache_sharded`、tp=8 时，src 本地 stride = `262144/8 = 32768`，还原全局 = `32768 × 8 = 262144`；dst（不开 CP、tp=8）同为 `262144`。**(a) 通过**——因为缺失发生在键维度而非字节维度。

该组合合法与否取决于 spec 类型（`PREFILL_CP + kv_cache_sharded` 下 `get_attn_tp_size()` 与 `cpSize()` **都取完整 tp_size**，代码中没有任何位置把 tp_size 拆成 head 因子 × CP 因子）：

| group 的 spec | 是否除 `get_attn_tp_size()` | 默认 `cp_mapping` | 该组合下 |
|---|---|---|---|
| `MLAKVCacheSpec` | 否（latent 复制） | `BLOCK_ROUND_ROBIN` | 完备 ✓ |
| `OpaqueKVCacheSpec` | 否（走 `cp_slice`） | 由 policy 指定 | 完备 ✓ |
| `LinearKVCacheSpec` | **是** | `NONE`（block 复制） | 完备 ✓ |
| `MHAKVCacheSpec` | **是** | `BLOCK_ROUND_ROBIN` | **不完备 ✗** ⇒ 被 (b) 拒绝 |

即 `PREFILL_CP + kv_cache_sharded` 本身是合法的生产配置（MLA + fixed-region 正是其目标），**只有 MHA FULL group 撞上它才是配置错误**，由 (b) 在编排期拦下。

### Step 2 — D2：逻辑位置归属（RR / 非 RR 的统一落点）

用**已有**的 `CPSlotMapper::physicalBlockPosition` + `layoutForGroup(...).slice` 建立归属函数，不新写映射数学：

```
owners(side, tag, logical_pos, logical_count) -> set<(cp_rank, physical_pos, slice_index)>
```

- `cp_slice != NONE` → **所有** cp_rank 参与，各持有以 `cp_rank` 索引的字节切片
- 否则按 `cp_mapping`：
  - `NONE`（非 RR）→ 每个 cp_rank 都持有全量，`physical_pos = logical_pos`
  - `BLOCK_ROUND_ROBIN`（RR）→ 唯一 owner `logical_pos % cp_size`，`physical_pos = logical_pos / cp_size`
  - `COMPACT_LAST_RANK` → owner 为 segment 尾（或序列最后一个 key），`physical_pos = logical_pos / cp_size`

三种 mapping + slice 收敛为同一个 `owners()` 查询，上层不再出现 `if (is_page_level_rr)` 分支。

### Step 3 — D2 配对：源 CP shard × 目的 CP shard

对每个目的 CP shard，把它需要的键集**按源 shard 身份切分**。注意不能简单地「交集非空就建 route」：
复制型 group（`cp_mapping = NONE`，每个 CP rank 都持全量，如 hybrid 里的 LINEAR）下每个 src_cp
都提供全量**同一份字节**，逐个建 route 会产生 `cpSize²` 条冗余 route。必须选举唯一源。

```
for dst_cp in [0, dst.cpSize()):
    needed = { pos : dst_cp ∈ owners(dst, tag, pos) }
    assign = {}                                    // (src_cp, slice_index) -> keys
    for pos in needed:
        providers = { src_cp : src_cp ∈ owners(src, tag, pos) }
        if src.groups[tag].policy.cp_slice != NONE:
            // 各 src_cp 持有的是不同字节切片 → 全取，slice_index = src_cp
            for src_cp in providers: assign[(src_cp, src_cp)] += pos
        else:
            // 各 provider 持有相同字节 → 选举唯一源（CP 轴上对偶于 MLA 的 headShard 0 选举）
            assign[(min(providers), 0)] += pos
    for (src_cp, slice_index) in assign: 记录 CP 级配对
```

产出 `KeyShardSpec` 前还需三步辅助处理（实现时才暴露出来的必需项）：

1. **剩余值折叠**：`assign` 里一组剩余值若构成模某个 `d | modulus` 的剩余类，折叠成 `{modulus=d, residue=r%d}`；
   全集折成 `{modulus=1, residue=0}`（即「全取」）。折不掉时**逐剩余值产出 route**——正确，只是 route 更多。
2. **FULL group 的无空洞校验**：`providers` 为空意味着该位置源端无人持有。FULL group 出现空洞是配置错误 ⇒ 报错；
   非 FULL（COMPACT/SWA）本就只在部分位置有 cache 条目 ⇒ 合法跳过。
3. **COMPACT 尾键唯一性**：`include_final_key` 只能挂在每个 `(dst_cp, slice_index)` 的**唯一一条** route 上，
   否则多条 route 的键集会在末位重叠。若某组合下会挂到多条，报错并提示该 src/dst CP 映射组合尚不支持。

各 mapping 下的行为：

| src `cp_mapping` | `providers` 规模 | 选举结果 | 每个 dst_cp 的 route 数 |
|---|---|---|---|
| `BLOCK_ROUND_ROBIN` | 恰好 1（`pos % N == cp_rank`） | 平凡 | = 有交集的 src_cp 数 |
| `COMPACT_LAST_RANK` | 全部 `cpSize` 个 | 取 cp_rank 最小者 | **1**（选举后） |
| `NONE`（复制） | 全部 `cpSize` 个 | 取 cp_rank 最小者 | **1**（选举后） |
| `cp_slice != NONE` | 全部 `cpSize` 个 | 不选举，全取 | `cpSize`（各持不同字节） |

> `COMPACT_LAST_RANK` 是**复制型子集**而非 rank 独占：`CPSlotMapper::physicalBlockPosition` 的 COMPACT
> 分支只判 `is_segment_tail || is_final_key`，**完全不看 `cp_rank`**，并对所有 rank 返回同一个
> `logical_position / cp_size`。因此它的 `providers` 是全部 cp_rank，走选举路径（除非叠加了 `cp_slice`）。

> 选举只决定**哪些 route 被产出**；`KeyShardSpec` 结构不变——route 已显式携带 `src_rank`，
> 未被选举的 rank 根本不会收到这条 route。

**CP 形态白名单（Step 1 的 (b0)）**：只允许两种形态——

| 形态 | 说明 |
|---|---|
| `dst.cpSize() == 1` | prefill CP N → decode 不分片。**本设计新增的能力**（今天被 §2.1 拒掉） |
| `dst.cpSize() == src.cpSize()` | 两侧完全相等。**今天 §2.1 唯一允许的形态** |

**两侧都分片且不相等（如 p cp=2 → d cp=4）明确不支持**。那需要「模 `lcm(src,dst)` 的剩余类 + 中国剩余定理求 residue」整套机制；vLLM 的 `_validate_remote_parallel_config` 与 SGLang 的 `common/conn.py` assert 同样禁止两侧 CP 并存。挡在白名单之后：

```
modulus = src.cpSize()          // lcm(src, dst) 在两种允许形态下都等于它
residue = src_cp                // dst_cp 或为 0（dst 不分片）或等于 src_cp（对称）
```

`KeyShardSpec` 仍用 `(modulus, residue)` 而不是 `(mapping, cp_size, cp_rank)`：前者让 `resolveKeys` 退化成 4 行、执行期不必再判 mapping 枚举，是更简的表示，与是否支持双侧 CP 无关。

**worked example**（prefill CP=4 RR → decode CP=1）：

```
needed(dst_cp=0)   = 全部
provided(src_cp=0) = {≡0 mod 4} → route r0  KeyShardSpec{modulus=4, residue=0}
provided(src_cp=1) = {≡1 mod 4} → route r1  KeyShardSpec{modulus=4, residue=1}
provided(src_cp=2) = {≡2 mod 4} → route r2
provided(src_cp=3) = {≡3 mod 4} → route r3
```

对称形态（prefill CP=4 → decode CP=4）下只有 `src_cp == dst_cp` 交集非空，退化成 4 条对角线 route，`residue = src_cp`——与今天的行为一致。

**`collapseResidues` 在白名单之后仍然必需**（唯一还需要它的场合）：源端 `cp_mapping = NONE`（复制）且 `cpSize() > 1` 时，选举出的那条 route 覆盖**全部**剩余类，必须折叠成 `{modulus=1, residue=0}`（全取），否则只会取到 `1/cpSize` 的键。见用例 B6。

decode rank0 在每个 `(layer, tag)` 上注册 **4** 个 recv task（key 分别为 `uk_L0_full_r0..r3`），
各自只装本 route 的键；两侧物理槽互不相同（prefill 侧 `logical/4`，decode 侧 `logical`），
join 键始终是 `cache_key`。对称 CP=4→CP=4 时只有 `j == k` 交集非空，退化为每个 dst_cp 恰好 1 条
route，与今天的行为一致（P1 影子比对即以此为等价基准）。

### Step 3b — 副本均分（ReplicaBalancer，可选优化）

Step 3 的 `min(providers)` 与 Step 5 的副本类选举是同一件事：**一组字节相同的副本里只选一个源**。既然是副本，就可以不选一个、而把键集**均分**给全部副本，把出口带宽摊到多张 NIC 上。

统一为 `ReplicaBalancer`：给定「字节相同的 provider 集合 + 键选择规则」，确定性地把键切给各 provider。

**必须是规则式，不能是显式键表**。prefill 不知道 decode 的 `needed`（取决于 prefix reuse 的 `block_range`），若为均分而下发显式 key 列表，就退回 §6 否掉的 100KB/请求 payload 问题。做法是给 `KeyShardSpec` 加一层：

```cpp
struct KeyShardSpec {
    int  modulus = 1, residue = 0;
    bool include_final_key = false;
    // 在上述剩余类之上再筛一层：仅保留 (pos / modulus) % count == index 的位置
    int  replica_split_count = 1, replica_split_index = 0;
};
```

零 payload 增长；prefix 部分命中时均分略有偏斜，但仍有效。

**收益分布（重要）**：

| 复制场景 | `needed` 规模 | 均分收益 |
|---|---|---|
| MLA head 副本（NP1D，prefill_tp=8→decode_tp=2） | 整条 block list（数百 key） | **大**：今天每 4 个 prefill rank 只有 1 个发，3/4 NIC 闲置 |
| `cp_mapping = NONE` 且 `cpSize > 1`（如 LINEAR） | `active_tail_blocks = 1` → **1 个 block** | **无**：1 个 key 分不给 4 个 provider |
| `cp_mapping = NONE` 因 `cpSize == 1` | 任意 | **无**：只有 1 个 provider |

即 CP 轴上 NONE 的均分机会很薄（LINEAR 只传尾块），真正的目标是 **MLA head 副本**。

**约束**：

- **确定性**：切分必须是 `(tag, dst_rank, provider 集合)` 的纯函数——不能有负载反馈、时间戳、随机数，否则两侧分歧 ⇒ key 不匹配 ⇒ TIMEOUT。
- **粒度** ≥ 1 个 cache_key，无法再细分。
- **route 数 × |providers|**：61 层 × 4 副本 = 244 个 recv task（vs 61）。需与 `kMaxOutstandingAsyncSendTasksPerRequest`（=8）及 recv store 压力权衡 ⇒ 建议 **per-group opt-in + 阈值**（仅当 `needed.size() >= threshold × |providers|` 才拆），默认关闭。
- 因此本项列为 **P5**，在 P0–P4 稳定之后再评估，不进入首版。

> **关于「复制 ⇒ 字节相同」这个前置条件**（此前列为待查，现已收敛）：
> - MLA 有代码背书：`P2PConnectorWorkerPrefill.cc:621`「KV cache is identical across all TP ranks」。
> - `COMPACT_LAST_RANK` **不涉及 rank 归属**：`physicalBlockPosition` 的 COMPACT 分支完全不看 `cp_rank`，
>   `buildCacheStorePlan` 的 compact 分支也没用 `cp_rank_`，即每个 rank 注册的 `(key_index, offset_index)`
>   完全一致。名字里的 `LAST_RANK` 指的是**键的选取**（用 RR 下本该归属 rank `N-1` 的那个逻辑位置作为
>   segment 代表，见 `canonicalCacheKeys` 从 `cp_size - 1` 起步），不是「数据只住在最后一个 rank」。
>   因此选 `cp_rank 0` 是对的。
> - 进一步：若各 rank 在那些槽位上字节不同，现有的 legacy cache-store 路径早已出错（它同样只从某一个
>   rank 读）。所以这不是本设计引入的新风险。
> - **真正需要区分的是有无 `cp_slice`**：COMPACT **带** `cp_slice` 时各 rank 持有不同字节段 ⇒ 不选举、
>   全取（用例 A9）；不带时才是复制 ⇒ 选举（用例 A8 / B6）。

关键结论：

> **decode 侧的 recv 注册必须按「源 shard 身份」拆分，而不是按 TP partition 下标拆分。**

今天 `recv_partition_count = remote_tp_size / local_tp_size` 之所以够用，仅因为 CP 被强制对称、TP 是唯一不对称维度。一般化后，decode rank `d` 在某个 `(layer, tag)` 上的 recv task 数 = 覆盖它的 route 数，每个 task 的键集 = `needed ∩ provided`（键集包含由构造保证）。

`needed ∩ provided` 编码为 `KeyShardSpec{modulus = src.cpSize(), residue = src_cp}`（经 `collapseResidues` 可能被折叠成更粗的模）：prefill 按此规则从自身投影筛选，decode 按此规则从本地 `layer_blocks` 筛选，规则一致 ⇒ 键集一致。

`mapping` 不进入 `KeyShardSpec`——它已经被 planner 折叠进 `(modulus, residue)`；只有 COMPACT 的非周期部分需要 `include_final_key` 单独承载。因此 `resolveKeys` 退化为极简且可证一致的实现：

```cpp
std::vector<size_t> resolveKeys(const KeyShardSpec& s, size_t count) {
    std::vector<size_t> out;
    for (size_t pos = s.residue; pos < count; pos += s.modulus) out.push_back(pos);
    if (s.include_final_key && count > 0 && (count - 1) % s.modulus != s.residue) out.push_back(count - 1);
    if (s.tail_count > 0 && out.size() > s.tail_count) out.erase(out.begin(), out.end() - s.tail_count);
    return out;   // 已升序、无重复
}
```

**`count` 必须是全序列的 `cache_keys` 数量，不是 `block_range` 窗口长度。** 这与
`LayerCacheBufferUtil::convertLayerTag` 的既有约定一致——它把 `cache_keys.size()` 传给
`physicalBlockPosition`，而 `[start_block_idx, +block_count)` 只是**额外**的窗口裁剪。
prefill 侧不知道 decode 的 `block_range`（prefix 部分命中的结果），若两侧用不同的 count，
`include_final_key` 与 `tail_count` 会算出不同的键，破坏键集包含契约。窗口裁剪只在 decode
侧叠加在 `resolveKeys` 结果之上。

> `active_tail_blocks`（LINEAR 只传尾块、SWA 传末两块）**必须由编排层折进 `KeyShardSpec::tail_count`，
> 不能让两侧各自筛**。`buildCacheStorePlan` 的 `start = total - tail_count` 里的 `total` 是**本侧**的
> 块数，而两侧 compact 程度可能不同（prefill compact `cp_size`→1、decode 不 compact），
> 「最后 N 个」在两侧会指向**不同的 key**，破坏键集包含契约。Step 1 因此还要校验两侧
> `active_tail_blocks` 相等。见用例 B7 / B8 / B9。

### Step 4 — D3：head 维度与字节切片

**逐 tag 处理**，设 `SH = src.headShardCount(tag)`、`DH = dst.headShardCount(tag)`。注意二者是 **per-group** 的（§4 `GroupLayout::head_shard_count`）：MLA / Opaque spec 恒为 1，MHA / Linear spec 为该侧的 `get_attn_tp_size()`。因此**不再需要 `is_mla || is_hybrid` 特判**——它自然落入下面的 `SH == DH == 1` 分支。

- **`SH == DH == 1`（head 轴退化：MLA / Opaque）**：partition 参数本就被 `MemoryLayoutStrategy` 忽略
  - `src_partition = dst_partition = {1,0}`，head 配对是唯一的 `(0,0)`
- **`SH == DH > 1`** → `src_partition = dst_partition = {1,0}`，head 配对为对角线 `(h,h)`
- **`SH > DH`（NP1D）**，`n = SH / DH`（要求整除）
  - `src_partition = {1,0}` ← **修正 §2.3**：源发整块
  - `dst_partition = {n, src_head_shard % n}`，配对 `dst_head_shard = src_head_shard / n`
- **`SH < DH`（ND1P）**，`n = DH / SH`（要求整除）
  - `src_partition = {n, dst_head_shard % n}`，`dst_partition = {1,0}`
  - 配对 `src_head_shard = dst_head_shard / n`

> **本步骤不含任何选举。** v2 曾在 `SH > DH` 分支里写「若 head 轴是复制的则副本集内选举」——
> 改成 per-group `head_shard_count` 之后该分支**永远不会触发**：head 轴复制的 spec（MLA / Opaque）
> 其 `head_shard_count` 恒为 1，故必然落在 `SH == DH == 1`，不可能出现 `SH > DH`。
> 真正的选举统一在 Step 5 的 rank 内容坐标层面完成。

**CP 字节切片正交叠加，但只施加在「持整块」的那一侧**，且用**对端**的 CP 几何：

`OpaqueKVCacheSpec::isPrefillCpSliced` 仅对 `RoleType::PREFILL` 返回 true —— 即 prefill 的 **spec 本身已经是切片后的**（本地 block 就是 1/N），decode 持整块。legacy `sliceCpDestinationForPeer` 也只切目的端、从不切源端。因此：

| 情形 | `src_slice` | `dst_slice` |
|---|---|---|
| src 预切片（prefill spec 已 sliced），dst 持整块 | `{NONE,1,0}`（本地 block 即切片） | `{mode, src.cpSize(), src_cp}` |
| src 持整块，dst 预切片 | `{mode, dst.cpSize(), dst_cp}` | `{NONE,1,0}` |
| 两侧同构（都整块 / 都预切片） | `{NONE,1,0}` | `{NONE,1,0}` |

`GroupLayout` 因此需要一个 `pre_sliced` 标志（来源同 `isPrefillCpSliced`），Step 1 的全局字节校验也要带上该因子：

```
effective_global(side, tag) = kv_block_stride_bytes
                            * (is_mla||is_hybrid ? 1 : headShardCount())
                            * (pre_sliced ? cpSize() : 1)
require effective_global(src, tag) == effective_global(dst, tag)
```

执行侧复用 `CPSlotMapper::sliceBlockForPeer`。

#### stride 与 payload：两个「一个 block 多少字节」

`PAYLOAD_BYTES` 是全链路唯一「两侧用不同量」的地方，所以必须把这两个量分清：

| 量 | 含义 | 计算 |
|---|---|---|
| **payload** | 实际有效数据 | `payloadBytes() = entry_count × entry_elems × sizeof(entry_dtype)` |
| **stride** | 相邻两个 block 在显存里的间距 | `blockStrideBytes()`：`override > 0` 时直取；否则 `alignment > 0 && entry_count >= min_entries` 时 `roundUp(payload, alignment)`；再否则 `= payload` |

`stride ≥ payload`，差值是尾部对齐填充。代码位置：

| 环节 | 位置 |
|---|---|
| desc 上的三个控制字段 | `KVCacheSpecDesc.h:70-72`（`block_stride_bytes_override` / `block_stride_bytes_alignment` / `block_stride_alignment_min_entries`） |
| 计算 | `OpaqueKVCacheSpec.h` 的 `blockStrideBytes()`；BLOCK_STRIDE 模式另走 `fixedStateBlockStrideBytes()` |
| 落到 group | `GroupBase::kv_block_stride_bytes`（`CacheTopology.h`），由 `CacheConfig.cc:67-68` 从 `spec->block_size_bytes()` 赋值 |
| payload 定义 | `KVCacheSpecBase.h:111` 的 `k_block_payload_bytes()` |
| 实际消费 | `MemoryLayoutStrategy.cc:221` `makeBlockInfo(..., config_.kv_block_stride_bytes)`；`getBlockPtr` 里的 `tensor.stride(0)` |

**错配的成因**：发送侧 `createBasicBlockInfo` 用的是 **stride**（含填充），而 `sliceBlockForPeer` 的
`PAYLOAD_BYTES` 分支用的是 **payload**。若预切片侧 `stride > payload`，源端会发出比目的端落点更大的
数据。举例（`alignment = 4096`）：

```
全局：payload 260000  ->  stride roundUp(260000,4096) = 262144   （填充 2144）
prefill 预切片(entries/=4)：payload 65000 -> stride roundUp(65000,4096) = 65536（填充 536）
prefill 发 65536 B，decode 为它留 payload_global/4 = 65000 B  ->  65536 != 65000  ->  size mismatch
```

Step 1 因此对预切片侧硬性要求 `stride == payload`。

#### DSv4 的实际配置：该校验不会误伤

`rtp_llm/models/deepseek_v4.py:107-146` 逐 tag 配置：

| tag | `prefill_slice_layout` | `cp.slice` | `block_stride_bytes_alignment` | 结论 |
|---|---|---|---|---|
| `indexer_state` | PAYLOAD | `PAYLOAD_BYTES` | **未设 → 0** | `stride == payload` ✓ |
| `csa_state` | PAYLOAD | `PAYLOAD_BYTES` | **未设 → 0** | `stride == payload` ✓ |
| `hca_state` | PAYLOAD | `PAYLOAD_BYTES` | **未设 → 0** | `stride == payload` ✓ |
| `swa_kv` | BLOCK_STRIDE | `EQUAL_BYTES` | 条件性设置 | 切 stride、两侧同量；`fixedStateBlockStrideBytes` 已 `CHECK(full_stride % cp_size == 0)` ✓ |

三个 PAYLOAD 模式的 tag 都**没有**设 `block_stride_bytes_alignment`，于是 `blockStrideBytes` 的对齐分支不生效、直接 `return payload_bytes`。（`deepseek_v4.py:148-150` 只兜底设了 `block_stride_alignment_min_entries`，**没设 alignment**，不触发对齐。）

⇒ **Step 1 的 `stride == payload` 校验是纯保险，不误伤任何现有配置**；若将来有人给 PAYLOAD tag 配上 alignment，会在编排期显式报错而不是到传输层变成 `BUFFER_MISMATCH`。

> 佐证 `tail_count` 不是假想需求：`hca_state` 同时设了 `desc.tail.active_tail_blocks = 1`
> （`deepseek_v4.py:137`），且它本身是 PAYLOAD 切片 group —— 尾块限制与字节切分会叠加，
> 正是用例 A9 + B7 覆盖的组合。

### Step 5 — 展开到 rank 对：内容坐标 + 副本类选举

**这是唯一做选举的地方。** 一个 rank 在某 tag 上的「内容坐标」= `(cpRank(r), headShard(r, tag))`；
坐标相同的 rank 互为字节相同的副本。目的端每个 rank 都必须被喂到（各自独立内存），
源端则在副本类内选举一个。

```
for dst_rank in [0, dst.rankCount()):
    dst_coord = (dst.cpRank(dst_rank), dst.headShard(dst_rank, tag))
    dst_class = 全部与 dst_coord 相同的 dst rank（升序）
    i = dst_rank 在 dst_class 中的下标；k = |dst_class|

    for cp_pair where cp_pair.dst_cp == dst_coord.cp:
      for head_pair where head_pair.dst_head == dst_coord.head:
        src_coord = (cp_pair.src_cp, head_pair.src_head)
        src_class = 全部与 src_coord 相同的 src rank（升序）；m = |src_class|
        if src_class 为空 -> 错误
        src_rank  = src_class[(i * m) / k]      // 按比例散开，而非恒取第一个
        emit route(route_id++, src_rank, dst_rank, tag, cp_pair.keys, head_pair.partitions, slices)
```

比例选举 `(i · m) / k` 的意义：把出口散开而不是全压在副本类的第一个 rank 上，
且**恰好复现今天的行为**。以 MLA `src.rankCount()=8, dst.rankCount()=4` 为例，
两侧 head 轴与 CP 轴均退化 ⇒ 单一副本类 `m=8, k=4` ⇒ `dst d ← src class[2d]`，
即 `dst0←src0, dst1←src2, dst2←src4, dst3←src6`，与 `decode_servers[tp_rank / local_partition_count]`
一致（用例 A6）。

> 为什么选举必须放在这一层而不是 Step 3/4：CP 轴与 head 轴各自的退化（`cpSize()==1` 或
> `head_shard_count==1`）都会把多个 rank 压到同一个内容坐标上，只有在 rank 全集上按
> **联合**坐标分类才能识别出真正的副本集。分轴选举会漏掉「两轴都退化、8 个 rank 全是副本」
> 这种情形（正是 MLA 无 CP 的常态）。

逐 route 校验（用 §4 推导出的 stride 数值精确计算）：

```
bytes(spec, partition, slice) = kv_block_stride_bytes / partition.count / slice.count （+ scale 分量同比例）
require bytes(src) == bytes(dst) && subBlockCount(src) == subBlockCount(dst)   // K/V (+ K/V scale)
```

任一条不成立 → 返回错误，`asyncRead` / `sendKVCache` 直接失败并附完整 route 描述。传输层的 `BUFFER_MISMATCH` 由此退化为纯粹的内存/协议损坏告警。

---

## 6. 协议改动：先加后删（分两个 slice）

**实现时修正**：字段删除**不能和新增放在同一次改动里**。`layer_blocks` / `remote_tp_size` /
`allow_empty_projection` 一旦删掉，`P2PBroadcastClient`、`P2PConnector::executeRead`、两侧 worker
会同时编不过 —— 整个重构被迫一次落地、无法分片验证。所以：

- **slice 1（= P1）**：只**新增** `TransferRoutePB` + `routes` + `plan_digest`，旧字段全部保留。
  新旧路径并存，plan 可以先影子比对而不驱动执行。本 slice 独立可编译、可测。
- **slice 2（= P2）**：切换执行路径之后，再删 `remote_tp_size`（唯一消费者
  `calculateRecvPartitionCount` 已删）与 `allow_empty_projection`（被「`routes` 为空」取代），
  并把 `layer_blocks` 移进 `TransferRoutePB`（blocks 改为 per-route）。

两个 slice 合起来仍是净减少，但必须分开落。

跨端协议（`GetPeerInfo` / `StartLoad`）零改动；intra-side 的 rank0→worker 广播 **+1 / −2**：

| 协议 / 字段 | 改动 | 理由 |
|---|---|---|
| `GetPeerInfoResponsePB` | **无** | `cp_size` 保留，但语义从「输入」改为「**断言**」：按 §3.2 decode 本可从本端配置推出 prefill 的 cp_size，此字段是唯一能独立验证「prefill 运行时配置 == decode 的假设」的信号，删掉会让本端推导变成不可验证的假设 |
| `P2PConnectorStartLoadRequestPB` | **无** | — |
| `P2PConnectorBroadcastTpRequestPB.routes` | **+1** | 新增 1 个 message + 1 个字段。使 worker 成为纯执行器 |
| `P2PConnectorBroadcastTpRequestPB.remote_tp_size`（8） | **−1** | 唯一消费者 `calculateRecvPartitionCount` 被删 |
| `P2PConnectorBroadcastTpRequestPB.allow_empty_projection`（10） | **−1** | 它只为区分「rank0 故意投影空集」与「请求畸形」。有了 routes：`routes` 空 = 权威的「本 worker 无任务」；`routes` 非空而 `layer_blocks` 空 = 畸形。`P2PConnector.cc:466` 的 `allow_empty_projection() \|\| cp_size <= 1` 条件塌缩为「routes 空 ⇒ 直接返回 Ok」 |
| `P2PConnectorBroadcastTpRequestPB.peer_workers`（2） | 留 | route 只带 `dst_rank`，仍需索引表解析成 `(ip, port)`；routes 数 ≥ workers 数，索引表比把端点内联进每条 route 更省 |

```proto
message TransferRoutePB {
    int32 route_id = 1; int32 src_rank = 2; int32 dst_rank = 3; string cache_tag = 4;
    int32 src_key_mapping = 5; int32 src_key_cp_size = 6; int32 src_key_cp_rank = 7;
    int32 src_partition_count = 8;  int32 src_partition_id = 9;
    int32 dst_partition_count = 10; int32 dst_partition_id = 11;
    int32 src_slice_mode = 12; int32 src_slice_count = 13; int32 src_slice_index = 14;
    int32 dst_slice_mode = 15; int32 dst_slice_count = 16; int32 dst_slice_index = 17;
}
```

**为什么 `routes` 值得加**：它是 **rank0 → 本侧 worker** 的同部署单元协议（同一二进制、同一发布单元），加字段没有滚动升级代价；而跨端的 `GetPeerInfo` / `StartLoad` 才是真正会版本错配的地方，那两个一动不动。加了它，worker 才是**真正的纯执行器**。

唯一的替代方案是让 worker 用同一个 planner 自行推导自己的切片（复用现有 `remote_tp_size`），可省掉这个字段——但决策逻辑会重回 worker，与「编排做在两侧 scheduler 里」的前提相悖。**取 `routes`，并把 `remote_tp_size` / `allow_empty_projection` 一并删掉**，wire 净变简单。

`TransferRoutePB` 体积：route 数 = `CP 配对 × head 配对 × tag 数`，**与 layer 数无关**。对称 CP=8/TP=8 → 8 条；CP4→CP1 + TP 不对称 → 数十条；每条 ~40B。全量通常 < 2KB，最坏 < 10KB。（若把 `cache_keys` 展开进 route，61 层 × 数百键会到 100KB+/请求，这是「只下发规则不下发键集」的原因。）

---

## 7. 代码改动清单

| 位置 | 变更 |
|---|---|
| `plan/ShardLayout.h`、`plan/TransferPlan.h`、`plan/KVCacheTransferPlanner.{h,cc}` | 新增。纯函数 + plan 缓存 |
| `P2PConnectorSchedulerConfig` | `+ ParallelismConfig pc`、`+ ModelConfig`（或所需子集）、`+ is_mla/is_hybrid`，供 `ShardLayout::forPeer` 推导对端 |
| **`P2PConnectorSchedulerDecode::asyncRead`** | 删除 §2.1 的 CP 拒绝分支；加 §3.2.4 的漂移断言；调 planner；per-worker buffer 构建从「按 `worker_rank % cp_size` 投影」改为「按 `forDecodeRank(d)` 的 route 集投影」；route 列表随 `broadcastPerRank` 下发 |
| **`P2PConnectorSchedulerPrefill::sendKVCache`** | 新增：由 `decode_transfer_servers.size()` 推出 decode layout → 调同一 planner → `broadcast` 改 `broadcastPerRank`，逐 prefill worker 下发 `forPrefillRank(p)` |
| `P2PBroadcastClient::broadcastPerRank` | 支持携带 `routes`（两个方向共用） |
| `P2PConnectorWorkerPrefill` | 删除 `asymmetric_tp_util_` 与 MLA 主 rank 跳过；`tp_partition_ctxs` 由 route 列表取代；`writeByLayer` 的硬编码 `cp_rank` 投影改为按 route 的 `src_keys` 规则投影 |
| `P2PConnectorWorkerDecode` | 删除 `calculateRecvPartitionCount`；recv task 按 route 列表生成；key 用 `makeRouteLayerKey` |
| `AsymmetricTpUtil` | 逻辑迁入 planner（NP1D 的 `src_partition` 修正为 `{1,0}`）；类与 `AsymmetricTpUtilTest.cc` 一并删除 |
| `LayerCacheBufferUtil::convertLayerTag` | `cp_rank/cp_size` 参数替换为 `(KeyShardSpec, SliceSpec)`，内部调 `CPSlotMapper::sliceBlockForPeer`（补齐 §2.2） |
| `P2PKeyUtil::makePartitionLayerKey` | 删除（不留过渡期） |
| `P2PConnectorWorkerConfig::cp_size`、`P2PConnectorSchedulerConfig::cp_rank` | 删除：worker 不再本地投影；scheduler 用完整 `pc` 而非单个 rank |

**关键取舍**：编排层不下发 prefill 的 block id（那属于 prefill 自己的 allocator），只下发逻辑键集规则；prefill 通过自身 `LayerCacheBuffer`（本就以 `cache_key` 为键）解析到本地 block id。plan 是**逻辑命名空间**的契约，物理寻址仍各自本地完成。

---

## 8. 分阶段落地

每阶段独立可测、可回滚，且都不改变对称场景的线上行为。

- **P0 — 修 §2.3（可独立发布）**：`handleNP1D` 的 `local_partition_count → 1`、`local_partition_id → 0`，保留 `remote_partition_id` 用于 key 与目的端切分。补端到端字节数断言。这是唯一的现网正确性缺陷，不该等重构。
- **P1 — 引入 planner，影子比对**：实现 `ShardLayout`/`TransferPlan`/planner，在两侧 scheduler 里算 plan 但**不上线**——与现有独立推导结果逐字段比对，不一致只打 WARNING + 上报 metric。跑满线上流量一个周期，证明 planner 在对称场景与现状等价。
- **P2 — 切换执行路径（硬切）**：下发 routes，两侧改为按 route 执行，`makeRouteLayerKey` 取代 `makePartitionLayerKey`。删除 `AsymmetricTpUtil` / `calculateRecvPartitionCount` / MLA 跳过分支 / `remote_tp_size` / `allow_empty_projection`。
  不引入灰度开关：`routes` 所在的是同部署单元协议，P/D 两端同时升级，无需双路径共存——留开关反而要维护两套 key 命名。P1 的影子比对是**正确性**闸门（不是兼容性闸门），仍建议保留一个周期后再删。
- **P3 — 放开非对称 CP**：删除 §2.1 拒绝分支。planner 的 Step 3 已天然覆盖，无新逻辑。
- **P4 — 补齐 CP 字节切分**：`LayerCacheBufferUtil` 接入 `sliceBlockForPeer`，打通 fixed-region / SWA 的 `cp_slice`，与 legacy `loadCache` 对齐。
- **P5 — 副本均分（§Step 3b，可选）**：先验证「复制 ⇒ 字节相同」不变量，再对 MLA head 副本启用 `replica_split`。per-group opt-in + 阈值，默认关。收益集中在 MLA NP1D 的出口带宽，LINEAR/NONE 无收益。

---

## 9. 测试规格

测试文件：`rtp_llm/cpp/cache/connector/p2p/plan/test/KVCacheTransferPlannerTest.cc`。planner 是纯函数，不依赖 GPU，是本设计的主要质量保障。

**公共基线**：8 个 KV head、head 维 128、fp16（无 scale）、每块 64 token，据此全局块大小固定，各 rank 的本地块大小 = 全局大小 ÷ `get_attn_tp_size()`。MLA 基线下 KV 是单份 latent，不随 TP 切分。除显式说明外，拓扑只有一个 FULL group，采用其默认 policy（RR 映射、无字节切片、无尾块限制）。

---

### 组 A：`plan()` 产出的 route 集与字段

#### 用例 A1: 对称 TP、无 CP 的基线
##### 输入：
两侧都是 TP=8、不开 CP、非 MLA 的布局。
##### 调用：
调用 planner 的 `plan` 函数。
##### 断言输出或副作用：
返回类型为"正常"。
产出的 route 数等于 TP 大小，且是 prefill rank 与 decode rank 的一一对角配对。
每条 route 的键选择规则为"取全部逻辑位置"，源端与目的端都不做 head 切分、不做字节切片。
每条 route 两侧的字节数与子块数相等。
route 编号不重复。

#### 用例 A2: RR CP 对称
##### 输入：
两侧都是 TP=8、开启 CP 且 KV cache 按 CP 分片的布局。此时 head 不再切分，序列方向按 RR 分片。
##### 调用：
调用 planner 的 `plan` 函数。
##### 断言输出或副作用：
返回类型为"正常"。
route 集恰好是对角线配对——RR 下只有源、目的 CP rank 相同时键集才有交集。
每条 route 的键选择规则是"RR、按本 CP rank 取模筛选"。
两侧都不做 head 切分，字节数相等。
本用例的输出即现状行为的等价基准，P1 影子比对以它为对照。

#### 用例 A3: 非对称 CP（prefill 按 CP 分 4 片，decode 不分片）
##### 输入：
prefill 侧 TP=4、开 CP 分片；decode 侧单 rank、不分片。
##### 调用：
调用 planner 的 `plan` 函数。
##### 断言输出或副作用：
返回类型为"正常"。
产出的 route 数等于 prefill 的 CP 片数，而不是 1 条——这是与现状最大的行为差异。
全部 route 都指向同一个 decode rank，每条 route 的键选择规则对应一个不同的源 CP rank。
从 decode rank 的视角查询到的 route 数等于源片数；从任一 prefill rank 的视角查询到的 route 数为 1。
两侧字节数相等。

#### 用例 A4: ND1P（prefill TP 小于 decode TP）
##### 输入：
prefill TP=4、decode TP=8，都不开 CP，非 MLA。
##### 调用：
调用 planner 的 `plan` 函数。
##### 断言输出或副作用：
返回类型为"正常"。
每个 decode rank 恰好对应一条 route。
源端按 head 维切分后只发自己负责的那一份，目的端接收整块。
每条 route 两侧字节数相等。

#### 用例 A5: NP1D（prefill TP 大于 decode TP）—— §2.3 回归
##### 输入：
prefill TP=8、decode TP=4，都不开 CP，非 MLA。
##### 调用：
调用 planner 的 `plan` 函数。
##### 断言输出或副作用：
返回类型为"正常"。
每个 prefill rank 恰好对应一条 route，多个 prefill rank 汇聚到同一个 decode rank。
**源端发送整块、不做 head 切分**；由目的端按 head 维切分决定落点。这是本用例的回归核心。
每条 route 两侧字节数相等。
反向断言：源端不应出现二次切分——若源端也按 head 维切分，两侧字节数会差一倍，即 §2.3 的缺陷。

#### 用例 A6: MLA 的 NP1D —— 按目的分组内选举
##### 输入：
prefill TP=8、decode TP=4，两侧都是 MLA。
##### 调用：
调用 planner 的 `plan` 函数。
##### 断言输出或副作用：
返回类型为"正常"。
route 数等于 decode rank 数：每个 decode rank 的若干 prefill 副本中只选出一个发送方。
两侧都不做 head 切分（MLA 下 partition 参数本就被内存布局层忽略），字节数与子块数相等。
未被选中的 prefill rank 查询到的 route 为空，即无任务。
反向断言：route 集不应只有一条——若在全局范围内只选一个发送方，除第一个以外的 decode rank 都拿不到数据。

#### 用例 A7a: PREFILL_CP + CP 分片，MLA group —— §3.3 回归
##### 输入：
prefill 侧配置为「CP 由 PD 分离的 prefill 角色承担」这一方式（该方式被 CP 判定逻辑排除在外，故 attention TP 大小不会被压成 1），同时开启 KV cache CP 分片，TP=8；decode 侧 TP=8、不开 CP 分片。
拓扑只有一个 MLA spec 的 FULL group。
##### 调用：
调用 planner 的 `plan` 函数。
##### 断言输出或副作用：
返回类型为"正常"。
该 group 在两侧的 head 分片数都为 **1**——head 分片数是 spec 类型的属性，MLA 的 latent 在各 rank 间复制、不随 attention TP 切分，与该侧的 attention TP 大小无关。
prefill 的 CP 片数等于 TP 大小，序列按 RR 分片；decode 不分片。
**route 数为 `src.rankCount() × dst.rankCount()`（此处 8×8 = 64）**：decode 侧 8 个 rank 各需全量序列
（MLA 下每个 decode rank 都持完整 KV），prefill 侧 8 个 rank 各持 1/8，因此每个 decode rank 必须向全部
8 个 prefill rank 取数。这是 MLA + 单侧 CP 分片的固有 O(tp²) 代价，不是编排引入的额外开销。
每个 decode rank 的 8 条 route 覆盖模 8 的全部 8 个剩余类，无重叠、无遗漏。
**两侧都不做 head 切分**。
反向断言：若把 head 分片数取成"该侧的 attention TP 大小"这一全局标量，MLA 会被误判为 8 份 head 分片，进而落入需要 head 切分的分支，两侧字节数不等。这正是必须 per-group 计算 head 分片数的原因。

#### 用例 A7b: PREFILL_CP + CP 分片，MHA group —— 单侧不完备，必须拒绝
##### 输入：
同 A7a 的 prefill 侧配置，但拓扑改为一个 MHA spec 的 FULL group（该 spec 会按 attention TP 切分 head，且 FULL 的默认映射是 RR）。
##### 调用：
调用 planner 的 `plan` 函数。
##### 断言输出或副作用：
返回类型为"错误"。
错误信息指出该 group 在同一侧被 head 与序列**双重分片**，导致本地持有的数据不完备：某个 rank 只持有"属于自己那份 head"且"落在自己那个序列剩余类"的块，因此绝大部分 (head, 块) 组合无人持有。
不产出任何 route。
**关键点：字节维度的校验通不住这个错误**——两侧还原出的全局块大小相等，缺失发生在键维度，所以必须有独立的单侧完备性校验。

#### 用例 A8: hybrid 下 RR 与非 RR group 共存
##### 输入：
两侧都是 TP=4、开 CP 分片的 hybrid 布局。拓扑含两个 group：一个 FULL group 用 RR 映射，一个 SWA group 用 COMPACT 映射且不带字节切片。
##### 调用：
调用 planner 的 `plan` 函数。
##### 断言输出或副作用：
返回类型为"正常"。
FULL group 的 route 是对角线配对，键规则为 RR 取模。
SWA group 的 route 全部来自同一个 prefill rank——COMPACT 是复制型子集，全部 CP rank 都是候选源，选举后统一收敛到最小的那个。
反向断言：SWA group 的 route 数不应等于 CP 片数的平方；若不做选举就会产生这么多冗余 route。

#### 用例 A9: CP 字节切分（prefill 预切片、decode 持整块）
##### 输入：
prefill 侧 TP=4、开 CP 分片，某 group 采用 COMPACT 映射并按 payload 做 CP 字节切片，其 spec 本身已经是切片后的小块；decode 侧单 rank、持完整块。
##### 调用：
调用 planner 的 `plan` 函数。
##### 断言输出或副作用：
返回类型为"正常"。
route 数等于 CP 片数——字节切片场景下各源持有互不相同的字节区间，因此不做选举、全部取用。
**源端不再做字节切片**（其本地块本身就是切片）；**目的端按源端的 CP 几何切出落点偏移**。
两侧字节数相等。

#### 用例 A10: CP 与 TP 同时不对称（prefill 按 CP 分 4 片、decode 按 head 分 4 份）
##### 输入：
prefill 侧 TP=4、开 CP 分片——序列切成 4 份，head 复制，每个 rank 持全部 head 但只有 1/4 的序列。
decode 侧 TP=4、不开 CP——head 切成 4 份，序列完整，每个 rank 持 1/4 的 head 但全部序列。
##### 调用：
调用 planner 的 `plan` 函数。
##### 断言输出或副作用：
返回类型为"正常"。
**route 数是两个维度的乘积，即 16 条**——decode 的每个 rank 都需要"全部序列位置上属于自己那份 head"，而任一序列位置只存在于一个 prefill rank 上，所以每个 decode rank 必须同时向全部 4 个 prefill rank 取数。
任一 prefill rank 查询到的 route 数为 4（发给 4 个 decode rank，各取不同 head 份），任一 decode rank 查询到的 route 数也为 4（从 4 个 prefill rank 取，各取不同序列子集）。
每条 route 的键规则由**源** rank 的 CP 位置决定，head 切分参数由**目的** rank 的 head 位置决定——两个维度各自独立取值，互不影响。
源端按 head 维切分后发送，目的端接收整块；两侧字节数相等。
本用例验证 §4.1 的乘积关系：route 数 = CP 配对数 × head 配对数。同时也提示运行时代价——16 条 route 乘层数是对称场景的 16 倍传输单元。

#### 用例 A11: 两侧都分片且不相等 —— 不支持，必须拒绝
##### 输入：
prefill 侧 TP=2、开 CP 分片（序列切 2 份）；decode 侧 TP=4、开 CP 分片（序列切 4 份）。
##### 调用：
调用 planner 的 `plan` 函数。
##### 断言输出或副作用：
返回类型为"错误"，错误信息指出 CP 形态不受支持并带上两侧的 CP 片数。
不产出任何 route。
理由见 Step 3 的白名单：该形态需要「模 lcm 的剩余类 + 中国剩余定理」整套机制，
而 vLLM / SGLang 同样禁止两侧 CP 并存。

#### 用例 A11b: 两侧 CP 相等仍然支持
##### 输入：
两侧都是 TP=4、开 CP 分片。这是今天唯一被允许的形态，不能被 A11 的白名单误伤。
##### 调用：
调用 planner 的 `plan` 函数。
##### 断言输出或副作用：
返回类型为"正常"，route 集为对角线配对。
键规则的模数等于源端 CP 片数，剩余值等于源 rank 的 CP 位置。
两侧字节数相等。

---

### 组 B：键选择规则的展开

#### 用例 B1: 剩余类规则展开
##### 输入：
一条模数为 4、剩余值为 1 的键规则，以及一个总长为 8 块的序列。
##### 调用：
调用 planner 的 `resolveKeys` 函数。
##### 断言输出或副作用：
返回值为以剩余值为首项、以模数为公差、不超出序列长度的等差序列。

#### 用例 B2: 全取规则展开
##### 输入：
一条模数为 1 的键规则（即不做筛选），以及一个总长为 8 块的序列。
##### 调用：
调用 planner 的 `resolveKeys` 函数。
##### 断言输出或副作用：
返回值为全部逻辑位置。

#### 用例 B3: 尾键例外
##### 输入：
一条带"必须包含序列末位"标记的键规则，且序列长度使末位不落在该剩余类上。
##### 调用：
调用 planner 的 `resolveKeys` 函数。
##### 断言输出或副作用：
返回值既包含该剩余类内的全部位置，也包含整条序列的最后一个位置。
返回值升序且无重复——当末位恰好落在剩余类内时不应出现重复项。

#### 用例 B4: 解析为空不算错误
##### 输入：
一条键规则，配一个短到该规则筛不出任何位置的序列长度。
##### 调用：
调用 planner 的 `resolveKeys` 函数。
##### 断言输出或副作用：
返回类型为"正常"，返回值为空。
这是预期行为：`plan` 阶段按剩余类保守产出 route，某些 route 在特定序列长度下解析为空；两侧规则一致，故会一致判空，decode 不注册、prefill 不发送。

#### 用例 B5: 非对称 CP 的 route 集在真实长度下完备且互斥
##### 输入：
用例 A3 产出的 plan，配一个具体的序列长度。
##### 调用：
对该 decode rank 的每条 route 依次调用 `resolveKeys` 函数。
##### 断言输出或副作用：
各 route 解析出的键集两两不相交。
各 route 键集的并集等于该 decode rank 实际需要的全部键，无遗漏。

#### 用例 B6: 复制型源的键集折叠成「全取」
##### 输入：
源端某 group 的 CP 映射为复制（非 RR）且 CP 片数大于 1，目的端不分片，配一个具体的序列长度。
##### 调用：
调用 `plan` 后，对被选中那条 route 调用 `resolveKeys` 函数。
##### 断言输出或副作用：
选举后每个 decode rank 只有一条 route，被选中的源是 CP 位置最小的那个。
该 route 的键规则被**折叠成「全取」**（模数为 1），解析结果覆盖全部逻辑位置。
反向断言：若不做折叠，键规则会停留在「模 CP 片数、取某一个剩余类」，只能取到 1/CP 片数 的键。
这是白名单收敛之后 `collapseResidues` 唯一还必需的场合。

---

### 组 C：编排期前置校验

#### 用例 C1: head 分片数不整除
##### 输入：
两侧 head 分片数不成整数倍关系的布局。
##### 调用：
调用 planner 的 `plan` 函数。
##### 断言输出或副作用：
返回类型为"错误"，错误信息指出是 head 分片数不整除并带上两侧的具体数值。
不产出任何 route，不进入传输层。

#### 用例 C2: 两侧全局块大小不一致
##### 输入：
人为把一侧的块大小改成与另一侧不匹配的值，使两侧还原出的全局块大小不等。
##### 调用：
调用 planner 的 `plan` 函数。
##### 断言输出或副作用：
返回类型为"错误"，错误信息带上两侧还原出的全局块大小。
这条校验把 §2.3 那类字节数缺陷从传输期的 `BUFFER_MISMATCH` 提前到编排期。

#### 用例 C3: 两侧 MLA 标志不一致
##### 输入：
一侧标记为 MLA、另一侧不是。
##### 调用：
调用 planner 的 `plan` 函数。
##### 断言输出或副作用：
返回类型为"错误"，错误信息指出 MLA 标志不一致。

#### 用例 C4: 同名 group 的类型不一致
##### 输入：
两侧存在同名 group，但 group 类型不同。
##### 调用：
调用 planner 的 `plan` 函数。
##### 断言输出或副作用：
返回类型为"错误"，错误信息指出该 group 名与类型冲突。

#### 用例 C5: CP 布局漂移断言（scheduler 层）
##### 输入：
对端上报的 CP 片数与本端配置推导出的 CP 片数互相矛盾的一次读请求。
##### 调用：
调用 `P2PConnectorSchedulerDecode` 的 `asyncRead` 函数。
##### 断言输出或副作用：
返回类型为"错误"，错误信息同时带上上报值与本端推导值。
未注册待检查的传输上下文，未发起任何 RPC——失败停在入口。

---

### 组 D：属性测试

#### 用例 D1: 完备且互斥
##### 输入：
批量随机生成能通过前置校验的两侧布局，并各配一个随机序列长度。
##### 调用：
先调用 `plan` 函数，再对每个 decode rank 的全部 route 调用 `resolveKeys` 函数。
##### 断言输出或副作用：
**互斥性必须按「目的端字节区间」判定，不能只按键判定。** 当 head 轴被切分（或 CP 字节切分生效）时，
同一个 cache_key 会**合法地**出现在多条 route 上，各自写入目的块的不同字节段——例如 head 轴 8 切 1 时，
decode rank 需要的同一个键会由 8 条 route 各写 1/8 字节。因此判据是 `(键, dst_partition.id, dst_slice.index)` 三元组：

- 完备（键维度）：被覆盖到的键集合等于该 decode rank 实际需要的键集合
- 互斥且无空洞（字节维度）：每个键上出现的 `(partition.id, slice.index)` 格子两两不同，
  且格子数恰好等于 `dst_partition.count × dst_slice.count`
- 同一个键上各 route 声明的 `(partition.count, slice.count)` 必须一致，否则目的块被按两种不同网格划分

#### 用例 D2: 两侧形状相等
##### 输入：
同上。
##### 调用：
调用 `plan` 函数。
##### 断言输出或副作用：
每条 route 两侧的字节数相等，且子块数量相等。

#### 用例 D3: 幂等与稳定
##### 输入：
同一组输入连续两次。
##### 调用：
调用 `plan` 函数两次。
##### 断言输出或副作用：
两次产出的 route 逐字段相等且顺序一致，摘要值相等。

#### 用例 D4: 两端独立求值结果一致
##### 输入：
同一组布局，分别构造成"decode 端持本端布局 + 从本端配置推导出的对端布局"和"prefill 端持本端布局 + 推导出的对端布局"两份输入。
##### 调用：
在两端各调用一次 `plan` 函数。
##### 断言输出或副作用：
两端产出的 route 逐字段相等（含 route 编号顺序），摘要值相等。
本用例是"跨端协议零改动"的正确性根据——它若失败，说明对端布局无法由本端配置推导，必须回退到由对端上报布局。

#### 用例 D5: 复制型 group 的选举结构（P5 前置）
##### 输入：
随机生成属于复制型、不带字节切片、且 CP 片数大于 1 的 group。
##### 调用：
调用 `plan` 函数。
##### 断言输出或副作用：
该 group 下每个 decode rank 只有一条 route，说明选举生效。
被选中的源始终是 CP rank 最小的那个。
本用例只验证选举结果的结构；"复制意味着字节相同"这个语义前提需在 spec 层另行断言（见 Step 3b 的说明）。

---

### 组 E：集成测试

#### 用例 E1: 非对称 CP 端到端
##### 输入：
在现有 `P2PConnectorTest` 里构造 prefill 按 CP 分片、decode 不分片的两端 mock。
##### 调用：
走完整的读取链路：发起异步读、通知 prefill 开始装载、prefill 侧广播、实际传输、目的端落盘拷贝。
##### 断言输出或副作用：
返回类型为"正常"。
拷贝阶段不出现缓冲区不匹配、也不出现"请求中缺少某个 cache key"。
decode 侧为每个层与 group 注册的接收任务数等于源 CP 片数。
传输 key 中带有 route 编号后缀，各 route 互不冲突。

#### 用例 E2: NP1D 端到端字节数（§2.3 回归，可在 P0 阶段独立落地）
##### 输入：
prefill TP 大于 decode TP、非 MLA 的两端 mock，单个逻辑块。
##### 调用：
走完整的读取链路。
##### 断言输出或副作用：
返回类型为"正常"。
每次传输声明的长度与目的端缓冲区大小一致。
拷贝阶段不出现尺寸不匹配的日志。

## 10. 收益小结

1. **删除**「两侧独立推导必须碰巧相同」这一整类耦合：`AsymmetricTpUtil` + `calculateRecvPartitionCount` + MLA 双端跳过规则 → 一个纯函数，在两侧 scheduler 各调一次。
2. **跨端协议零改动**：对端 layout 由本端 `ParallelismConfig` 替换后重算 `localKvHeadNumForSpec` 得到，复用「`prefill_cp_config` 部署级同配」这个既有前提，并加显式漂移断言。
3. **解锁**非对称 CP（§2.1）与 CP 字节切分（§2.2），使 P2P 链路达到 legacy cache-store 链路的能力覆盖。
4. **修复**非 MLA NP1D 字节数缺陷（§2.3）。
5. **纠正**五处推导错误（均由具体化用例逼出，见 §9）：
   - §3.3 rank 坐标：CP 与 head 分片是两条**独立**轴，不是 `tp/cp` 的嵌套关系（用例 A7）
   - Step 3 复制型 group 必须**选举**唯一源，否则 `cpSize²` 条冗余 route（用例 A8）
   - `COMPACT_LAST_RANK` 是**复制型子集**而非 rank 独占——`physicalBlockPosition` 的 COMPACT 分支完全不看 `cp_rank`（用例 A8）
   - MLA 选举必须**按 dst 分组内**进行，全局选 `headShard==0` 会让 `DH-1` 个 decode rank 拿不到数据（用例 A6）
   - `cp_slice` 只施加在**持整块的那一侧**、且用**对端**的 CP 几何；prefill spec 本身已预切片（用例 A9）
   - **CP 形态收敛为白名单**：只支持 `dst.cpSize() ∈ {1, src.cpSize()}`。两侧都分片且不相等不予支持（与 vLLM / SGLang 同样禁止两侧 CP 并存的取舍一致），因此不需要 lcm / CRT（用例 A11 / A11b）
   - `headShardCount` 必须 **per-group** 而非 per-side：是否按 attention TP 切分 head 是 **spec 类型的属性**（`MHAKVCacheSpec` / `LinearKVCacheSpec` 会除 `get_attn_tp_size()`，`MLAKVCacheSpec` / `OpaqueKVCacheSpec` 不会）。改成 per-group 后 Step 4 的 `is_mla || is_hybrid` 特判自然消失（用例 A7a）
   - Step 1 需要一条**单侧完备性**校验：同一 group 不能同时被 head 分片与 block RR 分片，否则缺 `(N-1)/N` 的数据。字节维度校验抓不到它（两侧还原出的全局块大小相等）（用例 A7b）
   - 选举必须在 **Step 5 的 rank 联合内容坐标** 层面做，不能拆成 CP 轴选举 + head 轴选举：改成 per-group `head_shard_count` 后，Step 4 里的 head 轴选举分支永远不触发，而「两轴都退化、全部 rank 互为副本」（MLA 无 CP 的常态）只有联合坐标才识别得出（用例 A6）
   - D1 的互斥判据是 `(键, dst_partition.id, dst_slice.index)` 三元组，不是键本身——head 轴被切分时同一个键合法地出现在多条 route 上（用例 D1）
   - 三处实现时才暴露的必需辅助逻辑：剩余值折叠、FULL group 无空洞校验、COMPACT 尾键唯一性（见 Step 3）

---

## 11. 容量估算：route 数放大的是控制面，不是数据面

放开非对称 CP（P3）后 route 数会成倍增长，需要先确认控制面撑得住。**字节总量不变**——
`总字节 = total_blocks × layers × block_bytes`，与切成几条 route 无关；`KeyBlockInfo` 总数也不变
（`= total_blocks × layers`），只是被分摊到更多 task 里。放大的是**每条 route × 每层产生一个
`transfer::SendRequest`（一次 arpc 调用）+ 一个 `TransferTask`（一个 map 条目）**。

以 A7a 为基准：MLA、61 层，**prefill 8 个 rank（开 CP 分片，即这 8 个 rank 全部用作 CP 分片）→ decode 8 个 rank（不开 CP 分片）**。

> **CP 不是额外一维，是从 TP 里划出来的**：`cpSize() = kv_cache_sharded ? tp_size : 1`，只能取
> 1 或 `tp_size`；`rankCount() = tp_size`。所以「tp=8 且开 CP 分片」= **8 个 rank**，不是 64 个。
> 下文的 `64` 是 `8 个 prefill rank × 8 个 decode rank` 的路由配对数。
>
> 两侧展开：prefill rank `r` 持有逻辑 block `≡ r mod 8` 的**完整 latent**（MLA 的
> `head_shard_count` 恒为 1，head 轴退化，只在序列方向分片）；decode 8 个 rank 的 `cpRank` 与
> `headShard` 全为 0，互为字节相同的副本，**每个都需要全量序列** ⇒ 每个 decode rank 必须向全部
> 8 个 prefill rank 取数。

| | 对称基线（A2） | A7a | 放大 |
|---|---|---|---|
| 全局 route 数 | 8 | 64 | 8× |
| 单个 decode rank 的 recv task 数 | 61 | **488** | 8× |
| 单个 prefill rank 的 send 次数 | 61 | **488** | 8× |

### 拓扑形态：这是一个 8×8 full mesh

每个 decode rank 都要向全部 8 个 prefill rank 取数才能凑齐全序列 —— 即 prefill ranks 与
decode ranks 之间形成 **full mesh**。但 "full mesh" 下面有四个量，代价差别很大：

| 量 | A7a | 说明 |
|---|---|---|
| 每个 prefill rank 的对端数 | **8** | `TcpClient::channel_map_` 按 `ip:port` 缓存 channel（带 idle TTL 淘汰），**跨请求复用、不是每请求建连** ⇒ 这项不是问题 |
| 集群级 `(src, dst)` 配对数 | **64** | mesh 的度 |
| 每个 prefill rank 的传输次数 | **488** | = 8 route × 61 层，被压力点 ① 串成 61 轮 |
| 每个 prefill rank 的总字节 | **1 份完整序列** | **与不开 CP 分片的基线相同**，见下 |
| 集群级网络字节 | 8 × 序列 KV | MLA 复制固有，**基线同样是 8×**，不是非对称 CP 引入的 |

**mesh 的成因可拆成两半，其中一半可优化：**

- **「8 个源」固有**：每个 prefill rank 只持 1/8 序列，要在任何一处凑齐全序列必须接触全部 8 个源。
- **「8 个目的」来自 MLA 复制，可优化**：decode 的 8 个 rank 各需一份完整 KV。可以只让 1 个 decode
  rank 从 8 个源拉齐（8 次网络传输），再经 NVLink 本地复制给其余 7 个 ⇒ **网络传输 64 → 8、
  网络字节 8× → 1×**，代价是多一次本地 broadcast 与一个同步点。**本设计不含此优化**，记录为后续方向。

**mesh 只出现在一个特定组合 —— 正是 P3 要放开的那个：**

| 场景 | 拓扑 |
|---|---|
| 不开 CP 分片（MLA，prefill 8 rank → decode 8 rank） | prefill 每 rank 都有全序列 ⇒ 每个 decode rank 只需 1 个源 ⇒ **8 条，无 mesh** |
| 对称 CP（用例 A2） | 两侧同样分片 ⇒ **对角线 8 条，无 mesh** |
| **源按序列分片 + 目的不分片（用例 A7a / A3）** | **full mesh** |

即 mesh 不是本编排引入的，是「非对称 CP」这个能力本身的拓扑代价。

### 五个压力点

1. **prefill dispatch 被 `kMaxOutstandingAsyncSendTasksPerRequest = 8` 串行化 —— 最实际的风险。**

   488 次传输**都携带真实 KV 数据**（TCP 后端字节在 `proto_block.content()` 里，RDMA 后端是一次
   RDMA op），但**总字节量与基线相同**：

   | | 基线（不开 CP 分片） | A7a | 差异 |
   |---|---|---|---|
   | 每 rank 传输次数 | 61（1 route × 61 层） | **488**（8 route × 61 层） | **8×** |
   | 每次传输的字节 | 一层的全序列 | 一层的 1/8 序列 | **1/8** |
   | 每 rank 总字节 | 1 份完整序列 | 1 份完整序列 | **不变** |

   （基线下每个 prefill rank 持完整序列、只发给配对的那个 decode rank；A7a 下只持 1/8 但要发给
   全部 8 个 decode rank —— 乘出来一样。）

   所以「控制面放大、数据面不变」的准确含义是：**同样的字节被切成 8 倍多的操作，每个 1/8 大小**。
   放大的是每次操作的固定开销：RPC framing、一次 `task_map_` 查找、一次回调、一个 outstanding 槽位。

   `outstanding = 8` 是 per-request（`transfer_result` 随 `sendKVCache` 一次调用）⇒ 基线 61/8 ≈ 8 轮，
   A7a 488/8 = **61 轮**，`waitForAsyncSendSlot` 走 cv 等待。但每轮只搬 1/8 字节，因此：

   - 纯带宽受限 ⇒ 墙钟**不变**（总字节相同）
   - 真实回归来自**每次操作的固定延迟不随大小缩小** ⇒ 最多多暴露 8 倍的**固定延迟分量**，
     而不是 8 倍总延迟

   风险大小取决于「一层的 1/8 序列」这个粒度落在 RDMA 的固定开销主导区还是带宽主导区。
   DSv4 MLA 单层 latent 不大，切成 1/8 后很可能进入固定开销主导区 —— **这是最该实测的点**。

   **必须同步修的一件事：outstanding 阈值要按 route 数缩放，否则 per-layer overlap 会失效。**

   现有实现已经有**真正的 per-layer 流水**（比 vLLM NIXL 的空实现 `wait_for_layer_load` /
   SGLang 的 chunk 级粒度都强，且不像 LMCache 那样要把 CUDA graph 切成 PIECEWISE）：

   ```
   prefill forward 第 k 层算完 -> writeByLayer(k, ..., torch::Event)
     -> StoreWaitContextChecker（1ms LoopThread）轮询 event->query()   // 非阻塞 CUDA event query
     -> ready 即进 ComputedLayerCacheBufferStore
     -> dispatchPendingLayerTransfers 取到就发，取不到则 waitChange(50ms)
   ```

   但 `sendLayerToPartitions` 里每条 route 都要过 `waitForAsyncSendSlot`，而该阈值
   **以「传输次数」计量，我们要的流水深度以「层数」计量**：

   | | 每层 route 数 R | outstanding 预算 | 有效流水深度 |
   |---|---|---|---|
   | 基线 | 1 | 8 | **8 层在飞** |
   | A7a | 8 | 8 | **~1 层在飞** |

   A7a 下一层的 8 条 route 正好填满窗口，第 `k+1` 层必须等第 `k` 层的回调开始返回才能 dispatch
   ⇒ 流水深度塌到 1 层，per-layer overlap 基本失效。

   修法：`outstanding_transfers = base × routes_per_layer`（等价于把阈值改成以**层**计量）。
   这不是内存回归——A7a 每次传输只有基线的 1/8 字节，8 倍 outstanding 数 ⇒ **in-flight 字节不变**，
   而该阈值的目的本就是限制在飞内存（每个 pending send 持一个 `LayerCacheBuffer` keepalive
   + 一个 `SendRequest`）。

   **这条属于 P2（接入层）而非 P3**：对称场景 R=1、阈值 8 恰好够用，问题不会暴露；等 P3 放开非对称
   CP 才发现流水没了，届时难以归因。

   > 仍存在的缺口（与 route 无关，今天就有）：decode 侧**没有** per-layer 消费——`buildRecvTasks`
   > 一次注册全部层，`waitRecvTasksWithReadDeadlinePolicy` 等全部 done 才返回。所以现有 overlap 是
   > 「prefill 计算 ∥ 传输」，不是「传输 ∥ decode 计算」。

2. **`TcpTransferService` 的 worker 队列硬编码为 20**（`TcpTransferService.cc:39-40`：
   `LockFreeThreadPool(worker_thread_count, 20, ...)` —— 注意**没有**用
   `cache_store_tcp_worker_queue_size`）。`waitCheckProc` 一次扫描可能收集到大量 ready context，
   `pushTask` 失败是**硬失败**（`"push transfer task to thread pool failed"`）而非背压。488 个
   recv task 在同一批匹配上时会踩到。仅影响 TCP 后端（生产走 RDMA）。

3. **decode 侧 `waitRecvTasksWithReadDeadlinePolicy` 的 O(n) 轮询**：每次 backoff 迭代（上限 8ms，
   `kBackoffCapMs`）扫描全部 task 调 `task->done()`，而 `done()` 要拿 `shared_lock`
   （`TransferTask.cc:12-15`）。488 task → 每 8ms 488 次锁获取；并发 32 请求就是 ~15.6k 次/8ms。

4. **`TransferTaskStore::task_map_` 是 `std::map<std::string, ...>` + `shared_mutex`**
   （`TransferTask.h:84-85`）。`addTask` 拿独占锁做红黑树插入（key 是字符串），488 次；
   deadline 前 `stealTask` 又要 488 次 erase。

5. **`queryLeaseStatus` 同样 O(n) 全扫**（rank0 轮询 lease 状态时），488 个 task 每次都扫。

### 不受影响的

`kSenderPoolQueueSize = 10000`、`rdma_transfer_worker_thread_count = 16`、
`messager_worker_thread_count = 16` 这些都够宽。

### 建议的验证与缓解

- 不必真跑 A7a：先做静态估算，再用现有 `P2PConnectorWorkerDecodeLeaseTest` 造一个 488-task 的用例，
  测轮询与锁开销。`TransferTaskStore::getTaskCount()` 已存在，可加 metric 观察 `task_map_` 峰值。
- 压力点 1 若成为瓶颈，`kMaxOutstandingAsyncSendTasksPerRequest` 是常量、可调；但要连带评估
  `kSenderPoolThreadCount = 4` 是否也该提。
- 压力点 3/5 的 O(n) 轮询可以改成「只扫未完成的 task」（维护一个未完成计数），属于局部优化。

### 对落地节奏的影响

**不阻塞 P0–P2**——那些阶段 route 数与今天一致（P1 是影子比对、P2 是对称场景切换执行路径）。
**只影响 P3**（放开非对称 CP）。可以在 P3 前把上面的估算做完，或者先上 P3 但对 route 数加一个
上限告警（`routes.size() > 阈值` 时打 WARNING + 上报 metric），先在小流量上观察。

---

## 12. 实现状态

已落地（P1 的 planner 部分）：

| 文件 | 内容 |
|---|---|
| `rtp_llm/cpp/cache/connector/p2p/plan/ShardLayout.h` | 两侧布局；per-group `head_shard_count`、`effectiveMapping/Slice`、`effectiveGlobalBlockBytes`、`forPeer` |
| `rtp_llm/cpp/cache/connector/p2p/plan/TransferPlan.h` | `PartitionSpec` / `SliceSpec` / `KeyShardSpec` / `TransferRoute` / `TransferPlan` / `PlanResult` |
| `rtp_llm/cpp/cache/connector/p2p/plan/KVCacheTransferPlanner.{h,cc}` | Step 1–5 + `resolveKeys` + `digest` |
| `rtp_llm/cpp/cache/connector/p2p/plan/test/KVCacheTransferPlannerTest.cc` | 31 个用例（A1–A11b、B1–B9、C1–C4/C6、D1–D5） |

测试 target **不带** `device_impl_target()` 与 `exec_properties gpu`，验证了「纯函数、CPU 单测」这一设计目标。

未落地：两侧 scheduler 的接入、worker 的执行改造、proto 的 `routes` 字段、§Step 3b 副本均分（P5）。
6. **前移**失败点：编排期确定性报错替代传输期 `BUFFER_MISMATCH`，绝大部分正确性验证搬到不需要 GPU 的纯函数单测里。
