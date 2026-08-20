# Staging Ring Admission 设计文档

## 目标

PD handoff 时，seq>16k 的长请求**准入瞬时**只分配 ~320 个 GPU 块（而非 ~1030 块），
使长请求在集群负载下能成功准入（消除 B0m 的 1 ok / 127 fail）。

## 现状（为什么 B 长请求全灭）

```
allocateResource()
  └─ initKVBlock() → 一次性 malloc 全前缀块（63k = 1030 块）作为 loadCache 落点
loadCacheFromPrefill()
  └─ loadCache() → 按 block_ids[] 逐块注册 GPU 地址 → cache_store RDMA 拉取写入
localGenerate()
  └─ decode 生成 ≥16 token 后 offloadPrefixBlocks() 收缩到 256 块
```

**问题**：瞬时峰值 1030 块在 W96 负载下永远超过 decode rank 可用容量（~2457 块总池，
短请求常驻 ~1000 块），RETRY_MS 窗口内等不到空窗 → 结构性失败。

## 方案：Staging 环准入

### 核心思路

handoff 时只分配**目标形态**的 GPU 块，前缀通过 staging 环**流式直落 host**：

```
准入分配（GPU 块，一次性）：
  block0 (1)  +  staging 环 (64)  +  16k 窗口尾部 (256)  =  321 块
  → 与请求长度无关！128k 请求也是 321 块

前缀传输路径（改 loadCache 内循环）：
  block_pos ∈ [0, N-256) 的前缀块：
    主 KV (5KB/token)  → staging 环块轮转写入 → D2H 到 host pinned store
    indexer-K (132B/token) → 直接写入 GPU 侧 indexer 池（打分要用）
  block_pos ∈ [N-256, N) 的尾部块：
    正常写入 16k 窗口 GPU 块（和 A 完全一致）
```

### 实现架构（所有权模型）

**为什么环形传输必须在引擎里做**：只有引擎能从 prefill 拉数据（cache_store RDMA/TCP）；
且 D2H+indexer-K 抽取必须发生在每次环刷新**之间**（环只有 64 块，整个前缀会把环覆盖
ceil(prefix/64) 次）。python 侧的 decode hook 在首个 decode 步才拿到请求，那时环早被覆盖多轮，
无法回收前面的批次。⇒ **准入镜像的产出必须由引擎的 loadCache 驱动。**

**两个构建产物的职责**：

| 产物 | 语言 | 职责 |
|------|------|------|
| 引擎 wheel（rtp-llm-rdma） | C++ | 准入：分配封顶、环块、loadCache 批量环形传输、host 镜像池 + GPU indexer 池的产出 |
| v32_ctx.so | python 侧 torch ext | decode 期：读引擎产出的镜像/池地址，跑打分/miss-fetch/写回（沿用现有机制） |

**跨语言边界接口**：引擎新增单例 `V32AdmissionStore`（C++），按 request_id 持有：
- host pinned 主 KV 镜像（per layer，[prefix_tokens, 576] bf16）
- GPU indexer-K 池块（per layer，[nb, 64, 132] uint8）
- watermark（已镜像的前缀 token 数）

经 pybind 暴露 `get_admission_mirror(request_id, layer) -> (host_tensor, idx_pool_tensor, watermark)`。
python 的 `v32_capacity._entry` 在首个 decode 步查询此接口，**adopt 引擎产出的张量**并注册进
v32_ctx（`ctx_register` 用引擎的 host 张量），替代原来自己跑 `_mirror_chunk` 镜像前缀的逻辑。

**为什么 host 镜像归引擎而非 python**：单一所有者、footprint 有界，正面改善 B6/B7 定罪的
"host pinned 多方争抢 + 无界增长"失效模式。

## 改动清单

### 关键实现发现（决定改动形态）

1. **`loadBuffers()` 是无状态的**：每次调用自带一组 `RequestBlockBuffer` 并返回独立
   `LoadContext`。⇒ 可以**分批串行调用**（每批 ≤64 个前缀块），批间插入 D2H/D2D 排空，
   环块随即复用。不需要改 cache_store 的回调链。
2. **现有内循环遇 NULL 哨兵自然跳过**：`loadCache` 里已有
   `if (isNullBlockIdx(block_id)) continue;`。⇒ 只要准入把前缀位置填成哨兵，
   原有传输路径**零改动**就只拉尾部窗口。

⇒ 改动形态：**不动 loadCache 内循环**，而是在其后追加一趟独立的"分批环形拉取前缀"。

| # | 文件 | 改动 |
|---|------|------|
| 1 | `V32AdmissionStore.{h,cc}`（新增） | 单例。per (request, layer) 持有 host pinned 主 KV 镜像 + GPU indexer-K 池块 + watermark；提供环块申请/归还；pybind 暴露给 python adopt |
| 2 | `StreamCacheResource.cc` `initKVBlock()` | 长请求走封顶分配：malloc 321 块 → 块表 resize 到 N，把 256 块搬到尾部位置、block0 留位置 0、前缀位置填 NULL(0) 哨兵；64 环块存 `staging_ring_`（不进块表） |
| 3 | `DecodeRpcServer.cc` | 新增 `loadPrefixThroughRing()`：在 `loadCache` 之后调用。按 64 块分批，每批构造指向环块地址的 `layer_caches` → `loadBuffers`+`waitDone` → 主 KV D2H 落 host 镜像、indexer-K D2D 落 GPU 池 → 环块复用下一批 |
| 4 | `StreamCacheResource.h/.cc` | 新增 `staging_ring_` 成员；`offloadPrefixBlocks()` 在准入已封顶时短路（准入即目标形态） |
| 5 | `v32_capacity.py` | `_entry()` 改为优先 adopt 引擎产出的镜像（`get_admission_mirror`），跳过自己的 `_mirror_chunk` 前缀镜像；`ctx_register` 用引擎的 host 张量 |

### 分批环形拉取伪代码

```cpp
// 在 loadCache() 成功返回后调用
ErrorInfo loadPrefixThroughRing(ctx, prefix_block_positions, ring_blocks) {
  auto& store = V32AdmissionStore::instance();
  store.prepare(request_id, layer_num, prefix_tokens);   // 分配 host 镜像 + idx 池
  const int R = ring_blocks.size();                      // 64
  for (size_t base = 0; base < prefix_block_positions.size(); base += R) {
    const size_t n = min(R, prefix_block_positions.size() - base);
    // 1) 构造本批 layer_caches：目标地址 = 环块，cache_key = 该前缀块的 key
    for (layer, tag) { for (j < n) {
        block_pos = prefix_block_positions[base + j];
        parts = convertIndexToBufferByTag(ring_blocks[j], layer, tag);  // 环块地址
        addBufBlock("kv_" + makeCacheKey(..., cache_keys[block_pos], layer, tag), parts[0]);
    }}
    // 2) 拉本批（阻塞到落地）
    loadBuffers(batch_caches, ...)->waitDone();  必须 success
    // 3) 排空环：主 KV D2H → host 镜像；indexer-K D2D → GPU idx 池
    store.drainRing(request_id, ring_blocks, n, base * 64 /*token offset*/);
    // 4) 环块自动复用（下一批覆盖写）
  }
}
```

瞬时 GPU 块占用 = 1 (block0) + 64 (环) + 256 (尾部窗) = **321 块，与请求长度无关**。

### 被否方案：host 直落

利用 `addBlock(..., gpu_mem=false)` 让 cache store 直接把前缀写进 host pinned
（cache store 原生支持 CPU 落点：RDMA 模式经 `regUserMr` 注册，TCP 模式直接 CPU tensor）。
GPU 块需求可压到 257 块（block0 + 256 尾部），且不用改 cache store 回调链。

**否决理由（两条，均致命）**：
1. **正撞稳定性墙**：B6/B7 的崩溃根因已定罪为 host pinned 镜像（3.9GB×并发长）与 PD 传输
   缓存店抢宿主资源 → `CACHE_STORE_PUSH_ITEM_FAILED` → 实例级联。host 直落把大量 host pinned
   在**准入瞬间**注册进 cache store，是把该失效模式直接放大。
2. **indexer-K 与主 KV 同物理块**（踩坑 3）：MLA 路径按整块拉取（单个 `kv_` key 覆盖整块，
   indexer-K 藏在 kv_scale 区），落 host 会把打分必需的 indexer-K 一起搬走，还需逐块 H2D 搬回。

### 选择：Staging 环（2026-08-19 定）

- host pinned footprint **有界**（复用现有 `_mirror_chunk` 的分片 D2H 机制），不放大 B6/B7 失效模式
- 环块是 GPU 块，整块拉取语义不变：主 KV 与 indexer-K 到齐后再在 GPU 上分流
  （主 KV→host 镜像，indexer-K→GPU 侧池），无需 H2D 回搬
- 代价：loadCache 需要"环块顺序填充 + D2H 交替"的分批调度，实现复杂度高于 host 直落

### 验证计划

1. 单机门禁：63k 请求 PREFIX 逐字一致 3/3，准入瞬时块占用 ≤330（目标形态 321 = 1+64+256）
2. 集群 B-new：秒拒语义下长请求成功率 >80%（vs B0m 的 0.8%）

### 环境变量

| 名称 | 默认 | 含义 |
|------|------|------|
| `RTP_KV_OFFLOAD_ADMISSION_THRESHOLD` | 0（禁用） | seq_len 超过此值启用 staging 准入 |
| `RTP_KV_OFFLOAD_KEEP_BLOCKS` | 256 | 尾部保留窗口 |
| `RTP_KV_OFFLOAD_STAGING_BLOCKS` | 32 | python staging LRU 块数（不变） |

### 时序与依赖

```
allocateResource()  →  只分配 257 块
    ↓
loadCacheFromPrefill()
    ├─ 尾部 256 块 → 正常 GPU 落点（RDMA→GPU）
    ├─ 前缀主 KV → host pinned store（RDMA→host / TCP→host）
    └─ 前缀 indexer-K → GPU indexer 池（RDMA→GPU / alloc from _ipool）
    ↓
localGenerate()
    ├─ prefix_offloaded_ = true（初始）
    ├─ python hook: store 已 populated，直接注册 ctx_register
    └─ 正常 decode（单波/双波打分均可）
```

## 判决记录（2026-08-19 晚）

### Br-smoke（64 请求）：链路全通
- 一条 63k 长请求完整走通：admission capped（353 块）→ mirror prepared（4.4GB host +
  0.5GB idxp）→ 环拉取 25s → python adopt → 解码 3.5min 完成 → release
- 发现并修复镜像释放 UAF（fetch 线程仍在读时引擎 free → 堆损坏）：release 改 30s 墓地（r2）

### Br0m（1000 请求，mixed W48 秒拒，keep256+ring64）
- 终值：长 **1 ok / 127 fail**（与 B0m 持平）；短 865/872
- cap 机制全程生效（拒绝日志 need_blocks=353 = 封顶目标），但被 reserve 预检拒：
  available 常年 150-453 < 353+reserve(128)=481

### 重大发现：decode 实例只有 dp_rank0 在接请求
- Br0m 全程 ranks 1-7 `reqs=0`（v32_sw 心跳）、instance_status 所有 running task dp_rank=0
- LB 只连 decode 单一 RPC 端口（27001=rank0），DP8 的 ranks 1-7 全程跑 dummy 行
- **decode 有效 KV 容量 = 名义值的 1/8（2451 块而非 19608 块）**——追溯解释：
  B2"单 rank 只容 ~7 并发长"、W96→W48 收益有限、"decode batch 含 dummy 行"等全部历史观测
- 修复路由（LB 分发到 8 个 rank RPC 或引擎内部 DP dispatch）= 8× 容量，独立课题

### Br3m（keep64 + ring32，2026-08-19 21:10 启动）
- 准入需求 1+32+64+32=129 块（+128 reserve = 257 阈值），压到单 rank 空闲底线之下
- **终值：长 2/128；短 551/872（321 个短 8211）；总 ok 553（Br0m 是 866）**
- 判读：更细粒度的准入让若干长请求入池并驻留数分钟，在 2451 块的单 rank 池里
  把短请求挤成大面积 8211——**池总量是硬约束，准入粒度只是重新分配失败对象**

### 总结论
1. **机制层面：staging 环准入已证明**（冒烟全链路 + Br0m 拒绝日志 need=353 确认封顶生效；
   准入瞬时需求 1030 → 353（16k 窗）/129（4k 窗），与请求长度无关）
2. **容量层面：判决被部署问题遮蔽**——decode 实例仅 rank0 接请求（LB decode 端点=每实例
   一个地址 27001，`rtp_cluster.py direct_endpoint`；引擎未做内部 DP 分发），
   W48 需求 ~3000 块 vs 单 rank 2451 块，任何准入策略都无法凭空造容量
3. **下一步排序**：修 decode 多 rank 路由/分发（8× 容量，独立课题）→ 之后 16k 窗
   ring 准入（need 481）在 ~2000 块/rank 空闲下将轻松通过，重跑 Br0m 即可出真判决
