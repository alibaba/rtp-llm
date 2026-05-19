# PD 分离 Gathered Cache Transfer 方案设计

## 1. 背景与问题

### 1.1 当前架构

RTP-LLM 的 PD（Prefill/Decode）分离架构中，Prefill 节点完成前向计算后，需要将 KV Cache 传输给 Decode 节点。传输通过 `CacheStore` 子系统完成，支持 TCP 和 RDMA 两种通信方式。

当前传输以 **单个 KV Cache block** 为基本单元：

- **Prefill 端**：`runtimeWriteCacheStore`（`ExecOps.cc`）逐层（layer）、逐请求（batch）地将每个 block 的 GPU 地址注册到 `RequestBlockBuffer` 中，然后交由 `NormalCacheStore::store` 处理。
- **Decode 端**：`DecodeRpcServer::loadCache`（`DecodeRpcServer.cc`）为每个 block 通过 `convertIndexToBuffer` 获取目标 GPU 地址，构建 `RequestBlockBuffer`，然后通过 `NormalCacheStore::loadBuffers` 拉取数据。

### 1.2 性能问题

在长序列场景下，单层的 block 数量可达数十甚至上百个。Per-block 的处理模式在两条链路上都存在显著开销：

| 链路 | 瓶颈 |
|------|------|
| **TCP** | Prefill 端每个 block 独立 `cudaHostAlloc` + `cudaMemcpyAsync D2H` + `cudaStreamSynchronize`；Decode 端每个 block 独立 `cudaMemcpyAsync H2D` + `cudaStreamSynchronize`。N 个 block = N 次 CUDA 操作 + N 次同步。 |
| **RDMA** | 每个 block 对应一个独立的 RDMA Write WQE。大量小 WQE 导致 NIC 处理效率低下。现有 `getConcatWriteBlocks` 优化只能合并物理地址连续的 block，而 Paged KV Cache 下 block 几乎不可能连续。 |

### 1.3 优化目标

**仅针对 RDMA 链路**进行优化；TCP 链路完全保持原有 per-block 路径不变。

1. **RDMA Server 端**：将每层每请求的所有分散 GPU block 按 partition slice **gather** 到一块连续 host memory（pool buffer），跳过 per-block `cudaHostAlloc` / per-block D2H / per-block MR 注册
2. **RDMA Write**：单次（或按 NIC 数 K 切片的 K 次）RDMA Write，将 pool buffer 整段写到 Decode 端 pool buffer，WQE 数从 N 降为 1~K
3. **RDMA Decode 端**：接收到连续内存后单次 H2D + 单次 scatter kernel，把数据 scatter 到各 GPU block
4. 支持**预分配 Host Memory Pool** 避免运行时内存分配 + MR 注册开销，适配不同层/region 的 block 大小差异
5. 通过配置项控制开关，支持关闭
6. **TCP 链路不在本方案范围内**：TCP 仍然走原有 per-block `makeValidBlock` + `writeResponseBlock` + per-block `execNoBlockCopy` 路径，本设计完全不影响 TCP 行为

## 2. 配置项设计

### 2.1 Python 配置层

在 `kv_cache_group_args.py` 的 `init_kv_cache_group_args` 中新增：

```python
kv_cache_group.add_argument(
    "--enable_gathered_cache_transfer",
    env_name="ENABLE_GATHERED_CACHE_TRANSFER",
    bind_to=(kv_cache_config, "enable_gathered_cache_transfer"),
    type=str2bool,
    default=False,
    help="PD 分离时是否启用 RDMA 聚合传输模式。开启后 RDMA server 在 serve 时将每层每请求的所有 KV block "
    "按 partition slice gather 到连续内存后再单次/分块 RDMA Write，显著减少 WQE 数量。仅影响 RDMA 链路，TCP 链路不受影响。",
)

kv_cache_group.add_argument(
    "--cache_transfer_buffer_size_mb",
    env_name="CACHE_TRANSFER_BUFFER_SIZE_MB",
    bind_to=(kv_cache_config, "cache_transfer_buffer_size_mb"),
    type=int,
    default=0,
    help="PD 分离聚合传输的 Host Memory Pool 预分配大小（MB），Prefill 和 Decode 各自独立分配。"
    "0 表示不预分配，按需临时申请 pinned memory。"
    "建议设为所有 region 中单次最大传输数据量的 4-8 倍。"
    "RDMA 模式下会对该 buffer 提前注册 MR，避免运行时注册开销。",
)
```

### 2.2 C++ 配置层

`rtp_llm/cpp/config/ConfigModules.h` — `KVCacheConfig` 新增字段：

```cpp
struct KVCacheConfig {
    // ... existing fields ...

    // Gathered cache transfer (RDMA-only): RDMA server gathers all blocks
    // per-layer-per-request into contiguous host memory before sending via
    // RDMA Write. TCP path is not affected.
    bool    enable_gathered_cache_transfer = false;

    // Pre-allocated host memory pool size in MB for gathered cache transfer.
    // 0 = allocate pinned memory on demand (no pre-allocation).
    // Non-zero = pre-allocate this amount and manage as a memory pool.
    // RDMA mode: the buffer is MR-registered at init time.
    int64_t cache_transfer_buffer_size_mb  = 0;
};
```

### 2.3 Pybind 注册

`rtp_llm/cpp/pybind/ConfigInit.cc` 中新增：

```cpp
.def_readwrite("enable_gathered_cache_transfer",
               &KVCacheConfig::enable_gathered_cache_transfer)
.def_readwrite("cache_transfer_buffer_size_mb",
               &KVCacheConfig::cache_transfer_buffer_size_mb)
```

### 2.4 开关生效矩阵

**注**：本开关仅影响 **RDMA 链路**。TCP 链路无论开关如何，始终走原有 per-block 路径。

| `enable_gathered_cache_transfer` | `cache_transfer_buffer_size_mb` | RDMA 链路行为 |
|---|---|---|
| `false` | 任意 | 走原有 per-block RDMA Write 路径，无改动 |
| `true` | `0` | RDMA 聚合模式，host staging 按需 `cudaHostAlloc` + 运行时 `regUserMr`。Server 端 partition-aware gathered D2H + 单/多 chunk RDMA Write，Decode 端 gathered H2D + scatter |
| `true` | `> 0` | RDMA 聚合模式 + Host Memory Pool 预分配 + MR 预注册 |

## 3. Host Memory Pool 设计

### 3.1 设计动机：为什么不用固定 Slot Ring Buffer

DSV4 等 HybridAttention 模型中，同一层可能对应多个 cache region，不同 region 的 block 大小差异巨大：

| Region（DSV4 七组） | Group ID | Block 大小特征 | 传输策略 |
|---|---|---|---|
| CSA_KV | 0 | KV entry × (tokens_per_block/4)，FP8 有 576B 对齐 | FULL: 传所有 block |
| HCA_KV | 1 | KV entry × (tokens_per_block/128)，远小于 CSA_KV | FULL: 传所有 block |
| INDEXER_KV | 2 | indexer_entry × (tokens_per_block/4) | FULL: 传所有 block |
| INDEXER_STATE | 3 | idx_state_dim × 2 × tokens_per_block (FP32) | SWA: 只传最后 2 个 block |
| CSA_STATE | 4 | csa_state_dim × 2 × tokens_per_block (FP32) | SWA: 只传最后 2 个 block |
| HCA_STATE | 5 | hca_state_dim × 2 × tokens_per_block (FP32) | SWA: 只传最后 2 个 block |
| SWA_KV | 6 | KV entry × tokens_per_block（所有层共享） | SWA: 只传最后 2 个 block |

一次 `runtimeWriteCacheStore` 调用对应**一个 layer_id + 一个 region_name**（即一个 group），每次的总传输字节数取决于该 region 的 block 大小 × 该次的 block 数量，在不同调用之间差异可达 **10-100 倍**。

固定 Slot 大小的 Ring Buffer 会出现两个问题：
1. Slot 按最大 region 设定 → 小 region（如 STATE 类只传 2 个小 block）浪费严重
2. Slot 按平均设定 → 大 region 无法放入，频繁降级

因此改用**标准内存池设计**，按需分配所需大小的连续缓冲区。

### 3.2 核心数据结构

新增文件 `rtp_llm/cpp/disaggregate/cache_store/CacheTransferBufferPool.h`：

```cpp
class CacheTransferBufferPool {
public:
    // 分配结果
    struct BufferHandle {
        void*  ptr   = nullptr;  // 缓冲区起始地址（在 pool 内部的偏移）
        size_t size  = 0;        // 实际分配大小（>= 请求大小，对齐后）
        size_t offset = 0;       // 在 pool 中的偏移（用于释放）
    };

    // pool_size: 预分配 pinned host memory 总大小
    // memory_util: 用于 RDMA MR 注册（可为 nullptr）
    CacheTransferBufferPool(size_t pool_size,
                            const std::shared_ptr<MemoryUtil>& memory_util);
    ~CacheTransferBufferPool();

    // 分配 size 字节的连续缓冲区（16B 对齐），非阻塞
    // 返回 nullptr 表示 pool 空间不足
    BufferHandle* tryAllocate(size_t size);

    // 阻塞分配，带超时（毫秒），超时返回 nullptr
    BufferHandle* allocateWithTimeout(size_t size, int64_t timeout_ms);

    // 释放缓冲区
    void free(BufferHandle* handle);

    // 状态查询
    size_t totalBytes() const;
    size_t freeBytes() const;

    // 基地址（RDMA MR 信息查询）
    void* baseAddr() const;

private:
    void*  base_ = nullptr;         // cudaHostAlloc 整块 pinned memory
    size_t pool_size_ = 0;

    // Free list: 按 offset 排序的空闲段列表
    struct FreeBlock {
        size_t offset;
        size_t size;
    };
    std::list<FreeBlock> free_list_;  // 按 offset 有序

    std::mutex              mutex_;
    std::condition_variable cv_;
    size_t                  free_bytes_ = 0;

    std::shared_ptr<MemoryUtil> memory_util_;
    bool mr_registered_ = false;
};
```

### 3.3 分配算法：Best-Fit + 合并

```
allocate(requested_size):
  aligned_size = ALIGN_16(requested_size)

  // Best-Fit: 从 free_list_ 中找能容纳 aligned_size 的最小空闲段
  best = free_list_.end()
  for it in free_list_:
    if it.size >= aligned_size:
      if best == end OR it.size < best.size:
        best = it

  if best == end:
    return nullptr  // pool 不足

  // 分割
  handle = new BufferHandle{base_ + best.offset, aligned_size, best.offset}
  if best.size > aligned_size:
    best.offset += aligned_size
    best.size   -= aligned_size
  else:
    free_list_.erase(best)

  free_bytes_ -= aligned_size
  return handle

free(handle):
  // 归还并与相邻空闲段合并（前后合并）
  insert FreeBlock{handle.offset, handle.size} into free_list_ (ordered by offset)
  merge with prev/next if contiguous
  free_bytes_ += handle.size
  cv_.notify_all()
  delete handle
```

### 3.4 内存布局示例

```
Pool (一块连续 pinned host memory, 整块 MR 注册):
┌──────────────────────────────────────────────────────────┐
│ [Alloc: CSA_KV 2MB] [Free 512KB] [Alloc: STATE 32KB]    │
│ [Free 1.5MB] [Alloc: HCA_KV 256KB] [Free 4MB]          │
└──────────────────────────────────────────────────────────┘
^                                                          ^
base_                                               base_ + pool_size
```

- 不同大小的请求从同一个 pool 分配
- 释放后自动合并相邻空闲段，减少碎片
- RDMA MR 覆盖整个 pool，任意子段都可直接用于 RDMA 操作

### 3.5 初始化

```cpp
if (kv_cache_config.enable_gathered_cache_transfer
    && kv_cache_config.cache_transfer_buffer_size_mb > 0) {
    size_t pool_size = kv_cache_config.cache_transfer_buffer_size_mb * 1024ULL * 1024ULL;
    buffer_pool_ = std::make_unique<CacheTransferBufferPool>(
        pool_size, cache_store_->getMemoryUtil());
}
```

### 3.6 Pool 满场景处理

#### 三级降级策略

```
tryAllocate(size)
  ├─ 成功 → 正常聚合路径
  └─ 失败 → allocateWithTimeout(size, 100ms)
               ├─ 成功 → 正常聚合路径（等待其他传输完成释放空间后可用）
               └─ 超时 → 降级到临时分配
                         ├─ cudaHostAlloc 临时 pinned memory
                         ├─ RDMA: memory_util->regUserMr 临时注册
                         ├─ 使用完毕后释放
                         └─ 上报 metric (gathered_transfer_pool_fallback_count)
```

#### 降级场景的资源管理

```cpp
struct TempStagingGuard {
    void* ptr = nullptr;
    size_t size = 0;
    std::shared_ptr<MemoryUtil> memory_util;
    bool is_rdma = false;

    ~TempStagingGuard() {
        if (ptr) {
            if (is_rdma) memory_util->deregUserMr(ptr, false);
            cudaFreeHost(ptr);
        }
    }
};
```

### 3.7 碎片控制

长时间运行可能产生外部碎片。缓解措施：

1. **Best-Fit 策略**：优先选择最小够用的空闲段，减少碎片产生
2. **前后合并**：释放时自动与相邻空闲段合并
3. **对齐粒度**：所有分配 16B 对齐，减少小碎片
4. **Pool 容量建议**：设为最大单次分配的 4-8 倍，保证并发下有足够余量

### 3.8 Pool 容量估算

#### Prefill 端（per fire 粒度）

Prefill server 每次 watch fire 处理一个 (layer, region) 的 blocks，pool 单次分配 = `N_blocks × ALIGN_16(slice_size)`。

DSV4 H20 32K 长序列示例（典型）：
- CSA_KV：~16 blocks × 32KB slice ≈ **512KB**（最大单次）
- HCA_KV：~16 blocks × 1KB slice ≈ 16KB
- INDEXER_KV：~16 blocks × 8KB slice ≈ 128KB
- STATE 类：2 blocks × ~4KB ≈ 8KB

**Prefill pool 建议大小** = `max_single_alloc × concurrent_writes × 4-8x`，例 `512KB × 8 并发 × 6 ≈ 24MB`。**`cache_transfer_buffer_size_mb=64` 起步**。

#### Decode 端（per request 粒度，注意 combine_load_=true）

Decode 在 `makeLoadRequest` 一次性为整个 request 分配 buffer，覆盖**所有 layer × 所有 region × 所有 block 的 slice**：

```
total_recv_bytes_per_request
    = Σ_layer Σ_region (N_blocks × ALIGN_16(slice_size))
    ≈ layer_num × avg_region_per_layer × avg_blocks × avg_slice
```

DSV4 80 层 32K 长序列示例：
- 80 × 5 × 16 × 平均 8KB ≈ **50MB / request**
- 极端长序列（128K）下可能到 **200MB / request**

**Decode pool 建议大小** = `avg_request_size × concurrent_requests × 1.5-2x`（碎片余量），例 `100MB × 16 并发 × 2 ≈ 3.2GB`。**`cache_transfer_buffer_size_mb=4096` 起步**。

#### 关键观察

- Decode 端 pool 单请求消耗远大于 Prefill 端（因 `combine_load_=true` 把全请求合到一次分配）
- Decode pool 的并发上限直接由 `pool_size / avg_request_size` 决定；高并发场景需相应增大
- 监控 `gathered_pool_used_bytes / total > 0.85` 时主动扩容
- 若实测 `gathered_pool_fallback_count` 持续上涨，说明 pool 太小，降级路径性能反而比原 per-block 更差

## 4. Gather/Scatter Kernel 复用分析

### 4.1 可用 Kernel

已有两套 gather/scatter kernel（`sm_copy_kernel.cu`），均可复用：

| Kernel | 适用场景 | 特点 |
|--------|---------|------|
| `launch_gather_copy_split` / `launch_scatter_copy_split` | 定长 block（同一模型同一层的 block 大小一致） | 接口简单，只需 kv_cache/scale 指针表和固定 stride |
| `launch_dsv4_memory_cache_gather_copy_var_nooffset` / `launch_dsv4_memory_cache_scatter_copy_var_nooffset` | 变长 block（DSV4 多 region/不同大小） | 更通用，需要 sizes[] 和 offsets[] 数组 |

### 4.2 推荐选择

- **常规场景**（MHA/MLA，同一层所有 block 大小一致）：使用 `split` 版本
- **DSV4 多 region 场景**（不同 region 的 block 大小不同）：使用 `var_nooffset` 版本
- 统一通过 `execStagedMemoryCopy`（`NoBlockCopy.cc`）调用，它已经封装了完整的 gather + D2H / H2D + scatter 流程

### 4.3 `execStagedMemoryCopy` 复用、`host_is_pinned` 扩展、async 接口

该函数已实现的完整流程：

```
D2H 方向：
  1. 构建 device metadata (ptrs, offsets, sizes)
  2. launch gather kernel: 多个 GPU block → device_staging
  3. cudaMemcpyAsync: device_staging → host_staging (pinned)
  4. cudaStreamSynchronize
  5. 如有 host_segments: unpack from host_staging to caller buffers

H2D 方向：
  1. 如有 host_segments: pack from caller buffers into host_staging
  2. cudaMemcpyAsync: host_staging → device_staging
  3. 构建 device metadata
  4. launch scatter kernel: device_staging → 多个 GPU block
  5. cudaStreamSynchronize
```

`StagedMemoryCopyScratch` 支持跨调用复用（容量只增不减），**不**持有任何 `cudaEvent_t` 字段。

#### Async 接口：`execStagedMemoryCopyAsync`

新增异步版本，与同步版本流程相同但**不调用 `cudaStreamSynchronize`**，并通过 `cudaLaunchHostFunc` 在 stream 完成时触发用户 callback：

```cpp
// 接口签名
bool execStagedMemoryCopyAsync(
    const StagedMemoryCopyParams& params,
    StagedMemoryCopyScratch* scratch,
    std::function<void(bool success)> on_done);
```

**实现要点**：

```cpp
bool execStagedMemoryCopyAsync(params, scratch, on_done) {
    // ... 准备 metadata、device_staging（同步版相同）...
    // ... launch gather/scatter + cudaMemcpyAsync（不 sync）...

    // 用 cudaLaunchHostFunc 在 stream 末尾插入 host callback
    // 依赖 stream 顺序：gather → memcpy → host_func 严格按序触发
    auto* cb_holder = new std::function<void(bool)>(std::move(on_done));
    cudaError_t err = cudaLaunchHostFunc(stream, [](void* ud) {
        auto* cb = static_cast<std::function<void(bool)>*>(ud);
        // stream 顺序保证之前的 op 都完成
        // 用 cudaGetLastError() 同时获取并清除 sticky error，避免污染后续调用
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            handleStreamError(e);                     // 见下
            (*cb)(false);
        } else {
            (*cb)(true);
        }
        delete cb;
    }, cb_holder);

    if (err != cudaSuccess) {
        // launch 时已有 sticky error；同样用 cudaGetLastError 清掉
        (void)cudaGetLastError();
        delete cb_holder;
        return false;
    }
    return true;
}
```

**`handleStreamError` 错误分类**（关键：必须区分 fatal vs recoverable，否则线程从此失效）：

```cpp
static void handleStreamError(cudaError_t e) {
    switch (e) {
        // === Fatal：context 已损坏，必须 abort ===
        case cudaErrorIllegalAddress:
        case cudaErrorLaunchFailure:
        case cudaErrorMisalignedAddress:
        case cudaErrorHardwareStackError:
        case cudaErrorIllegalInstruction:
        case cudaErrorInvalidPc:
        case cudaErrorECCUncorrectable:
            RTP_LLM_LOG_FATAL("unrecoverable CUDA error: %s", cudaGetErrorString(e));
            std::abort();   // CUDA context 已损坏，无法恢复

        // === Recoverable：清掉 sticky 后线程可继续 ===
        case cudaErrorMemoryAllocation:               // OOM，可降级重试
        case cudaErrorTooManyPeers:
        case cudaErrorInvalidValue:                   // 参数错误，单次失败
        default:
            RTP_LLM_LOG_WARNING("recoverable CUDA error: %s, stream cleared",
                                cudaGetErrorString(e));
            // sticky 已通过 cudaGetLastError 清除，下次调用可正常继续
            return;
    }
}
```

**为什么不用 `cudaEventRecord`**：之前考虑过在 scratch 中持有 `completion_event`，由 callback 等待。但同一线程在 callback 未触发时再次发起 async 调用会覆盖 event，破坏 callback 的语义。`cudaLaunchHostFunc` 本身按 stream 顺序排队，每次调用插入一个独立 host func，driver 保证按顺序触发，无需 event。`StagedMemoryCopyScratch` 不需要 `completion_event` 字段。

**Stream 错误处理总结**：
- 用 `cudaGetLastError()`（不是 `cudaPeekAtLastError()`）同时**获取并清除** sticky error，避免错误污染线程后续所有 CUDA 调用
- 区分 fatal vs recoverable：fatal（IllegalAddress / LaunchFailure 等）context 已损坏，立即 `abort` 防止后续静默错误结果；recoverable（OOM / InvalidValue 等）单次失败上报后继续

#### `host_is_pinned` 扩展

聚合路径中 `params.host_base` 来自 Memory Pool（`cudaHostAlloc` 预分配），本身已是 pinned memory。原 `execStagedMemoryCopy` 实现会无条件 `memcpy` 到 scratch 的 `host_staging`（另一块 pinned memory）后再做 `cudaMemcpyAsync`，造成冗余拷贝。

扩展 `StagedMemoryCopyParams` 增加 `host_is_pinned` 标记：

```cpp
struct StagedMemoryCopyParams {
    void*                                    host_base      = nullptr;
    size_t                                   host_bytes     = 0;
    bool                                     host_is_pinned = false;  // NEW
    std::vector<StagedMemoryCopyHostSegment> host_segments;
    std::vector<StagedMemoryCopyTile>        tiles;
    int                                      device_index   = -1;
    StagedMemoryCopyDirection                direction      = StagedMemoryCopyDirection::H2D;
};
```

`execStagedMemoryCopy` 内部行为变化：

```
if params.host_is_pinned AND params.host_segments.empty():
  // host_base 已是 pinned memory，跳过 scratch.host_staging 中转
  D2H: cudaMemcpyAsync(params.host_base, device_staging, D2H)
  H2D: cudaMemcpyAsync(device_staging, params.host_base, H2D)
else:
  // 原路径：通过 scratch.host_staging 中转
  D2H: cudaMemcpyAsync(scratch.host_staging, device_staging, D2H)
       → cudaStreamSync → copyPinnedStagingToHost(params)
  H2D: copyHostToPinnedStaging(params, scratch.host_staging)
       → cudaMemcpyAsync(device_staging, scratch.host_staging, H2D)
```

当 `host_is_pinned = true` 时，scratch 不需要分配 `host_staging`（只需 `device_staging` + metadata），减少 pinned memory 占用并消除一次 MB 级 memcpy。

## 5. 数据流设计

### 5.1 数据流

#### RDMA 路径（完整 gathered 流程，partition-aware）

```
═══════════════════════ Prefill 端 ═══════════════════════

  1. Store 阶段（setRequestBlockBuffer）:
     GPU blocks 注册为 gathered-ready → watch 立即触发 server

  2. Server 收到 load request（已知 partition_count / partition_id）:

  GPU Paged KV Blocks: [blk0] [blk7] [blk3] ...    ← 分散，每个 block 含所有 partition 数据
          │
          │  partition-aware gather kernel:
          │  从每个 block 的 [partition_id * slice_size] 偏移取 slice_size 字节
          ▼
  GPU Device Staging:  [====连续 N slices====]      ← 只含该 partition 的数据
          │
          │  cudaMemcpyAsync D2H (1 次，host_is_pinned 跳过 scratch 中转)
          ▼
  Host Pool Buffer:    [====连续 N slices====]      ← pinned, MR 已注册
          │                                           size = N × ALIGN_16(slice_size)
          └──[RDMA]─→ 1 次 RDMA Write（proto 协商: gathered_recv_info 指定目标地址）
                      绕过 getConcatWriteBlocks，直接 doWriteBlock

═══════════════════════ Decode 端 ═══════════════════════

  RDMA Write 直接写入 Host Pool Buffer（地址由 gathered_recv_info 指定）
          │
          ▼
  Host Pool Buffer:    [====连续 N slices====]      ← pinned, MR 已注册
          │                                           size = N × ALIGN_16(slice_size)
          │  cudaMemcpyAsync H2D (1 次，host_is_pinned 跳过 scratch 中转)
          ▼
  GPU Device Staging:  [====连续 N slices====]      ← StagedMemoryCopyScratch.device_staging
          │
          │  scatter kernel (D2D, 1 次 launch)
          ▼
  GPU Paged KV Blocks: [blk0] [blk7] [blk3] ...    ← 分散
```

**TP 不对称场景**：Prefill TP=P, Decode TP=D (P ≠ D)。`partition_count = P/D`（或 D/P），每个 Decode 节点请求自己的 partition slice。Prefill server 在 D2H gather 时按 `partition_id` 偏移取 `full_block_size / partition_count` 字节，pool buffer 天然是该 partition 的连续数据，无需二次 compact。

**TCP 链路**：保持原 per-block 路径，本节不再赘述。

### 5.2 与原路径的操作数对比（仅 RDMA）

| 指标 | 原路径 (per-block) | 聚合路径 (per-layer) |
|------|-------------------|---------------------|
| CUDA kernel launch | 0 | 2（gather + scatter） |
| cudaMemcpyAsync 次数 | N（每 block 一次） | 2（一次 D2H + 一次 H2D） |
| cudaStreamSynchronize | N | 2 |
| cudaHostAlloc 次数 | N（逐 block 分配） | 0（Memory Pool 预分配） |
| RDMA WQE 数 | N | 1（或按 NIC 切片为 K） |
| pinned memory 碎片 | 严重（N 块小内存） | 无（Memory Pool 整块预分配） |
| MR 注册次数 | 每请求 N 次查找 | 0（预注册） |

## 6. 各模块改造设计

### 6.1 Prefill 端 Store 路径

#### 改造点：`RequestBlockBufferStore::setRequestBlockBuffer`

这是 GPU blocks → 合法传输 blocks 的唯一汇聚点，在 CUDA event 完成后被调用。

#### `BlockBuffer` 增加 `kind_` 字段

为了支持 RDMA 聚合路径下的 GPU blocks 直接注册（绕过 `isValidBlock` / `makeValidBlock`），在 `BlockBuffer` 增加 `kind_` 字段显式区分 block 类型：

```cpp
class BlockBuffer {
public:
    enum class Kind {
        HOST_PINNED,         // 传统路径：host pinned memory（TCP / 原 RDMA）
        GPU_GATHERED_READY,  // RDMA 聚合路径：GPU 地址，server 在 serve 时 D2H
        GPU_DIRECT,          // 原 RDMA 直写 GPU 路径（如已存在的 GPUDirect）
    };
    // ... existing fields ...
    Kind kind_ = Kind::HOST_PINNED;  // 默认保持现状
};
```

`isValidBlock` 适配：

```cpp
bool RequestBlockBufferStore::isValidBlock(const std::shared_ptr<BlockBuffer>& block) {
    if (block->kind_ == BlockBuffer::Kind::GPU_GATHERED_READY) {
        return true;   // GPU blocks 不需要 host 转换，直接 pass through
    }
    if (memory_util_->isRdmaMode()) {
        return memory_util_->isMemoryMr(block->addr.get(), block->len, block->gpu_mem, block->adopted);
    }
    return block->gpu_mem == false;
}
```

> **注**：拿到 `kind_ == GPU_GATHERED_READY` block 的下游模块（metrics / prefix cache 复用 / 等等）必须按 `kind_` 分支处理；此类 block 仅供 `RdmaCacheStoreServiceImplContext::loadBlockOnRdma` 在 serve 时消费。其他模块通过 `RequestBlockBufferStore::getBlockBuffer` 拿到 GPU 地址时若误用会段错误，需在调用处加 assert。

#### TCP 链路：完全不改动

TCP 链路始终走原 `makeValidBlock` per-block 路径（per-block `cudaHostAlloc` + per-block D2H + per-block 注册到 `RequestBlockBuffer`）。本聚合方案不引入任何 TCP 侧的改动。`is_tcp_mode` 由 `!memory_util_->isRdmaMode()` 判定。

#### RDMA 链路：延迟 D2H，注册 GPU blocks 为 gathered-ready

RDMA gathered 模式下，D2H 延迟到 server 收到 load request 后再执行。此时 `partition_count` 和 `partition_id` 已知，可以在 gather 阶段就做 partition 切分，pool buffer 只存放该 partition 的连续 slices。

```cpp
setRequestBlockBuffer(request_block_buffer):
  blocks = request_block_buffer->getBlocks()

  bool is_tcp_mode = !memory_util_->isRdmaMode()
  if (!enable_gathered_cache_transfer || is_tcp_mode) {
      → 原路径（逐 block isValidBlock / makeValidBlock）
      return
  }

  // ===== RDMA 聚合路径：不做 D2H，标记为 GPU_GATHERED_READY =====
  std::vector<std::shared_ptr<BlockBuffer>> gathered_blocks
  for (auto& [key, block] : blocks) {
      // 复制一份并改 kind_（不修改原 block，避免影响其他持有者）
      auto gb = std::make_shared<BlockBuffer>(*block)
      gb->kind_ = BlockBuffer::Kind::GPU_GATHERED_READY
      gathered_blocks.push_back(gb)
  }
  store_buffer->addBlocks(gathered_blocks)
  return
```

`addBlocks(gathered_blocks)` 走的是**已有的** `RequestBlockBuffer::addBlocks` 接口（无需新增重载）。`isValidBlock` 因 `kind_ == GPU_GATHERED_READY` 直接返回 true，跳过 `makeValidBlock`。watch 回调立即触发，server 收到这些 blocks 后通过 `kind_` 识别走 RDMA 聚合 serve 路径，在收到 load request 后做 partition-aware D2H。

`execStagedMemoryCopyAsync` 接口与实现见 §4.3。

关键设计决策：

- **D2H 延迟到 serve 时执行**：partition 配置在 load request 到达后才已知；延迟后可一次性 partition-aware gather，pool buffer 大小恰为 N × ALIGN_16(slice_size)。
- **store 路径轻量化**：`setRequestBlockBuffer` 仅做注册，不做 GPU 数据搬运，store 线程立即返回。
- **保持 per-key 的 `BlockBuffer` 结构不变**：上层 watch 机制、key 匹配逻辑不需要改动。
- **GPU staging scratch 按 thread_local 管理**：每个线程拥有独立的 scratch 实例，无锁竞争（见 §9.2）。

#### Store callback 语义变更（重要）

`NormalCacheStore::store(...)` 的 callback 触发时机在 RDMA 聚合模式下含义变化：

| 模式 | callback(true) 时机 | 含义 |
|------|--------------------|------|
| TCP / 原 RDMA per-block | `setRequestBlockBuffer` 同步返回后 | 数据已 staging 到 host pinned，可被 server serve |
| **RDMA 聚合** | blocks 已注册到 RequestBlockBuffer 后 | **blocks 已注册（GPU 地址）；实际 D2H 由 server 在收到 load request 后执行** |

**影响**：
- 上层若有"store callback = GPU paged blocks 可释放"的假设（典型场景：prefill 完成后立即重用 GPU 显存），在 RDMA 聚合模式下**不再成立** — 此时数据仍在 GPU，提前释放会导致 RDMA serve 时读到脏数据。
- **强烈建议**：上层切到 RDMA 聚合模式时，将"store done"信号改为"store done + load done"双信号，等 Decode 端 load 完成后再释放 GPU 块。
- 上层若仅用 callback 做 metrics/log，无影响。

### 6.2 Prefill Server 端响应（仅 RDMA）

> **TCP 链路不在本节范围**：TCP server 走原 `TcpCacheStoreServiceImplContext::loadBlockOnTcp` + `writeResponseBlock` 路径，不受本方案影响。

#### `loadImpl` 入口校验

`RdmaCacheStoreServiceImpl::loadImpl` 入口需校验 Prefill 与 Decode 的协商是否一致：

```cpp
void RdmaCacheStoreServiceImpl::loadImpl(...) {
    // 已有：每个 block 必须有 rdma_info（非 gathered 模式）
    // 改为：根据 gathered_mode 分支校验

    if (request->gathered_mode()) {
        if (!kv_cache_config_.enable_gathered_cache_transfer) {
            RTP_LLM_LOG_WARNING("peer requests gathered_mode but feature disabled, "
                                "request_id=%s peer=%s",
                                request->requestid().c_str(), request->client_ip().c_str())
            response->set_error_code(EC_FAILED_INVALID_REQ)
            done->Run(); return
        }
        if (!request->has_gathered_recv_info() ||
            request->gathered_block_metas_size() == 0) {
            response->set_error_code(EC_FAILED_INVALID_REQ)
            done->Run(); return
        }
        // gathered 模式不需要 per-block rdma_info（peer 只用合并写）
    } else {
        // 原校验：每个 block 必须有 rdma_info
        for (int i = 0; i < request->blocks_size(); i++) {
            if (!request->blocks(i).has_rdma_info()) {
                response->set_error_code(EC_FAILED_INVALID_REQ)
                done->Run(); return
            }
        }
    }
    // ... 后续 loadWriteBlocks（gathered 走 6.2 RDMA 聚合路径） ...
}
```

#### RDMA 路径（`RdmaCacheStoreServiceImplContext`）

当前 `loadBlockOnRdma` 对每个 block 构建 `RdmaConnectionWriteBlock` 并调用 `writeBlocks`。每个 block 有独立 callback → `onWriteDone` → `++write_cnt_`。

**方案：partition-aware gathered D2H + 单次 RDMA Write**

RDMA 聚合模式下，server 收到 `gathered_ready` 的 GPU blocks 后，在已知 `partition_count_` / `partition_id_` 的上下文中执行 partition-aware gathered D2H，pool buffer 中只存放该 partition 的连续 slices，然后单次 RDMA Write。

改造 `RdmaCacheStoreServiceImplContext::loadBlockOnRdma` 为**异步流程**：D2H 通过 `execStagedMemoryCopyAsync` 发起后立即返回，host callback 中再发起 RDMA Write。store/watch 线程不被同步 D2H 阻塞。

```
loadBlockOnRdma(ok, blocks):
  if done_run_: return
  if !ok: runFailed(...); return

  // ===== 非聚合模式：原路径 =====
  if NOT request_->gathered_mode():
    → 原路径（逐 block 构建 RdmaConnectionWriteBlock，逐 callback onWriteDone）
    return

  // ===== 聚合模式：async partition-aware D2H + RDMA Write in callback =====
  // 1. 消费所有 key（维护 unloaded_blocks_ 状态），收集 GPU 地址
  matched_count = 0
  std::vector<GpuBlockInfo> gpu_blocks  // {gpu_addr, full_len, key}

  for block in blocks:
    auto block_info = getAndEraseUnLoadedBlock(block->key)
    if block_info == nullptr: continue
    matched_count++
    gpu_blocks.push_back({block->addr.get(), block->len, block->key})

  if matched_count == 0: return

  // 2. 计算 partition slice 大小，分配 pool buffer 并 RAII 包装
  //    关键：release_guard 必须在 async 发起前创建，确保任何路径下 buf 都能释放
  size_t total_send_bytes = 0
  for b in gpu_blocks:
    total_send_bytes += ALIGN_16(b.full_len / partition_count_)

  BufferHandle* buf_handle = buffer_pool_->allocateWithTimeout(total_send_bytes, 100)
  void* send_staging = nullptr
  std::shared_ptr<void> release_guard

  if (buf_handle) {
      send_staging = buf_handle->ptr
      release_guard = std::shared_ptr<void>(buf_handle,
          [pool=buffer_pool_](BufferHandle* h) { pool->free(h); })
  } else {
      // 降级：临时 pinned + MR 注册
      cudaHostAlloc(&send_staging, total_send_bytes, ...)
      memory_util_->regUserMr(send_staging, total_send_bytes, false)
      auto util = memory_util_; auto ptr = send_staging;
      release_guard = std::shared_ptr<void>(send_staging,
          [util, ptr](void*) {
              util->deregUserMr(ptr, false);
              cudaFreeHost(ptr);
          })
  }

  // 3. 构造 D2H 参数：partition-aware gather
  StagedMemoryCopyParams params
  params.direction = D2H
  params.host_base = send_staging
  params.host_bytes = total_send_bytes
  params.host_is_pinned = true
  params.device_index = device_index_
  size_t offset = 0
  for b in gpu_blocks:
    size_t slice_size = b.full_len / partition_count_
    void* slice_gpu_addr = (char*)b.gpu_addr + partition_id_ * slice_size
    params.tiles.push_back({gpu=slice_gpu_addr, host_offset=offset, bytes=slice_size})
    offset += ALIGN_16(slice_size)

  // 4. 异步 D2H + host callback 中发起 RDMA Write
  //    release_guard 通过 capture 传入 callback，确保 buf 在 RDMA Write 完成后才释放
  auto self = shared_from_this()
  auto state = std::make_shared<GatheredWriteState>{
      .release_guard = release_guard,
      .send_staging = send_staging,
      .total_send_bytes = total_send_bytes,
      .matched_count = matched_count,
      .start_us = currentTimeUs(),
  }

  bool launched = execStagedMemoryCopyAsync(
      params,
      &threadLocalScratch(device_index_),
      [self, state](bool d2h_ok) {
          // === CUDA host callback 线程（D2H 完成后）===
          auto ctx = std::dynamic_pointer_cast<RdmaCacheStoreServiceImplContext>(self)
          if (!d2h_ok) {
              ctx->runFailed(EC_FAILED_INTERNAL)
              return                              // release_guard 析构归还 buf
          }
          ctx->dispatchGatheredRdmaWrite(state)   // 见下方
      })

  if (!launched) {
      runFailed(EC_FAILED_INTERNAL)
      return                                       // release_guard 析构归还 buf
  }
  // loadBlockOnRdma 立即返回，watch 线程不阻塞
```

**`dispatchGatheredRdmaWrite`**（在 D2H callback 中执行）：

> **关键**：每次 watch fire 只处理一个 (layer, region) 的 blocks，不同 fire 对应 decode pool buffer 中**不同的 offset 区间**。RDMA Write 的目标地址必须按 metas 中的 offset 计算，不能写到 pool 起点（否则不同 fire 互相覆盖）。

`RdmaCacheStoreServiceImplContext` 在构造时基于 `request_->gathered_block_metas()` 建立 `key→meta` map，用于在每次 dispatch 时查找目标 offset：

```cpp
// context 构造时
std::unordered_map<std::string, GatheredBlockMeta> meta_map_;
for (const auto& m : request_->gathered_block_metas()) {
    meta_map_[m.key()] = m;
}
```

```cpp
dispatchGatheredRdmaWrite(state):
  // 1. 通过第一个 block 的 key 查 meta，得到 decode pool 中的 destination offset
  auto& first_key = state->gpu_blocks[0].key
  auto first_it = meta_map_.find(first_key)
  if (first_it == meta_map_.end()) {
      runFailed(EC_FAILED_INTERNAL); return       // metas 中找不到 → 协商错误
  }
  size_t dst_offset = first_it->second.offset()

  // 2. 校验本次 gather 的 blocks 在 metas 中连续（local compact 布局必须与 remote 一致）
  size_t expect_offset = dst_offset
  for (size_t i = 0; i < state->gpu_blocks.size(); ++i) {
      auto it = meta_map_.find(state->gpu_blocks[i].key)
      if (it == meta_map_.end() || it->second.offset() != expect_offset) {
          // 不连续：本次 fire 的 blocks 不能合并为单次 RDMA Write
          // 这通常意味着 watch fire 顺序与 metas 顺序不一致（违反调度约束，见 §13.6）
          runFailed(EC_FAILED_INTERNAL); return
      }
      expect_offset += ALIGN_16(it->second.length())
  }

  // 3. local block：alias ptr 共享 release_guard 生命周期
  auto merged_local = std::make_shared<BlockBuffer>(
      "__gathered_" + request_id_ + "_off" + std::to_string(dst_offset) + "__",
      std::shared_ptr<void>(state->release_guard, state->send_staging),
      state->total_send_bytes, false/*cpu*/, true/*adopted*/)

  // 4. peer block：rdma_info 从 gathered_recv_info copy，但 addr 偏移到 dst_offset
  auto merged_peer = std::make_shared<BlockBufferInfo>()
  merged_peer->set_key(merged_local->key)
  merged_peer->set_len(state->total_send_bytes)
  auto* rdma_info = merged_peer->mutable_rdma_info()
  rdma_info->CopyFrom(request_->gathered_recv_info())
  rdma_info->set_addr(request_->gathered_recv_info().addr() + dst_offset)
  // ↑ 关键修正：peer addr = decode pool base + 本批 blocks 的 first offset

  // 5. 单一 callback：一次性推进 write_cnt_
  auto self = shared_from_this()
  auto callback = [self, state, collector=collector_](bool success) {
      auto ctx = std::dynamic_pointer_cast<RdmaCacheStoreServiceImplContext>(self)
      collector->setWriteInfo(state->matched_count, state->total_send_bytes,
                              currentTimeUs() - state->start_us)
      if (!success) {
          ctx->runFailed(EC_FAILED_RDMA_WRITE)
          return                                   // release_guard 析构归还 buf
      }
      int new_cnt = ctx->write_cnt_.fetch_add(state->matched_count)
                    + state->matched_count
      if (new_cnt == ctx->total_block_count_) ctx->runSuccess(true)
  }

  // 6. 单次 RDMA Write，绕过 getConcatWriteBlocks
  auto write_block = std::make_shared<RdmaConnectionWriteBlock>(
      merged_local, merged_peer, std::move(callback))
  rdma_connection_->doWriteBlock(write_block)
```

**正确性依赖**：本次 fire 内的 blocks 在 metas 中是连续的子区间。该假设由调度顺序约束（§13.6）保证：watch fire 是 per-(layer, region) 粒度，Decode 端 metas 的 block 顺序为 `for layer → for group → for block_pos`，因此同一 (layer, region) 的 blocks 在 metas 中天然连续。

**关键设计决策**：

- **Async D2H + 异步 RDMA**：`loadBlockOnRdma` 立即返回，所有耗时操作通过 callback 链推进，watch/store 线程不阻塞。
- **`release_guard` 在 async 发起前创建**：用 RAII shared_ptr 包装 pool buf 或临时 pinned 分配；通过 capture 传入 D2H callback 与 RDMA Write callback；任何失败/异常路径都能正确归还。
- **共享 release_guard 通过 shared_ptr aliasing**：`merged_local.addr` 用 alias 共享 `release_guard` 生命周期，RDMA Write callback 完成后才真正归还 pool buf。
- **Partition-aware D2H**：gather kernel 从每个 GPU block 的 `partition_id * slice_size` 偏移处取 `slice_size` 字节。pool buffer 中数据是该 partition 的连续 slices，无需二次 compact。
- **Pool buffer 大小 = N × ALIGN_16(slice_size)**：TP 不对称场景下内存占用最小化。
- **绕过 `getConcatWriteBlocks`**：直接调用 `doWriteBlock`，避免走 per-block 的合并逻辑。
- **`write_cnt_` 一次性递增**：callback 中用 `fetch_add(matched_count)` 而非 N 次 `++write_cnt_`，与 `total_block_count_` 比较正确。
- **保持 `unloaded_blocks_` 正确性**：仍然逐 key 调用 `getAndEraseUnLoadedBlock`，metrics 不受影响。
- **`gathered_recv_info` 来自 proto**：decode 端在发请求时预分配 pool buffer 并通过 proto 传递地址 + rkey。

### 6.3 Decode 端 Load 路径（仅 RDMA）

> **TCP 链路不在本节范围**：TCP decode 走原 `TcpCacheStoreLoadServiceClosure::Run` 逐 block `execNoBlockCopy` 路径，不受本方案影响。

#### RDMA Decode 端（`RdmaMessager` + `RdmaCacheStoreLoadServiceClosure`）

当前 RDMA 路径：Decode 端在 `makeLoadRequest` 中为每个 block 填入 GPU 地址作为 RDMA Write 目标，prefill 直写 decode GPU，Closure::Run 不做拷贝。

聚合模式下改为**三阶段**：

**阶段 1：构建请求（`RdmaMessager::makeLoadRequest`）**

```
makeLoadRequest(rdma_connection, request):
  load_request = Messager::makeLoadRequest(request)  // 填充 per-block key/len

  if NOT enable_gathered_cache_transfer:
    → 原路径：为每个 block 填 rdma_info（GPU 地址 + MR）
    return

  // ===== 聚合模式 =====
  auto blocks = request->request_block_buffer->getBlocks()

  // 1. 计算总接收大小
  //    注意：Decode 端 block->len 已经是 slice_size（每个 Decode 节点的 GPU
  //    block 本就只放本节点 partition 的数据，大小等于 Prefill 的 slice）。
  //    现有 generateBlockInfo 调用传入 partition_count=1 也印证了这一点。
  //    因此这里**不要再除 partition_count**，直接累加 ALIGN_16(block->len)。
  total_recv_bytes = 0
  for (key, block) in blocks:
    total_recv_bytes += ALIGN_16(block->len)

  // 2. 从 decode 端 pool 分配接收缓冲区，立即 RAII shared_ptr 包装
  void* recv_ptr = nullptr
  std::shared_ptr<void> release_guard

  BufferHandle* buf_handle = decode_buffer_pool_->allocateWithTimeout(total_recv_bytes, timeout_ms)
  if (buf_handle) {
      recv_ptr = buf_handle->ptr
      release_guard = std::shared_ptr<void>(buf_handle,
          [pool=decode_buffer_pool_](BufferHandle* h) { pool->free(h); })
  } else {
      // 降级：临时 pinned + MR 注册
      cudaHostAlloc(&recv_ptr, total_recv_bytes, ...)
      memory_util_->regUserMr(recv_ptr, total_recv_bytes, false)
      auto util = memory_util_; auto ptr = recv_ptr;
      release_guard = std::shared_ptr<void>(recv_ptr,
          [util, ptr](void*) {
              util->deregUserMr(ptr, false);
              cudaFreeHost(ptr);
          })
  }

  // 3. 设置 gathered 协商字段
  load_request->set_gathered_mode(true)

  auto* gathered_info = load_request->mutable_gathered_recv_info()
  gathered_info->set_addr(reinterpret_cast<uint64_t>(recv_ptr))
  // 查找 pool 整块 MR，获取所有 NIC 的 rkey
  ::accl::barex::memp_t mem
  memory_util_->findMemoryMr(&mem, recv_ptr, total_recv_bytes, false, true)
  for (nicid, mr) in mem.mrs:
    auto* nk = gathered_info->add_nic_rkeys()
    nk->set_nicid(nicid)
    nk->set_rkey(mr->rkey)

  // 4. 填充 block_metas（偏移布局，使用 slice size）
  offset = 0
  for i in [0..load_request->blocks_size()):
    auto& block_msg = load_request->blocks(i)
    size_t slice_len = block_msg.len()  // request 中的 len 已是 slice size
    auto* meta = load_request->add_gathered_block_metas()
    meta->set_key(block_msg.key())
    meta->set_offset(offset)
    meta->set_length(slice_len)
    offset += ALIGN_16(slice_len)

  // 5. per-block 的 rdma_info 不再填充 GPU 地址（prefill 不用），保留 blocks 列表用于 key/len 匹配

  // 6. 将 recv_ptr / release_guard / 元信息传给 closure
  closure->recv_staging_     = recv_ptr
  closure->total_recv_bytes_ = total_recv_bytes
  closure->release_guard_    = release_guard      // RAII 接管 buf 生命周期
  closure->gathered_mode_    = true
  closure->device_index_     = device_index
  return load_request
```

**阶段 2：Prefill Server partition-aware D2H + RDMA Write（见 6.2 节 RDMA 路径）**

Prefill server 检测 `gathered_mode()` 后，从 GPU blocks 做 partition-aware gathered D2H（只取该 partition 的 slice），pool buffer 中为连续 slices，通过**单次 RDMA Write** 写到 decode 端的 pool buffer。

**阶段 3：接收完成后 H2D + scatter（`RdmaCacheStoreLoadServiceClosure::Run`）**

```cpp
Run():
  // RPC 返回，检查 controller / error_code
  if (controller_->Failed() || response_->error_code() != EC_SUCCESS) {
      // 失败：destroy 连接（不可信任），release_guard_ 析构归还 buf
      recycleRdmaConnection(false)
      end(false, ...)
      return
  }

  if (NOT gathered_mode_) {
      // 原路径：RDMA 已直写 GPU，释放连接即可
      recycleRdmaConnection(true)
      end(true)
      return
  }

  // ===== 聚合模式：RDMA Write 已完成，数据在 release_guard_ 持有的 pool buffer 中 =====
  // 注意：连接 recycle 必须延后到 H2D 结束后，按 H2D 结果决定 success 实参

  // 1. 根据 gathered_block_metas 构建 scatter 参数
  StagedMemoryCopyParams params
  params.direction = H2D
  params.host_base = recv_staging_      // 裸指针，由 release_guard_ 持有生命周期
  params.host_bytes = total_recv_bytes_
  params.host_is_pinned = true          // pool buffer 已是 pinned，跳过 scratch 中转

  for (meta in request_->gathered_block_metas()) {
      auto gpu_block = request_block_buffer_->getBlock(meta.key())
      if (gpu_block == nullptr) {
          recycleRdmaConnection(false)
          end(false, LoadBufferTimeout)
          return                         // release_guard_ 析构归还 buf
      }
      params.tiles.push_back({
          gpu = gpu_block->addr.get(),
          host_offset = meta.offset(),
          bytes = meta.length()
      })
  }
  params.device_index = device_index_

  // 2. 一次 execStagedMemoryCopy(H2D) 完成 H2D + scatter，必须检查返回值
  bool ok = execStagedMemoryCopy(params, &threadLocalScratch(device_index_))

  // 3. 按 H2D 结果决定 recycle 与 callback 状态
  if (!ok) {
      recycleRdmaConnection(false)
      end(false, LoadFailed)
      return                             // release_guard_ 析构归还 buf
  }

  recycleRdmaConnection(true)
  end(true)
  // release_guard_ 在 closure 析构时归还 buf
```

**Closure 成员变量**：

```cpp
class RdmaCacheStoreLoadServiceClosure : public RPCClosure {
    // ... existing members ...

    // 聚合模式新增
    bool                                       gathered_mode_      = false;
    void*                                      recv_staging_       = nullptr;  // 裸指针
    size_t                                     total_recv_bytes_   = 0;
    int                                        device_index_       = 0;

    // RAII：buf 归还/MR 注销由 shared_ptr 析构链统一处理
    // 来源可以是 pool 分配（pool->free）或临时 cudaHostAlloc + regUserMr
    std::shared_ptr<void>                      release_guard_;

    // ❌ 不持有 scratch_ 成员：使用 threadLocalScratch(device_index_) 替代
    //   原因：closure 寿命跨多线程；per-closure scratch 让每请求各持几 MB 资源，
    //   1000 并发即 GB 级浪费。设计原则：scratch 资源跟着线程走，不跟请求走。
};
```

**关键修复点**：

1. **`recycleRdmaConnection` 时序**：`Run()` 中 recycle 必须延后到 H2D 完成后，按 H2D 结果决定 `success` 实参。失败时统一 `recycleRdmaConnection(false)`（destroy）防止连接被误用。
2. **H2D 失败传播**：`execStagedMemoryCopy` 返回 `bool`，必须检查；失败时 `end(false, LoadFailed)`，避免上报"成功但 GPU 数据是脏的"。
3. **RAII 资源管理**：`release_guard_` 是 `shared_ptr<void>`，构造时用 custom deleter 包装 `pool->free` 或 `cudaFreeHost + deregUserMr`。closure 析构、Run() 各失败路径均自动归还，无需手写释放代码。
4. **`scratch_` 移除**：closure 不持有 scratch；`Run()` 直接用 `threadLocalScratch(device_index_)`，scratch 跟着 RPC 回调线程走，跨请求复用。

### 6.4 Proto 扩展

```protobuf
// 新增 message：描述 gathered buffer 中每个 block 的布局
message GatheredBlockMeta {
    optional string key    = 1;  // block key，用于 decode 端匹配 GPU 目标地址
    optional uint64 offset = 2;  // 在 gathered buffer 中的字节偏移
    optional uint32 length = 3;  // 字节数
}

// CacheLoadRequest 扩展（RDMA gathered 协商）
message CacheLoadRequest {
    // ... existing fields (1-8) ...

    // RDMA 聚合传输协商
    optional bool gathered_mode = 9;                            // 是否使用 RDMA 聚合模式
    optional RdmaBlockBufferAddrInfo gathered_recv_info = 10;   // RDMA: decode 端接收 buffer 地址 + MR rkey
    repeated GatheredBlockMeta gathered_block_metas = 11;       // 各 block 在 gathered buffer 中的偏移布局
}

// CacheLoadResponse 不需要扩展
// TCP 保持原有 repeated BlockBufferInfo blocks 响应格式
// RDMA 聚合模式下 response 只需返回 error_code + direct_write_response=true
```

**向后兼容**：
- `gathered_mode` 不设置时默认 `false`，走原有逐 block 路径
- TCP 路径完全不受影响（不使用 `gathered_mode` 字段）
- RDMA 路径：旧版 Prefill 不会识别 `gathered_recv_info`，仍然尝试逐 block 直写 GPU，但由于 per-block `rdma_info` 未填充会失败 → Decode 端需要在升级时确保 Prefill 也升级

## 7. Buffer 生命周期

本节描述 RDMA 链路下 pool buffer 的生命周期。TCP 链路不使用 Memory Pool，不在本节范围。

### 7.1 Prefill 端 Buffer（仅 RDMA）

```
// D2H 延迟到 server 收到 load request
pool.allocate(total_send_bytes)         // total = N × ALIGN_16(slice_size)
  → BufferHandle (ptr, size, offset)
     │
     │ 1. execStagedMemoryCopy(D2H): partition-aware gather
     │ 2. RDMA Write
     │ 3. RDMA Write callback 完成
     ↓
pool.free(handle)  ← RDMA Write callback 中 shared_ptr 析构触发
```

释放触发：`BlockBuffer` 的 `shared_ptr` custom deleter，当所有引用（包括 RDMA Write 期间的引用）都释放后自动触发。pool 会自动合并相邻空闲段。

### 7.2 Decode 端 Buffer（仅 RDMA）

```
pool.allocate(total_bytes)
  → BufferHandle (ptr, size, offset)
     │
     │ 1. RDMA Write 直接写入 pool buffer
     │ 2. execStagedMemoryCopy(H2D, host_is_pinned=true): H2D + scatter
     ↓
pool.free(handle)  ← scatter 完成后立即释放
```

释放触发：`execStagedMemoryCopy` 完成（包含 `cudaStreamSynchronize`）后立即释放。

## 8. `combine_load_` 说明

聚合模式不改变 RDMA 链路 `combine_load_ = true` 的行为，仍然在 per-layer-per-region 粒度减少 WQE 数量。TCP 链路与本方案无关。

## 9. 关键实现细节

### 9.1 对齐要求

所有 block 在 gathered buffer 中的偏移需 **16 字节对齐**，确保 gather/scatter kernel 走 `int4` 向量化路径：

```cpp
constexpr size_t kGatherAlignBytes = 16;
size_t alignedOffset(size_t offset) {
    return (offset + kGatherAlignBytes - 1) & ~(kGatherAlignBytes - 1);
}
```

### 9.2 GPU Staging Scratch 管理

使用 `thread_local` 管理 scratch 实例，每个线程拥有独立的 scratch，彻底消除并发竞争：

```cpp
// 全局 thread_local scratch（per-thread per-device）
StagedMemoryCopyScratch& threadLocalScratch(int device_index) {
    struct PerDeviceScratch {
        std::map<int, StagedMemoryCopyScratch> scratches;
        ~PerDeviceScratch() {
            for (auto& [_, s] : scratches) {
                releaseStagedMemoryCopyScratch(s);
            }
        }
    };
    static thread_local PerDeviceScratch tls;
    return tls.scratches[device_index];
}
```

设计依据：
- `execStagedMemoryCopy` 内部已使用 `thread_local` CUDA stream（`getNoBlockCopyStream()`），scratch 与 stream 生命周期一致
- store 线程池中多线程并发调用 `setRequestBlockBuffer` 时，各线程使用各自的 scratch，无锁无竞争
- scratch 容量只增不减（跨请求复用），避免反复 `cudaMalloc`/`cudaFreeHost`
- 线程退出时析构函数自动释放 GPU/host 资源

### 9.3 线程安全

- `CacheTransferBufferPool` 的 `allocate/free` 使用 `mutex + condition_variable`，支持多线程并发访问
- `StagedMemoryCopyScratch` 使用 `thread_local` 管理，每线程独立实例，无锁无竞争
- `execStagedMemoryCopy` / `execStagedMemoryCopyAsync` 内部使用 `thread_local` CUDA stream（`getNoBlockCopyStream()`），与 scratch 的 thread_local 生命周期一致
- CUDA event callback（`cudaLaunchHostFunc`）在 CUDA 内部线程执行，`addBlocks` 操作本身已通过 `RequestBlockBuffer` 内部锁保护

### 9.4 Warmup

在 `EngineBase::init` 中已有 `warmupNoBlockCopy()` 调用，会预热 split-KV kernel。如果选用 `var_nooffset` kernel，需要增加对应的 warmup 调用：

```cpp
// warmupNoBlockCopy 中增加：
if (kv_cache_config.enable_gathered_cache_transfer) {
    if (!sDevMPS::warmup_sm_copy_var_nooffset_kernels(stream)) {
        RTP_LLM_LOG_WARNING("warmup var_nooffset kernels failed");
    }
}
```

### 9.5 Metrics

聚合传输引入若干新增 metric，便于运维观测和性能分析：

| Metric | 类型 | 含义 |
|--------|------|------|
| `gathered_pool_used_bytes` | gauge | Memory Pool 已使用字节数（瞬时） |
| `gathered_pool_total_bytes` | gauge | Memory Pool 总容量 |
| `gathered_pool_largest_free_bytes` | gauge | 最大空闲段（碎片度指标） |
| `gathered_pool_alloc_latency_us` | histogram | 单次 `allocate` 耗时（含等待） |
| `gathered_pool_alloc_wait_us` | histogram | `allocateWithTimeout` 等待时间 |
| `gathered_pool_fallback_count` | counter | pool 满降级到临时 `cudaHostAlloc` 次数 |
| `gathered_pool_fallback_register_us` | histogram | 降级路径 MR 注册延迟 |
| `gathered_d2h_latency_us` | histogram | gather + D2H 延迟（kernel launch → memcpy 完成） |
| `gathered_h2d_latency_us` | histogram | H2D + scatter 延迟（Decode 端） |
| `gathered_rdma_write_latency_us` | histogram | 单次 RDMA Write callback 延迟 |
| `gathered_async_callback_drift_us` | histogram | CUDA host callback 触发到 RDMA Write 发起的延迟（衡量回调线程调度滞后） |
| `gathered_request_bytes` | histogram | 单 request 单 layer 单 region 的传输字节数分布（用于 pool sizing） |
| `gathered_d2h_failure_count` | counter | D2H 失败次数（CUDA error / OOM 等） |

部分 metric 的告警阈值建议：
- `gathered_pool_used_bytes / total > 0.85`：长期高水位 → 增大 pool 或检查 buf 释放是否及时
- `gathered_pool_fallback_count` 增长率 > 10/min：pool 容量不足
- `gathered_async_callback_drift_us` p99 > 1ms：CUDA host callback 线程被阻塞，可能是 store/RPC 线程池过载

### 9.6 初始化顺序约束

Memory Pool 在 RDMA 模式下需要对整块 buffer 注册 MR，依赖 `MemoryUtil` 已被 `accl::barex` 初始化。**初始化顺序必须**：

```
1. NormalCacheStore::createNormalCacheStore(params)
   ├─ memory_util_ = make_shared<RdmaMemoryUtilImpl>(...)
   ├─ messager_->init(...)              // 内部初始化 accl::barex
   │                                     // 此时 memory_util_ 才能注册 MR
   └─ NormalCacheStore::init() 完成（messager_ 已就绪）

2. NormalCacheStore::initBufferPool(kv_cache_config)
   ├─ 检查 enable_gathered_cache_transfer 与 cache_transfer_buffer_size_mb
   └─ buffer_pool_ = make_unique<CacheTransferBufferPool>(
          pool_size, memory_util_)     // 一次性 cudaHostAlloc + regUserMr
```

**约束**：
- pool 构造**必须在 `NormalCacheStore::init()` 之后**调用，确保 `memory_util_` 已可用
- 构造失败（OOM / MR 注册失败）时：log warning + 降级为 0-size pool（即纯运行时按需 alloc 的降级路径）；不阻塞 NormalCacheStore 的初始化

**错误顺序的后果**：
- 若在 messager init 之前构造 pool：`regUserMr` 失败，pool 退化为不可 RDMA 的 pinned memory，每次 RDMA 调用都会走降级路径 → 性能预期完全落空
- 若 `enable_gathered_cache_transfer = true` 而 `cache_transfer_buffer_size_mb = 0`：跳过预分配，全程走运行时 alloc + reg，需要监控 `gathered_pool_fallback_count`

**实施建议**：在 `NormalCacheStore::init()` 末尾增加 `initBufferPool()` 调用，由其负责检查配置并构造 pool；构造失败时 log warning 并降级。

## 10. 回调与计数正确性分析

### 10.1 `write_cnt_` 机制回顾

Prefill server 端 `CacheStoreServiceImplContext` 通过 `write_cnt_` 和 `total_block_count_` 判断传输完成：

```cpp
// total_block_count_ = request_->blocks_size()（decode RPC 中的 block 数量 N）
// write_cnt_ 每消费一个 key 递增
// write_cnt_ == total_block_count_ → runSuccess()
```

### 10.2 RDMA 聚合模式

RDMA 聚合路径将 N 个 block 合并为**单次 RDMA Write**，因此只有**一个 callback**。`write_cnt_` 使用 `fetch_add(matched_count)` 一次性递增：

```
loadBlockOnRdma(blocks):
  if gathered_mode:
    // 消费所有 key（matched_count 个）
    for block in blocks:
      getAndEraseUnLoadedBlock(block->key)
      matched_count++

    // 单次 RDMA Write → 单个 callback
    callback = [matched_count](bool ok) {
      write_cnt_.fetch_add(matched_count)
      if write_cnt_ == total_block_count_: runSuccess()
    }
    doWriteBlock(merged_write_block)
```

**正确性保证**：`fetch_add(matched_count)` 原子递增，最终值 == N == `total_block_count_`。

### 10.3 watch 批量投递的安全性

`RequestBlockBuffer::addBlocks(valid_blocks)` 在聚合路径下一次投递 N 个 block。watch 回调收到完整的 N 个 block batch。但如果 store 端分多次调用 `addBlocks`（如不同 batch 的 block 先后到达），watch 可能被多次触发。

`getAndEraseUnLoadedBlock` 的幂等性（已消费 key 返回 nullptr → skip）确保不会重复计数：
- 第一次触发：消费 M 个 key → `write_cnt_ += M`
- 第二次触发：消费 K 个 key → `write_cnt_ += K`
- 最终 `M + K == N == total_block_count_`

### 10.4 Decode 客户端校验（仅 RDMA）

`RdmaCacheStoreLoadServiceClosure::Run` 当前不做 block count 校验（只检查 `error_code`），聚合模式下同样只需检查 `error_code`。

## 11. 分阶段实施计划

### Phase 1: 核心框架 + Proto 协商 + RDMA 聚合

- [ ] Proto 扩展：`gathered_mode`, `gathered_recv_info`, `GatheredBlockMeta`
- [ ] `KVCacheConfig` 新增 `enable_gathered_cache_transfer` 字段
- [ ] pybind 注册 + Python 参数
- [ ] `StagedMemoryCopyParams` 增加 `host_is_pinned` 字段，`execStagedMemoryCopy` 适配
- [ ] `execStagedMemoryCopyAsync` 实现（非阻塞 D2H + CUDA event callback）
- [ ] `threadLocalScratch` thread_local scratch 管理
- [ ] Prefill store 端：`RequestBlockBufferStore::setRequestBlockBuffer` 异步聚合 D2H 路径
- [ ] RDMA Prefill server：`RdmaCacheStoreServiceImplContext::loadBlockOnRdma` 聚合单次写入
- [ ] RDMA Decode client：`RdmaMessager::makeLoadRequest` 填充 `gathered_recv_info`
- [ ] RDMA Decode client：`RdmaCacheStoreLoadServiceClosure::Run` H2D + scatter
- [ ] 临时 `cudaHostAlloc` / `cudaFreeHost`（无 Memory Pool）
- [ ] 功能测试 + Smoke 测试验证（RDMA 聚合路径功能与性能；TCP 链路回归确认未被影响）

### Phase 2: Host Memory Pool + 性能优化

- [ ] `CacheTransferBufferPool` 实现（Best-Fit + 合并）
- [ ] `KVCacheConfig` 新增 `cache_transfer_buffer_size_mb`
- [ ] `NormalCacheStore` 集成 Memory Pool
- [ ] Pool 满降级逻辑
- [ ] RDMA MR 预注册（整块 pool 一次注册）
- [ ] RDMA 性能测试验证

### Phase 3: 进一步优化（可选）

- [ ] GPU staging Memory Pool（RDMA GPUDirect 场景，跳过 host 中转）
- [ ] 调度器反压集成（Memory Pool 满时暂停 context batch 调度）

## 12. 风险与注意事项

| 风险 | 缓解措施 |
|------|---------|
| Memory Pool 内存占用 | 通过 `cache_transfer_buffer_size_mb` 控制，默认 0（不预分配） |
| Memory Pool 碎片 | Best-Fit + 合并策略控制碎片；设置充足 pool 大小（4-8x 最大单次分配） |
| 聚合模式下单次传输数据量大 | 仍按 per-layer-per-request 粒度聚合（不是全部层合并），单次数据量可控 |
| RDMA 降级到 host staging 性能下降 | 相比原有 per-block GPU→GPU，增加了 D2H + H2D。但减少了 WQE 数量。Net 效果需性能测试验证 |
| 向后兼容 | `enable_gathered_cache_transfer` 默认 false，不影响现有行为 |
| Partition 支持 | D2H gather 在 server 端执行，已知 partition_id，直接取 slice |
| MTP/Eagle 模型 | Decode 端 `loadCache` 中 MTP block 同样适用聚合路径 |

## 13. DSV4 HybridAttention 正确性分析

### 13.1 DSV4 Cache 架构概述

DeepSeek V4 使用 HybridAttention，通过 `layer_compress_ratios` 将模型的所有层分为三种类型：

| 层类型 | compress_ratio | 触及的 Region | 对应 Group ID |
|--------|---------------|---------------|---------------|
| CSA 层 | 4 | CSA_KV, INDEXER_KV, INDEXER_STATE, CSA_STATE, SWA_KV | g0, g2, g3, g4, g6 |
| HCA 层 | 128 | HCA_KV, HCA_STATE, SWA_KV | g1, g5, g6 |
| SWA-only 层 | 0 | SWA_KV | g6 |

七个 Group 的详细配置由 `DSV4CacheConfigHelper::applyConfig` 生成：

| Group | Region | CacheGroupType | Block 大小特征 | 是否 Paged |
|-------|--------|----------------|---------------|-----------|
| g0 | CSA_KV | FULL | `kv_entry × (tokens_per_block/4)`, FP8 有 576B 对齐 | 是 |
| g1 | HCA_KV | FULL | `kv_entry × (tokens_per_block/128)` | 是 |
| g2 | INDEXER_KV | FULL | `indexer_entry × (tokens_per_block/4)` | 是 |
| g3 | INDEXER_STATE | SWA | `idx_state_dim × 2 × tokens_per_block` (FP32) | 否（固定池） |
| g4 | CSA_STATE | SWA | `csa_state_dim × 2 × tokens_per_block` (FP32) | 否（固定池） |
| g5 | HCA_STATE | SWA | `hca_state_dim × 2 × tokens_per_block` (FP32) | 否（固定池） |
| g6 | SWA_KV | SWA | `kv_entry × tokens_per_block` | 否（固定池） |

### 13.2 传输粒度：每次 `runtimeWriteCacheStore` = 1 个 (layer, region)

Python 侧 `getLayerCaches(layer_idx)` 返回该层所涉及的**每个 region** 的独立 `LayerKVCache`，每个 `LayerKVCache` 对应一次独立的 `runtimeWriteCacheStore` 调用。

例如一个 CSA 层（layer 0），会产生 5 次 `runtimeWriteCacheStore` 调用：

| 调用序号 | layer_id | region_name | group_id | block 大小 | 传输 block 数 |
|---------|----------|-------------|----------|-----------|-------------|
| 1 | 0 | CSA_KV | 0 | 大（KV entry × N/4） | 所有 block（FULL） |
| 2 | 0 | INDEXER_KV | 2 | 中（indexer × N/4） | 所有 block（FULL） |
| 3 | 0 | INDEXER_STATE | 3 | 小（state dim × N） | 最后 2 个（SWA） |
| 4 | 0 | CSA_STATE | 4 | 小（state dim × N） | 最后 2 个（SWA） |
| 5 | 0 | SWA_KV | 6 | 中（KV entry × N） | 最后 2 个（SWA） |

**关键结论**：每次 gather 操作只处理**同一个 group** 的 blocks，所有 block 的大小一致（由该 group 的 `cache_spec` 决定）。不存在"一次 gather 中混合不同大小 block"的情况。

但**不同次 gather 的总字节数差异很大**（CSA_KV 的 FULL 传输可能 2MB，而 STATE 的 SWA 传输可能只有 32KB），这正是需要 Memory Pool（而非固定 Slot）的原因。

### 13.3 Cache Key 的唯一性保证

Prefill 端和 Decode 端通过 **cache key** 匹配 block 数据与 GPU 目标地址。key 的格式为：

```
makeCacheKey(model_id, token_id_str, layer_id, region_name)
  → "model_id_{M}_token_id_str_{T}_layer_id_{L}_region_{R}"
```

其中 `region_name` 是 `KVCacheRegionName` 枚举值（CSA_KV=1, HCA_KV=2, ...）。

**Prefill 端生成 key**（`ExecOps.cc::runtimeWriteCacheStore`）：

```
cache_key = makeCacheKey(model_id, cache_keys[block_pos], layer_id, region_name)
block_key = "kv_" + cache_key       // 或 "kv_scale_" + cache_key
```

**Decode 端生成 key**（`DecodeRpcServer.cc::loadCache`）：

```
cache_key = makeCacheKey(model_id, cache_keys[block_pos], layer_id, region_name)
block_key = "kv_" + cache_key       // 或 "kv_scale_" + cache_key
```

两端使用**完全相同的** `makeCacheKey` 函数和前缀，因此 key 一一对应。

### 13.4 Gather/Scatter 正确性端到端验证

以 CSA 层 layer_id=0, region=CSA_KV, group_id=0 为例，假设该请求有 5 个 block（block_ids = [12, 3, 47, 8, 31]），每个 block 大小为 `B` 字节：

#### Prefill 端 Gather

```
ExecOps.cc → runtimeWriteCacheStore(layer_id=0, region=CSA_KV):
  gid = layer_region_to_group_id[0][CSA_KV] = 0
  block_ids = block_ids_by_group[0] → [12, 3, 47, 8, 31]
  kv_block_stride_bytes = group_kv_block_stride_bytes[0] = B

  for block_pos in [0,1,2,3,4]:
    block_id = block_ids[block_pos]
    gpu_addr = convertIndexToBuffer(block_id, layer_id=0, region=CSA_KV)
    cache_key = makeCacheKey(model_id, token_str[block_pos], 0, CSA_KV)
    addBlock(key="kv_" + cache_key, addr=gpu_addr, len=B, gpu_mem=true)
```

聚合路径处理：

```
RequestBlockBufferStore → setRequestBlockBuffer:
  gpu_blocks = 5 个 GPU blocks, 每个 B 字节
  total_bytes = 5 * ALIGN_16(B)

  buffer_pool_.allocate(total_bytes) → BufferHandle {ptr, size}

  execStagedMemoryCopy(D2H):
    gather kernel: 5 个 GPU 地址 → GPU staging (连续 5*B)
    cudaMemcpyAsync: GPU staging → host pool buffer
    cudaStreamSync

  Host Pool Buffer 布局:
  [  kv_blk0 (B)  |  kv_blk1 (B)  |  kv_blk2 (B)  |  kv_blk3 (B)  |  kv_blk4 (B)  ]
  ^offset=0       ^offset=B       ^offset=2B      ^offset=3B      ^offset=4B

  构造 5 个 valid BlockBuffer:
    key="kv_{cache_key_0}", addr=pool+0,   len=B, cpu_mem
    key="kv_{cache_key_1}", addr=pool+B,   len=B, cpu_mem
    key="kv_{cache_key_2}", addr=pool+2B,  len=B, cpu_mem
    key="kv_{cache_key_3}", addr=pool+3B,  len=B, cpu_mem
    key="kv_{cache_key_4}", addr=pool+4B,  len=B, cpu_mem
```

#### Decode 端 Scatter（RDMA 聚合路径）

RDMA Write 将连续 slices 写入 decode pool buffer，之后一次 `execStagedMemoryCopy(H2D)` 完成 H2D + scatter 到各 GPU block。

当 `partition_count > 1` 时，Prefill server 的 partition-aware D2H 只传 slice，Decode 端 pool buffer 和 GPU block 大小均为 `slice_size = full_block_size / partition_count`，scatter 正确。

#### Decode 端 key 匹配流程

```
DecodeRpcServer::loadCache:
  for layer_id in [0..layer_num):
    layer_gids = layer_to_group_ids[layer_id]  // CSA 层 → [0, 2, 3, 4, 6]
    for gid in layer_gids:
      region_name = group_region_names[gid]    // gid=0 → CSA_KV
      block_ids = block_ids_by_group[gid]
      block_pos_list = blockPositionsForCacheTransfer(...)  // FULL → all

      for block_pos in block_pos_list:
        cache_key = makeCacheKey(model_id, cache_keys[block_pos], layer_id, region_name)
        parts = cache_manager.convertIndexToBuffer(block_id, layer_id, CSA_KV, ...)
        addBufBlock("kv_" + cache_key, parts[0])   // GPU 地址
        // ↑ 这里的 key 与 prefill 端完全一致
```

### 13.5 结论：正确性保证

1. **Key 唯一性**：`makeCacheKey(model_id, token_str, layer_id, region_name)` 四元组保证全局唯一。不同 region 的 block 有不同的 key（因 region_name 不同），不会混淆。

2. **每次 gather 只处理同一 group**：由 `runtimeWriteCacheStore` 的调用粒度决定（一个 layer + 一个 region = 一个 group），所有 block 大小一致，gather 操作正确。

3. **block 大小来自 per-group 配置**：`WriteCacheStoreOp.cc` 从 `LayerKVCache.kv_cache_base` tensor 的行大小推导 `kv_block_stride_bytes`，这是该 region 的实际 stride，而非全局 max stride。gather/scatter 使用的就是这个 per-region 大小。

4. **Decode 端目标地址正确**：`convertIndexToBuffer(block_id, layer_id, region_name, ...)` 通过 `groupIdForLayerRegion` 路由到正确的 group，返回正确大小的 `BlockInfo`。key 匹配确保 scatter 写入正确的 GPU block。

5. **Scale 块同步处理**：`parts.size() == 2` 时，`kv_scale_` 前缀的 block 独立传输，其大小由 `kv_scale_stride_bytes` 决定。gather 操作将 kv 和 scale 作为独立 block 处理，无对齐问题。

6. **Memory Pool 适配性**：不同 region 的 gather 操作从同一个 pool 按需分配不同大小的 buffer（CSA_KV FULL 传输 ~MB 级，STATE SWA 传输 ~KB 级），不存在浪费或不足。

### 13.6 调度顺序约束（RDMA 聚合路径正确性的隐含前提）

§6.2 `dispatchGatheredRdmaWrite` 中"通过第一个 block 的 key 查 meta，然后 RDMA Write 到 `recv_addr + first_offset`"这一逻辑，依赖以下三条**调度顺序约束**：

#### C1：watch fire 粒度 = 一个 (layer, region)

Prefill 端 `runtimeWriteCacheStore` 一次调用对应一个 `(layer_id, region_name)`，内部一次 `setRequestBlockBuffer(blocks)` + `addBlocks` 触发一次 watch fire。`loadBlockOnRdma` 拿到的 `blocks` 参数就是这一 fire 的 blocks，**等于**该 (layer, region) 下需要传输的所有 blocks。

⇒ 一次 fire 内的 blocks 同属一个 (layer, region)。

#### C2：Decode 端 metas 顺序 = `for layer → for group → for block_pos`

Decode 端 `DecodeRpcServer::loadCache` 按以下三层循环构造 `request_block_buffer`：

```
for layer_id in [0..layer_num):
  for gid in layer_to_group_ids[layer_id]:        // 一个 layer 内的 group 顺序固定
    for block_pos in block_pos_list:
      addBufBlock(makeCacheKey(..., layer_id, region_name), ...)
```

`RdmaMessager::makeLoadRequest` 在 `gathered_block_metas` 中按相同顺序填充。

⇒ metas 中同一 (layer_id, region_name) 的 blocks 是**连续的子区间**。

#### C3：fire 内 block 顺序 = metas 内对应子区间顺序

Prefill 端 `runtimeWriteCacheStore` 内对 block_pos 的迭代顺序与 Decode 端构造 `request_block_buffer` 时的顺序一致（两侧都是按 `block_pos_list` 升序）。

⇒ 一次 fire 内 blocks 的 key 序列与 metas 中对应子区间的 key 序列**逐项相等**。

#### 推论：local compact 布局 ≡ remote 子区间布局

满足 C1 + C2 + C3 时：
- 本次 fire 内 N 个 blocks 在 local `send_staging` 中的偏移 = 0, ALIGN_16(slice), 2·ALIGN_16(slice), ...
- 对应到 metas 中是 first_offset, first_offset + ALIGN_16(slice), first_offset + 2·ALIGN_16(slice), ...

local 中相对偏移 = remote 中相对偏移 ⇒ 单次 RDMA Write 到 `recv_addr + first_offset` 就能正确放置所有 blocks。

#### 校验机制

§6.2 `dispatchGatheredRdmaWrite` 对每次 fire 显式校验上述约束：

```cpp
size_t expect_offset = first_meta.offset()
for (i in [0..gpu_blocks.size())):
    if (meta_map_[gpu_blocks[i].key].offset() != expect_offset):
        runFailed(EC_FAILED_INTERNAL); return  // 违反约束 → 拒绝
    expect_offset += ALIGN_16(meta.length())
```

不连续即失败而非降级到逐 block，避免静默错误数据。

#### 何时约束可能失败

理论上以下情况会破坏约束：
- 上层对 `runtimeWriteCacheStore` 的调用顺序变更（例如 layer 乱序）
- Decode 端 `loadCache` 的循环顺序变更
- 启用按 cache key 去重 / prefix cache 复用导致 fire 内 block 集合非全集

若未来引入这些特性，需要扩展协议（如 metas 中携带 fire 标识）或改用多次 RDMA Write。
