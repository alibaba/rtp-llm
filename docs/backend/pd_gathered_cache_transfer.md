# PD 分离 Gathered Cache Transfer

## 概述

Gathered Cache Transfer 是 PD 分离场景下 KV Cache 传输的优化方案。核心思想是将 Prefill 节点上 N 个离散的 GPU KV Cache block 聚合(gather)到一块连续的 pinned memory staging buffer，通过单次 RDMA Write 发送到 Decode 端，再由 Decode 端 scatter 回各 GPU block 地址。

**核心收益**：将 N 次 per-block RDMA Write 合并为 1 次，消除 WQE(Work Queue Entry) 瓶颈和网络 RTT 累积。

## 架构图

```
Prefill 节点 (Server端)                          Decode 节点 (Client端)
┌────────────────────────────┐                  ┌────────────────────────────┐
│  GPU KV Cache (N blocks)   │                  │  GPU KV Cache (N blocks)   │
│           │                │                  │           ▲                │
│     SM Gather Kernel       │                  │     SM Scatter Kernel      │
│     (D2D → staging buf)    │                  │     (staging buf → D2D)    │
│           │                │                  │           ▲                │
│     cudaMemcpyAsync        │                  │     cudaMemcpyAsync        │
│     (D2H to pinned mem)    │                  │     (H2D from pinned mem)  │
│           │                │                  │           ▲                │
│  ┌─────────────────────┐   │     1 RDMA       │  ┌─────────────────────┐   │
│  │ Pin Memory Staging  │───┼────Write────────▶│  │ Pin Memory Staging  │   │
│  │ (Pool / Fallback)   │   │                  │  │ (Pool / Fallback)   │   │
│  └─────────────────────┘   │                  │  └─────────────────────┘   │
└────────────────────────────┘                  └────────────────────────────┘
```

## 全流程详解

### Phase 1: Decode 端发起请求

**入口**: `RdmaMessager::load()` → `makeLoadRequest()`

当 `buffer_pool_` 存在时进入 gathered 分支：

1. **计算总接收 buffer 大小**：遍历所有 block，按 16B 对齐累加
   ```cpp
   total_recv_bytes += alignGatherSize(block_msg.len());
   ```

2. **分配 staging buffer**：
   ```cpp
   auto staging_guard = allocateStaging(buffer_pool_, memory_util_, total_recv_bytes);
   ```
   - `tryAllocate` 成功 → 使用 pool 中预注册的 pinned memory
   - 失败 → fallback 到 `cudaHostAlloc` + 动态 `regUserMr`（RDMA MR 注册）

3. **设置 protobuf 请求字段**：
   - `gathered_mode = true`
   - `gathered_recv_info`：接收端 staging buffer 的地址 + 各 NIC 的 rkey
   - `gathered_block_metas`：每个 block 的 key、在 staging buffer 中的 offset、length

4. **创建 RPC Closure 并设置 gathered 模式**：
   ```cpp
   closure->setGatheredMode(recv_guard, recv_guard->is_fallback, pool_free);
   ```

5. **发送 RPC 请求**

### Phase 2: Prefill 端处理请求

**入口**: `RdmaCacheStoreServiceImplContext::loadBlockOnRdma()` → `loadBlockOnRdmaGathered()`

#### 2a. 准备阶段

1. **匹配 block**：对每个 block 调用 `getAndEraseUnLoadedBlock`，收集 GPU 地址和大小
2. **计算总发送大小**：
   ```cpp
   total_send_bytes = Σ alignGatherSize(full_lens[i] / partition_count)
   ```
3. **分配本端 staging buffer**：
   ```cpp
   auto staging_guard = allocateStaging(buffer_pool_, memory_util_, total_send_bytes);
   ```
4. **构造 D2H 拷贝参数** (`StagedMemoryCopyParams`)：
   - 方向 = D2H
   - 每个 block 对应一个 tile：GPU 源地址 + host_offset + bytes
   - `host_is_pinned = true`（跳过中间 host staging 直接拷贝到目标 pinned memory）

5. **发起异步 D2H**：`execStagedMemoryCopyAsync(params, scratch, callback)`

#### 2b. D2H 内部实现

`execStagedMemoryCopyAsync` 在单个 CUDA stream 上依次执行：

```
cudaMemcpyAsync(tile metadata → GPU scratch)    // 上传 tile 元数据到 GPU
launch SM gather kernel                          // GPU D2D: 各 block → 连续 device staging
cudaMemcpyAsync(device staging → host pinned)    // D2H
cudaStreamAddCallback → completion callback      // 通知完成
```

**SM Gather Kernel** (`dsv4_memory_cache_gather_copy_var_nooffset_kernel`)：
- Launch: `<<<num_tiles, 512, 0, stream>>>`
- 每个 thread block 负责一个 tile（一个 block 的数据）
- 内部 `dsv4_copy_region`：按地址对齐选择 int4/int2/uint/ushort/byte 向量化拷贝

#### 2c. RDMA Write

D2H 完成后触发 `dispatchGatheredRdmaWrite`：

1. **计算目标偏移**：从 `meta_map_` 获取第一个 block 的 offset
2. **构造 merged block**：
   - local: 整个 staging buffer 作为一个连续块
   - peer: 目标地址 = `gathered_recv_info.addr + dst_offset`
3. **发起单次 RDMA Write**：`rdma_connection_->writeBlocks({write_block})`
4. **Write 完成回调**：上报 metrics，批量计数完成 block 数

### 完成计数机制

`loadBlockOnRdma` 可能被多次调用（block 分批就绪），需要追踪所有 block 全部传输完成。两种路径的计数方式不同：

**Per-block 路径**：每个 block 独立一个 RDMA Write callback，逐个递增：

```cpp
auto callback = [...](bool success) {
    if (++write_cnt_ == total_block_count_) {
        runSuccess(true);
    }
};
```

N 个 block → N 次 callback，每次 +1。

**Gathered 路径**：每批 block 合并为一次 RDMA Write，callback 中批量计数：

```cpp
auto callback = [..., state](bool success) {
    int new_cnt = ctx->write_cnt_.fetch_add(state->matched_count) + state->matched_count;
    if (new_cnt == static_cast<int>(ctx->total_block_count_)) {
        ctx->runSuccess(true);
    }
};
```

N 个 block 分 M 批就绪（M <= N）→ M 次 callback，每次加该批次的 `matched_count`。

两种路径最终都要累积到 `total_block_count_` 才触发 `runSuccess`，标记整个请求完成。

### Phase 3: Decode 端接收完成

**入口**: `RdmaCacheStoreLoadServiceClosure::Run()`

Prefill RDMA Write 完成后 Server 返回 RPC response，触发 Closure：

1. **检查 response** 无错误
2. **构造 H2D scatter 参数**：
   - 方向 = H2D
   - 从 `gathered_block_metas` 重建 tile：每个 block 的 GPU 目标地址 + staging 中的 offset + length
3. **执行同步 H2D scatter**：`execStagedMemoryCopy(params, scratch)`
   - SM scatter kernel 将连续 pinned memory 分发回各 block 的 GPU 地址
4. **回收 RDMA connection**，标记传输完成

## 关键数据结构

| 结构 | 文件 | 作用 |
|------|------|------|
| `CacheTransferBufferPool` | `CacheTransferBufferPool.h` | Best-fit free-list 管理预分配 pinned memory |
| `TempStagingGuard` | `TempStagingGuard.h` | RAII 守卫：析构时归还 pool 或释放 fallback 内存 |
| `GatheredWriteState` | `RdmaCacheStoreServiceImplContext.h` | 异步生命周期管理：跨 D2H 和 RDMA callback 持有 staging + 计时 |
| `GatheredBlockMeta` | proto | Block 在 staging buffer 中的 layout：key + offset + length |
| `StagedMemoryCopyParams` / `Tile` | `NoBlockCopy.h` | SM copy kernel 的输入描述 |

## Buffer Pool 与 Fallback 机制

### Pool 路径（正常）
- 初始化时 `cudaHostAlloc` 分配整块 pinned memory + `regUserMr` 一次性注册 RDMA MR
- 运行时 `tryAllocate` 从 free-list 中分配，`free` 归还并合并相邻空闲块
- 零额外系统调用开销

### Fallback 路径（pool 容量不足）
- `cudaHostAlloc` 临时分配 + `regUserMr` 动态注册
- 析构时 `deregUserMr` + `cudaFreeHost`
- 有系统调用开销，但保证功能正确

配置项：`cache_transfer_buffer_size_mb`（`NormalCacheStore::initBufferPool`）

## TP 不对称处理

PD 分离场景下 Prefill 和 Decode 可以使用不同的 TP 大小（如 Prefill TP=2, Decode TP=4）。通过 `partition_count` 和 `partition_id` 实现逻辑切片。

### 核心逻辑

Decode 端在 `DecodeRpcServer::constructRemoteLoadRequest` 中根据两端 TP 拓扑计算分片参数：

**Case 1: D >= P（Decode 并行度 >= Prefill）**

例如 Prefill TP=2, Decode TP=4：

```cpp
int part_cnt = D / P;  // 4/2 = 2
request.set_partition_count(part_cnt);          // 每个 Prefill block 切 2 份
request.set_partition_id(index % part_cnt);     // Decode rank 0,1 取 slice 0,1
request.add_peer_addrs(peer_addrs[index / part_cnt]); // rank 0,1 → Prefill rank 0
```

多个 Decode rank 从同一个 Prefill rank 取数据，各取 block 的不同切片：

```
Prefill rank 0: [===== full block =====]
                  ↓ slice 0    ↓ slice 1
              Decode rank 0  Decode rank 1

Prefill rank 1: [===== full block =====]
                  ↓ slice 0    ↓ slice 1
              Decode rank 2  Decode rank 3
```

**Case 2: P > D（Prefill 并行度 > Decode）**

例如 Prefill TP=4, Decode TP=2：

```cpp
request.set_partition_count(1);   // 不切片，取完整 block
request.set_partition_id(0);
int group_num = P / D;  // 4/2 = 2
// Decode rank 向多个 Prefill rank 发请求
for (int i = 0; i < group_num; i++) {
    request.add_peer_addrs(peer_addrs[index * group_num + i]);
}
```

每个 Decode rank 从多个 Prefill rank 各取完整 block 再拼接：

```
Decode rank 0 ← Prefill rank 0 (full block) + Prefill rank 1 (full block)
Decode rank 1 ← Prefill rank 2 (full block) + Prefill rank 3 (full block)
```

**Case 3: Prefill CP（Context Parallelism）模式**

CP 模式下每个 Prefill rank 持有完整 KV（已做 all-gather），Decode 每个 rank 直接取 1/D 切片：

```cpp
request.set_partition_count(D);
request.set_partition_id(index % D);
request.add_peer_addrs(peer_addrs[index % P]);
```

### Server 端切片执行

Prefill 端收到请求后，在 `loadBlockOnRdmaGathered` 中按 partition 参数只 gather 对应切片：

```cpp
size_t slice_size     = full_lens[i] / partition_count_;
void*  slice_gpu_addr = static_cast<char*>(gpu_addrs[i]) + partition_id_ * slice_size;
```

Gather 只收集当前 partition 的数据到 staging buffer，RDMA Write 只传输该切片。Decode 端接收后 scatter 到自己的 GPU block（block 大小 = slice 大小）。

## 与原始 Per-block 路径对比

| 维度 | Per-block | Gathered |
|------|-----------|----------|
| RDMA Write 次数 | N（每 block 一次） | 1 |
| 额外内存拷贝 | 无（GPU block 直接注册 MR） | D2H gather + H2D scatter |
| 适用前提 | block 已在 host 或已注册 GPU MR | block 在 GPU 且未注册 MR |
| 延迟构成 | N × RDMA RTT | SM kernel + D2H + 1×RDMA RTT + H2D + SM kernel |
| 吞吐优势 | 少量 block | 大量 block（WQE 瓶颈消除） |
| 启用条件 | 默认 | `buffer_pool_` 存在（配置 `cache_transfer_buffer_size_mb > 0`） |

## 可观测性（Metrics）

| Metric | 类型 | 上报端 | 含义 |
|--------|------|--------|------|
| `rtp_llm.cache_store.gathered.pool_free_bytes` | Gauge | 两端 | Pin Memory 缓存池剩余字节 |
| `rtp_llm.cache_store.gathered.staging_fallback_qps` | QPS | 两端 | Fallback 到临时 cudaHostAlloc 的频率 |
| `rtp_llm.cache_store.gathered.d2h_latency_us` | Gauge | Prefill | D2H gather 拷贝耗时 (μs) |
| `rtp_llm.cache_store.gathered.h2d_latency_us` | Gauge | Decode | H2D scatter 拷贝耗时 (μs) |

## 相关文件索引

| 文件 | 职责 |
|------|------|
| `rtp_llm/cpp/disaggregate/cache_store/CacheTransferBufferPool.h/cpp` | Pool 实现 |
| `rtp_llm/cpp/disaggregate/cache_store/TempStagingGuard.h` | allocateStaging + RAII guard |
| `rtp_llm/models_py/bindings/cuda/NoBlockCopy.cc` | execStagedMemoryCopy / Async |
| `rtp_llm/models_py/bindings/common/kernels/sm_copy_kernel.cu` | SM gather/scatter kernel |
| `internal:RdmaCacheStoreServiceImplContext.cpp` | Prefill 端 gathered 逻辑 |
| `internal:RdmaCacheStoreLoadServiceClosure.cpp` | Decode 端 H2D scatter |
| `internal:RdmaMessager.cpp` | Decode 端请求构造 |
| `rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.cpp` | Pool 初始化 (`initBufferPool`) |
