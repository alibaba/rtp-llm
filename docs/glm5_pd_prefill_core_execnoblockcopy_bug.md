# GLM-5 线上 PD+CP+MTP Prefill Core Dump 调查

调查日期: 2026-05-29
对应分支: feature/glm5_cu13
Good commit (用户给出): `7c85d3c3e` (`opt - sparse attention kernels: fast_topk v2 + triton kv scatter`)
Bad commit (定位结论): `706818802` (`fix - fix mtp draft model cuda graph`)

---

## 1. 现象

线上 GLM-5 (PD 分离 + Prefill CP 跨机 + MTP) 在 prefill 节点出现 core，同机 8 rank 同时挂掉。

10.46.54.51 上 rank 0 的 stack：

```
@ rtp_llm::CpuTpBroadcaster::broadcast()
@ rtp_llm::execBroadcastCpu()
@ rtp_llm::tpSyncModelInputs()
@ rtp_llm::MtpExecutor::prefillStep()
@ rtp_llm::MtpExecutor::process()
@ rtp_llm::NormalEngine::step()
@ rtp_llm::NormalEngine::loop()
```

错误信息：

- RANK 0: `CpuTpBroadcaster write to rank 6 (96 bytes) failed: Broken pipe` (CpuTpBroadcaster.cc:413)
- RANK 1/2/3/4/5/7: `CpuTpBroadcaster read from rank 0 (96 bytes) failed: Success` (CpuTpBroadcaster.cc:421)
- RANK 6 (pid 2490589): **没有独立 stack trace**

---

## 2. Stack 解读

### 2.1 雪崩源是 rank 6

- Rank 0 写 rank 6 时 `Broken pipe` -> rank 6 早就死了, 进程已退出, kernel 把 socket 半关
- 其他 rank 读 rank 0 时 errno=Success -> read() 返回 0 字节 (EOF) -> rank 0 也 assert 退出后, 它们跟着挂

整个 crash 是级联结果, **真正第一个死的是 rank 6**.

### 2.2 96 bytes 对应 prefillStep 第一个 tpSync

`rtp_llm/cpp/models/ModelTypes.cc:24-98` 里 `tpSyncModelInputs` 第一步 broadcast 一个 `shape_hints_t` int32 张量, 大小 = `gptModelInputLength`.

数 `ModelTypes.h:61-89` 的 enum:

```
0  comboTokens             12 textTokensMask
1  inputLengths            13 mmFeaturesLocs
2  sequenceLengths         14 mmFeaturesNum
3  prefixLengths           15 mmFeaturesSize
4  maxKernelBlocksPerBatch 16 mmFeaturesDtype
5  maxBlocksPerBatch       17 needAllLogits
6  kvCacheGroupNum         18 mtpHiddenStates
7  kvCacheLayerToGroupLen  19 mtpHiddenStatesDtype
8  kvCacheGroupTypesLen    20 skipRun
9  kvCacheUpdateCopyNum    21 gptModelRequestLength
10 lmOutputIndexes         22 isFakeStream
11 comboPositionIds        23 tensorDeviceMap
                           24 gptModelInputLength <- count
```

24 * sizeof(int32) = **96 bytes**, 完全吻合.

所以 crash 命中的是 `MtpExecutor.cc:884` 的第一段 `tpSyncModelInputs` (target forward 之前). Rank 6 死的时候连本次 prefill 的 forward 都还没跑.

### 2.3 Rank 6 为什么没 stack

业务在 `StackTrace.cc` 里注册的是 SIGSEGV 系列 sighandler. 但 `std::terminate()` (未捕获 C++ 异常 / `__cxa_throw` 没人接) 会走 `abort()` -> SIGABRT, 这条路径**不经过业务 sighandler**, 因此没有应用层 stack 打印.

后台 worker 线程里抛 C++ 异常没人 catch, 完全符合这一现象.

---

## 3. 根因 commit: `706818802`

```
fix - fix mtp draft model cuda graph
 rtp_llm/cpp/models/ModelTypes.h                    |  8 +-
 rtp_llm/cpp/models/PyWrappedModel.cc               |  5 +-
 rtp_llm/cpp/models/PyWrappedModel.h                |  6 +
 rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc   | 22 +-
 rtp_llm/models_py/bindings/core/ExecOps.cc         | 56 +++-
 rtp_llm/models_py/kernels/cuda/fast_topk/fast_topk.py  | 20 +-
```

其他文件都是加日志 / 加 virtual getter / fast_topk cuda graph capture 兜底, 不会造成静默 abort. 真正引入风险的是 `ExecOps.cc` 里对 `execNoBlockCopy` 的改写.

### 3.1 改动前 (good)

`rtp_llm/models_py/bindings/core/ExecOps.cc` (commit `706818802^`):

```cpp
at::cuda::CUDAStream& getNoBlockCopyStream() {
    static thread_local auto stream = at::cuda::getStreamFromPool(/*isHighPriority=*/false);
    return stream;
}

void execNoBlockCopy(const CopyParams& params) {
    params.check();
    const auto& src = params.src;
    const auto& dst = params.dst;
#if USING_CUDA
    auto stream = getNoBlockCopyStream().stream();
    check_cuda_value(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(),
                                     src.nbytes(), cudaMemcpyDefault, stream));
    check_cuda_value(cudaStreamSynchronize(stream));
    check_cuda_error();
#else
    dst.copy_(src);
#endif
}
```

- `getNoBlockCopyStream` 用 `thread_local` 缓存 stream
- 没有 `DeviceGuard`, 直接用调用者线程的 current device
- 即使 device / stream 不匹配, 至少是"一直一致地错", 大多数情况下 cudaMemcpyDefault 还能容忍

### 3.2 改动后 (bad, HEAD)

`rtp_llm/models_py/bindings/core/ExecOps.cc:554-619`:

```cpp
int getDevicePointerDevice(const void* ptr) {
    cudaPointerAttributes attr;
    auto status = cudaPointerGetAttributes(&attr, ptr);
    if (status != cudaSuccess) { cudaGetLastError(); return -1; }
    if (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged) {
        return attr.device;
    }
    return -1;
}

int resolveNoBlockCopyDevice(const torch::Tensor& dst, const torch::Tensor& src) {
    if (!src.is_cuda() && !dst.is_cuda()) return -1;
    // Prefer the real pointer owner. Cache-store worker threads may not inherit
    // the rank's CUDA current device, and from_blob(cuda) can therefore carry a
    // stale device index even though the pointer itself is valid on this rank.
    if (dst.is_cuda()) {
        auto device = getDevicePointerDevice(dst.data_ptr());
        if (device >= 0) return device;
    }
    if (src.is_cuda()) {
        auto device = getDevicePointerDevice(src.data_ptr());
        if (device >= 0) return device;
    }
    return static_cast<int>(getDeviceId());
}

void execNoBlockCopy(const CopyParams& params) {
    params.check();
    const auto& src = params.src;
    const auto& dst = params.dst;
#if USING_CUDA
    if (src.data_ptr() == dst.data_ptr()) return;
    if (!src.is_cuda() && !dst.is_cuda()) {
        std::memcpy(dst.data_ptr(), src.data_ptr(), src.nbytes());
        return;
    }

    auto copy_device = resolveNoBlockCopyDevice(dst, src);
    RTP_LLM_CHECK_WITH_INFO(copy_device >= 0,
        "execNoBlockCopy failed to resolve CUDA copy device.");   // <-- 抛异常点 (A)
    DeviceGuard guard(copy_device);                                // <-- 切到真实 device
    auto        stream = getNoBlockCopyStream().stream();          // <-- 仍 thread_local
    check_cuda_value(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(),
                                     src.nbytes(), cudaMemcpyDefault, stream));
    check_cuda_value(cudaStreamSynchronize(stream));
    check_cuda_error();
#else
    dst.copy_(src);
#endif
}
```

### 3.3 调用方都是后台 worker 线程

```
rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreServiceImpl.cpp:133
rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreLoadServiceClosure.cpp:63
rtp_llm/cpp/disaggregate/cache_store/TcpBlockReadClosure.cpp:69
rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.cpp:190
rtp_llm/cpp/cache/connector/p2p/transfer/tcp/CudaCopyUtil.cc:33
rtp_llm/cpp/cache/connector/p2p/transfer/tcp/CudaCopyUtil.cc:55
rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.cc:824
```

注释 (`resolveNoBlockCopyDevice` 上面) 自述了这个改动的目标场景:
> Cache-store worker threads may not inherit the rank's CUDA current device

也就是 **PD 分离 + cache_store** 路径. 跟线上部署形态完全一致.

---

## 4. 两条具体崩溃路径

### 路径 A: `copy_device < 0` 触发 RTP_LLM_CHECK_WITH_INFO

发生条件:
- `resolveNoBlockCopyDevice` 走 `cudaPointerGetAttributes`. 对于 host pinned / non-CUDA pointer / 部分 managed 边界状态可能返回 `cudaErrorInvalidValue`
- 函数返回 -1
- `RTP_LLM_CHECK_WITH_INFO` 抛 `std::runtime_error`
- worker 线程没有 try/catch -> `std::terminate()` -> `abort()` -> SIGABRT
- **SIGABRT 不走业务 SIGSEGV sighandler**, 进程静默退出, 无业务 stack

### 路径 B: thread_local stream 与 DeviceGuard 切的 device 失配

发生条件:
- 同一个 worker 线程先后服务多个 device 的 copy (cache_store 跨 GPU 转 KV 很常见)
- 线程第一次调用时, `getNoBlockCopyStream()` 用 `thread_local` 把 stream 锁死在**当时 current device** (假设是 device X) 的 pool
- 后续调用 `DeviceGuard(Y)` 切到 device Y, 但 `getNoBlockCopyStream()` 返回的**仍是 device X 的 stream**
- `cudaMemcpyAsync(dst_on_Y, src_on_Y, stream_on_X)` 在 CUDA 上是非法用法, 返回 `cudaErrorInvalidResourceHandle` 之类
- `check_cuda_value` 抛 -> 同样 `std::terminate` -> SIGABRT -> 无 stack

两条路径都满足"后台线程静默 abort, rank 进程随之退出"的现象.

---

## 5. 与 sync 失败的因果链

```
worker thread (cache_store) on rank 6
   |
   v
execNoBlockCopy 失败 -> std::terminate -> SIGABRT
   |
   v
rank 6 进程退出 (无业务 stack)
   |
   v
某一刻 rank 0 推进到 prefillStep
   -> tpSyncModelInputs 第一段 (shape_hints, 96 bytes)
   -> writeAll(peer_fds_[6], 96 bytes) 发现 EPIPE -> broken pipe
   -> RTP_LLM_CHECK_WITH_INFO -> rank 0 abort
   |
   v
rank 1/2/3/4/5/7 在 readAll(peer_fds_[0], 96 bytes) 上等
   -> read 返回 0 (errno 没改, "Success") -> n != 96 -> 全部 abort
```

`CpuTpBroadcaster` 是 UDS intra-node, 这条链全在 10.46.54.51 一台机上发生.

跟另一台 10.46.54.68 上看到的 `SparseMlaParams::fillCpPlanParams` "All q indices were filtered out" warning **没有因果关系**:
- 不同机器, 时间还晚 3 分钟
- 那个 warning 是 forward 内部的 CP plan 阶段, 远在 tpSync 之后, 不可能往前打死同机别的 rank
- 那条线属于独立的 CP 退化情况, 应单独跟踪 (e.g. cp_size=8 时遇到 padded_total > 实际 length 导致某 rank 全 padding)

---

## 6. 修复建议

### 方案 1 (推荐, 最小改动): stream 不再 thread_local

```cpp
void execNoBlockCopy(const CopyParams& params) {
    params.check();
    const auto& src = params.src;
    const auto& dst = params.dst;
#if USING_CUDA
    if (src.data_ptr() == dst.data_ptr()) return;
    if (!src.is_cuda() && !dst.is_cuda()) {
        std::memcpy(dst.data_ptr(), src.data_ptr(), src.nbytes());
        return;
    }

    auto copy_device = resolveNoBlockCopyDevice(dst, src);
    if (copy_device < 0) {
        copy_device = static_cast<int>(getDeviceId());   // 不要 CHECK, 兜底到 caller device
    }
    DeviceGuard guard(copy_device);
    // 关键: 在 DeviceGuard 之后, 从 current device 的 pool 取 stream
    auto stream = at::cuda::getStreamFromPool(/*isHighPriority=*/false).stream();
    check_cuda_value(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(),
                                     src.nbytes(), cudaMemcpyDefault, stream));
    check_cuda_value(cudaStreamSynchronize(stream));
    check_cuda_error();
#else
    dst.copy_(src);
#endif
}
```

要点:
- 不再 `RTP_LLM_CHECK_WITH_INFO(copy_device >= 0)`, 改成兜底到 `getDeviceId()`, 避免 worker 线程因解析失败 abort
- stream 在 `DeviceGuard` 之后每次现取, 跟 current device 绑定一致
- `getNoBlockCopyStream` 这个 helper 在新语义下没意义, 可以直接删

### 方案 2: 保留 stream pool 缓存, 改成 per-device

```cpp
at::cuda::CUDAStream getNoBlockCopyStream(int device) {
    static std::array<std::optional<at::cuda::CUDAStream>, 16> cache{};
    static std::array<std::once_flag, 16> init_flags{};
    std::call_once(init_flags[device], [device] {
        DeviceGuard g(device);
        cache[device] = at::cuda::getStreamFromPool(false);
    });
    return *cache[device];
}
```

避免每次 copy 新建 stream 的开销, 同时保证 stream 与 device 一致.

### 兜底配套
- worker 线程顶层 (TcpCacheStoreServiceImpl / TcpCacheStoreLoadServiceClosure 的 closure 入口) 加一层 try/catch, 把异常变 LOG_ERROR + 返回 status, 而不是让 worker 线程 `std::terminate`. 这是更深层的健壮性改进, 跟本次 bug 修复并行做.

---

## 7. 二级嫌疑 (本次不直接相关, 留档备查)

### 7.1 commit `3a29591e6` 改了 MtpExecutor.cc 第二个 tpSync 前的逻辑

```cpp
// before
if (cp_enabled) {
    model_input.last_hidden_states = torch::Tensor();
}

// after (HEAD)
if (cp_enabled) {
    auto target_mtp_hidden = model_->getMtpTargetHiddenStates(-1);
    if (target_mtp_hidden.defined() && target_mtp_hidden.numel() > 0) {
        model_input.last_hidden_states = torch::Tensor();
    }
}
```

风险点: 不同 rank 独立判定 `getMtpTargetHiddenStates(-1)` 是否存在, 若各 rank 结论不一致 (rank 0 跑完 sampler 把 buffer 消费 / 旋转, 其他 rank 没跑), 则 `last_hidden_states.numel()` 在第二段 tpSync shape_hints 里分歧 -> packed buffer 大小对不上 -> 第二段 tpSync crash.

但本次 crash 命中的是**第一段** tpSync (96 bytes 已对上), 这块不是本次根因. 修完 `execNoBlockCopy` 后, 如果还有别的 sync crash, 再回来看这里.

### 7.2 commit `13a48d8b0` Optimize CP MLA q gather

直接改 `flashmla_sparse_cp_impl.py`, 跟 10.46.54.68 上 `SparseMlaParams::fillCpPlanParams` warning 是同一文件邻近代码, 可能跟那条独立的 CP 退化告警相关, 跟本次 crash 无直接关系.

---

## 8. 后续验证清单

- [ ] 应用方案 1 的 patch, 本地至少跑通 `smoke_h20_mla` 或带 PD 分离的烟测
- [ ] 线上灰度复现场景验证, 观察 cache-store worker 线程是否还会触发 abort (建议在 `execNoBlockCopy` 内部临时加 INFO 日志: thread_id + resolved device + src/dst.is_cuda + tensor.device, 方便事后归因)
- [ ] 给 cache_store 后台 worker 线程加 try/catch 兜底
- [ ] 跟踪二级嫌疑 7.1, 在多个 step 跑过之后看是否还有 96 bytes 之外的 tpSync 字节数不一致
