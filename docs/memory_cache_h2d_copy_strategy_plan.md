# RTP-LLM Memory Cache H2D/D2H Copy 多方案选路与 CUDA 3D Batch 实施计划

## 1. 目标

为 `KVCacheMemoryConnector` 的 Memory↔Device 双向路径增加显式、可观测、可回退的 copy 方案选路，并为单 group MiniMax-M3 + MTP 增加 `cudaMemcpy3DBatchAsync` 实验路径。

本计划独立于 `memory_cache_remote_eviction_plan.md`，不修改 Memory→Remote 淘汰语义。

目标：

- 通过环境变量分别指定 H2D、D2H copy 方案。
- 保留当前 generic、split-KV SM、staged SM、`cudaMemcpyBatchAsync` 路径。
- 新增 `cudaMemcpy3DBatchAsync` 路径。
- MiniMax-M3 + MTP 按真实 Device layout 分成 main KV、main idx_K、MTP KV、MTP idx_K 四类规则段。
- eligibility 不满足或运行失败时安全回退。
- 日志和指标明确记录 requested/effective/fallback mode。
- D2H 复用同一套 regular-run 构造及 CUDA 3D Batch executor，仅交换 source/destination 与 source access order。
- 不改变 cache match、in-flight、Memory→Remote 淘汰和成功后释放 Memory backing 的语义。

### 1.1 D2H 对称扩展

D2H 使用独立开关，避免 H2D 灰度策略影响 GPU→Memory 写回：

```text
MEMORY_CACHE_D2H_COPY_MODE=auto
MEMORY_CACHE_D2H_COPY_STRICT=0
ENABLE_MEMORY_CACHE_D2H_3D_BATCH_AUTO=0
```

`MEMORY_CACHE_D2H_COPY_MODE` 支持 `auto`、`generic`、`memcpy_batch`、`memcpy3d_batch`、`staged_sm`。第一版 `auto` 默认保持原选路；只有打开 D2H auto 灰度开关时才优先尝试 3D Batch。显式 `memcpy3d_batch` 配合 strict 可用于单测和 smoke，确保没有静默 fallback。

CUDA 3D Batch 的 D2H source 位于 GPU，因此 descriptor 使用 `cudaMemcpySrcAccessOrderDuringApiCall`；H2D source 位于 host，继续使用 `cudaMemcpySrcAccessOrderStream`。

## 2. 当前 H2D 链路

```text
KVCacheMemoryConnector::asyncRead()
    ↓
buildCopyPlanForRead()/buildPrefixCopyPlanForRead()
    ↓
startCopyAsync()
    ↓
sendCopyPlan()
    ↓ TP broadcast
KVCacheMemoryConnector::copyCache()
    ↓
选择具体copy实现
    ↓
stream完成并返回worker response
    ↓
所有TP rank成功
    ↓
setMemoryReuseBlockNum()
    ↓
移除Memory cache entry并释放backing
```

当前 `copyCache()` 主要选路：

```text
prefix typed item       → copyPrefixMemoryItems
包含Disk backing        → generic Memory + Disk copy
DSV4 typed layout       → staged SM copy
typed slots             → cudaMemcpyBatchAsync
其他                    → execNoBlockCopy
                           ├── eligible + ENABLE_MEMORY_CACHE_SM_COPY=1 → split-KV SM
                           └── 否则 → 多次cudaMemcpyAsync
```

## 3. 新增环境变量

### 3.1 主选路变量

```text
MEMORY_CACHE_H2D_COPY_MODE=auto
```

合法值：

| 值 | 含义 |
|---|---|
| `auto` | 根据 layout 和 CUDA 能力自动选择 |
| `generic` | 使用 `execNoBlockCopy`，不启用 split-KV SM |
| `memcpy_batch` | 使用 `cudaMemcpyBatchAsync` |
| `memcpy3d_batch` | 使用 `cudaMemcpy3DBatchAsync` |
| `staged_sm` | 连续 H2D 到 Device staging，再用 SM scatter |
| `split_kv_sm` | 使用现有 split-KV staging/scatter 路径 |

解析时统一转小写；未知值在初始化阶段报错，不允许运行时静默解释成 `auto`。

建议在 `KVCacheConfig` 中新增枚举，而不是在热路径重复解析字符串：

```cpp
enum class MemoryCacheH2DCopyMode {
    AUTO,
    GENERIC,
    MEMCPY_BATCH,
    MEMCPY3D_BATCH,
    STAGED_SM,
    SPLIT_KV_SM,
};
```

### 3.2 严格模式

```text
MEMORY_CACHE_H2D_COPY_STRICT=0
```

- `0`：指定方案不满足 eligibility 或执行失败时，按安全 fallback 链继续。
- `1`：指定方案不可用或失败时，本次 H2D 失败。用于单测和 benchmark，防止实际跑了 fallback 却误记为目标方案性能。

`auto` 模式忽略 strict，因为 auto 本身就允许选路。

### 3.3 3D Batch auto 灰度开关

```text
ENABLE_MEMORY_CACHE_H2D_3D_BATCH_AUTO=0
```

- `0`：`auto` 保持当前选路，不主动选择 `memcpy3d_batch`；仍可通过 `MEMORY_CACHE_H2D_COPY_MODE=memcpy3d_batch` 强制验证。
- `1`：`auto` 在满足 CUDA 版本、单 group 和 regular-run eligibility 时允许选择 `memcpy3d_batch`。
- 第一版默认值必须为 `0`，待正确性、性能和稳定性验证通过后再灰度开启。

### 3.4 与旧开关的关系

现有：

```text
ENABLE_MEMORY_CACHE_SM_COPY
```

兼容优先级：

```text
显式 MEMORY_CACHE_H2D_COPY_MODE != auto
    → 新变量优先，旧bool不参与H2D选路

MEMORY_CACHE_H2D_COPY_MODE == auto
    → auto选路
    → 对legacy非typed layout，仅当ENABLE_MEMORY_CACHE_SM_COPY=1时考虑split_kv_sm
```

第一版不删除旧开关。稳定后可将其标记 deprecated。

## 4. 统一选路框架

新增入口：

```cpp
struct MemoryCopyExecutionResult {
    bool                   success{false};
    MemoryCacheH2DCopyMode requested_mode;
    MemoryCacheH2DCopyMode effective_mode;
    std::string            fallback_reason;
    size_t                 logical_bytes{0};
    size_t                 op_count{0};
};

MemoryCopyExecutionResult executeMemoryH2DCopy(
    const MemoryOperationRequestPB& request,
    const std::vector<LayerRegionSlot>& slots);
```

H2D 与 D2H 各自使用独立配置进入选路；两者共享 regular-run builder 和 executor。以下情况保持原有专用路径：

- Disk backing 或 Memory/Disk 混合。
- prefix-tree 的 `COMPRESSED_KV`、`STATE_SWA_KV`，除非后续单独验证。
- 包含 host-only target slot。

建议的 auto 顺序：

```text
DSV4 typed且staged eligible
    → staged_sm
否则 ENABLE_MEMORY_CACHE_H2D_3D_BATCH_AUTO=1
    且单group且3D layout eligible且CUDA >= 13.0
    → memcpy3d_batch
否则typed layout且CUDA >= 12.8
    → memcpy_batch
否则legacy split-KV eligible且旧SM开关开启
    → split_kv_sm
否则
    → generic
```

强制模式 fallback 链：

```text
memcpy3d_batch → memcpy_batch → generic
memcpy_batch   → generic
staged_sm      → memcpy3d_batch → memcpy_batch → generic
split_kv_sm    → generic
generic        → 无fallback
```

若 `STRICT=1`，强制模式只执行第一项。

## 5. MiniMax-M3 + MTP 的真实布局

当前 `BlockPoolConfigHelper::createConfig()` 将 Device 大 buffer 排列为：

```text
[main_kv]
[main_scale / main_idx_K]
[mtp_kv]
[mtp_scale / mtp_idx_K]
```

主模型内部固定某个 block ID 时：

```text
main KV相邻层pitch
    = device_block_num × main_kv_block_stride

main idx_K相邻层pitch
    = device_block_num × main_scale_block_stride
```

MTP 使用独立 `MemoryLayoutConfig`。即使单层 shape 与主模型相同，main→MTP 边界中间隔着完整的 main idx_K pool，不能沿用主模型 layer pitch。

Memory cache 一个逻辑 block 则按 `layerRegionSlots()` 顺序平铺：

```text
[main L0 KV][main L0 idx_K]
[main L1 KV][main L1 idx_K]
...
[main LN-1 KV][main LN-1 idx_K]
[MTP L0 KV][MTP L0 idx_K]
```

因此每个逻辑 block 应形成最多四个 3D segment：

```text
segment 0：main KV，depth=main_layer_num
segment 1：main idx_K，depth=main_layer_num
segment 2：MTP KV，depth=1
segment 3：MTP idx_K，depth=1
```

若未启用 idx_K/scale，则删除对应 segment。若没有 MTP，则只保留 main segments。

## 6. 不硬编码 MiniMax：按地址形成规则 run

实现不应写死“四段”或模型名。对每个 Memory block，先沿 slot/component 顺序生成逻辑 tile：

```cpp
struct H2DLogicalTile {
    const void* host;
    void*       device;
    size_t      bytes;
    int         global_layer_id;
    int         component_index;
};
```

把相邻 tile 合并为 maximal regular run。只有同时满足以下条件才能进入同一 run：

```text
component_index相同
bytes相同且非0
host地址差恒定
device地址差恒定
host/device地址范围均不重叠
没有NULL_BLOCK_IDX或被跳过slot形成的空洞
```

以首两个 tile 得到：

```cpp
host_layer_pitch   = host[1]   - host[0];
device_layer_pitch = device[1] - device[0];
```

后续 tile 必须满足：

```cpp
host[i]   == host[0]   + i * host_layer_pitch;
device[i] == device[0] + i * device_layer_pitch;
```

main→MTP 地址差不符合主模型 pitch，会自然切断为新 run。这样同一实现也能处理无 MTP、多个 MTP layout 或未来其他单 group 模型。

depth=1 的 singleton run 仍可生成一个 3D op；也可以统一放入普通 BatchAsync。第一版为简化控制面，全部生成 `cudaMemcpy3DBatchOp`。

## 7. `cudaMemcpy3DBatchAsync` 参数构造

CUDA 13 pointer-to-pointer 3D operand 以 byte 为 element。对一个 regular run：

```cpp
cudaMemcpy3DBatchOp op{};

op.src.type = cudaMemcpyOperandTypePointer;
op.src.op.ptr.ptr = const_cast<void*>(run.host_base);
op.src.op.ptr.rowLength = run.host_layer_pitch;
op.src.op.ptr.layerHeight = 1;

op.dst.type = cudaMemcpyOperandTypePointer;
op.dst.op.ptr.ptr = run.device_base;
op.dst.op.ptr.rowLength = run.device_layer_pitch;
op.dst.op.ptr.layerHeight = 1;

op.extent = make_cudaExtent(
    run.bytes,
    1,
    run.depth);

op.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
op.flags = 0;
```

约束：

- `rowLength >= extent.width`。
- `layerHeight >= extent.height`。
- width/height/depth 均不得为 0。
- batch 内 copy 不得存在依赖或 src/dst overlap。
- stream 不能是 legacy NULL stream。
- Memory cache backing 必须保持有效直到 stream 完成。

多个 blocks：

```text
block0：最多4 ops
block1：最多4 ops
...
```

一次提交：

```cpp
cudaMemcpy3DBatchAsync(ops.size(), ops.data(), 0, stream);
cudaStreamSynchronize(stream);
```

假设主模型 88 层、1 层 MTP、KV+idx_K 两个 component，复制 B 个 blocks：

```text
当前逐tile描述数：B × 89 × 2
3D Batch op数：   B × 4
```

例如 B=8：从 1424 个逻辑 tiles 压缩成约 32 个 3D ops。

注意：descriptor 数减少不等于底层只产生 32 个连续 DMA。跨层目标仍有大 pitch，最终收益必须由 profiler/benchmark 验证。

## 8. CUDA 版本和编译隔离

110 当前 CUDA 13.0 头文件包含 `cudaMemcpy3DBatchAsync`。新增实现使用：

```cpp
#if CUDART_VERSION >= 13000
bool exec3DBatchedMemoryCopy(...);
#else
bool exec3DBatchedMemoryCopy(...) {
    return false;
}
#endif
```

不能复用当前 `cudaMemcpyBatchAsync` 的 `CUDART_VERSION >= 12080` 条件。

建议修改：

```text
rtp_llm/models_py/bindings/NoBlockCopy.h
rtp_llm/models_py/bindings/cuda/NoBlockCopy.cc
rtp_llm/models_py/bindings/NoBlockCopyDefault.cc
```

新增结构：

```cpp
struct BatchedMemoryCopy3DRun {
    const void* src{nullptr};
    void*       dst{nullptr};
    size_t      width_bytes{0};
    size_t      src_layer_pitch_bytes{0};
    size_t      dst_layer_pitch_bytes{0};
    size_t      depth{0};
};

struct BatchedMemoryCopy3DParams {
    std::vector<BatchedMemoryCopy3DRun> runs;
    int device_index{-1};
};
```

## 9. Connector 修改点

建议修改：

```text
rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h
rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.cc
rtp_llm/cpp/config/ConfigModules.h
rtp_llm/server/server_args/kv_cache_group_args.py
rtp_llm/cpp/pybind/ConfigInit.cc
rtp_llm/ops/libth_transformer_config.pyi
```

新增：

```cpp
bool tryCopyCacheWith3DBatchedMemoryCopy(
    const MemoryOperationRequestPB& request,
    CopyDirection direction,
    const std::vector<LayerRegionSlot>& slots,
    MemoryCopyExecutionResult& result);
```

第一版明确限制：

```cpp
if (direction != CopyDirection::H2D) return false;
if (cache_config_.groupNums() != 1) return false;
if (hasDiskItem(request)) return false;
if (hasPrefixKindItem(request)) return false;
```

`groupNums()==1` 是必要但不充分条件；最终由实际地址 run 检查决定是否 eligible。

## 10. 完成与生命周期

保持当前同步边界：

```text
在专用NoBlockCopy stream提交
    ↓
cudaStreamSynchronize
    ↓
worker response success
    ↓
TP broadcast全部成功
    ↓
Memory read_done(success=true)
    ↓
更新memoryReuseBlockNum
    ↓
removeIfMatch + releaseCacheBacking
```

第一版不改变为 event-only 异步完成，避免 worker 提前返回后 Memory backing 被释放。

若 `cudaMemcpy3DBatchAsync` API 返回失败，必须先同步/清理当前 stream 状态，再决定 fallback。为避免部分 copy 已经执行后与 fallback 重叠，推荐：

- 参数校验和 eligibility 在提交前完成。
- API 提交失败时查询错误；只有确认没有入队成功才 fallback。
- API 已成功入队但 stream 执行失败时，本次 H2D 直接失败，不在同一目标地址上重跑另一方案。

## 11. 可观测性

日志：

```text
memory h2d copy:
requested_mode
effective_mode
fallback_reason
block_count
logical_tile_count
submitted_op_count
logical_bytes
elapsed_us
cuda_runtime_version
```

建议指标：

```text
memory_cache_h2d_copy_qps{mode}
memory_cache_h2d_copy_fail_qps{mode}
memory_cache_h2d_copy_fallback_qps{requested,effective,reason}
memory_cache_h2d_copy_latency_us{mode}
memory_cache_h2d_copy_bytes{mode}
memory_cache_h2d_copy_op_count{mode}
```

## 12. 单元测试

### 12.1 环境变量解析

- 六种合法值正确映射。
- 大小写归一化。
- 非法值初始化失败。
- 新变量显式设置时覆盖 `ENABLE_MEMORY_CACHE_SM_COPY`。
- strict/non-strict 行为正确。

### 12.2 Regular run 划分

- 等宽、等 host/device pitch 的 N 层合并为一个 run。
- component 不同必须拆分。
- payload bytes 不同必须拆分。
- main→MTP pitch 改变时必须拆分。
- `NULL_BLOCK_IDX`、跳过 slot、地址空洞必须拆分或判定不 eligible。
- 不允许任何 batch 内 src/dst overlap。

### 12.3 MiniMax-M3 + MTP

构造：

```text
main layers = 4
mtp layers = 1
components = KV + idx_K
blocks = 3
```

预期：

```text
每block 4 runs
总3D ops = 12
main KV/idx_K depth=4
MTP KV/idx_K depth=1
```

逐字节比较 H2D 后所有 main/MTP KV 与 idx_K。

### 12.4 Fallback

- CUDA < 13：3D mode fallback 到 Batch/generic。
- 多 group：3D mode 不 eligible。
- 不规则 layer stride：不合并错误的 run。
- API 提交前校验失败可 fallback。
- stream 执行失败不可重复覆盖，整个 context 失败。
- strict 模式不 fallback。

### 12.5 生命周期

- stream 完成前 Memory cache item 保持 in-flight。
- TP 任一 rank 失败时不移除 Memory cache entry。
- 全部成功后只释放一次 backing。
- timeout/cancel 不产生 use-after-free。

## 13. Benchmark

比较：

```text
generic
memcpy_batch
memcpy3d_batch
staged_sm
split_kv_sm（eligible场景）
```

变量：

```text
block_count：1, 2, 4, 8, 16, 32
main layers：真实MiniMax配置
MTP：off/on
KV dtype：BF16/FP8
Host backing：pageable/pinned
并发H2D任务：1, 2, 4
```

观察：

```text
端到端memory read latency
有效GB/s
CPU API提交耗时
CUDA memcpy engine利用率
SM利用率
descriptor/op数量
与模型compute overlap程度
```

所有强制方案 benchmark 使用：

```text
MEMORY_CACHE_H2D_COPY_STRICT=1
```

并校验日志中的 `effective_mode`。

## 14. 分阶段落地

### 阶段一：选路框架

- 增加枚举和环境变量。
- 把现有路径接入统一 dispatcher。
- 增加 effective/fallback 日志和指标。
- 默认结果与现状一致。

### 阶段二：3D Batch 原型

- 实现 regular run builder。
- 实现 CUDA 13 `exec3DBatchedMemoryCopy()`。
- 开放单 group、Memory backing 的 H2D/D2H；两个方向分别灰度。
- 完成 MiniMax-M3 + MTP 精度测试。

### 阶段三：性能验证

- 在 110 环境 microbenchmark。
- 对比 current BatchAsync 和 staged SM。
- 确定不同 block_count 的收益区间。

### 阶段四：auto 策略

- 原型期保持 H2D/D2H 两个 3D Batch auto 开关均为 `0`，`auto` 默认沿用当前选路。
- 性能和稳定性达标后，再灰度设置为 `1`，将 eligible 单 group 纳入 auto。
- 保留强制环境变量用于回滚。

## 15. 验收标准

- 所有模式输出与 generic baseline 逐字节一致。
- MiniMax 主模型、MTP KV、idx_K 全部正确。
- 3D 模式不会跨 main/MTP 非等 pitch 边界错误合并。
- 不支持场景有确定 fallback 或 strict failure。
- 没有 Memory backing 提前释放。
- TP rank 对 effective mode 和执行结果一致。
- 默认配置上线初期不改变现有选路。
- benchmark 明确给出 3D Batch 相对 BatchAsync/staged SM 的收益或不采用结论。

## 16. 最终建议

第一版把 `cudaMemcpy3DBatchAsync` 作为实验性强制模式，不直接替换当前 auto 路径：

```text
MEMORY_CACHE_H2D_COPY_MODE=memcpy3d_batch
MEMORY_CACHE_H2D_COPY_STRICT=1
```

确认 MiniMax-M3 + MTP 的正确性和收益后，再加入 auto。单 group 只作为快速前置过滤，真正安全条件必须来自逐地址 regular-run 验证。
