# Phase 1: 内存管理与设备运行时 — 分步执行验证方案 (v0.3)

> **基于 v0.2 代码审核结果修正**
> 
> v0.2 存在以下问题：目录路径错误（计划用 `rtp_llm/cpp/ascend/` 但实际在 `rtp_llm/models_py/bindings/ascend/`）；引用的 `DeviceData.h` DeviceType 枚举和 `allocator.h` 不存在；约 40% 内容描述的代码已实现；API 常量名 `torch::kNPU` 应为 `torch::kPrivateUse1`。本版已全部修正。

**Goal:** 补齐 Ascend 内存管理与设备运行时缺失环节，使 tensor 可正确分配到 NPU 设备。

**参考架构（实际）：**
- Ascend 适配代码路径：`rtp_llm/models_py/bindings/ascend/`（已存在 `ascend_types_hdr.h`、`ascend_host_utils.{h,cc}`）
- ROCm 适配代码路径：`rtp_llm/models_py/bindings/rocm/`
- CUDA 适配代码路径：`rtp_llm/models_py/bindings/cuda/`
- 核心定义：`rtp_llm/models_py/bindings/core/`
- 缓存层（BlockPool等）：`rtp_llm/cpp/cache/`

**内存类型定义（已有）：** `rtp_llm/models_py/bindings/core/Types.h:10-15`
```cpp
typedef enum memorytype_enum {
    MEMORY_CPU,
    MEMORY_CPU_PINNED,
    MEMORY_GPU,
    MEMORY_NPU        // ← 已有 Ascend 条目
} MemoryType;
```

**设备类型机制：** 不使用独立的 `DeviceType` 枚举。设备选择通过编译宏 `USING_CUDA` / `USING_ROCM` / `USING_ASCEND` + `#if/#elif` 条件编译实现。Ascend 使用 `torch::kPrivateUse1`（而非 `torch::kNPU`）作为 PyTorch 设备类型。

**BUILD 基础设施（Phase 0 已完成）：**
- `def.bzl` 已有 `ascend_copts()` 和 `if_ascend()`，`copts()` 中包含 `-DUSING_ASCEND=1`
- `@local_config_ascend//ascend:ascend_headers` 等仓库可用
- `bazel/device_defs.bzl` 已有 `using_ascend` 分支

---

## 总览：Phase 1 实际需要完成的任务

```
Phase 1
├── Task 1: Torch_ext.h — 添加 Ascend 分支 (StreamType/GET_CURRENT_STREAM/CHECK_CUDA)
├── Task 2: CudaOps.cc — 实现 Ascend runtimeCopy (参考 ROCm 方案)
├── Task 3: BlockPool.cc — 参数化 torch::kCUDA + .is_cuda()
├── Task 4: MemoryEvaluationHelper.cc — 添加 Ascend 内存查询分支
├── Task 5: ExecOps.cc — 修复残留 torch::kCUDA (execCreateMoeExpertStates)
├── Task 6: KVCacheManager.cc / MemoryLayoutStrategy.cc / TypeConvert.h / LayerBlockConverterImpl.h — 适配 .is_cuda()
├── Task 7: [跳过] BlockInfo.h — 使用 is_cuda 兼容性判断替代新增 is_npu
├── Task 8: BUILD 文件 select() 扩展 (补充遗漏)
├── Task 9: 集成验证
└── Task 10: (可选) Ascend Allocator 实现 — 基于 KVCacheAllocator 接口
```

---

## Task 1: Torch_ext.h — 添加 Ascend 分支

**Files:**
- Modify: `rtp_llm/models_py/bindings/common/Torch_ext.h:7-29`

### Step 1: 当前代码

```cpp
// Torch_ext.h:7-29
#if USING_ROCM
#include <rtp_llm/models_py/bindings/rocm/amd_bfloat16.h>
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>
#include "rtp_llm/models_py/bindings/rocm/kernels/fused_qk_rmsnorm.h"
#include "rtp_llm/models_py/bindings/rocm/kernels/layernorm_kernels.h"
using bf16_type  = amd_bfloat16;
using StreamType = hipStream_t;
#define GET_CURRENT_STREAM() at::hip::getCurrentHIPStream().stream()
#elif USING_CUDA
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
...
using bf16_type  = nv_bfloat16;
using StreamType = cudaStream_t;
#define GET_CURRENT_STREAM() at::cuda::getCurrentCUDAStream(at::cuda::current_device()).stream()
#endif

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
```

### Step 2: 修改为

在 `#elif USING_CUDA` 块后、`#endif` 前插入 (或在 `#elif USING_CUDA` 改为 `#elif` 链):

```cpp
#elif USING_CUDA
...
using bf16_type  = nv_bfloat16;
using StreamType = cudaStream_t;
#define GET_CURRENT_STREAM() at::cuda::getCurrentCUDAStream(at::cuda::current_device()).stream()
#elif USING_ASCEND
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
using bf16_type  = at::BFloat16;  // 或使用 aclFloat16 当需要
using StreamType = aclrtStream;
#define GET_CURRENT_STREAM() c10_npu::getCurrentNPUStream().stream()
#endif

// 将 CHECK_CUDA 修改为同时兼容 CUDA/ROCm/NPU:
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda() || x.is_privateuseone(), #x " must be a CUDA or NPU tensor")
```

**注意:** `c10_npu::getCurrentNPUStream()` 的 API 签名在不同 `torch_npu` 版本中可能有差异。执行前运行兼容性检查（见附录）。

### Step 3: 验证编译

```bash
bazel build //rtp_llm/models_py/bindings:torch_ext --config=ascend
```

**验收标准:** 编译通过，`GET_CURRENT_STREAM` 在 Ascend 下返回 `aclrtStream`。

---

## Task 2: CudaOps.cc — 实现 Ascend runtimeCopy

**Files:**
- Modify: `rtp_llm/models_py/bindings/core/CudaOps.cc:234-262`

### Step 1: 当前代码

Ascend 分支（行 234-262）全部 stub throw `OpException(ERROR_UNIMPLEMENTED)`。

### Step 2: 修改为（参考同一文件中 ROCm 分支行 264-338 的模式）

```cpp
#elif USING_ASCEND

// ============================================================
// Copy ops (Ascend)
// ============================================================

namespace rtp_llm {

void runtimeCopy(const CopyParams& params) {
    params.check();
    const auto& src = params.src;
    const auto& dst = params.dst;
    if (src.data_ptr() == dst.data_ptr()) {
        return;
    }
    // Ascend: 使用 PyTorch tensor copy 分发到 torch_npu
    // params.overlapped 被有意忽略 — Ascend 暂无专用 overlap stream
    dst.copy_(src, /*non_blocking=*/src.is_privateuseone() && dst.is_privateuseone());
}

void multiMergeCopy(const MultiMergeCopyParams& params) {
    // MultiMergeCopy 目标在 HOST 上，与 CUDA 行为一致
    for (size_t i = 0; i < params.src_ptrs.size(); i++) {
        auto dst = static_cast<char*>(params.dst_ptr) + params.dst_offsets[i];
        std::memcpy(dst, params.src_ptrs[i], params.copy_size[i]);
    }
}

static void batchCopyFallback(const BatchCopyParams& params) {
    // 与 ROCm batchCopyFallback 相同逻辑（参考行 290-328）
    for (uint32_t copy_type_enum = 0; copy_type_enum < BatchCopyParams::TYPE_SIZE; ++copy_type_enum) {
        auto   copy_type       = BatchCopyParams::CopyType(copy_type_enum);
        auto&  buffers         = params.copy_buffers[copy_type];
        size_t copy_batch_size = buffers.sizes.size();
        if (copy_batch_size == 0)
            continue;
        for (size_t i = 0; i < copy_batch_size; ++i) {
            size_t        bytes      = buffers.sizes[i];
            torch::Device dst_device = torch::kCPU, src_device = torch::kCPU;
            switch (copy_type) {
                case BatchCopyParams::D2D:
                    dst_device = torch::Device(torch::kPrivateUse1);
                    src_device = torch::Device(torch::kPrivateUse1);
                    break;
                case BatchCopyParams::D2H:
                    dst_device = torch::kCPU;
                    src_device = torch::Device(torch::kPrivateUse1);
                    break;
                case BatchCopyParams::H2D:
                    dst_device = torch::Device(torch::kPrivateUse1);
                    src_device = torch::kCPU;
                    break;
                case BatchCopyParams::H2H:
                    break;
                default:
                    RTP_LLM_FAIL("Unexpected CopyType %d", copy_type);
                    break;
            }
            auto dst_tensor =
                torch::from_blob(buffers.dst_ptr[i], {(int64_t)bytes}, torch::dtype(torch::kUInt8).device(dst_device));
            auto src_tensor = torch::from_blob(const_cast<void*>(buffers.src_ptr[i]),
                                               {(int64_t)bytes},
                                               torch::dtype(torch::kUInt8).device(src_device));
            runtimeCopy({dst_tensor, src_tensor, params.overlapped});
        }
    }
}

void runtimeBatchCopy(const BatchCopyParams& params) {
    batchCopyFallback(params);
}

void runtimeMaskLogits(torch::Tensor& logits, const torch::Tensor& mask) {
    auto masked = logits.clone();
    auto mask_float = mask.to(logits.dtype());
    logits.masked_fill_(mask_float.to(torch::kBool) == 0, -1e9f);
}

}  // namespace rtp_llm
```

### Step 3: 验证编译

```bash
bazel build //rtp_llm/models_py/bindings/core:exec_ops --config=ascend
```

**验收标准:** 编译通过，不再 throw `ERROR_UNIMPLEMENTED`。

---

## Task 3: BlockPool.cc — 参数化 torch::kCUDA + .is_cuda()

**Files:**
- Modify: `rtp_llm/cpp/cache/BlockPool.cc:39-47`（initializeCacheBuffer）
- Modify: `rtp_llm/cpp/cache/BlockPool.cc:502-503`（where()）

### Step 1: initializeCacheBuffer (行 39-47)

```cpp
void BlockPool::initializeCacheBuffer() {
    torch::Device device = torch::kCPU;
    if (allocation_type_ == AllocationType::HOST) {
        device = torch::kCPU;
    } else {
#if USING_CUDA
        device = torch::kCUDA;
#elif USING_ASCEND
        device = torch::Device(torch::kPrivateUse1);
#elif USING_ROCM
        device = torch::kCUDA;  // HIP 使用 CUDA device type
#else
        device = torch::kCPU;
#endif
    }
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    cache_aligned_buffer_ = torch::empty({static_cast<int64_t>(config_.total_size_bytes)}, options);
    if (allocation_type_ == AllocationType::HOST) {
        cache_aligned_buffer_ = cache_aligned_buffer_.pin_memory();
    }
    cache_base_ptr_ = cache_aligned_buffer_.data_ptr();
    RTP_LLM_CHECK_WITH_INFO(cache_base_ptr_ != nullptr, "block pool allocate cache aligned buffer is null");
}
```

### Step 2: where() (行 502-503)

```cpp
MemoryType BlockPool::where() const {
#if USING_ASCEND
    return cache_aligned_buffer_.is_privateuseone() ? MemoryType::MEMORY_NPU : MemoryType::MEMORY_CPU;
#else
    return cache_aligned_buffer_.is_cuda() ? MemoryType::MEMORY_GPU : MemoryType::MEMORY_CPU;
#endif
}
```

### Step 3: 验证编译

```bash
bazel build //rtp_llm/cpp/cache:cache_core --config=ascend
```

**验收标准:** 编译通过，`BlockPool` 在 Ascend 配置下使用 `torch::kPrivateUse1` 创建 tensor。

---

## Task 4: MemoryEvaluationHelper.cc — 添加 Ascend 内存查询分支

**Files:**
- Modify: `rtp_llm/cpp/cache/MemoryEvaluationHelper.cc:5-10`（includes）
- Modify: `rtp_llm/cpp/cache/MemoryEvaluationHelper.cc:58-62`（cudaMemGetInfo）

### Step 1: 修改 includes (行 5-10)

```cpp
#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include <hip/hip_runtime.h>
#include "rtp_llm/models_py/bindings/rocm/hip_host_utils.h"
#elif USING_ASCEND
#include "rtp_llm/models_py/bindings/ascend/ascend_host_utils.h"
#endif
```

### Step 2: 修改 getDefaultRuntimeMemorySize 中的内存查询 (行 58-62)

```cpp
#if USING_CUDA
    check_cuda_value(cudaMemGetInfo(&free_gpu_bytes, &total_gpu_bytes));
#elif USING_ROCM
    ROCM_CHECK(hipMemGetInfo(&free_gpu_bytes, &total_gpu_bytes));
#elif USING_ASCEND
    auto [used_bytes, free_bytes] = rtp_llm::ascend::getDeviceMemoryInfo(false);
    free_gpu_bytes  = free_bytes;
    total_gpu_bytes = used_bytes + free_bytes;
#endif
```

> **说明:** `ascend::getDeviceMemoryInfo()` 返回值语义为 `(used, free)`（见 `ascend_host_utils.cc:48-53`），与 CUDA `(free, total)` 不同。需要这里的转换。

### Step 3: 验证编译

```bash
bazel build //rtp_llm/cpp/cache:cache_core --config=ascend
```

**验收标准:** 编译通过，Ascend 下不再使用未初始化的 `free_gpu_bytes`/`total_gpu_bytes`。

---

## Task 5: ExecOps.cc — 修复残留 torch::kCUDA

**Files:**
- Modify: `rtp_llm/models_py/bindings/core/ExecOps.cc:562-565`

### Step 1: execCreateMoeExpertStates（行 562-565）

当前代码：
```cpp
states.stats_buf.log_stats_buf = torch::zeros({(int64_t)params.layer_num, (int64_t)params.log_exp_num},
                                              torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
states.stats_buf.gpu_loads_buf = torch::zeros({(int64_t)params.layer_num, (int64_t)params.ep_size},
                                              torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
```

修改为：
```cpp
torch::Device moe_device = 
#if USING_ASCEND
    torch::Device(torch::kPrivateUse1);
#else
    torch::Device(torch::kCUDA);
#endif

states.stats_buf.log_stats_buf = torch::zeros({(int64_t)params.layer_num, (int64_t)params.log_exp_num},
                                              torch::TensorOptions(torch::kInt32).device(moe_device));
states.stats_buf.gpu_loads_buf = torch::zeros({(int64_t)params.layer_num, (int64_t)params.ep_size},
                                              torch::TensorOptions(torch::kInt32).device(moe_device));
```

### Step 2: 验证编译

```bash
bazel build //rtp_llm/models_py/bindings/core:exec_ops --config=ascend
```

---

## Task 6: 适配 .is_cuda() 在缓存层中的使用

以下文件中有 `t.is_cuda()` / `dev.is_cuda()` 调用，需要为 Ascend 添加 `is_privateuseone()` 检查。

### Step 6a: KVCacheManager.cc (行 193-194)

当前：
```cpp
auto  dst_device = dst_block.is_cuda ? torch::kCUDA : torch::kCPU;
auto  src_device = src_tensor.is_cuda() ? torch::kCUDA : torch::kCPU;
```

修改（由于 Task 7 跳过，不依赖 `is_npu`，直接通过 tensor 判断）：
```cpp
auto  src_device = src_tensor.is_privateuseone() ? torch::Device(torch::kPrivateUse1) :
                   src_tensor.is_cuda() ? torch::kCUDA : torch::kCPU;
// dst_block.is_cuda 已在设置时标记为 true（见 6b/6d），保持原有 dst_block.is_cuda ? kCUDA : kCPU 逻辑即可
auto  dst_device = dst_block.is_cuda ? torch::kCUDA : torch::kCPU;
```

### Step 6b: MemoryLayoutStrategy.cc (行 285)

当前：
```cpp
info.is_cuda = dev.is_cuda();
```

修改（is_cuda 承担"加速器设备"语义，配合 Task 7 跳过）：
```cpp
info.is_cuda = dev.is_cuda() || dev.is_privateuseone();
```

### Step 6c: TypeConvert.h (行 82, 86)

当前：
```cpp
inline MemoryType torchDeviceToMemoryType(const c10::Device& device) {
    return device.is_cuda() ? MemoryType::MEMORY_GPU : MemoryType::MEMORY_CPU;
}
inline c10::Device memoryTypeToTorchDevice(const MemoryType& memory_type) {
    return memory_type == MemoryType::MEMORY_GPU ? torch::DeviceType::CUDA : torch::DeviceType::CPU;
}
```

修改：
```cpp
inline MemoryType torchDeviceToMemoryType(const c10::Device& device) {
    if (device.is_cuda()) return MemoryType::MEMORY_GPU;
    if (device.is_privateuseone()) return MemoryType::MEMORY_NPU;
    return MemoryType::MEMORY_CPU;
}
inline c10::Device memoryTypeToTorchDevice(const MemoryType& memory_type) {
    switch (memory_type) {
        case MemoryType::MEMORY_GPU: return torch::DeviceType::CUDA;
        case MemoryType::MEMORY_NPU: return torch::Device(torch::kPrivateUse1);
        default: return torch::DeviceType::CPU;
    }
}
```

### Step 6d: LayerBlockConverterImpl.h (行 36-37)

当前：
```cpp
info.is_cuda = t.is_cuda();
info.device_index = t.is_cuda() ? static_cast<int32_t>(t.get_device()) : 0;
```

修改（配合 Task 7 跳过，不设置 is_npu）：
```cpp
info.is_cuda = t.is_cuda() || t.is_privateuseone();
info.device_index = t.is_cuda() || t.is_privateuseone() ? static_cast<int32_t>(t.get_device()) : 0;
```

---

## Task 7: [跳过] 不新增 is_npu，is_cuda 承担"加速器设备"语义

**决策理由：** Task 7 已跳过。不向 `BlockInfo.h` 新增 `is_npu` 字段。改用 `is_cuda` 兼容性判断：在 Ascend 路径下设置 `is_cuda = true`，使其表达"在任意加速器设备上"的语义。

### 影响范围

| 位置 | 原本 | 修改后 |
|------|------|--------|
| 6b MemoryLayoutStrategy.cc | `info.is_cuda = dev.is_cuda()` | `info.is_cuda = dev.is_cuda() \|\| dev.is_privateuseone()` |
| 6d LayerBlockConverterImpl.h | `info.is_cuda = t.is_cuda()` | `info.is_cuda = t.is_cuda() \|\| t.is_privateuseone()` |
| 6a KVCacheManager.cc | `dst_block.is_cuda ? kCUDA : kCPU` | 改用 tensor 直接判断：`src_tensor.is_privateuseone() ? kPrivateUse1 : src_tensor.is_cuda() ? kCUDA : kCPU` |

> **风险提示：** 如果存在其他代码依赖 `is_cuda == false` 来排除 NPU 设备，可能导致误判。当前代码库中 `is_cuda` 仅用于判断是否位于加速器上执行 copy/alloc，语义上兼容此修改。Task 8 的 BUILD 扩展仍按计划执行。

---

## Task 8: BUILD 文件 select() 扩展

**Files:**
- Verify/Modify: 以下文件的 select() 块需要检查是否包含 `@//:using_ascend` 分支

### Step 1: 搜索所有 select() 中引用 device 的地方

```bash
rg "using_cuda.*using_rocm\|select(" --type build rtp_llm/cpp/cache/BUILD rtp_llm/models_py/bindings/core/BUILD rtp_llm/cpp/cuda/BUILD
```

### Step 2: 已知需要修改的 BUILD 文件

- `rtp_llm/cpp/cache/BUILD` — 引用 `cuda_host_utils` / `hip_host_utils` 的地方需添加 `ascend_host_utils` 分支
- `rtp_llm/models_py/bindings/core/BUILD` — 需要添加 `ascend_types_hdr` / `ascend_host_utils` 依赖

示例 cache/BUILD：
```python
deps = select({
    "@//:using_cuda": ["//rtp_llm/models_py/bindings/cuda:cuda_host_utils"],
    "@//:using_rocm": ["//rtp_llm/models_py/bindings/rocm:hip_host_utils"],
    "@//:using_ascend": ["//rtp_llm/models_py/bindings/ascend:ascend_host_utils"],
    "//conditions:default": [],
}),
```

---

## Task 9: 集成验证

### Step 9.1: 编译全量验证

```bash
bazel build //rtp_llm/models_py/bindings/core:exec_ops --config=ascend
bazel build //rtp_llm/cpp/cache:cache_core --config=ascend
bazel build //rtp_llm/cpp/ascend/... --config=ascend
```

### Step 9.2: 端到端内存验证脚本

```python
# tests/ascend/test_memory_e2e.py
"""Phase 1 端到端内存验证（基于实际代码库）"""
import torch
import torch_npu

def test_npu_basic_allocation():
    """验证 NPU 设备可见且可分配 tensor"""
    assert torch.npu.is_available(), "NPU not available"
    assert torch.npu.device_count() > 0, "No NPU devices found"
    device = torch.device("npu:0")
    t = torch.randn(100, 100, device=device)
    assert t.device.type == "npu"

def test_tensor_to_npu():
    """验证 .to() 重定向到 NPU"""
    t = torch.randn(10, 10)
    t_npu = t.to(torch.device("npu:0"))
    assert "npu" in str(t_npu.device)

def test_npu_stream():
    """验证 NPU Stream 可创建和同步"""
    s = torch.npu.Stream()
    with torch.npu.stream(s):
        t = torch.randn(1000, 1000, device="npu:0")
        _ = t @ t
    s.synchronize()

if __name__ == "__main__":
    test_npu_basic_allocation()
    test_tensor_to_npu()
    test_npu_stream()
    print("=== Phase 1 端到端验证全部通过 ===")
```

### Step 9.3: 验收清单

| 验收项 | 验收标准 | 验证方法 |
|--------|---------|---------|
| Torch_ext.h | `GET_CURRENT_STREAM` 在 Ascend 编译通过 | 编译 |
| CudaOps.cc | `runtimeCopy` 使用 `dst.copy_()` 而非 throw | 编译 + 集成测试 |
| BlockPool | `initializeCacheBuffer` 使用 `kPrivateUse1` 创建 tensor | 编译 + log 检查 |
| BlockPool::where() | 返回 `MEMORY_NPU` 而非 `MEMORY_GPU` | 编译 |
| MemoryEvaluationHelper | `getDefaultRuntimeMemorySize` 使用 `ascend::getDeviceMemoryInfo` | 编译 |
| ExecOps | `execCreateMoeExpertStates` 无 `kCUDA` | 编译 |
| TypeConvert.h | `torchDeviceToMemoryType` 处理 NPU | 编译 |
| BlockInfo.h | [跳过] `is_cuda` 在 Ascend 下也设为 true，不新增字段 | 编译 |
| BUILD select | 包含 `using_ascend` 分支 | `rg` 搜索确认 |
| 端到端 | tensor 可分配到 NPU，stream 可同步 | Python 测试脚本 |

---

## Task 10: (可选) Ascend Allocator 包装

> **说明:** 代码库无 `AllocatorType` 枚举或 `Allocator<T>` 模板类。KV Cache 分配通过 `KVCacheAllocator` + `BlockPool` 完成。当前 `KVCacheAllocator` 使用 `BlockPool` 的 `torch::empty()` 直接分配，不依赖独立 allocator。
>
> 如果后续需要封装 Ascend 显存管理的独立 allocator（用于非 KV Cache 场景），可参考以下结构：

```cpp
// rtp_llm/models_py/bindings/ascend/ascend_allocator.h
#pragma once
#if USING_ASCEND
#include <acl/acl.h>
#include <cstddef>

namespace rtp_llm {
namespace ascend {

// 简单封装 aclrtMalloc/aclrtFree
void* deviceMalloc(size_t size);
void  deviceFree(void* ptr);
void* hostPinnedMalloc(size_t size);
void  hostPinnedFree(void* ptr);

}  // namespace ascend
}  // namespace rtp_llm
#endif
```

```cpp
// rtp_llm/models_py/bindings/ascend/ascend_allocator.cc
#include "ascend_allocator.h"
#if USING_ASCEND
#include <acl/acl.h>
#include "rtp_llm/models_py/bindings/ascend/ascend_types_hdr.h"

namespace rtp_llm {
namespace ascend {

void* deviceMalloc(size_t size) {
    void* ptr = nullptr;
    ASCEND_CHECK(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
    return ptr;
}

void deviceFree(void* ptr) {
    ASCEND_CHECK(aclrtFree(ptr));
}

void* hostPinnedMalloc(size_t size) {
    void* ptr = nullptr;
    ASCEND_CHECK(aclrtMallocHost(&ptr, size));
    return ptr;
}

void hostPinnedFree(void* ptr) {
    ASCEND_CHECK(aclrtFreeHost(ptr));
}

}  // namespace ascend
}  // namespace rtp_llm
#endif
```

---

## 已实现但计划未覆盖的代码（v0.2 重复描述的代码）

以下是 v0.2 中列为"待实现"但在代码库中 **已存在** 的内容，此处不再重复：

| 已有功能 | 文件位置 | 状态 |
|---------|---------|------|
| Ascend 头文件类型定义 | `bindings/ascend/ascend_types_hdr.h` | ✅ 存在（加速流类型 + 错误检查宏） |
| Ascend host utils | `bindings/ascend/ascend_host_utils.{h,cc}` | ✅ 存在（设备查询、内存信息、同步检查） |
| Ascend BUILD 文件 | `bindings/ascend/BUILD` | ✅ 存在（ascend_types_hdr, ascend_host_utils, ascend_bindings_register） |
| ExecOps Ascend 分支 | `bindings/core/ExecOps.cc` | ✅ 存在（runtimeSyncAndCheck、runtimeCreateEvent、getGpuExecStatus、getTorchCudaDevice、initRuntime） |
| GET_CURRENT_STREAM 已有 CUDA/ROCm | `bindings/common/Torch_ext.h` | ❌ 缺少 Ascend 分支（Task 1） |
| CudaOps.cc Ascend 分支 | `bindings/core/CudaOps.cc` | ⚠️ 存在但全为 stub（Task 2 实现） |
| bazel/device_defs.bzl | 已有 `using_ascend` | ✅ 存在 |
| def.bzl ascend_copts | `def.bzl:162` | ✅ 存在 |
| MemoryType::MEMORY_NPU | `bindings/core/Types.h:14` | ✅ 存在 |
| AscendRegister.cc | `bindings/ascend/AscendRegister.cc` | ⚠️ 空文件占位（Phase 5 填充） |

---

## 依赖关系图

```
Task 1 (Torch_ext.h) ───────── 独立
Task 2 (CudaOps.cc)  ───────── 独立
Task 3 (BlockPool)   ───────── 依赖 Phase 0 BUILD 基础设施（已完成）
    └→ Task 6a,b,d (KVCacheManager/LayerBlockConverter/MemoryLayoutStrategy) ←── 依赖 Task 7
         └→ Task 6c (TypeConvert.h)
Task 4 (MemoryEvalHelper) ──── 依赖 Task 3（同一 target）
Task 5 (ExecOps.cc) ────────── 独立
Task 7: [跳过] — 不新增 is_npu，is_cuda 承载"加速器设备"语义
Task 8 (BUILD select) ──────── 贯穿全部
Task 9 (集成验证) ───────────── 依赖全部
Task 10 (可选 Allocator) ───── 独立
```

**可并行执行：** Task 1 + Task 2 + Task 7 可并行；Task 3 + Task 4 + Task 5 在一个 target 可串行。

---

## 附录：执行前兼容性检查

### 1. torch_npu stream API 检查

```bash
python3 -c "
import torch_npu
# 检查 getCurrentNPUStream 可用性
try:
    from torch_npu.csrc.core.npu.NPUStream import getCurrentNPUStream
    print('getCurrentNPUStream available via torch_npu.csrc.core.npu.NPUStream')
except ImportError:
    pass
try:
    stream = torch_npu.npu.current_stream()
    print(f'current_stream() works: {stream}')
except Exception as e:
    print(f'current_stream() failed: {e}')

# 检查 c10_npu namespace
try:
    import c10_npu
    print('c10_npu module available')
    stream = c10_npu.getCurrentNPUStream()
    print(f'c10_npu.getCurrentNPUStream() works: {stream}')
except Exception as e:
    print(f'c10_npu check: {e}')
"
```

### 2. BlockInfo 修改影响范围

```bash
# 搜索 BlockInfo.is_cuda 的所有读取方
rg "\.is_cuda" rtp_llm/cpp/ --type cpp
# 确认修改后未遗漏调用方
```

### 3. 残留 kCUDA 搜索

```bash
# 搜索对 Phase 1 范围关键但尚未被 guard 的 torch::kCUDA
rg "torch::kCUDA" --type cpp rtp_llm/cpp/ rtp_llm/models_py/bindings/core/ rtp_llm/models_py/bindings/common/
```

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| `torch_npu` 中 `c10_npu::getCurrentNPUStream().stream()` 返回类型非 `aclrtStream` | Torch_ext.h 编译失败 | 使用 `aclrtGetCurrentStream()` 作为 fallback：`#define GET_CURRENT_STREAM() ({ aclrtStream s; aclrtGetCurrentStream(&s); s; })` |
| `BlockInfo.is_npu` 新增字段导致第三方/RPC 序列化不兼容 | 缓存传输可能失败 | `is_npu` 默认 false，不影响现有 CUDA/ROCm 路径；RPC 序列化需独立适配 |
| `is_privateuseone()` 在旧版 PyTorch 中不可用 | 编译失败 | PyTorch >= 2.0 支持；检查 `torch.is_privateuseone_backend_available()` |
| `MemoryType::MEMORY_NPU` 在上层代码中未处理 | cache 层使用 `MEMORY_GPU` 路径操作 NPU 内存 | 在 `torchDeviceToMemoryType` / `memoryTypeToTorchDevice` 全量映射后，上层无需感知 NPU 类型 |
