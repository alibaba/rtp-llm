# rtp-llm 华为 Ascend NPU 全量适配计划

## 一、项目概况与现状分析

### 1.1 当前架构的 GPU 耦合度

rtp-llm 的 GPU 依赖可分为以下几个层次：

| 层次 | 耦合程度 | 说明 |
|------|---------|------|
| **编译构建层 (Bazel)** | 高 | `cuda_configure.bzl` 生成 `@local_config_cuda`，全项目 BUILD 文件通过 `select()` 按 `using_cuda/using_rocm` 切换依赖 |
| **设备抽象层 (C++)** | 中 | 无统一 Device 接口，使用 `#if USING_CUDA` / `#if USING_ROCM` 编译时宏 + Bazel `select()` 分派 |
| **内存管理 (C++)** | 高 | `Allocator<CUDA>` 直接调用 `cudaMalloc/Free`，`ICudaAllocator` 暴露 `cudaStream_t` |
| **GEMM 层 (C++)** | 极高 | `cublasMMWrapper` / `cublasFP8MMWrapper` 直接封装 cuBLAS Lt API，含算法调优体系 |
| **Attention 层 (C++)** | 极高 | FlashInfer (CUDA only)、XQA (SM90 only) 深度耦合 |
| **自定义 CUDA 算子 (Python C++ Extension)** | 高 | ~15 个 `.cu` 算子（activation、layernorm、kv_cache、copy 等），使用 CUDA 编程模型 |
| **分布式通信** | 低 | 已迁移至 `torch.distributed` c10d ProcessGroup 抽象 |
| **KV Cache 管理** | 中 | C++ 层基于 `torch::Tensor` 设备无关，但内存查询和 block 拷贝依赖 CUDA API |
| **CUDA Graph** | 中 | 深入集成，但 CANN 有对应 Graph 模式 |
| **FP8/量化** | 高 | 涉及 GEMM、Attention、KV Cache 全栈 |

### 1.2 现有 ROCm 参考模式

ROCm 适配方案提供了良好参考：

- **宏映射**：`rtp_llm/cpp/rocm/cuda_shims.h` (~128 个 `#define cudaXxx hipXxx`)
- **构建分选**：`.bazelrc` 中 `build:rocm` 配置段，`def.bzl` 中 `if_rocm()` 函数
- **实现目录**：`rtp_llm/cpp/rocm/`、`rtp_llm/models_py/bindings/rocm/`
- **Python 工厂模式**：Attention/Linear/MoE 通过 Factory 注册表选择设备实现

### 1.3 关键代码路径索引

| 模块 | 关键文件 | 说明 |
|------|---------|------|
| 设备类型枚举 | `rtp_llm/cpp/core/DeviceData.h` | `DeviceType` 枚举，需新增 `Ascend` |
| 分配器接口 | `rtp_llm/cpp/core/allocator.h` | `AllocatorType` 枚举，需新增 `ASCEND` |
| CUDA 分配器 | `rtp_llm/cpp/cuda/allocator_cuda.h` | `Allocator<CUDA>` 模板特化，需新增 `Allocator<ASCEND>` |
| GEMM 封装 | `rtp_llm/cpp/cuda/cublas/cublasMMWrapper.h` | cuBLAS Lt 封装，需 CANN 等价物 |
| FP8 GEMM | `rtp_llm/cpp/cuda/cublas/cublasFP8MMWrapper.h` | FP8 MatMul，需 CANN FP8 GEMM |
| CUDA Host Utils | `rtp_llm/cpp/cuda/cuda_host_utils.h` | 设备属性/内存查询 |
| ROCm Shim | `rtp_llm/cpp/rocm/cuda_shims.h` | CUDA→HIP 宏映射参考 |
| FlashInfer Ops | `rtp_llm/cpp/cuda/ops/CudaFlashInfer.cc` | Prefill/Decode Attention |
| XQA Ops | `rtp_llm/cpp/cuda/ops/CudaXqa.cc` | SM90 Decode Attention（跳过） |
| Beam Search | `rtp_llm/cpp/cuda/ops/BeamSearchOp.h` | 纯 PyTorch，设备无关 |
| 分布式通信 | `rtp_llm/cpp/core/DistributedComm.h` | c10d ProcessGroup 抽象 |
| Python 分布式 | `rtp_llm/models_py/distributed/collective_torch.py` | torch.distributed，需 HCCL |
| Attention Factory | `rtp_llm/models_py/modules/factory/attention/attn_factory.py` | 设备实现注册 |
| Linear Factory | `rtp_llm/models_py/modules/factory/linear/` | 设备实现注册 |
| MoE Factory | `rtp_llm/models_py/modules/factory/fused_moe/` | 设备实现注册 |
| KV Cache 管理 | `rtp_llm/cpp/cache/` | BlockPool / MemoryLayoutStrategy |
| CUDA Graph Copy | `rtp_llm/models_py/bindings/common/kernels/cuda_graph_copy_kernel.cu` | Graph 模式拷贝 |
| 构建配置 | `.bazelrc` | `build:cuda` / `build:rocm` 配置段 |
| 顶层定义 | `def.bzl` | `if_cuda()` / `if_rocm()` / `copts()` |
| 设备构建辅助 | `bazel/device_defs.bzl` | `device_impl_target()` / `device_test_envs()` |
| CUDA Configure | `3rdparty/gpus/cuda_configure.bzl` | CUDA 自动配置（参考模板） |
| ROCm Configure | `3rdparty/gpus/rocm_configure.bzl` | ROCm 自动配置（参考模板） |

---

## 二、适配总体策略

### 2.1 技术路线选择

```
方案A: Shim 层映射（类似 ROCm）   → 适用场景：API 级兼容
方案B: 独立实现层 + Factory 注册  → 适用场景：API 不兼容
方案C: PyTorch 抽象层（torch_npu）→ 适用场景：PyTorch 原生支持的操作
```

**推荐策略**：三管齐下

- **PyTorch 标准操作** → 通过 `torch_npu` 适配，修改 device placement（`cuda→npu`）
- **CANN 有对应 API 的操作** → 新建 `rtp_llm/cpp/ascend/` 独立实现 + `#if USING_ASCEND` 守卫
- **CANN 无对应 API 的自定义算子** → 使用 Ascend C 重写或通过 `aclnn` 操作 API 封装

### 2.2 总体架构变更

```
                    ┌──────────────────────────────────────┐
                    │           rtp-llm Framework           │
                    ├──────────────────────────────────────┤
                    │  Python Factory Layer                 │
                    │  ├─ AttentionFactory                  │
                    │  ├─ LinearFactory                     │
                    │  └─ FusedMoEFactory                   │
                    ├──────────────────────────────────────┤
                    │  C++ Device Abstraction (新增)        │
                    │  ├─ IDevice interface                 │
                    │  ├─ CudaDevice / RocmDevice           │
                    │  └─ AscendDevice (NPU)                │
                    ├──────────────────────────────────────┤
                    │  C++ Ops Layer                        │
                    │  ├─ GemmOps → cuBLAS / hipBLAS / aclblas│
                    │  ├─ AttentionOps → FlashInfer/aiter/AscendAttn│
                    │  ├─ Allocator → cuda/hip/aclrt        │
                    │  └─ Stream/Event → cuda/hip/aclrt     │
                    ├──────────────────────────────────────┤
                    │  Build System (Bazel)                 │
                    │  ├─ @local_config_cuda                │
                    │  ├─ @local_config_rocm                │
                    │  └─ @local_config_ascend (新增)       │
                    └──────────────────────────────────────┘
```

---

## 三、分阶段实施计划

### Phase 0: 编译基础设施搭建（~2 周）

**目标**：打通 Bazel 编译链路，使项目可在 Ascend 环境下编译通过。

#### 0.1 创建 `ascend_configure.bzl`

参考 `3rdparty/gpus/cuda_configure.bzl`（1306行），创建 `3rdparty/gpus/ascend_configure.bzl`：

| 任务 | 说明 |
|------|------|
| 环境变量 | `TF_NEED_ASCEND`, `ASCEND_TOOLKIT_PATH`, `ASCEND_VERSION` |
| 自动探测 | CANN 安装路径、acl/aclblas/aclnn 头文件和库 |
| 生成模板 | `ascend/BUILD.tpl`, `ascend/build_defs.bzl.tpl`, `ascend/ascend_config.h.tpl` |
| 产物 | `@local_config_ascend` 仓库，提供 `ascend_headers`, `ascend_libs`, `crosstool` 等目标 |

**关键文件修改**：

- `WORKSPACE`：添加 `load("//3rdparty/gpus:ascend_configure.bzl", "ascend_configure")` + `ascend_configure(name = "local_config_ascend")`
- `BUILD`：添加 `config_setting(name = "using_ascend")`
- `def.bzl`：添加 `if_ascend()` 函数、`ascend_copts()` 函数、`USING_ASCEND` 宏

#### 0.2 `.bazelrc` 添加 `build:ascend` 配置

```starlark
build:ascend --copt="-DUSING_ASCEND=1"
build:ascend --define=using_cuda=false --define=using_cuda_nvcc=false
build:ascend --define=using_ascend=true
build:ascend --action_env TF_NEED_CUDA="0"
build:ascend --action_env TF_NEED_ASCEND="1"
build:ascend --action_env ASCEND_TOOLKIT_PATH="/usr/local/Ascend/ascend-toolkit/latest"
build:ascend --crosstool_top=@local_config_ascend//crosstool:toolchain
```

#### 0.3 扩展 `DeviceType` 枚举和编译守卫

- `rtp_llm/cpp/core/DeviceData.h`：添加 `Ascend = 6`
- `rtp_llm/cpp/core/allocator.h`：添加 `ASCEND`, `ASCEND_HOST` 到 `AllocatorType`
- 全项目 `select()` 块中添加 `"@//:using_ascend"` 分支

#### 0.4 创建 Ascend 兼容层头文件

参考 `rtp_llm/cpp/rocm/cuda_shims.h`，创建 `rtp_llm/cpp/ascend/ascend_shims.h`：

```
rtp_llm/cpp/ascend/
├── ascend_shims.h          # cudaStream_t → aclrtStream 等宏映射
├── ascend_host_utils.h     # NPU 设备属性查询、内存信息
├── ascend_host_utils.cc    # getDeviceMemoryInfo() 用 aclrtGetMemInfo
├── ascend_type_utils.h     # BF16/FP16/FP8 类型适配
└── BUILD                   # Bazel 构建目标
```

**核心 API 映射**：

| CUDA API | CANN (aclrt) API |
|----------|------------------|
| `cudaStream_t` | `aclrtStream` |
| `cudaEvent_t` | `aclrtEvent` |
| `cudaMalloc` | `aclrtMalloc` |
| `cudaFree` | `aclrtFree` |
| `cudaMemcpyAsync` | `aclrtMemcpyAsync` |
| `cudaStreamSynchronize` | `aclrtSynchronizeStream` |
| `cudaEventCreate/Record/Synchronize` | `aclrtCreateEvent/RecordEvent/SynchronizeEvent` |
| `cudaMemGetInfo` | `aclrtGetMemInfo` |
| `cudaGetDevice/SetDevice` | `aclrtGetDevice/SetDevice` |

---

### Phase 1: 内存管理与设备运行时（~1.5 周）

**目标**：替换 GPU 内存分配器，使 tensor 可正确分配到 NPU。

#### 1.1 Ascend Allocator

创建 `rtp_llm/cpp/ascend/allocator_ascend.h`：

```cpp
template<>
class Allocator<AllocatorType::ASCEND>: public ... {
    void* doMalloc(size_t size) override;  // aclrtMalloc
    void  doFree(void* ptr) override;      // aclrtFree
};
```

- 修改 `rtp_llm/cpp/cache/BUILD`：`select()` 中添加 ascend 分支链接 `ascend_host_utils`
- 修改 `BlockPool` 中的内存查询：`getDeviceMemoryInfo()` 添加 `#if USING_ASCEND` 分支

#### 1.2 Device Placement 参数化

C++ 层面：

- `rtp_llm/cpp/models/PyWrappedModel.cc`：将 `.cuda()` / `.to(torch::kCUDA)` 改为 `.to(device_)`，其中 `device_` 由 `DeviceType` 决定
- `rtp_llm/cpp/models/Sampler.cc`：同样参数化
- `rtp_llm/cpp/cuda/ops/CudaFlashInfer.cc`：`at::cuda::getCurrentCUDAStream()` 需要提供 NPU 等价方法

Python 层面：

- 添加 `rtp_llm/utils/device_utils.py`：提供 `get_device_str()` / `get_torch_device()` 工具函数
- 全项目搜索 `device="cuda"` / `torch::kCUDA` 并参数化（~580+ 处 C++、~340+ 处 Python）
- 利用 `torch_npu` 的 `torch.npu` 接口

#### 1.3 Stream/Event 管理

- `rtp_llm/cpp/core/CudaOps.cc` 中的 `cudaEventCreate/Record` → `aclrtCreateEvent/RecordEvent`
- `rtp_llm/cpp/cuda/allocator_cuda.cc` 中的 `cudaStreamSynchronize` → `aclrtSynchronizeStream`
- 考虑通过 `ascend_shims.h` 宏映射来减少侵入式修改

---

### Phase 2: GEMM 算子适配（~2 周）

**目标**：替换 cuBLAS，实现基于 CANN 的 GEMM 操作。

#### 2.1 Ascend GEMM Wrapper

创建 `rtp_llm/cpp/ascend/ascend_gemm/`：

```
rtp_llm/cpp/ascend/ascend_gemm/
├── AscendGemmWrapper.h       # 对应 cublasMMWrapper.h
├── AscendGemmWrapper.cc      # 使用 aclnnMatMul / aclblasGemmEx API
├── AscendFP8GemmWrapper.h    # 对应 cublasFP8MMWrapper.h
├── AscendFP8GemmWrapper.cc   # FP8 GEMM 支持
└── BUILD
```

**需要替换的 cuBLAS 操作**：

| cuBLAS 操作 | CANN 替代方案 |
|-------------|--------------|
| `cublasLtMatmul` (通用矩阵乘) | `aclnnMatMul` / `aclblasGemmEx` |
| `cublasGemmStridedBatchedEx` (批量矩阵乘) | `aclnnBatchMatMul` |
| `cublasGemmBatchedEx` | `aclnnBatchMatMul` |
| `cublasLtMatmul` + FP8 | `aclnnMatMul` + `ACL_FLOAT8_E4M3FN` |
| 算法调优 (algo_map) | CANN 算子调优或使用默认策略 |
| Epilogue (bias/gelu/relu) | 分步实现或 CANN fusion 算子 |

#### 2.2 GEMM 接口抽象（可选但推荐）

在 `cublasMMWrapper` 上提取接口 `IGemmWrapper`，让 `cublasMMWrapper` 和 `AscendGemmWrapper` 实现同一接口：

```cpp
class IGemmWrapper {
public:
    virtual void Gemm(...) = 0;
    virtual void GemmFP8(...) = 0;
    virtual void setStream(aclStream_t stream) = 0;
};
```

#### 2.3 替换 CUTLASS / DeepGEMM

CUTLASS (`rtp_llm/cpp/cuda/cutlass/`) 和 DeepGEMM (`rtp_llm/cpp/cuda/deep_gemm/`) 是 NVIDIA 专用库：

- **INT8/Weight-only GEMM**：使用 CANN 的量化 MatMul 算子
- **MoE Grouped GEMM**：使用 CANN 的 `aclnnGroupedMatMul` 或循环调用 `aclnnMatMul`
- **FP4 GEMM**：如果 CANN 不支持 FP4，降级为 FP8/FP16

---

### Phase 3: Attention 算子适配（~3 周，核心难点）

**目标**：实现 NPU 上的 Paged Attention，替换 FlashInfer/XQA。

#### 3.1 分析现有 Attention 路径

```
Prefill Attention:
  ├─ FlashInfer BatchPrefill (主要路径)
  ├─ TRT FMHA (备选路径)
  └─ XQA (不支持，仅 decode)

Decode Attention:
  ├─ FlashInfer BatchDecode (主要路径)
  └─ XQA SM90 (SM90 专用)

MLA Attention:
  ├─ FlashMLA (FlashInfer 扩展)
  └─ 无备选
```

#### 3.2 Ascend Attention 实现方案

创建 `rtp_llm/cpp/ascend/attention/`：

```
rtp_llm/cpp/ascend/attention/
├── AscendFlashAttention.h      # 封装 aclnnFlashAttentionScore
├── AscendFlashAttention.cc     # Prefill + Decode 统一实现
├── AscendPagedAttention.h      # Paged KV Cache Attention
├── AscendPagedAttention.cc     # 使用 Ascend C 自定义算子或 aclnnPagedAttention
└── BUILD
```

**技术选型**：

- **优先方案**：使用 CANN 内置的 `aclnnFlashAttentionScore` / `aclnnFlashAttentionScoreGrad` 操作，配合 Paged Attention 参数
- **备选方案**：使用 Ascend C 开发自定义 Paged Attention 算子（如果 CANN 内置算子不支持 paged 模式）
- **MLA**：需要单独适配，CANN 可能没有原生 MLA 支持，需自定义实现

#### 3.3 Attention Factory 注册

修改 `rtp_llm/models_py/modules/factory/attention/attn_factory.py`：

```python
# 新增注册
if device_type == "npu":
    PREFILL_MHA_IMPS.append(AscendPrefillImpl)
    DECODE_MHA_IMPS.append(AscendDecodeImpl)
```

创建：

```
rtp_llm/models_py/modules/factory/attention/
├── ascend_impl/
│   ├── __init__.py
│   ├── ascend_prefill.py    # Ascend Prefill 实现
│   └── ascend_decode.py     # Ascend Decode 实现
```

#### 3.4 XQA 处理

- XQA 是 SM90 专用，Ascend 无等价物
- **跳过 XQA**，所有 decode 走 `AscendPagedAttention` 路径
- `supportXqa()` 在 Ascend 编译时返回 false（已有 fallback 机制）

---

### Phase 4: KV Cache 适配（~1.5 周）

**目标**：使 KV Cache 在 NPU 上正确工作，支持 Paged KV Cache 格式。

#### 4.1 KV Cache 内存管理

- `BlockPool` 基于 `torch::Tensor`，设备无关性好 → 只需确保 tensor 创建在 NPU 设备上
- `getDeviceMemoryInfo()` → 添加 `#if USING_ASCEND` 使用 `aclrtGetMemInfo(ACL_HBM_MEM, ...)`
- Block 拷贝 → `aclrtMemcpyAsync` 替代 `cudaMemcpyAsync`

#### 4.2 KV Cache Index 格式

- `rtp_llm/models_py/bindings/common/kernels/kv_cache/` 中的 `KVBlockArray` 等结构基于 CUDA 假设
- 需要评估 Ascend NPU 的内存模型差异：
  - Ascend 使用 Unified Buffer / L1 Cache 架构，与 GPU Shared Memory 不同
  - Paged KV Cache 的 block 大小可能需要调整以匹配 Ascend 内存层级

#### 4.3 KV Cache Copy Kernel 替换

`kv_cache_kernels.cu` 中的 CUDA kernel：

- `convertOffsetAndSize2IdxKernel` → 使用 `aclnn` 张量操作或 Ascend C 重写
- `reuseCacheKernel` → 同上
- FP8 KV Cache dequantization → 使用 CANN 的量化/反量化算子

---

### Phase 5: 自定义 CUDA 算子迁移（~3 周）

**目标**：将 ~15 个自定义 CUDA 算子迁移到 Ascend。

#### 5.1 算子迁移清单

| 算子 | 源文件 | 迁移策略 | 优先级 |
|------|--------|---------|--------|
| Activation (bias+softmax, silu) | `activation_kernels.cu` | `aclnnSilu` / `aclnnSoftmax` | P0 |
| LayerNorm | `layernorm_kernels.cu` | `aclnnLayerNorm` | P0 |
| Sampling penalty | `sampling_penalty_kernels.cu` | `aclnn` 组合或 Ascend C | P0 |
| Mask logits | `mask_logits.cu` | `aclnn` 组合 | P0 |
| KV Cache ops | `kv_cache_kernels.cu` | Ascend C 或 `aclnn` | P0 |
| Batch copy | `batch_copy.cu` | `aclrtMemcpyAsync` | P1 |
| CUDA Graph copy | `cuda_graph_copy_kernel.cu` | Ascend Graph 模式 | P1 |
| Embedding | `embedding_kernels.cu` | `aclnnEmbedding` | P1 |
| MoE dispatch/reorder | `moe_kernels.cu` | Ascend C | P1 |
| TopK | `fast_topk/` | `aclnnTopk` | P1 |
| FP8 quant | `scaled_fp8_quant` | `aclnnCast` + scale | P2 |
| FP4 kernel | `fp4_kernel/` | 暂不支持，降级 | P3 |
| Debug kernel | `debug_kernel.cu` | 可选 | P3 |

#### 5.2 迁移方法

**方法一：使用 CANN 算子 API（aclnn）**

- 适用于有标准对应的算子（LayerNorm、Softmax、TopK、Embedding 等）
- 调用模式：`aclnnXxxGetWorkspaceSize()` + `aclnnXxxExecute()`

**方法二：Ascend C 自定义算子**

- 适用于无标准对应的算子（sampling penalty、CUDA graph copy 等）
- 需要按照 Ascend C 编程模型重写 kernel
- 编译为 `.o` 算子文件后通过 `aclOpCompiler` 加载

**方法三：PyTorch 原生操作**

- 适用于简单算子，直接用 PyTorch 组合实现
- 通过 `torch_npu` 自动调度到 NPU

创建 `rtp_llm/models_py/bindings/ascend/` 目录结构：

```
rtp_llm/models_py/bindings/ascend/
├── BUILD
├── RegisterAscendOps.cc        # 算子注册入口
├── AscendAttentionOp.cc        # Attention 算子
├── AscendGemmOp.cc             # GEMM 算子
├── AscendQuantizeOp.cc         # 量化算子
├── AscendSamplingOp.cc         # 采样算子
├── kernels/
│   ├── ascend_activation.cc    # 激活函数
│   ├── ascend_layernorm.cc     # LayerNorm
│   ├── ascend_kv_cache.cc      # KV Cache 操作
│   └── ascend_copy.cc          # 拷贝操作
└── ...
```

---

### Phase 6: 分布式通信适配（~1 周）

**目标**：使 TP/DP 分布式推理在多卡 NPU 上工作。

#### 6.1 torch.distributed 后端切换

- `collective_torch.py`：将 `backend="nccl"` 改为 `backend="hccl"`（华为 HCCL）
- `torch_npu` 已集成 HCCL 后端，`torch.distributed.init_process_group(backend="hccl")` 可直接使用
- 条件选择：根据 `device_type` 动态选择 backend

#### 6.2 C++ ProcessGroup 适配

- `DistributedComm.cc` 中的 `ensureCuda()` → 添加 `ensureNpu()`：

```cpp
#if USING_ASCEND
static at::Tensor ensureNpu(const at::Tensor& t, int device_id) {
    if (t.is_npu()) return t;
    return t.to(at::Device(at::kNPU, device_id));
}
#endif
```

- `at::cuda::CUDAGuard` → 使用 `c10_npu::NPUGuard` 或类似机制

#### 6.3 Expert Parallelism (DeepEP)

- DeepEP 是基于 CUDA 的 Expert Parallelism 库，无 CANN 等价物
- 短期方案：不使用 DeepEP，使用朴素的 EP 实现
- 长期方案：基于 HCCL 实现自定义 EP 通信

---

### Phase 7: CUDA Graph → Ascend Graph 适配（~1.5 周）

**目标**：将 CUDA Graph 功能映射到 Ascend Graph。

#### 7.1 Graph 模式映射

| CUDA Graph | Ascend Graph |
|-----------|--------------|
| `at::cuda::currentStreamCaptureStatus()` | 自定义标志位 |
| `cuda_graph_copy_kernel.cu` | Ascend Graph 兼容的拷贝算子 |
| `prefill_cuda_graph_copy_params` | 适配 Ascend Graph 的固定大小 tensor |
| FlashInfer `enable_cuda_graph` | Ascend Attention 的 graph 模式 |

#### 7.2 Graph Capture/Replay

CANN 支持 `aclGraph` 模式：

1. `aclrtGraphCreate()` → 创建 graph
2. 在 graph 模式下执行操作序列
3. `aclrtGraphExecute()` → 重放 graph

需要适配的文件：

- `rtp_llm/models_py/modules/factory/attention/` 中的 `prepare_cuda_graph()` / `support_cuda_graph()`
- `rtp_llm/cpp/cuda/ops/CudaFlashInfer.cc` 中的 graph 检测逻辑

---

### Phase 8: FP8/量化支持（~2 周）

**目标**：在 Ascend NPU 上支持 FP8 量化推理。

#### 8.1 FP8 数据类型映射

| NVIDIA | CANN |
|--------|------|
| `CUDA_R_8F_E4M3` | `ACL_FLOAT8_E4M3FN` |
| `__nv_fp8_e4m3` | CANN FP8 类型 |
| `torch.float8_e4m3fn` | `torch_npu` 支持 |

#### 8.2 FP8 GEMM

- 使用 `aclnnMatMul` + FP8 数据类型
- Scale pointer 模式需要映射到 CANN 的量化参数传递方式
- `fast_accum` 模式 → CANN 可能有自己的高精度/高性能模式

#### 8.3 FP8 KV Cache

- `is_kv_cache_fp8` 路径需适配 CANN 的量化/反量化算子
- Dequantization kernel 需用 Ascend C 重写

#### 8.4 Triton 内核

- `rtp_llm/models_py/triton_kernels/` 使用 Triton 编程模型，与 Ascend 不兼容
- 需要：
  - 评估哪些 Triton kernel 有 CANN 等价算子
  - 无等价算子的需要用 Ascend C 重写
  - FLA (Flash Linear Attention) 相关的 Triton kernel 需特别处理

---

### Phase 9: Python 侧全量适配（~2 周）

**目标**：Python 层面全面支持 NPU 设备。

#### 9.1 设备选择参数化

- `rtp_llm/server/server_args/` 中的 `device` 参数支持 `"npu"` 值
- `GpuInitParameters` 等命名中性的参数结构
- `model_loader.py` 中权重加载使用 `torch_npu`

#### 9.2 Factory 注册扩展

| Factory | 新增实现 |
|---------|---------|
| `AttentionFactory` | `ascend_impl/` (prefill + decode) |
| `LinearFactory` | `ascend_impl/` (f16_linear, fp8_linear) |
| `FusedMoEFactory` | `ascend_impl/` (ascend_moe) |

#### 9.3 测试适配

- 所有 `device="cuda"` 的测试参数化
- 添加 NPU CI pipeline
- 性能 benchmark 在 NPU 上建立 baseline

---

### Phase 10: 性能优化与集成测试（~3 周）

**目标**：优化性能，确保功能完整性。

#### 10.1 性能优化

| 优化方向 | 措施 |
|---------|------|
| GEMM 调优 | CANN 算子调优工具 (msprof) |
| Attention 调优 | 调整 block size、匹配 Ascend 硬件特性 |
| KV Cache | 优化 block 布局以匹配 NPU 内存层级 |
| 通信优化 | HCCL 调优、通信-计算重叠 |
| Graph 模式 | 充分利用 Ascend Graph 减少 launch 开销 |

#### 10.2 集成测试

- 单卡推理（多种模型架构）
- 多卡 TP 推理
- FP8 量化推理
- Speculative Decoding
- CUDA Graph (Ascend Graph) 模式
- 长序列 KV Cache 管理
- Beam Search

---

## 四、关键风险与缓解措施

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| CANN 不支持 Paged Attention | Attention 核心功能不可用 | 优先用 `aclnnFlashAttention` + 自定义 paged wrapper；最坏用 Ascend C 自研 |
| CANN FP8 GEMM 性能不足 | 量化推理性能不达标 | 先跑通 FP16，再逐步启用 FP8 |
| `torch_npu` API 不完整 | 部分 PyTorch 操作不可用 | 降级为 CPU 计算或 CANN 算子替代 |
| 自定义算子迁移工作量大 | 周期延长 | 优先使用 CANN 内置算子，仅对无等价算子的使用 Ascend C |
| CANN Graph 模式与 CUDA Graph 差异大 | Graph 加速不可用 | Phase 7 中详细评估兼容性 |
| MoE Grouped GEMM 无等价 | MoE 模型性能差 | 用循环 MatMul 临时替代，性能通过 overlap 优化 |

---

## 五、文件变更范围估算

| 类别 | 新增文件 | 修改文件 |
|------|---------|---------|
| Bazel 构建系统 | ~15 | ~50 |
| C++ Ascend 实现 | ~30 | ~40 |
| Python Ascend 实现 | ~20 | ~60 |
| 头文件/Shim | ~10 | ~20 |
| 测试 | ~15 | ~30 |
| 配置文件 | ~5 | ~10 |
| **合计** | **~95** | **~210** |

---

## 六、里程碑与时间线

| 里程碑 | 阶段 | 预计周期 | 交付物 |
|--------|------|---------|--------|
| M1 | Phase 0 | 第 1-2 周 | Bazel 编译通过 |
| M2 | Phase 1-2 | 第 3-6 周 | 单卡 FP16 推理可跑通（不含自定义算子） |
| M3 | Phase 3-4 | 第 7-11 周 | Attention + KV Cache 可用，支持主流模型推理 |
| M4 | Phase 5-6 | 第 12-16 周 | 自定义算子完成、多卡分布式可用 |
| M5 | Phase 7-8 | 第 17-20 周 | Graph 模式 + FP8 量化支持 |
| M6 | Phase 9-10 | 第 21-25 周 | 全量功能、性能优化、集成测试 |

**总预计周期：~25 周（6 个月）**

---

## 七、新增目录结构总览

```
rtp-llm/
├── 3rdparty/gpus/
│   └── ascend_configure.bzl                  # CANN 自动配置 [新增]
├── rtp_llm/cpp/ascend/                       # C++ Ascend 实现层 [新增]
│   ├── ascend_shims.h                        # API 宏映射
│   ├── ascend_host_utils.{h,cc}              # 设备属性/内存查询
│   ├── ascend_type_utils.h                   # 数据类型适配
│   ├── allocator_ascend.{h,cc}               # NPU 内存分配器
│   ├── ascend_gemm/                          # GEMM 封装
│   │   ├── AscendGemmWrapper.{h,cc}
│   │   ├── AscendFP8GemmWrapper.{h,cc}
│   │   └── BUILD
│   ├── attention/                            # Attention 封装
│   │   ├── AscendFlashAttention.{h,cc}
│   │   ├── AscendPagedAttention.{h,cc}
│   │   └── BUILD
│   └── BUILD
├── rtp_llm/models_py/bindings/ascend/        # Python C++ 扩展 [新增]
│   ├── BUILD
│   ├── RegisterAscendOps.cc
│   ├── AscendAttentionOp.cc
│   ├── AscendGemmOp.cc
│   ├── AscendQuantizeOp.cc
│   ├── AscendSamplingOp.cc
│   └── kernels/
│       ├── ascend_activation.cc
│       ├── ascend_layernorm.cc
│       ├── ascend_kv_cache.cc
│       └── ascend_copy.cc
├── rtp_llm/models_py/modules/factory/
│   ├── attention/ascend_impl/                # Attention 实现 [新增]
│   │   ├── __init__.py
│   │   ├── ascend_prefill.py
│   │   └── ascend_decode.py
│   ├── linear/impl/ascend/                   # Linear 实现 [新增]
│   └── fused_moe/impl/ascend/                # MoE 实现 [新增]
├── rtp_llm/models_py/kernels/ascend/         # Ascend Python 算子 [新增]
└── rtp_llm/utils/device_utils.py             # 设备工具函数 [新增]
```
