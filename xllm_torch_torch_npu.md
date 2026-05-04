# xLLM 在 Ascend NPU 上使用 Torch 的方式分析

## 1. 总体架构：纯 C++ + libtorch + torch_npu

xllm **不使用 Python 层的 torch**，所有 torch 调用都在 C++ 层通过 **libtorch C++ API** 完成。NPU 后端通过 `torch_npu`（华为提供的 PyTorch Ascend 扩展）实现。编译时通过 `-DUSE_NPU` 宏切换后端。

## 2. 设备抽象层（`xllm/core/platform/`）

| 文件 | 关键机制 |
|------|---------|
| `device.cpp` | NPU 映射为 `torch::kPrivateUse1`，通过 `c10_npu::set_device()` / `torch_npu::init_npu()` 初始化 |
| `stream.cpp` | 封装 `c10_npu::NPUStream`，通过 `aclrtSynchronizeStreamWithTimeout()` 同步 |
| `npu/npu_layer_synchronizer.cpp` | 使用原始 `aclrtEvent` 做多流间同步 |
| `vmm_torch_allocator.h` | 与 torch 内存分配器集成 |

## 3. 双轨 Layer 实现策略

xllm 有**两套并行的 Layer 实现**：

- **ATB 层** (`xllm/core/layers/npu/`)：使用华为 **ATB (Ascend Transformer Boost)** 推理库，通过 `atb_speed::Utils::AtTensor2Tensor()` 在 `torch::Tensor` 和 `atb::Tensor` 之间转换，执行通过 `atb::Operation::Execute()`。支持 DeepSeek V2/V3、Qwen2/3、LLaMA、GLM4 等模型。

- **Torch 原生层** (`xllm/core/layers/npu_torch/`)：纯 `torch_npu` 自定义算子，通过 `xllm::kernel::npu::` 命名空间分派。用于 ATB 尚未支持的模型/特性。

## 4. 算子分派机制（`xllm/core/kernels/npu/`）

三类 NPU 算子来源：

| 类型 | 示例 | 来源 |
|------|------|------|
| **torch_npu 自定义算子** | `npu_rms_norm`, `npu_swiglu`, `npu_grouped_matmul`, `npu_apply_rotary_pos_emb` | `at_npu::native::custom_ops::*` |
| **ATB 算子** | `npu_flash_attention`, `npu_paged_attention` | `atb::` 命名空间 |
| **AscendC 自定义算子** | `npu_gemma_rms_norm` | `third_party/xllm_ops/` |

算子调用模式：通用接口（`ops_api.cpp`）→ `#ifdef USE_NPU` → NPU 命名空间实现 → 调用 torch_npu custom ops 或 ATB ops。

## 5. ACL Graph 执行器（`xllm/core/runtime/acl_graph_executor_impl.cpp`）

这是最核心的 NPU 特有组件（1131 行），实现了类似 CUDA Graph 的优化：

1. **捕获阶段**：在非默认流上调用 `c10_npu::NPUGraph::capture_begin()`，执行模型 forward，`capture_end()`
2. **重放阶段**：更新预分配的持久化 tensor 数据，调用 `graph_.replay()`
3. **分桶策略**：按 token 数量分桶（1, 2, 4, 8, 16, 32, 48, 64...）避免每个输入大小都重新捕获
4. **自定义 Paged Attention tiling**：预计算 tiling 数据避免 graph 内部 `.to(kCPU)` 操作破坏图

## 6. Torch 与 ACL 混合调用

代码中频繁**混用 torch_npu 调用和原始 ACL 运行时调用**：

```cpp
// torch_npu 调用
c10_npu::set_device(index);
c10_npu::NPUCachingAllocator::emptyCache();

// 直接 ACL 调用
aclrtGetMemInfo(ACL_HBM_MEM, &free, &total);
aclrtCreateEventWithFlag(&events_[i], ACL_EVENT_SYNC);
aclrtSynchronizeEventWithTimeout(events_[layer_index], timeout_);
```

任务分派通过 `at_npu::native::OpCommand` 封装，确保操作正确入队到 NPU stream。

## 7. 构建依赖（`CMakeLists.txt`）

关键环境变量和库：

| 环境变量 | 用途 |
|---------|------|
| `PYTORCH_INSTALL_PATH` | PyTorch / libtorch 原生安装路径 |
| `PYTORCH_NPU_INSTALL_PATH` | torch_npu（华为 Ascend 扩展）安装路径 |
| `NPU_HOME_PATH` | Ascend NPU 工具包 |
| `ATB_HOME_PATH` | ATB 库 |

链接库：`torch_npu`, `ascendcl`, `atb_customize`, `hccl`, `cust_opapi`, `xllm_atb_layers`

---

## `at_npu::native::custom_ops` 的引入方式

通过 **单个头文件** 引入：

```cpp
#include <torch_npu/csrc/aten/CustomFunctions.h>
```

这个头文件来自 **torch_npu**（PyTorch Ascend 扩展库），其安装路径通过 CMake 配置：

```cmake
# CMakeLists.txt:340
include_directories($ENV{PYTORCH_NPU_INSTALL_PATH}/include)
```

即环境变量 `PYTORCH_NPU_INSTALL_PATH` 指向的 torch_npu 安装目录的 `include/` 下，存放着 `torch_npu/csrc/aten/CustomFunctions.h`。

### 本质

`at_npu::native::custom_ops` 是 torch_npu 库内建的命名空间，不是 xllm 自己定义的。`CustomFunctions.h` 声明了华为预置的一系列 NPU 优化算子，包括：

| 算子 | 功能 |
|------|------|
| `npu_rms_norm` | RMS 归一化 |
| `npu_add_rms_norm` | 融合 Add + RMS Norm |
| `npu_swiglu` | SwiGLU 激活 |
| `npu_apply_rotary_pos_emb` | 旋转位置编码 |
| `npu_grouped_matmul` | 分组矩阵乘 |
| `npu_moe_gating_top_k_softmax` | MoE 门控 TopK |
| `npu_moe_init_routing_v2` | MoE 路由初始化 |
| `npu_moe_token_unpermute` | MoE token 反重排 |
| `npu_fusion_attention` | 融合注意力 |
| `npu_rotary_mul` | 旋转乘法 |

### 使用模式

xllm 的 NPU kernel 层（`xllm/core/kernels/npu/`）直接 `#include` 该头文件后调用，例如 `fused_layernorm.cpp:16-31`：

```cpp
#include <torch_npu/csrc/aten/CustomFunctions.h>
// ...
auto result = at_npu::native::custom_ops::npu_rms_norm(input, weight, eps);
```

这些算子是 **torch_npu 编译时内建的**（通过 `torch_npu` 库链接），不需要 xllm 额外编译或注册。CMake 中链接 `torch_npu` 库即可使用：

```cmake
# 链接 torch_npu 库
set(COMMON_LIBS torch_npu ...)
```

---

## xllm 使用的 torch_npu 主要特性

xllm 共使用了 torch_npu 的 **6 大类特性**：

### 1. 设备初始化与管理（`torch_npu::`）

| API | 位置 | 用途 |
|-----|------|------|
| `torch_npu::init_npu(index)` | `device.cpp:130` | 初始化 NPU 设备上下文 |
| `torch_npu::finalize_npu()` | test 文件 | 反初始化 NPU |
| `torch_npu::NPUStorageImpl` | `base_manual_loader.cpp:310` | 直接操作 tensor 底层存储，强制设置 NZ 格式 |

### 2. 设备/流/内存管理（`c10_npu::`）

| API | 位置 | 用途 |
|-----|------|------|
| `c10_npu::set_device(idx)` | `device.cpp:95` | 设置当前 NPU 设备 |
| `c10_npu::device_count()` | `device.cpp:136` | 获取 NPU 数量 |
| `c10_npu::getCurrentNPUStream(idx)` | 全局 ~30 处 | 获取当前 NPU 流（转成 `aclrtStream` 传给 ACL/ATB） |
| `c10_npu::getNPUStreamFromPool()` | `stream.cpp:24`, `npu_process_group.cpp:84` | 从池中分配新流 |
| `c10_npu::getStreamFromPool(high_prio, idx)` | `acl_graph_executor_impl.cpp:918` | 获取高优先级流用于 graph capture |
| `c10_npu::setCurrentNPUStream()` | `acl_graph_executor_impl.cpp:873,900` | 切换当前流（graph capture 前后） |
| `c10_npu::getDefaultNPUStream(idx)` | `acl_graph_executor_impl.cpp:872,901` | 获取默认流，与 capture 流做比较 |
| `c10_npu::NPUCachingAllocator::emptyCache()` | `device.cpp:206` | 释放 NPU 缓存内存 |
| `c10_npu::NPUCachingAllocator::cacheInfo()` | `worker_impl.cpp:372` | 查询缓存信息 |
| `c10_npu::NPUGraph` | `acl_graph_executor_impl.h:286` | ACL Graph 捕获/重放 |

**这是用量最大的特性**——几乎所有 NPU 操作都需要通过 `c10_npu::getCurrentNPUStream()` 拿到 `aclrtStream`，再传给 ATB/ACL 底层 API。

### 3. 内存分配（`at_npu::native::`）

| API | 位置 | 用途 |
|-----|------|------|
| `at_npu::native::empty_with_format()` | `atb_buffer.cpp:69` | 按指定 ACL 格式创建空 tensor |
| `at_npu::detail::getDefaultNPUGenerator()` | `device.cpp:108` | 获取 NPU 随机数生成器（设 seed） |

### 4. 张量格式转换（`at_npu::native::npu_format_cast`）

这是 NPU 特有的需求——Ascend NPU 有多种数据排布格式（ND、NZ 等），需要显式转换：

| 场景 | 位置 | 说明 |
|------|------|------|
| 权重加载 | `base_manual_loader.cpp:204` | host ND → device NZ 格式 |
| 权重加载 | `qwen2/qwen3/deepseek/glm4_loader.cpp` 多处 | 权重转为格式 2（ND）或 29（NZ）以适配 ATB |
| KV Cache 创建 | `kv_cache_utils.cpp:37,42,74` | KV cache 指定 ACL 格式 |
| KV Cache 迁移 | `kv_cache.cpp:162,163` | KV cache 转为 ND 格式 |
| 模型推理 | `onerec_npu_impl.h:231` | 运行时格式转换确保 ND |
| MoE 权重 | `npu_deepseek_v2_decoder_layer_impl.cpp:618` | gateup/down 权重格式转换 |

### 5. 自定义算子（`at_npu::native::custom_ops::`）

通过 `#include <torch_npu/csrc/aten/CustomFunctions.h>` 引入：

| 算子 | 用途 | 调用位置 |
|------|------|---------|
| `npu_rms_norm` | RMS 归一化 | `fused_layernorm.cpp`, `glm4v.h` |
| `npu_add_rms_norm` | 融合 Add+RMSNorm | `fused_layernorm.cpp` |
| `npu_swiglu` | SwiGLU 激活 | `active.cpp` |
| `npu_apply_rotary_pos_emb` | 旋转位置编码 | `rope.cpp` |
| `npu_grouped_matmul` | 分组矩阵乘（MoE） | `npu_grouped_matmul.cpp` |
| `npu_moe_gating_top_k_softmax` | MoE 门控 | `npu_moe_gating_topk_softmax.cpp` |
| `npu_moe_init_routing_v2` | MoE 路由 | `npu_moe_init_routing_v2.cpp` |
| `npu_moe_token_unpermute` | MoE token 反排列 | `npu_moe_token_unpermute.cpp` |
| `npu_fusion_attention` | 融合注意力 | `transformer_flux.h`, `transformer_qwen_image.h` |
| `npu_rotary_mul` | 旋转乘法 | `transformer_flux.h` |

### 6. OpCommand 异步任务分派（`at_npu::native::OpCommand`）

```cpp
// npu_base_layer.cpp:146
at_npu::native::OpCommand cmd;
cmd.Name(taskName);
cmd.SetCustomHandler(task);  // 注册 ATB 操作回调
cmd.Run();                    // 异步提交到 NPU stream
```

这是 ATB 层执行操作的核心机制——将 ATB 的同步 `Execute()` 调用包装为 torch_npu 的异步 OpCommand，确保与 NPU stream 正确集成。

### 特性重要性总结

按重要性排序：

1. **`c10_npu` Stream 管理**——用量最大，所有 NPU 操作的基础
2. **`NPUGraph`**——ACL Graph 捕获/重放，decode 阶段的核心优化
3. **`custom_ops` 自定义算子**——替代 CUDA kernel 的 NPU 高性能算子
4. **`npu_format_cast` 格式转换**——NPU 特有的数据排布管理
5. **`OpCommand` 任务分派**——ATB 层与 torch_npu 的异步桥接
6. **`NPUStorageImpl` / 设备初始化**——底层存储和设备管理

---

## PYTORCH_INSTALL_PATH 与 PYTORCH_NPU_INSTALL_PATH 的区别

从 CMakeLists.txt 的使用方式来看：

```cmake
# CMakeLists.txt:336-356
include_directories(
    $ENV{PYTORCH_INSTALL_PATH}/include
    $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
    $ENV{PYTORCH_NPU_INSTALL_PATH}/include     # torch_npu 头文件
    $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/distributed
    ...
)
link_directories(
    $ENV{PYTORCH_INSTALL_PATH}/lib              # libtorch 库
    $ENV{PYTORCH_NPU_INSTALL_PATH}/lib          # torch_npu 库
)
```

**`PYTORCH_INSTALL_PATH`** → **PyTorch / libtorch** 原生安装路径

提供：
- `include/torch/` — libtorch C++ 头文件（`<torch/torch.h>`, `<torch/nn.h>` 等）
- `lib/libtorch.so`, `lib/libtorch_python.so` 等
- 也就是标准 PyTorch C++ SDK

**`PYTORCH_NPU_INSTALL_PATH`** → **torch_npu**（华为 Ascend 扩展）安装路径

提供：
- `include/torch_npu/` — NPU 扩展头文件（`<torch_npu/torch_npu.h>`, `<torch_npu/csrc/aten/CustomFunctions.h>` 等）
- `lib/libtorch_npu.so` — torch_npu 库
- 即华为在 PyTorch 之上追加的 NPU 适配层

**本质上是两层关系**：PyTorch 是基础框架，torch_npu 是其 Ascend NPU 后端插件。对应到 Python 世界就是 `import torch` 和 `import torch_npu` 的关系，只是 xllm 在 C++ 层面链接了这两个库。

---

## PyTorch 与 torch_npu 在 C++ 层面的配合机制

### 核心配合原理：PyTorch 的 PrivateUse1 扩展机制

PyTorch 在 C++ 层预留了 **`PrivateUse1`** 这个设备后端槽位，供第三方硬件厂商注册自己的设备实现。torch_npu 就是注册到这个槽位上的 Ascend NPU 后端。

配合发生在 **4 个层面**：

### 1. 设备类型映射

```cpp
// device.cpp:160-162
torch::DeviceType Device::type_torch() {
    return torch::kPrivateUse1;  // PyTorch 不知道"NPU"，只知道 PrivateUse1
}
```

torch_npu 在初始化时将 `PrivateUse1` 注册为 NPU 设备。之后所有 `torch::Tensor` 上 `.device(PrivateUse1)` 的操作，PyTorch 都会分派给 torch_npu 的实现。

### 2. 算子自动分派（Dispatch）

这是最关键的配合——xllm 直接写 **标准 PyTorch C++ API**，torch_npu 负责让这些 API 在 NPU 上执行：

```cpp
// matmul.cpp:25 — 写的是标准 torch API
torch::nn::functional::linear(a, b);

// multi_head_attention.cpp:59-61 — 标准张量操作
torch::matmul(q, k.transpose(-2, -1));
torch::softmax(attn_output, -1);

// tensor 创建
torch::zeros({max_tokens, hidden_size}, torch::dtype(dtype).device(device));
```

**调用链**：`torch::matmul()` → PyTorch dispatch 机制 → 发现 tensor 在 `PrivateUse1` 设备上 → 路由到 torch_npu 注册的 NPU kernel → 实际在 Ascend NPU 上执行。

xllm **不需要写 NPU 版本的 matmul/softmax/linear**，因为 torch_npu 已经为这些标准算子注册了 NPU 实现。

### 3. 存储和内存分配器注册

```cpp
// base_manual_loader.cpp:284-297
c10::DeviceType device_type = c10::DeviceType::PrivateUse1;

// 从 PyTorch 的全局注册表中取出 torch_npu 注册的工厂函数
auto fptr = c10::GetStorageImplCreate(device_type);    // → torch_npu 注册的 NPUStorageImpl 构造器
auto allocator = c10::GetAllocator(device_type);        // → torch_npu 注册的 NPU 内存分配器

storage = fptr(...);  // 创建的是 NPUStorageImpl（torch_npu 的 Storage 子类）
```

torch_npu 在 `init_npu()` 时向 PyTorch 注册了：
- **`NPUStorageImpl`**：带 NPU 格式描述（`npu_desc_.npu_format_`）的存储实现
- **`NPUCachingAllocator`**：NPU 显存管理器

这样 `torch::empty()` 在 `PrivateUse1` 设备上就自动走 NPU 分配器。

### 4. NPU 扩展能力（torch_npu 独有 API）

标准 PyTorch API 覆盖不到的场景，直接调用 torch_npu 扩展：

| 扩展能力 | 为什么要扩展 |
|---------|------------|
| `c10_npu::NPUStream` | PyTorch 只有 CUDAStream，NPU 需要自己的流管理 |
| `c10_npu::NPUGraph` | PyTorch 只有 CUDAGraph，NPU 需要自己的图捕获 |
| `at_npu::native::npu_format_cast` | Ascend 特有的 NZ/ND 数据排布格式，PyTorch 没有这个概念 |
| `at_npu::native::custom_ops::*` | NPU 高性能融合算子（npu_rms_norm 等），PyTorch 标准库没有 |
| `at_npu::native::OpCommand` | NPU 异步任务提交机制 |
| `torch_npu::NPUStorageImpl` | 扩展 Storage 增加 NPU 格式字段 |

### 配合关系图

```
xllm 代码调用
    │
    ├── 标准 PyTorch API (matmul, softmax, linear, zeros...)
    │       │
    │       ▼
    │   PyTorch Dispatch 机制 ──→ 发现 device=PrivateUse1
    │       │
    │       ▼
    │   torch_npu 注册的 NPU kernel 实现执行
    │
    └── torch_npu 扩展 API (NPUStream, NPUGraph, custom_ops...)
            │
            ▼
        直接走 NPU 底层 (ACL runtime)
```

简单说：**PyTorch 提供框架和标准算子分派机制，torch_npu 填充 NPU 的具体实现。** xllm 大部分代码写的是标准 PyTorch API，靠 dispatch 自动路由到 NPU；只有 NPU 特有能力才直接调 torch_npu。
