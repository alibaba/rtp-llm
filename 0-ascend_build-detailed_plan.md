# Phase 0 详细实施方案：Ascend NPU 编译基础设施搭建

> **目标**：打通 Bazel 编译链路，使 rtp-llm 项目可在 Ascend NPU 环境下使用 `--config=ascend` 编译通过。
>
> **预计周期**：2 周
>
> **前置条件**：Ascend CANN Toolkit 已安装（默认路径 `/usr/local/Ascend/ascend-toolkit/latest`）

---

## 目录

- [Step 0.1 创建 ascend_configure.bzl 及模板文件](#step-01-创建-ascend_configurebzl-及模板文件)
- [Step 0.2 修改 WORKSPACE 注册 Ascend 仓库](#step-02-修改-workspace-注册-ascend-仓库)
- [Step 0.3 BUILD 根文件添加 using_ascend 配置](#step-03-build-根文件添加-using_ascend-配置)
- [Step 0.4 修改 def.bzl 添加 Ascend 分支](#step-04-修改-defbzl-添加-ascend-分支)
- [Step 0.5 修改 .bazelrc 添加 build:ascend 配置段](#step-05-修改-bazelrc-添加-buildascend-配置段)
- [Step 0.6 扩展类型枚举和编译守卫](#step-06-扩展类型枚举和编译守卫)
- [Step 0.7 修改 arch_config/arch_select.bzl 设备选择中枢](#step-07-修改-arch_configarch_selectbzl-设备选择中枢)
- [Step 0.8 修改 bazel/device_defs.bzl 设备定义](#step-08-修改-bazeldevice_defsbzl-设备定义)
- [Step 0.9 添加 pip/依赖管理](#step-09-添加-pip依赖管理)
- [Step 0.10 创建 Ascend 兼容层头文件和 BUILD](#step-010-创建-ascend-兼容层头文件和-build)
- [Step 0.11 批量修改 BUILD 文件添加 select() 分支](#step-011-批量修改-build-文件添加-select-分支)
- [Step 0.12 阶段性验证计划](#step-012-阶段性验证计划)
- [附录 A：完整新建文件清单](#附录-a完整新建文件清单)
- [附录 B：完整修改文件清单](#附录-b完整修改文件清单)
- [附录 C：核心 API 映射参考](#附录-c核心-api-映射参考)

---

## Step 0.1 创建 `ascend_configure.bzl` 及模板文件

### 0.1.1 新建 `3rdparty/gpus/ascend_configure.bzl`

**参照范本**：`3rdparty/gpus/rocm_configure.bzl`（1017 行，复用了 `cuda_configure.bzl` 的公共函数）

**核心职责**：

| 任务 | 说明 |
|------|------|
| 环境变量读取 | `TF_NEED_ASCEND`、`ASCEND_TOOLKIT_PATH`（默认 `/usr/local/Ascend/ascend-toolkit/latest`）、`ASCEND_VERSION` |
| 自动探测 CANN | 扫描头文件 `acl/acl.h`、`aclrt/aclrt.h` 和库文件 `lib64/libascendcl.so` |
| 生成 `@local_config_ascend` 仓库 | 包含 BUILD、build_defs.bzl、ascend_config.h |
| 生成 copy_rules | 将 CANN 头文件和库文件拷贝到构建沙箱 |

**详细实现骨架**：

```python
"""Repository rule for Ascend CANN autoconfiguration.

`ascend_configure` depends on the following environment variables:

  * `TF_NEED_ASCEND`: Whether to enable building with Ascend CANN.
  * `ASCEND_TOOLKIT_PATH`: The path to the Ascend CANN toolkit. Default is
    `/usr/local/Ascend/ascend-toolkit/latest`.
  * `ASCEND_VERSION`: The version of the CANN toolkit.
"""

load(
    ":cuda_configure.bzl",
    "make_copy_dir_rule",
    "make_copy_files_rule",
    "to_list_of_strings",
    "verify_build_defines",
)

_ASCEND_TOOLKIT_PATH = "ASCEND_TOOLKIT_PATH"
_TF_ASCEND_VERSION = "TF_ASCEND_VERSION"
_TF_NEED_ASCEND = "TF_NEED_ASCEND"
_DEFAULT_ASCEND_TOOLKIT_PATH = "/usr/local/Ascend/ascend-toolkit/latest"

def _ascend_autoconf_impl(repository_ctx):
    """Implementation of the ascend_configure repository rule."""

    # 1. 检查 TF_NEED_ASCEND 环境变量
    tf_need_ascend = repository_ctx.os.environ.get(_TF_NEED_ASCEND, "0")
    if tf_need_ascend != "1":
        # Ascend 未启用，创建空仓库
        _create_dummy_repo(repository_ctx)
        return

    # 2. 确定 CANN 安装路径
    ascend_toolkit_path = repository_ctx.os.environ.get(
        _ASCEND_TOOLKIT_PATH, _DEFAULT_ASCEND_TOOLKIT_PATH
    )

    # 3. 探测 CANN 头文件和库文件
    include_path = ascend_toolkit_path + "/include"
    lib_path = ascend_toolkit_path + "/lib64"

    _check_file_exists(repository_ctx, include_path + "/acl/acl.h", "CANN header")
    _check_file_exists(repository_ctx, lib_path + "/libascendcl.so", "CANN library")

    # 4. 获取版本号
    ascend_version = repository_ctx.os.environ.get(_TF_ASCEND_VERSION, "")

    # 5. 生成模板文件
    _ascend_create(repository_ctx, ascend_toolkit_path, include_path,
                   lib_path, ascend_version)


def _check_file_exists(repository_ctx, path, label):
    if not repository_ctx.path(path).exists:
        fail("Cannot find %s at %s" % (label, path))


def _create_dummy_repo(repository_ctx):
    """Create empty BUILD and build_defs.bzl when Ascend is not configured."""
    repository_ctx.file("ascend/BUILD", """
package(default_visibility = ["//visibility:public"])
config_setting(name = "using_ascend", values = {"define": "using_ascend=true"})
""")
    repository_ctx.file("ascend/build_defs.bzl", """
def if_ascend(if_true, if_false = []):
    return select({
        "@local_config_ascend//ascend:using_ascend": if_true,
        "//conditions:default": if_false,
    })

def ascend_default_copts():
    return if_ascend([])

def ascend_is_configured():
    return False
""")


def _ascend_create(repository_ctx, toolkit_path, include_path, lib_path, version):
    """Generate the @local_config_ascend repository from CANN installation."""

    # --- 拷贝头文件 ---
    copy_rules = []

    # make_copy_dir_rule/make_copy_files_rule 返回 genrule 字符串
    # 参照 rocm_configure.bzl 的 _create_local_rocm_repository 模式
    copy_rules.append(
        make_copy_dir_rule(
            repository_ctx,
            name = "ascend-include",
            src_dir = include_path,
            out_dir = "ascend/include",
            exceptions = None,  # 参数名为 exceptions，而非 excludes
        )
    )

    # --- 拷贝库文件 ---
    lib_files = {
        "libascendcl.so": "ascend",
        "libaclblas.so":  "aclblas",
        "libhccl.so":     "hccl",
    }

    lib_srcs = []
    lib_outs = []
    for lib_name, target_name in lib_files.items():
        src = lib_path + "/" + lib_name
        if repository_ctx.path(src).exists:
            lib_srcs.append(src)
            lib_outs.append("ascend/lib/" + lib_name)

    copy_rules.append(
        make_copy_files_rule(
            repository_ctx,
            name = "ascend-lib",
            srcs = lib_srcs,
            outs = lib_outs,
        )
    )

    # --- 生成 BUILD 内容 ---
    # 参照 rocm/rocm_configure.bzl: copy_rules 通过 %{copy_rules} 模板变量注入
    build_content = """package(default_visibility = ["//visibility:public"])

config_setting(
    name = "using_ascend",
    values = {{"define": "using_ascend=true"}},
)

cc_library(
    name = "ascend_headers",
    hdrs = glob(["ascend/include/**/*.h"]),
    includes = [".", "ascend/include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ascend",
    srcs = ["ascend/lib/libascendcl.so"],
    data = ["ascend/lib/libascendcl.so"],
    includes = [".", "ascend/include"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "aclblas",
    srcs = ["ascend/lib/libaclblas.so"],
    data = ["ascend/lib/libaclblas.so"],
    includes = [".", "ascend/include"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "hccl",
    srcs = ["ascend/lib/libhccl.so"],
    data = ["ascend/lib/libhccl.so"],
    includes = [".", "ascend/include"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

""" + "\n".join(copy_rules)

    repository_ctx.file("ascend/BUILD", build_content)

    # --- 生成 build_defs.bzl ---
    repository_ctx.file("ascend/build_defs.bzl", """def if_ascend(if_true, if_false = []):
    return select({{
        "@local_config_ascend//ascend:using_ascend": if_true,
        "//conditions:default": if_false,
    }})

def ascend_default_copts():
    return if_ascend([])

def ascend_is_configured():
    return True
""")

    # --- 生成 ascend_config.h ---
    repository_ctx.file("ascend/ascend_config.h", """#ifndef ASCEND_ASCEND_CONFIG_H_
#define ASCEND_ASCEND_CONFIG_H_
#define TF_ASCEND_TOOLKIT_PATH "%s"
#define TF_ASCEND_VERSION "%s"
#endif
""" % (toolkit_path, version))


ascend_configure = repository_rule(
    implementation = _ascend_autoconf_impl,
    environ = [
        _TF_NEED_ASCEND,
        _ASCEND_TOOLKIT_PATH,
        _TF_ASCEND_VERSION,
    ],
)
```

### 0.1.2 模板目录文件（由 `ascend_configure.bzl` 在运行时生成）

以下文件**不需要手动创建**，由 `ascend_configure.bzl` 生成到 `@local_config_ascend` 外部仓库中：

| 生成文件 | 用途 |
|---------|------|
| `ascend/BUILD` | Bazel 构建目标（ascend_headers, ascend, aclblas, hccl） |
| `ascend/build_defs.bzl` | `if_ascend()` / `ascend_default_copts()` / `ascend_is_configured()` |
| `ascend/ascend_config.h` | `TF_ASCEND_TOOLKIT_PATH` / `TF_ASCEND_VERSION` 宏 |

> **注意**：`3rdparty/gpus/` 目录下不需要创建 `ascend/` 模板目录（与 ROCm 不同），因为 Ascend 配置直接在 `.bzl` 中内联生成 BUILD 内容，模板文件更简单。

---

## Step 0.2 修改 WORKSPACE 注册 Ascend 仓库

**文件**：`WORKSPACE`（根目录）

**当前状态**（第 3-9 行）：
```python
load("//3rdparty/cuda_config:cuda_configure.bzl", "cuda_configure")
load("//3rdparty/gpus:rocm_configure.bzl", "rocm_configure")
load("//3rdparty/py:python_configure.bzl", "python_configure")

cuda_configure(name = "local_config_cuda")
rocm_configure(name = "local_config_rocm")
python_configure(name = "local_config_python")
```

**修改内容**：在 `rocm_configure` 的 `load` 之后添加 Ascend 的 `load`，调用放在 `rocm_configure()` 之后：

```python
load("//3rdparty/gpus:ascend_configure.bzl", "ascend_configure")
ascend_configure(name = "local_config_ascend")
```

**完整 diff**（与实际 WORKSPACE 文件结构一致）：

```diff
--- a/WORKSPACE
+++ b/WORKSPACE
@@ -3,11 +3,14 @@
 load("//3rdparty/cuda_config:cuda_configure.bzl", "cuda_configure")
 load("//3rdparty/gpus:rocm_configure.bzl", "rocm_configure")
+load("//3rdparty/gpus:ascend_configure.bzl", "ascend_configure")
 load("//3rdparty/py:python_configure.bzl", "python_configure")
 
 cuda_configure(name = "local_config_cuda")
 
 rocm_configure(name = "local_config_rocm")
 
+ascend_configure(name = "local_config_ascend")
+
 python_configure(name = "local_config_python")
```

> **注意**：与 `cuda_configure` / `rocm_configure` 模式一致，`load` 放在 `workspace()` 之后、`ascend_configure()` 调用放在 `rocm_configure()` 之后。

---

## Step 0.3 BUILD 根文件添加 `using_ascend` 配置

**文件**：`BUILD`（根目录）

**当前状态**（第 48-51 行）：
```python
config_setting(
    name = "using_rocm",
    values = {"define": "using_rocm=true"},
)
```

**修改内容**：在 `using_rocm` 之后（约第 51 行后）添加：

```python
config_setting(
    name = "using_ascend",
    values = {"define": "using_ascend=true"},
)
```

**完整 diff**：

```diff
--- a/BUILD
+++ b/BUILD
@@ -50,6 +50,11 @@ config_setting(
     values = {"define": "using_rocm=true"},
 )
 
+config_setting(
+    name = "using_ascend",
+    values = {"define": "using_ascend=true"},
+)
+
 
 config_setting(
     name = "rocm_gfx950",
```

**`rtp_compute_ops` 目标**（第 108-126 行）：当前 `select` 只有 `using_cuda12` 和 `default`。Ascend 编译时走 `default`（空 deps），**无需修改此 target**。但后续阶段需在此添加 Ascend 分支。

> **额外注意**：根 `BUILD` 第 5 行无条件调用 `flashinfer_deps()`，该函数创建 `@flashinfer_cpp//:flashinfer` 别名。由于 `@flashinfer_cpp` 依赖 CUDA 仓库，在 Ascend 模式下可能导致 Bazel 解析失败。需要在 `arch_select.bzl` 的 `flashinfer_deps()` 中添加 Ascend 分支（见 Step 0.7.7）。

---

## Step 0.4 修改 `def.bzl` 添加 Ascend 分支

**文件**：`def.bzl`（根目录）

### 0.4.1 添加 `if_ascend` 导入

**当前状态**（第 1-14 行）：
```python
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "cuda_default_copts",
    _if_cuda = "if_cuda",
)

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "rocm_default_copts",
    _if_rocm = "if_rocm",
)

if_rocm = _if_rocm
if_cuda = _if_cuda
```

**修改后**：
```python
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "cuda_default_copts",
    _if_cuda = "if_cuda",
)

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "rocm_default_copts",
    _if_rocm = "if_rocm",
)

load(
    "@local_config_ascend//ascend:build_defs.bzl",
    "ascend_default_copts",
    _if_ascend = "if_ascend",
)

if_rocm = _if_rocm
if_cuda = _if_cuda
if_ascend = _if_ascend
```

### 0.4.2 修改 `copts()` 函数

**当前状态**（第 135-145 行）：
```python
def copts():
    return [
        "-DTORCH_CUDA",
    ] + if_cuda([
        "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
        "-DUSE_C10D_NCCL",
        "-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE",
    ]) + if_rocm([
        "-x", "rocm",
        "-DUSE_C10D_NCCL",
    ])
```

**修改后**：
```python
def copts():
    return [
        "-DTORCH_CUDA",
    ] + if_cuda([
        "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
        "-DUSE_C10D_NCCL",
        "-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE",
    ]) + if_rocm([
        "-x", "rocm",
        "-DUSE_C10D_NCCL",
    ]) + if_ascend([
        "-DUSING_ASCEND=1",
        "-DUSE_C10D_HCCL",
    ])
```

> **关于 `-DTORCH_CUDA`**：此宏定义在 `copts()` 无条件返回的列表中（非 `if_cuda()` 内部），意味着 Ascend 模式下也会生效。`-DTORCH_CUDA` 可能导致 PyTorch 的 CUDA 相关头文件被错误包含（如 `torch/csrc/cuda/Stream.h`），Ascend 编译时很可能需要移除或通过 `if_ascend([])` 反向条件化。Phase 0 计划以 `-DUSING_CUDA=0` 配合 `#if USING_CUDA` 守卫拦截大部分 CUDA 代码，但如果看到类似 `/torch/csrc/cuda/Stream.h` not found 的编译错误，应优先通过 `if_ascend([])` 在 Ascend 下移除 `-DTORCH_CUDA`。

### 0.4.3 添加 `ascend_copts()` 函数

在 `rocm_copts()` 函数之后（第 152 行后）添加：

```python
def ascend_copts():
    return copts() + ascend_default_copts()
```

**完整 diff**：

```diff
--- a/def.bzl
+++ b/def.bzl
@@ -10,9 +10,16 @@ load(
     _if_rocm = "if_rocm",
 )
 
+load(
+    "@local_config_ascend//ascend:build_defs.bzl",
+    "ascend_default_copts",
+    _if_ascend = "if_ascend",
+)
+
 if_rocm = _if_rocm
 if_cuda = _if_cuda
+if_ascend = _if_ascend
 
 def rpm_library(
         name,
@@ -142,6 +149,9 @@ def copts():
     ]) + if_rocm([
         "-x", "rocm",
         "-DUSE_C10D_NCCL",
+    ]) + if_ascend([
+        "-DUSING_ASCEND=1",
+        "-DUSE_C10D_HCCL",
     ])
 
 def cuda_copts():
@@ -153,6 +163,10 @@ def rocm_copts():
     return copts() + rocm_default_copts() + if_rocm(["-Wc++17-extensions"])
 
+def ascend_copts():
+    return copts() + ascend_default_copts()
+
 def any_cuda_copts():
     return copts() + cuda_default_copts() + if_cuda(["-nvcc_options=objdir-as-tempdir"]) + rocm_default_copts() + if_rocm(["-Wc++17-extensions"])
```

---

## Step 0.5 修改 `.bazelrc` 添加 `build:ascend` 配置段

**文件**：`.bazelrc`（根目录）

**插入位置**：在 `build:rocm` 配置段结束之后（第 178 行 `test:rocm` 之后）、`build:arm` 配置段之前（第 180 行）。

**添加内容**：

```starlark
# ==================== Ascend NPU ====================
build:ascend --copt="-DENABLE_BF16=1"
build:ascend --action_env TF_NEED_CUDA="0"
build:ascend --host_action_env TF_NEED_CUDA="0"
build:ascend --crosstool_top=@bazel_tools//tools/cpp:toolchain
build:ascend --host_crosstool_top=@bazel_tools//tools/cpp:toolchain
build:ascend --define=using_cuda=false --define=using_cuda_nvcc=false
build:ascend --define=using_ascend=true
build:ascend --action_env TF_NEED_ASCEND=1
build:ascend --host_action_env TF_NEED_ASCEND=1
build:ascend --action_env ASCEND_TOOLKIT_PATH="/usr/local/Ascend/ascend-toolkit/latest"
build:ascend --host_action_env ASCEND_TOOLKIT_PATH="/usr/local/Ascend/ascend-toolkit/latest"
build:ascend --copt="-DUSING_CUDA=0"
build:ascend --copt="-DUSING_ASCEND=1"
build:ascend --copt="-D_GLIBCXX_USE_CXX11_ABI=1"
build:ascend --linkopt="-L/usr/local/Ascend/ascend-toolkit/latest/lib64"
build:ascend --linkopt="-lascendcl"
build:ascend --linkopt="-laclblas"
build:ascend --linkopt="-lhccl"
build:ascend --copt=-Wno-deprecated-declarations
build:ascend --action_env LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/add-ons/"
build:ascend --host_action_env LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/add-ons/"

test:ascend --test_env LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/add-ons/"
test:ascend --test_env TEST_USING_DEVICE="ASCEND"
```

**关键设计说明**：

| 配置项 | 说明 |
|--------|------|
| `crosstool_top=@bazel_tools//tools/cpp:toolchain` | Ascend 使用标准 g++ 编译，**不需要自定义 crosstool**（与 `build:cpu` 一致）。只有使用 Ascend C 编译器 (`opc`) 编译自定义算子时才需要 |
| `define=using_ascend=true` | 触发根 BUILD 中的 `config_setting(name = "using_ascend")` |
| `linkopt=-lascendcl` 等 | 链接 CANN 运行时库 |
| `copt=-DUSING_CUDA=0` | 禁用 CUDA 代码路径 |
| `copt=-D_GLIBCXX_USE_CXX11_ABI=1` | 与 ROCm 配置一致。CPU 配置使用 `=0`（旧 ABI），Ascend 使用 CXX11 ABI 以匹配 PyTorch。如果编译中出现 `undefined reference to std::__cxx11::...` 相关链接错误，需确认该值与编译环境使用的 libstdc++ 版本匹配。 |

---

## Step 0.6 扩展类型枚举和编译守卫

### 0.6.1 `MemoryType` 枚举扩展

**文件**：`rtp_llm/models_py/bindings/core/Types.h`（第 10-14 行）

**当前**：
```cpp
typedef enum memorytype_enum {
    MEMORY_CPU,
    MEMORY_CPU_PINNED,
    MEMORY_GPU
} MemoryType;
```

**修改后**：
```cpp
typedef enum memorytype_enum {
    MEMORY_CPU,
    MEMORY_CPU_PINNED,
    MEMORY_GPU,
    MEMORY_NPU
} MemoryType;
```

### 0.6.2 `Types.cc` 添加 Ascend 类型支持

**文件**：`rtp_llm/models_py/bindings/core/Types.cc`

**当前**（第 9-19 行）：
```cpp
#if USING_CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

#if USING_ROCM
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#endif
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
```

**修改后**：
```cpp
#if USING_CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

#if USING_ROCM
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#endif

#if USING_ASCEND
// Ascend: 使用 PyTorch 的 half/bfloat16 类型
// 暂时使用 fake 类型，待 torch_npu 集成后替换
#endif

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
```

**同时修改第 38-53 行的 `FT_FOREACH_DEVICE_TYPE` 宏**：

```cpp
// 当前：
#if USING_CUDA || USING_ROCM
#define FT_FOREACH_DEVICE_TYPE(F)                                                                                      \
    F(DataType::TYPE_FP16, half);                                                                                      \
    F(DataType::TYPE_BF16, __nv_bfloat16);

#else
struct fake_half { uint16_t x; };
struct fake_bfloat16 { uint16_t x; };
#define FT_FOREACH_DEVICE_TYPE(F)                                                                                      \
    F(DataType::TYPE_FP16, fake_half);                                                                                 \
    F(DataType::TYPE_BF16, fake_bfloat16);
#endif

// 修改后：
#if USING_CUDA || USING_ROCM
#define FT_FOREACH_DEVICE_TYPE(F)                                                                                      \
    F(DataType::TYPE_FP16, half);                                                                                      \
    F(DataType::TYPE_BF16, __nv_bfloat16);

#elif USING_ASCEND
struct fake_half { uint16_t x; };
struct fake_bfloat16 { uint16_t x; };
#define FT_FOREACH_DEVICE_TYPE(F)                                                                                      \
    F(DataType::TYPE_FP16, fake_half);                                                                                 \
    F(DataType::TYPE_BF16, fake_bfloat16);
#else
struct fake_half { uint16_t x; };
struct fake_bfloat16 { uint16_t x; };
#define FT_FOREACH_DEVICE_TYPE(F)                                                                                      \
    F(DataType::TYPE_FP16, fake_half);                                                                                 \
    F(DataType::TYPE_BF16, fake_bfloat16);
#endif
```

### 0.6.3 `cuda_host_utils.h` 添加 Ascend 分支

**文件**：`rtp_llm/models_py/bindings/cuda/cuda_host_utils.h`（第 24-28 行）

**当前**：
```cpp
#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include <hip/hip_runtime.h>
#endif
```

**修改后**：
```cpp
#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include <hip/hip_runtime.h>
#elif USING_ASCEND
// Ascend host utils are in a separate header, not here
// See rtp_llm/models_py/bindings/ascend/ascend_host_utils.h
#endif
```

> **设计决策**：`cuda_host_utils.h` 暂不在 Phase 0 修改太多。Ascend 的 host utils 放在独立的 `rtp_llm/models_py/bindings/ascend/ascend_host_utils.h` 中，通过 BUILD 文件的 `select()` 选择。这样可以避免大面积修改现有代码。
>
> **关键限制**：`cuda_host_utils.h` 中第 34-66 行的函数声明（如 `check(cudaError_t, ...)`、`timing_function(..., cudaStream_t)`）在其签名中直接使用了 CUDA 类型。即使添加 `#elif USING_ASCEND` include 守卫，这些函数声明的签名在 Ascend 模式下仍会因 `cudaStream_t` 等类型不可用而编译失败。因此 Ascend 模式下**该文件不应被编译**——依赖于该文件的 BUILD target 通过 `select()` 条件化，Ascend 分支指向 `ascend_host_utils.h` 替代。Phase 1 将逐步将 CUDA 特定函数提取为接口/虚函数以彻底解决此问题。

---

## Step 0.7 修改 `arch_config/arch_select.bzl` 设备选择中枢

**文件**：`arch_config/arch_select.bzl`

这是**最关键的修改文件**，几乎所有 BUILD 文件都通过它间接选择设备相关依赖。

### 0.7.1 添加 Ascend pip requirement 加载

**当前**（第 1-7 行）：
```python
load("@pip_cpu_torch//:requirements.bzl", requirement_cpu="requirement")
load("@pip_arm_torch//:requirements.bzl", requirement_arm="requirement")
load("@pip_gpu_cuda12_torch//:requirements.bzl", requirement_gpu_cuda12="requirement")
load("@pip_gpu_cuda12_9_torch//:requirements.bzl", requirement_gpu_cuda12_9="requirement")
load("@pip_gpu_rocm_torch//:requirements.bzl", requirement_gpu_rocm="requirement")
```

**修改后**：
```python
load("@pip_cpu_torch//:requirements.bzl", requirement_cpu="requirement")
load("@pip_arm_torch//:requirements.bzl", requirement_arm="requirement")
load("@pip_gpu_cuda12_torch//:requirements.bzl", requirement_gpu_cuda12="requirement")
load("@pip_gpu_cuda12_9_torch//:requirements.bzl", requirement_gpu_cuda12_9="requirement")
load("@pip_gpu_rocm_torch//:requirements.bzl", requirement_gpu_rocm="requirement")
load("@pip_ascend_torch//:requirements.bzl", requirement_ascend="requirement")
```

> **关键约束**：`load()` 是 Bazel 全局语句，`@pip_ascend_torch` 仓库在 WORKSPACE 加载时必须存在，否则整个文件报错且无法恢复。因此 `pip_ascend_torch` 必须在 `pip_deps()` 中**无条件创建**（即使 `TF_NEED_ASCEND=0`），让 `arch_select.bzl` 始终能成功加载。这是与本代码库其他 pip 仓库一致的策略。

### 0.7.2 修改 `requirement()` 函数

**当前**（第 14-26 行）：
```python
def requirement(names):
    for name in names:
        native.py_library(
            name = name,
            deps = select({
                "@rtp_llm//:cuda_pre_12_9": [requirement_gpu_cuda12(name)],
                "@rtp_llm//:using_cuda12_9_x86": [requirement_gpu_cuda12_9(name)],
                "@rtp_llm//:using_rocm": [requirement_gpu_rocm(name)],
                "@rtp_llm//:using_arm": [requirement_arm(name)],
                "//conditions:default": [requirement_cpu(name)],
            }),
            visibility = ["//visibility:public"],
        )
```

**修改后**：
```python
def requirement(names):
    for name in names:
        native.py_library(
            name = name,
            deps = select({
                "@rtp_llm//:cuda_pre_12_9": [requirement_gpu_cuda12(name)],
                "@rtp_llm//:using_cuda12_9_x86": [requirement_gpu_cuda12_9(name)],
                "@rtp_llm//:using_rocm": [requirement_gpu_rocm(name)],
                "@rtp_llm//:using_arm": [requirement_arm(name)],
                "@rtp_llm//:using_ascend": [requirement_ascend(name)],
                "//conditions:default": [requirement_cpu(name)],
            }),
            visibility = ["//visibility:public"],
        )
```

### 0.7.3 修改 `whl_deps()` 函数

**当前**（第 58-63 行）：
```python
def whl_deps():
    return select({
        "@rtp_llm//:using_cuda12": ["torch==2.6.0+cu126"],
        "@rtp_llm//:using_rocm": ["pyrsmi==0.2.0", ...],
        "//conditions:default": ["torch==2.1.2"],
    })
```

**修改后**：
```python
def whl_deps():
    return select({
        "@rtp_llm//:using_cuda12": ["torch==2.6.0+cu126"],
        "@rtp_llm//:using_rocm": ["pyrsmi==0.2.0", "amdsmi@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis%2FAMD%2Famd_smi%2Fali%2Famd_smi.tar", "aiter@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/RTP/aiter-0.1.13.dev14%2Bgfa35072d0.d20260402-cp310-cp310-linux_x86_64.whl"],
        "@rtp_llm//:using_ascend": ["torch==2.5.1", "torch_npu==2.5.1"],
        "//conditions:default": ["torch==2.1.2"],
    })
```

### 0.7.4 修改 `torch_deps()` 函数

**当前**（第 73-101 行）：
```python
def torch_deps():
    deps = select({
        "@rtp_llm//:using_rocm": [
            "@torch_rocm//:torch_api", "@torch_rocm//:torch", "@torch_rocm//:torch_libs",
        ],
        "@rtp_llm//:using_arm": [
            "@torch_2.3_py310_cpu_aarch64//:torch_api", ...
        ],
        "@rtp_llm//:cuda_pre_12_9": [
            "@torch_2.6_py310_cuda//:torch_api", ...
        ],
        "@rtp_llm//:using_cuda12_9_x86": [
            "@torch_2.8_py310_cuda//:torch_api", ...
        ],
        "//conditions:default": [
            "@torch_2.1_py310_cpu//:torch_api", ...
        ]
    })
    return deps
```

**修改后**：
```python
def torch_deps():
    deps = select({
        "@rtp_llm//:using_rocm": [
            "@torch_rocm//:torch_api",
            "@torch_rocm//:torch",
            "@torch_rocm//:torch_libs",
        ],
        "@rtp_llm//:using_arm": [
            "@torch_2.3_py310_cpu_aarch64//:torch_api",
            "@torch_2.3_py310_cpu_aarch64//:torch",
            "@torch_2.3_py310_cpu_aarch64//:torch_libs",
        ],
        "@rtp_llm//:cuda_pre_12_9": [
            "@torch_2.6_py310_cuda//:torch_api",
            "@torch_2.6_py310_cuda//:torch",
            "@torch_2.6_py310_cuda//:torch_libs",
        ],
        "@rtp_llm//:using_cuda12_9_x86": [
            "@torch_2.8_py310_cuda//:torch_api",
            "@torch_2.8_py310_cuda//:torch",
            "@torch_2.8_py310_cuda//:torch_libs",
        ],
        "@rtp_llm//:using_ascend": [
            "@torch_ascend//:torch_api",
            "@torch_ascend//:torch",
            "@torch_ascend//:torch_libs",
        ],
        "//conditions:default": [
            "@torch_2.1_py310_cpu//:torch_api",
            "@torch_2.1_py310_cpu//:torch",
            "@torch_2.1_py310_cpu//:torch_libs",
        ]
    })
    return deps
```

> **注意**：`@torch_ascend` 仓库需在 `deps/http.bzl` 中注册（见 Step 0.9.3）。

### 0.7.5 修改 `select_py_bindings()` 函数

**当前**（第 141-152 行）：
```python
def select_py_bindings():
    return select({
        "@rtp_llm//:using_cuda12": [
            "@rtp_llm//rtp_llm/models_py/bindings/cuda:cuda_bindings_register"
        ],
        "@rtp_llm//:using_rocm": [
            "@rtp_llm//rtp_llm/models_py/bindings/rocm:rocm_bindings_register"
        ],
        "//conditions:default": [
            "@rtp_llm//rtp_llm/models_py/bindings:dummy_register",
        ],
    })
```

**修改后**：
```python
def select_py_bindings():
    return select({
        "@rtp_llm//:using_cuda12": [
            "@rtp_llm//rtp_llm/models_py/bindings/cuda:cuda_bindings_register"
        ],
        "@rtp_llm//:using_rocm": [
            "@rtp_llm//rtp_llm/models_py/bindings/rocm:rocm_bindings_register"
        ],
        "@rtp_llm//:using_ascend": [
            "@rtp_llm//rtp_llm/models_py/bindings:dummy_register",
        ],
        "//conditions:default": [
            "@rtp_llm//rtp_llm/models_py/bindings:dummy_register",
        ],
    })
```

> Ascend 初期使用 `dummy_register`（已有空实现），Phase 5 替换为 `ascend_bindings_register`。

### 0.7.6 修改 `no_block_copy_link_deps()` 函数

**当前**（第 154-166 行）：
```python
def no_block_copy_link_deps():
    return select({
        "@rtp_llm//:using_cuda12": [
            "@rtp_llm//rtp_llm/models_py/bindings/cuda:no_block_copy",
        ],
        "@rtp_llm//:using_rocm": [
            "@rtp_llm//rtp_llm/models_py/bindings:no_block_copy_default",
        ],
        "//conditions:default": [
            "@rtp_llm//rtp_llm/models_py/bindings:no_block_copy_default",
        ],
    })
```

**修改后**：
```python
def no_block_copy_link_deps():
    return select({
        "@rtp_llm//:using_cuda12": [
            "@rtp_llm//rtp_llm/models_py/bindings/cuda:no_block_copy",
        ],
        "@rtp_llm//:using_rocm": [
            "@rtp_llm//rtp_llm/models_py/bindings:no_block_copy_default",
        ],
        "@rtp_llm//:using_ascend": [
            "@rtp_llm//rtp_llm/models_py/bindings:no_block_copy_default",
        ],
        "//conditions:default": [
            "@rtp_llm//rtp_llm/models_py/bindings:no_block_copy_default",
        ],
    })
```

### 0.7.7 修改 `flashinfer_deps()` 函数

**当前**（第 103-107 行）：
```python
def flashinfer_deps():
    native.alias(
        name = "flashinfer",
        actual = "@flashinfer_cpp//:flashinfer"
    )
```

**修改后**：
```python
def flashinfer_deps():
    native.alias(
        name = "flashinfer",
        actual = select({
            "@rtp_llm//:using_ascend": "@rtp_llm//rtp_llm/models_py/bindings:dummy_register",
            "//conditions:default": "@flashinfer_cpp//:flashinfer",
        }),
    )
```

> **关键说明**：`flashinfer_deps()` 在根 `BUILD` 第 5 行被**无条件调用**，创建 `//external:flashinfer` 别名。如果 `@flashinfer_cpp` 依赖的 CUDA 仓库在 Ascend 模式下不可用，Bazel 解析阶段就会失败。此处通过 `select()` 将 Ascend 分支重定向到一个已知存在的空目标（`dummy_register`），避免解析失败。注意：`native.alias` 的 `actual` 参数通常不支持 `select()`，若 Bazel 报错，替代方案是将 `flashinfer_deps()` 的调用点也条件化——在根 BUILD 中包裹 `if not ascend_is_configured()` 或改用其他方式。

### 0.7.8 修改 `platform_deps()` 函数

**当前**（第 65-71 行）：
```python
def platform_deps():
    return select({
        "@rtp_llm//:using_arm": [],
        "@rtp_llm//:using_cuda12_arm": [],
        "@rtp_llm//:using_rocm": ["pyyaml==6.0.2","decord==0.6.0"],
        "//conditions:default": ["decord==0.6.0"],
    })
```

**修改后**：
```python
def platform_deps():
    return select({
        "@rtp_llm//:using_arm": [],
        "@rtp_llm//:using_cuda12_arm": [],
        "@rtp_llm//:using_rocm": ["pyyaml==6.0.2","decord==0.6.0"],
        "@rtp_llm//:using_ascend": ["decord==0.6.0"],
        "//conditions:default": ["decord==0.6.0"],
    })
```

---

## Step 0.8 修改 `bazel/device_defs.bzl` 设备定义

**文件**：`bazel/device_defs.bzl`

### 0.8.1 修改 `device_test_envs()`

**当前**：
```python
def device_test_envs():
    return select({
        "@//:using_cuda": {
            "TEST_USING_DEVICE": "CUDA",
            "LD_PRELOAD": "libtorch_cpu.so",
        },
        "@//:using_rocm": {
            "TEST_USING_DEVICE": "ROCM",
        },
        "//conditions:default": {
            "TEST_USING_DEVICE": "CUDA",
            "LD_PRELOAD": "libtorch_cpu.so",
        },
    })
```

**修改后**：
```python
def device_test_envs():
    return select({
        "@//:using_cuda": {
            "TEST_USING_DEVICE": "CUDA",
            "LD_PRELOAD": "libtorch_cpu.so",
        },
        "@//:using_rocm": {
            "TEST_USING_DEVICE": "ROCM",
        },
        "@//:using_ascend": {
            "TEST_USING_DEVICE": "ASCEND",
        },
        "//conditions:default": {
            "TEST_USING_DEVICE": "CUDA",
            "LD_PRELOAD": "libtorch_cpu.so",
        },
    })
```

### 0.8.2 修改 `device_impl_target()`

**当前**：
```python
def device_impl_target():
    return select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        ],
        "//conditions:default": [],
    })
```

**修改后**：
```python
def device_impl_target():
    return select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        ],
        "@//:using_ascend": [
            # Phase 5 实现: "//rtp_llm/models_py/bindings/ascend/ops:ascend_impl",
        ],
        "//conditions:default": [],
    })
```

---

## Step 0.9 添加 pip/依赖管理

### 0.9.1 新建 `deps/requirements_ascend.txt`

```
# Ascend NPU dependencies
torch==2.5.1
torch_npu==2.5.1
numpy==1.26.4
```

> **版本说明**：`torch_npu` 版本需与 CANN 版本匹配。具体版本号需根据目标 CANN 版本确定。建议使用 CANN 8.0 + torch_npu 2.5.1。

### 0.9.2 修改 `deps/pip.bzl`

**文件**：`deps/pip.bzl`

在文件末尾（第 67 行后）添加：

```python
    pip_parse(
        name = "pip_ascend_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_ascend.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 12000,
    )
```

> **关键说明**：此 `pip_parse` **无条件执行**（不依赖 `TF_NEED_ASCEND` 环境变量），因为 `arch_select.bzl` 在文件顶部 `load()` 了 `@pip_ascend_torch`，Bazel 要求在解析时仓库必须存在。即使 Ascend 未启用，`pip_ascend_torch` 也会创建，但只有当 `--config=ascend` 时 `select()` 才会选择它。这与本库现有 `pip_gpu_rocm_torch` 等其他 pip 仓库的模式一致。

### 0.9.3 修改 `deps/http.bzl` 注册 torch_npu wheel

**文件**：`deps/http.bzl`

在 `torch_rocm` 条目之后（第 69 行后）添加：

```python
    http_archive(
        name = "torch_ascend",
        sha256 = "<TO_BE_FILLED_AFTER_WHEEL_DOWNLOAD>",
        urls = [
            # torch_npu wheel URL - 需要根据实际分发方式填入
            # 华为官方 PyPI: https://pypi.org/project/torch-npu/
            # 或本地 wheel: "file:///path/to/torch_npu-2.5.1-cp310-cp310-linux_x86_64.whl"
        ],
        type = "zip",
        build_file = clean_dep("@rtp_llm//:BUILD.pytorch"),
    )
```

> **注意**：
> - `sha256` 和 `urls` 需要在实际部署时填入。
> - **`BUILD.pytorch` 兼容性**：`torch_npu` wheel 内部结构可能与 PyTorch wheel 不同（它是 torch 的 ABI 兼容插件，内含 `torch_npu/_C.cpython-310-*.so` 等独立符号）。直接复用 `BUILD.pytorch` **可能需要调整**。建议先下载 wheel 解压检查目录树，确认 `torch_npu/` 目录结构后再决定是修改 `BUILD.pytorch` 还是创建单独的 `BUILD.torch_npu`。如果 torch_npu 作为独立 `.so` 存在（不侵入 torch 包），也可以不在 `http.bzl` 中 archive，而是通过 `requirements_ascend.txt` pip 安装。

### 0.9.4 修改 `deps/BUILD` 添加 requirements 编译目标

**文件**：`deps/BUILD`

在 `requirements_rocm` 之后（第 81 行后）添加：

```python
compile_pip_requirements(
    name = "requirements_ascend",
    src = "requirements_ascend.txt",
    extra_args = PIP_EXTRA_ARGS,
    extra_data = ["//:requirements_base.txt"],
    requirements_txt = "requirements_lock_ascend.txt",
    tags = ["manual"],
)
```

### 0.9.5 修改 `WORKSPACE` 添加 pip_ascend_torch 安装

在 `WORKSPACE` 文件中 `pip_gpu_rocm_torch_install_deps()` 之后（第 58 行后）添加：

```python
load("@pip_ascend_torch//:requirements.bzl", pip_ascend_torch_install_deps = "install_deps")
pip_ascend_torch_install_deps()
```

---

## Step 0.10 创建 Ascend 兼容层头文件和 BUILD

### 0.10.1 新建目录

```
rtp_llm/models_py/bindings/ascend/
├── BUILD
├── ascend_types_hdr.h
├── ascend_host_utils.h
├── ascend_host_utils.cc
└── AscendRegister.cc
```

### 0.10.2 新建 `rtp_llm/models_py/bindings/ascend/ascend_types_hdr.h`

```cpp
#pragma once

#if USING_ASCEND
#include <acl/acl.h>

namespace rtp_llm {
namespace ascend {

using ascendStream_t = aclrtStream;
using ascendEvent_t  = aclrtEvent;

template<typename T>
void check(T result, const char* const file, int const line);

void syncAndCheckInDebug(const char* const file, int const line);

}  // namespace ascend
}  // namespace rtp_llm

#define ASCEND_CHECK(val) rtp_llm::ascend::check((val), __FILE__, __LINE__)
#define ASCEND_CHECK_ERROR() rtp_llm::ascend::syncAndCheckInDebug(__FILE__, __LINE__)

#endif  // USING_ASCEND
```

### 0.10.3 新建 `rtp_llm/models_py/bindings/ascend/ascend_host_utils.h`

```cpp
#pragma once

#if USING_ASCEND
#include <acl/acl.h>
#include "ascend_types_hdr.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <cstddef>
#include <tuple>
#include <string>

namespace rtp_llm {
namespace ascend {

int  getDevice();
int  getDeviceCount();
int  currentDeviceId();
std::tuple<size_t, size_t> getDeviceMemoryInfo(bool const useUvm);
int  getMultiProcessorCount(int device_id = -1);
void setAscendGraphCaptureEnabled(bool enabled);
bool isAscendGraphCaptureEnabled();

}  // namespace ascend
}  // namespace rtp_llm

#endif  // USING_ASCEND
```

### 0.10.4 新建 `rtp_llm/models_py/bindings/ascend/ascend_host_utils.cc`

```cpp
#include "ascend_host_utils.h"

#if USING_ASCEND
#include <acl/acl.h>
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace ascend {

static bool ascend_graph_capture_enabled = false;

template<typename T>
void check(T result, const char* const file, int const line) {
    if (result != ACL_SUCCESS) {
        RTP_LLM_LOG_ERROR("Ascend error at %s:%d, error code: %d", file, line,
                          static_cast<int>(result));
        throw std::runtime_error("Ascend runtime error");
    }
}

template void check<aclError>(aclError result, const char* const file, int const line);

void syncAndCheckInDebug(const char* const file, int const line) {
    aclError err = aclrtSynchronizeDevice();
    if (err != ACL_SUCCESS) {
        RTP_LLM_LOG_ERROR("Ascend sync error at %s:%d, error code: %d", file, line,
                          static_cast<int>(err));
        throw std::runtime_error("Ascend sync error");
    }
}

int getDevice() {
    int32_t device_id = 0;
    aclrtGetDevice(&device_id);
    return device_id;
}

int getDeviceCount() {
    uint32_t count = 0;
    aclrtGetDeviceCount(&count);
    return static_cast<int>(count);
}

int currentDeviceId() {
    return getDevice();
}

std::tuple<size_t, size_t> getDeviceMemoryInfo(bool const useUvm) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    aclrtGetMemInfo(ACL_HBM_MEM, &free_bytes, &total_bytes);
    return {total_bytes - free_bytes, free_bytes};
}

int getMultiProcessorCount(int device_id) {
    // Ascend NPU 的 AI Core 数量
    // 使用 aclrtGetDeviceCapability 查询
    return 0;
}

void setAscendGraphCaptureEnabled(bool enabled) {
    ascend_graph_capture_enabled = enabled;
}

bool isAscendGraphCaptureEnabled() {
    return ascend_graph_capture_enabled;
}

}  // namespace ascend
}  // namespace rtp_llm

#endif  // USING_ASCEND
```

### 0.10.5 新建 `rtp_llm/models_py/bindings/ascend/AscendRegister.cc`

```cpp
// Phase 0: 空 Ascend 注册文件
// Phase 5 时替换为实际的算子注册
```

### 0.10.6 新建 `rtp_llm/models_py/bindings/ascend/BUILD`

**参照范本**：`rtp_llm/models_py/bindings/rocm/BUILD`

```python
load("//:def.bzl", "copts", "ascend_copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_library(
    name = "ascend_types_hdr",
    hdrs = [
        "ascend_types_hdr.h",
    ],
    deps = select({
        "@//:using_ascend": [
            "@local_config_ascend//ascend:ascend_headers",
            "//rtp_llm/cpp/utils:core_utils",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ascend_host_utils",
    srcs = [
        "ascend_host_utils.cc",
    ],
    hdrs = [
        "ascend_host_utils.h",
        "ascend_types_hdr.h",
    ],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
    ] + select({
        "@//:using_ascend": [
            "@local_config_ascend//ascend:ascend_headers",
            "@local_config_ascend//ascend:ascend",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ascend",
    deps = [
        ":ascend_host_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ascend_bindings_register",
    srcs = [
        "AscendRegister.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:register_ops_hdr",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
```

---

## Step 0.11 批量修改 BUILD 文件添加 `select()` 分支

### 核心策略

> **初期在 `select()` 中为 Ascend 添加的分支全部指向 `"//conditions:default": []`（空列表），确保编译不会因缺少目标而失败。** 后续阶段逐步替换为实际的 Ascend 实现目标。

### 第一优先级（核心编译链路）- Phase 0 必须完成

#### 0.11.1 `rtp_llm/cpp/pybind/BUILD`（第 49-72 行）

**`th_compute_lib` 中的两个 `select()`** 需添加 Ascend 分支：

**select 1**（第 53-56 行）：
```python
    ] + select({
        "@//:using_cuda12": ["//rtp_llm/models_py/bindings/core:exec_ops_srcs"],
        "@//:using_ascend": ["//rtp_llm/models_py/bindings/core:exec_ops_srcs"],  # [新增]
        "//conditions:default": [],
    }),
```

**select 2**（第 64-68 行）：
```python
    ) + select({
        "@//:using_cuda12": ["//rtp_llm/models_py/bindings/cuda/ops:gpu_base"],
        "@//:using_rocm": ["//rtp_llm/models_py/bindings/core:exec_ctx_rocm"],
        "@//:using_ascend": [],  # [新增] Phase 0: 空 deps
        "//conditions:default": [],
    }) + no_block_copy_link_deps(),
```

**select 3**（`th_transformer_lib` 第 95-109 行）：
```python
    ] + select({
        "//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
            "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        ],
        "@//:using_rocm": [
            "//rtp_llm/models_py/bindings/core:exec_ctx_ops",
            "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
            "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        ],
        "@//:using_ascend": [  # [新增] Phase 0: 最小 deps
            "//rtp_llm/models_py/bindings/core:exec_ctx_ops",
            "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
        ],
        "//conditions:default": [],
    }),
```

#### 0.11.2 `rtp_llm/models_py/bindings/core/BUILD`

**`types` target**（第 17-33 行）：
```python
    ] + select({
        "@//:using_cuda": ["@local_config_cuda//cuda:cuda_headers",
                           "@local_config_cuda//cuda:cudart"],
        "@//:using_rocm": ["@local_config_rocm//rocm:rocm_headers",
                           "@local_config_rocm//rocm:rocm",
                           "//rtp_llm/models_py/bindings/rocm:rocm_types_hdr"],
        "@//:using_ascend": ["@local_config_ascend//ascend:ascend_headers"],  # [新增]
        "//conditions:default": [],
    }),
```

**`exec_ops_hdr` target**（第 93-96 行）：
```python
    ] + torch_deps() + select({
        "@//:using_rocm": ["@local_config_rocm//rocm:rocm_headers"],
        "@//:using_ascend": ["@local_config_ascend//ascend:ascend_headers"],  # [新增]
        "//conditions:default": [],
    }),
```

**`exec_ctx_state` target**（第 218-226 行）：
```python
    ] + torch_deps() + select({
        "@//:using_rocm": [
            "//rtp_llm/models_py/bindings/rocm/kernels/sampling:sampling",
            "//rtp_llm/models_py/bindings/rocm:rocm_host_utils",
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:rocm",
        ],
        "@//:using_ascend": [  # [新增]
            "//rtp_llm/models_py/bindings/ascend:ascend_host_utils",
            "@local_config_ascend//ascend:ascend_headers",
            "@local_config_ascend//ascend:ascend",
        ],
        "//conditions:default": [],
    }),
```

**`exec_ctx_ops` target**（第 263-276 行）：与 `exec_ctx_state` 相同模式添加 Ascend 分支。

#### 0.11.3 `rtp_llm/cpp/cache/BUILD`

**`cache_core` target**（第 164-177 行）：
```python
    ] + torch_deps() + select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        ],
        "@//:using_rocm": [
            "//rtp_llm/models_py/bindings/rocm:rocm_host_utils",
        ],
        "@//:using_ascend": [  # [新增]
            "//rtp_llm/models_py/bindings/ascend:ascend_host_utils",
        ],
        "//conditions:default": [],
    }) + select({
```

#### 0.11.4 `rtp_llm/models_py/bindings/common/kernels/BUILD`

**`any_cuda_deps` 变量**（第 5-20 行）：

添加 `ascend_deps` 变量并修改使用 `any_cuda_deps` 的 targets：

```python
any_cuda_deps = select({
    "@//:using_cuda": [
        "//rtp_llm/models_py/bindings/cuda/kernels:cuda_utils_common",
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    "@//:using_rocm": [
        "//rtp_llm/models_py/bindings/rocm/kernels:rocm_utils",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:hip",
        "//rtp_llm/models_py/bindings/rocm:rocm_types_hdr",
        "//rtp_llm/models_py/bindings/rocm:rocm_host_utils",
    ],
    "//conditions:default": [],
})

ascend_deps = select({
    "@//:using_ascend": [
        "//rtp_llm/models_py/bindings/ascend:ascend_host_utils",
        "@local_config_ascend//ascend:ascend_headers",
    ],
    "//conditions:default": [],
})
```

> **注意**：`common/kernels/` 中的 `.cu` 文件（activation、sampling、kv_cache 等）在 Ascend 编译时**不能被编译**（无 nvcc/hipcc）。这些 target 的 `srcs` 中包含 `.cu` 文件，需要通过 `select()` 条件化或使用 `glob()` 排除。
> 
> **Phase 0 策略**：
> 1. `ascend_deps` 仅声明 head-only 依赖（`ascend_host_utils`、`ascend_headers`）
> 2. 所有 `.cu` targets（`kernels_activation`、`kernels_sampling` 等）**在 Ascend 模式下不被依赖到**，通过上层 BUILD 的 `select()` 分支指向空列表绕过
> 3. `fuse_copy_util`（head-only 的 target）使用 `ascend_deps` 替代 `any_cuda_deps`
> 4. `kernels_cu` target 的 select 已包含 `@//:using_ascend: []`（空列表）

### 第二优先级（其他 BUILD 文件）- Phase 0 尽量完成

以下文件需逐一检查，为包含 `select()` 的位置添加 `@//:using_ascend` 分支。**通用模式**：

```python
# 在现有 select 字典中添加：
"@//:using_ascend": [],  # Phase 0: 空 deps
```

| # | 文件 | 关键 target | 预期修改 |
|---|------|------------|---------|
| 1 | `rtp_llm/cpp/models/BUILD` | 检查是否有 `select()` | 添加 ascend 分支 |
| 2 | `rtp_llm/cpp/normal_engine/BUILD` | 同上 | 同上 |
| 3 | `rtp_llm/cpp/embedding_engine/BUILD` | 同上 | 同上 |
| 4 | `rtp_llm/cpp/cuda_graph/BUILD` | 同上 | 同上 |
| 5 | `rtp_llm/cpp/utils/BUILD` | 同上 | 同上 |
| 6 | `rtp_llm/models_py/bindings/cuda/BUILD` | 同上 | 同上 |
| 7 | `rtp_llm/models_py/bindings/cuda/ops/BUILD` | 同上 | 同上 |
| 8 | `rtp_llm/models_py/bindings/common/BUILD` | 同上 | 同上 |
| 9 | `rtp_llm/models_py/BUILD` | 同上 | 同上 |
| 10 | `rtp_llm/models_py/standalone/BUILD` | 同上 | 同上 |
| 11 | `rtp_llm/models_py/distributed/BUILD` | 同上 | 同上 |

### 搜索 BUILD 文件中所有 select 块的命令

```bash
# 查找所有包含 select 的 BUILD 文件
grep -rn "select({" --include="BUILD" --include="*.bzl" | grep -E "(using_cuda|using_rocm|local_config_cuda|local_config_rocm)"
```

建议使用此命令逐一确认所有需要修改的位置。

---

## Step 0.12 阶段性验证计划

### 验证环境准备

```bash
# 确保 CANN 环境变量已设置
export ASCEND_TOOLKIT_PATH=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=$ASCEND_TOOLKIT_PATH/lib64:$LD_LIBRARY_PATH
export TF_NEED_ASCEND=1

# 验证 CANN 安装
ls $ASCEND_TOOLKIT_PATH/include/acl/acl.h
ls $ASCEND_TOOLKIT_PATH/lib64/libascendcl.so

# 生成 requirements lock 文件（在完成 Step 0.9.1 和 0.9.4 后执行）
bazel run //deps:requirements_ascend -- --upgrade
```

### 逐步验证步骤

| 步骤 | 验证命令 | 预期结果 | 失败排查 |
|------|---------|---------|---------|
| **Step 1** | `bazel query @local_config_ascend//...` | 能查询到 ascend 仓库的所有目标（ascend_headers, ascend, aclblas, hccl） | 检查 `ascend_configure.bzl` 是否正确检测到 CANN 安装路径 |
| **Step 2** | `bazel build --config=ascend //rtp_llm/models_py/bindings/ascend:ascend_host_utils` | Ascend 兼容层编译通过 | 检查 `acl/acl.h` 头文件路径是否正确 |
| **Step 3** | `bazel build --config=ascend //rtp_llm/models_py/bindings/core:types_hdr` | 核心头文件编译通过 | 检查 `Types.h` 中 `MEMORY_NPU` 枚举是否引入语法错误 |
| **Step 4** | `bazel build --config=ascend //:th_transformer_config` | 配置库编译通过 | 检查 `def.bzl` 中 `if_ascend` 是否正确导入 |
| **Step 5** | `bazel build --config=ascend //:rtp_compute_ops` | 计算 ops 库编译通过 | 此步大概率会遇到 CUDA 特定代码的编译错误，参见下方"常见编译错误处理" |
| **Step 6** | `bazel build --config=ascend //:th_transformer` | 最终产物编译通过 | 如果 Step 5 通过，此步通常也能通过 |

### 常见编译错误及处理

#### 错误 1：`#include <cuda_runtime.h>` 找不到

**原因**：Ascend 模式下 `USING_CUDA=0`，但部分 `.cc` 文件无条件包含了 CUDA 头文件。

**修复**：在对应 `.cc` 文件中添加 `#if USING_CUDA` 守卫：

```cpp
#if USING_CUDA
#include <cuda_runtime.h>
#endif
```

**需要排查的文件**（高概率出现此错误）：

- `rtp_llm/models_py/bindings/core/CudaOps.cc` — 核心文件，包含大量 CUDA 调用
- `rtp_llm/models_py/bindings/core/CudaSampleOp.cc`
- `rtp_llm/cpp/cache/BlockPool.cc` — 可能包含 `getDeviceMemoryInfo()` CUDA 实现
- `rtp_llm/cpp/cache/MemoryEvaluationHelper.cc`

#### 错误 2：`undefined reference to cudaXxx`

**原因**：`.cc` 文件中的 CUDA 函数调用在 Ascend 模式下无对应实现。

**修复**：添加 `#if USING_CUDA` / `#elif USING_ASCEND` 条件编译：

```cpp
#if USING_CUDA
    cudaStreamSynchronize(stream);
#elif USING_ASCEND
    aclrtSynchronizeStream(stream);
#endif
```

#### 错误 3：`nvcc not found`

**原因**：BUILD 文件中的 `.cu` 文件尝试用 nvcc 编译。

**修复**：确保这些 targets 在 Ascend select 分支中被排除或替换为 `.cc` 文件。

#### 错误 4：`torch_npu` pip 包安装失败

**原因**：`torch_npu` 需要匹配的 CANN 版本和 Python 版本。

**修复**：确认 `requirements_ascend.txt` 中的版本号与实际 CANN 版本匹配。

### 编译错误绕过策略

Phase 0 的核心目标是**让编译错误减少到可控范围内**。遇到难以解决的编译错误时，可以：

1. **在 `select()` 中添加 Ascend 分支指向空列表** — 跳过该 target
2. **使用 `#if USING_CUDA` 守卫包裹 CUDA 特定代码** — 最小化修改
3. **在 Ascend 分支使用 dummy/stub 实现** — 保持接口一致
4. **暂时从 `srcs` 中排除 `.cu` 文件** — 使用 `glob(exclude=[])` 模式

---

## 附录 A：完整新建文件清单

| # | 文件路径 | 说明 |
|---|---------|------|
| 1 | `3rdparty/gpus/ascend_configure.bzl` | CANN 自动配置 repository rule |
| 2 | `deps/requirements_ascend.txt` | Ascend pip 依赖声明 |
| 3 | `rtp_llm/models_py/bindings/ascend/BUILD` | Ascend Bazel 构建目标 |
| 4 | `rtp_llm/models_py/bindings/ascend/ascend_types_hdr.h` | Ascend 类型适配头文件 |
| 5 | `rtp_llm/models_py/bindings/ascend/ascend_host_utils.h` | NPU 设备属性/内存查询头文件 |
| 6 | `rtp_llm/models_py/bindings/ascend/ascend_host_utils.cc` | NPU 设备属性/内存查询实现 |
| 7 | `rtp_llm/models_py/bindings/ascend/AscendRegister.cc` | Ascend 注册空文件 |

**合计**：7 个新建文件（+ `deps/requirements_ascend.txt` 和 `deps/requirements_lock_ascend.txt` 为第 8、9 个）

---

## 附录 B：完整修改文件清单

| # | 文件路径 | 修改内容 | 优先级 |
|---|---------|---------|--------|
| 1 | `WORKSPACE` | 添加 `ascend_configure` + `pip_ascend_torch_install_deps` | P0 |
| 2 | `BUILD`（根目录） | 添加 `config_setting(name = "using_ascend")` | P0 |
| 3 | `def.bzl` | 添加 `if_ascend`、修改 `copts()`、添加 `ascend_copts()` | P0 |
| 4 | `.bazelrc` | 添加 `build:ascend` / `test:ascend` 配置段 | P0 |
| 5 | `arch_config/arch_select.bzl` | 添加 Ascend 分支到所有 `select()` | P0 |
| 6 | `bazel/device_defs.bzl` | 添加 Ascend 到 `device_test_envs()` / `device_impl_target()` | P0 |
| 7 | `deps/pip.bzl` | 添加 `pip_ascend_torch` | P0 |
| 8 | `deps/BUILD` | 添加 `requirements_ascend` | P0 |
| 9 | `deps/http.bzl` | 添加 `torch_ascend` http_archive | P0 |
| 10 | `rtp_llm/models_py/bindings/core/Types.h` | 添加 `MEMORY_NPU` | P0 |
| 11 | `rtp_llm/models_py/bindings/core/Types.cc` | 添加 Ascend 条件编译分支 | P0 |
| 12 | `rtp_llm/models_py/bindings/cuda/cuda_host_utils.h` | 添加 `#elif USING_ASCEND`（含注释，Ascend 下不编译该文件） | P0 |
| 13 | `rtp_llm/cpp/pybind/BUILD` | 添加 Ascend 到 3 个 `select()` | P0 |
| 14 | `arch_config/arch_select.bzl` | 修改 `flashinfer_deps()` 添加 Ascend 分支 | P0 |
| 15 | `deps/requirements_lock_ascend.txt` | 生成 lock 文件（运行 `bazel run //deps:requirements_ascend -- --upgrade`） | P0 |
| 16 | `rtp_llm/models_py/bindings/core/BUILD` | 添加 Ascend 到 `types`、`exec_ops_hdr`、`exec_ctx_state`、`exec_ctx_ops` 等 select | P0 |
| 17 | `rtp_llm/cpp/cache/BUILD` | 添加 Ascend 到 `cache_core` select | P1 |
| 18 | `rtp_llm/models_py/bindings/common/kernels/BUILD` | 添加 `ascend_deps` 变量 | P1 |
| 19-27 | 其他 ~9 个 BUILD 文件 | 添加 `@//:using_ascend` 空 select 分支 | P1 |

**合计**：~27 个修改文件

---

## 附录 C：核心 API 映射参考

### CANN (aclrt) ↔ CUDA API 映射表

| CUDA API | CANN (aclrt) API | 参数差异 | Phase 0 状态 |
|----------|------------------|---------|-------------|
| `cudaStream_t` | `aclrtStream` | 类型别名 | ✅ 已定义 |
| `cudaEvent_t` | `aclrtEvent` | 类型别名 | ✅ 已定义 |
| `cudaMalloc` | `aclrtMalloc` | aclrtMalloc 多 `aclrtMallocPolicy` 参数 | Phase 1 实现 |
| `cudaFree` | `aclrtFree` | 直接映射 | Phase 1 实现 |
| `cudaMemcpyAsync` | `aclrtMemcpyAsync` | 枚举值不同 (ACL_MEMCPY_*) | Phase 1 实现 |
| `cudaStreamSynchronize` | `aclrtSynchronizeStream` | 直接映射 | ✅ 已使用 |
| `cudaEventCreate` | `aclrtCreateEvent` | 直接映射 | Phase 1 实现 |
| `cudaEventRecord` | `aclrtRecordEvent` | 直接映射 | Phase 1 实现 |
| `cudaMemGetInfo` | `aclrtGetMemInfo(ACL_HBM_MEM, ...)` | 包装函数 | ✅ 已实现 |
| `cudaGetDevice` | `aclrtGetDevice` | 参数为 `int32_t*` | ✅ 已实现 |
| `cudaSetDevice` | `aclrtSetDevice` | 直接映射 | Phase 1 实现 |
| `cudaDeviceSynchronize` | `aclrtSynchronizeDevice` | 直接映射 | ✅ 已实现 |
| `cudaGetDeviceCount` | `aclrtGetDeviceCount` | 参数为 `uint32_t*` | ✅ 已实现 |

### CANN 库文件映射

| CUDA 库 | CANN 对应库 | 用途 |
|---------|------------|------|
| `libcublas.so` | `libaclblas.so` | 矩阵运算 (GEMM) |
| `libcuda.so` (runtime) | `libascendcl.so` | 设备运行时 |
| `libnccl.so` | `libhccl.so` | 集合通信 |
| N/A | `libaclnn.so` | NN 算子 (LayerNorm, Softmax 等) |

---

## 执行顺序建议

```
Week 1:
  Day 1-2: Step 0.1 → 0.5 (构建系统核心文件)
  Day 3:   Step 0.6 (类型枚举扩展)
  Day 4-5: Step 0.7 → 0.8 (设备选择中枢)

Week 2:
  Day 1:   Step 0.9 (pip/依赖管理)
  Day 2-3: Step 0.10 (Ascend 兼容层)
  Day 4:   Step 0.11 (批量 BUILD 文件修改)
  Day 5:   Step 0.12 (验证 + 修复编译错误)
```

### 关键里程碑

| 里程碑 | 完成标准 |
|--------|---------|
| M0.1 | `bazel query @local_config_ascend//...` 成功 |
| M0.2 | `bazel build --config=ascend //rtp_llm/models_py/bindings/ascend:ascend_host_utils` 成功 |
| M0.3 | `bazel build --config=ascend //:th_transformer_config` 成功 |
| M0.4 | `bazel build --config=ascend //:th_transformer` 成功（允许有未解决的非关键编译错误） |
