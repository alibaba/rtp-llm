load("@rules_python//python:defs.bzl", "py_library")
load("@//:def.bzl", "copts", "cuda_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts")
load("@//bazel:arch_select.bzl", "requirement", "torch_deps")

package(default_visibility = ["//visibility:public"])

requirement([
    "torch",
    "setuptools",
])

# DeepGEMM 头文件库
cc_library(
    name = "deep_gemm_hdrs",
    hdrs = glob([
        "deep_gemm/include/**/*.h",
        "deep_gemm/include/**/*.hpp", 
        "deep_gemm/include/**/*.cuh",
        "third-party/cutlass/include/**/*.h",
        "third-party/cutlass/include/**/*.hpp",
        "third-party/fmt/include/**/*.h",
        "third-party/pybind11/include/**/*.h",
        "csrc/**/*.hpp",
        "csrc/**/*.cuh",
    ]),
    includes = [
        "deep_gemm/include",
        "third-party/cutlass/include", 
        "third-party/fmt/include",
        "third-party/pybind11/include",
        "csrc",
        ".",
    ],
    deps = torch_deps() + [
        "@local_config_cuda//cuda:cuda_headers",
    ],
    copts = cuda_copts() + [
        "-DWITH_CUDA",
        "-DCUDA_VERSION=12060",
        "-Wno-deprecated-declarations",
        "-Wno-unused-function",
        "-Wno-reorder",
        "-Wno-unused-variable",
    ],
    visibility = ["//visibility:public"],
)

# CUDA编译设置
cuda_copts_extended = cuda_default_copts() + [
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr", 
    "--use_fast_math",
    "-DWITH_CUDA",
    "-DCUDA_VERSION=12060",
    "-Wno-deprecated-declarations",
]

# 主要的C++扩展库
cc_binary(
    name = "deep_gemm_cpp.so",
    srcs = [
        "csrc/python_api.cpp",
    ] + glob([
        "csrc/jit/*.hpp",
        "csrc/jit_kernels/**/*.hpp",
        "csrc/utils/*.hpp",
        "csrc/indexing/*.cu",
    ]),
    copts = cuda_copts_extended + [
        "-std=c++17",
        "-O3",
        "-fPIC",
        "-D_GLIBCXX_USE_CXX11_ABI=1",
    ],
    includes = [
        "deep_gemm/include",
        "third-party/cutlass/include",
        "third-party/fmt/include", 
        "third-party/pybind11/include",
        "csrc",
        ".",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":deep_gemm_hdrs",
    ] + torch_deps() + select({
        "@//:using_cuda": [
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "//conditions:default": [],
    }),
    linkopts = select({
        "@//:using_cuda": [
            "-lcuda",
            "-lcudart",
            "-lcublas",
            "-lnvrtc",
            "-lcupti",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

# 别名保持兼容性
alias(
    name = "build_deep_gemm_cpp", 
    actual = ":deep_gemm_cpp.so",
    visibility = ["//visibility:public"],
)

# 创建一个genrule来重组文件结构，包括符号链接到include目录
genrule(
    name = "prepare_deep_gemm_sources",
    srcs = [
        "deep_gemm/__init__.py",
        "deep_gemm/testing/__init__.py", 
        "deep_gemm/testing/bench.py",
        "deep_gemm/testing/numeric.py",
        "deep_gemm/utils/__init__.py",
        "deep_gemm/utils/layout.py", 
        "deep_gemm/utils/math.py",
        ":create_deep_gemm_init",
    ] + glob([
        "deep_gemm/include/**/*",
    ]),
    outs = [
        "__init__.py",
        "testing/__init__.py",
        "testing/bench.py", 
        "testing/numeric.py",
        "utils/__init__.py",
        "utils/layout.py",
        "utils/math.py",

    ],
    cmd = """
    # First prepare the modified __init__.py with path setup
    echo "# Auto-generated path setup for DeepGEMM" > $(location __init__.py)
    cat $(location :create_deep_gemm_init) >> $(location __init__.py)
    echo "" >> $(location __init__.py)
    cat $(location deep_gemm/__init__.py) >> $(location __init__.py)
    
    # 复制其他Python文件到正确位置
    cp $(location deep_gemm/testing/__init__.py) $(location testing/__init__.py)  
    cp $(location deep_gemm/testing/bench.py) $(location testing/bench.py)
    cp $(location deep_gemm/testing/numeric.py) $(location testing/numeric.py)
    cp $(location deep_gemm/utils/__init__.py) $(location utils/__init__.py)
    cp $(location deep_gemm/utils/layout.py) $(location utils/layout.py)
    cp $(location deep_gemm/utils/math.py) $(location utils/math.py)

    """,
)





# 创建一个Python初始化脚本来处理路径问题
genrule(
    name = "create_deep_gemm_init",
    srcs = [],
    outs = ["deep_gemm_init_wrapper.py"],
    cmd = """
cat > $(location deep_gemm_init_wrapper.py) << 'EOF'
import os
import sys

def setup_deep_gemm_paths():
    \"\"\"在运行时设置正确的DeepGEMM路径结构\"\"\"
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 检查并创建include符号链接
    include_target = os.path.join(current_dir, "include")
    include_source = os.path.join(current_dir, "deep_gemm", "include")
    
    if os.path.exists(include_source) and not os.path.exists(include_target):
        try:
            os.symlink(include_source, include_target)
        except (OSError, NotImplementedError):
            # 如果符号链接失败，尝试复制目录
            import shutil
            try:
                shutil.copytree(include_source, include_target)
            except Exception:
                pass

# 在导入时自动设置路径
setup_deep_gemm_paths()
EOF
    """,
)

# Python库 - 包含所有必需的文件和目录结构
py_library(
    name = "deep_gemm",
    srcs = [":prepare_deep_gemm_sources", ":create_deep_gemm_init"],
    data = [
        ":deep_gemm_cpp.so",
    ] + glob([
        "deep_gemm/include/**/*",
        "third-party/**/*", 
        "csrc/**/*.hpp",
        "csrc/**/*.cuh",
    ]) + select({
        "@//:using_cuda": [
            "@local_config_cuda//cuda:cuda_headers",
        ],
        "//conditions:default": [],
    }),
    imports = [
        ".",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "deep_gemm_files",
    srcs = glob([
        "deep_gemm/**/*",
        "third-party/**/*",
        "csrc/**/*",
    ], exclude = [
        "**/*.o",
        "**/*.so",
        "**/*.a",
        "**/BUILD*",
    ]),
    visibility = ["//visibility:public"],
)