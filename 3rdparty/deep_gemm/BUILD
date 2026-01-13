load("@rules_python//python:defs.bzl", "py_library")
load("@//:def.bzl", "copts", "cuda_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts")
load("@//bazel:arch_select.bzl", "torch_deps")

package(default_visibility = ["//visibility:public"])

# DeepGEMM headers
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

# CUDA compilation settings
cuda_copts_extended = cuda_default_copts() + [
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr",
    "--use_fast_math",
    "-DWITH_CUDA",
    "-DCUDA_VERSION=12060",
    "-Wno-deprecated-declarations",
]

# Main C++ extension library
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
            # "-lcudart",
            "-lcublas",
            "-lnvrtc",
            # "-lcupti",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

# Alias for compatibility
alias(
    name = "build_deep_gemm_cpp",
    actual = ":deep_gemm_cpp.so",
    visibility = ["//visibility:public"],
)

genrule(
    name = "deep_gemm_so_in_pkg",
    srcs = [":deep_gemm_cpp.so"],
    outs = ["deep_gemm/deep_gemm_cpp.so"],
    cmd = "mkdir -p $(RULEDIR)/deep_gemm && cp $(location :deep_gemm_cpp.so) $@",
    visibility = ["//visibility:public"],
)

py_library(
    name = "deep_gemm_pkg",
    srcs = glob(["deep_gemm/**/*.py", "deep_gemm/*.py"]),
    data = [
        ] + glob(["deep_gemm/include/**/*"]) + [
        ":deep_gemm_so_in_pkg",
        ":deep_gemm_cpp.so",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
)

alias(
    name = "deep_gemm",
    actual = ":deep_gemm_pkg",
    visibility = ["//visibility:public"],
)

# All files for external access
filegroup(
    name = "all",
    srcs = glob([
        "**/*",
    ], exclude = [
        "**/*.o",
        "**/*.so",
        "**/*.a",
        "**/BUILD*",
    ]),
    visibility = ["//visibility:public"],
)
