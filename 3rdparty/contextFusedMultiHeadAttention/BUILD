load("//:def.bzl", "rpm_library", "copts", "cuda_copts",)
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch", "if_cuda")
load("//bazel:arch_select.bzl", "torch_deps")
load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
package(default_visibility = ["//visibility:public"])

## How to upgrade trt:
## generate from commit 88ced50ca7 in TensorRT-LLM for now and generate below:
## 1. change `reinterpret_cast<bert::Fused_multihead_attention_params_v2 &>(params)` into `convertKernelParmas2BertParams(params)` in cpp/kernels/fmha_v2/setup.py
## 2. run setup.py to get *.cu
## 3. use `USE_PPU` to cover cuda func declaration.
## 4. change `CudaDevice` to `TrtCudaDevice` and avoid ambigous definition
cc_library(
    name = "trt_fmha_common_define_header",
    hdrs = glob([
        "multiHeadAttentionCommon.h"
    ]),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "fmha_v2_header",
    hdrs = glob([
        "*.h",
    ], exclude = ["multiHeadAttentionCommon.h"]),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        ":trt_fmha_common_define_header",
    ],
)

cc_library(
    name = "fmha_v2_impl_src_header",
    hdrs = glob([
        "src/**/*.h",
    ]),
    deps = [
        ":fmha_v2_header",
        "//rtp_llm/cpp/kernels:attn_utils",
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "//rtp_llm/cpp/utils:math_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
    strip_include_prefix = "src",
)

base_cuda_copts = copts() + cuda_default_copts_without_arch() + if_cuda(["-nvcc_options=objdir-as-tempdir"])

sm80_cuda_copts = [
    '--cuda-include-ptx=sm_80', '--cuda-gpu-arch=sm_80',
]

sm86_cuda_copts = [
    '--cuda-include-ptx=sm_86', '--cuda-gpu-arch=sm_86',
]

sm89_cuda_copts = [
    '--cuda-include-ptx=sm_89', '--cuda-gpu-arch=sm_89',
]

sm90_cuda_copts = [
    '--cuda-include-ptx=sm_90a', '--cuda-gpu-arch=sm_90a',
]

cc_library(
    name = "context_attention_kernels_sm80",
    srcs = glob([
        "fmha_v2_cu/*sm80.cu",
    ]),
    deps = [
        ":fmha_v2_impl_src_header",
    ],
    copts = sm80_cuda_copts + base_cuda_copts,
    visibility = ["//visibility:public"],
    alwayslink = True
)

cc_library(
    name = "context_attention_kernels_sm86",
    srcs = glob([
        "fmha_v2_cu/*sm86.cu",
    ]),
    deps = [
        ":fmha_v2_impl_src_header",
    ],
    copts = sm86_cuda_copts + base_cuda_copts,
    visibility = ["//visibility:public"],
    alwayslink = True
)

cc_library(
    name = "context_attention_kernels_sm89",
    srcs = glob([
        "fmha_v2_cu/*sm89.cu",
    ]),
    deps = [
        ":fmha_v2_impl_src_header",
    ],
    copts = sm89_cuda_copts + base_cuda_copts,
    visibility = ["//visibility:public"],
    alwayslink = True
)

cc_library(
    name = "context_attention_kernels_sm90",
    srcs = glob([
        "fmha_v2_cu/*sm90.cu",
    ]),
    deps = [
        ":fmha_v2_impl_src_header",
    ],
    copts = sm90_cuda_copts + base_cuda_copts,
    visibility = ["//visibility:public"],
    alwayslink = True
)

cc_library(
    name = "fmha_v2_cubin",
    hdrs = glob([
        "cubin/fmha_cubin.h"
    ]),
    srcs = glob([
        "cubin/*.cpp",
    ]),
    deps = [
        ":fmha_v2_impl_src_header",
    ],
    copts = copts() + ["-DEXCLUDE_SM_100=1", "-DEXCLUDE_SM_120=1"],
    visibility = ["//visibility:public"],
    alwayslink = True,
)

cc_library(
    name = "trt_fmha_impl",
    srcs = glob([
        "*.cpp",
    ]),
    deps = [
        ":fmha_v2_cubin",
        "//3rdparty:cuda_driver",
    ] + select({
        "@//:using_cuda12": [
            ":context_attention_kernels_sm80",
            ":context_attention_kernels_sm86",
            ":context_attention_kernels_sm89",
            ":context_attention_kernels_sm90",
        ],
        "//conditions:default": [],
    }),
    copts = cuda_copts() + ["-DEXCLUDE_SM_100=1", "-DEXCLUDE_SM_120=1"],
    visibility = ["//visibility:public"],
)
