load("//:def.bzl", "copts")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch")

cc_library(
    name = "reduce_gate",
    srcs = ["reduce_gate.cu"],
    hdrs = ["reduce_gate.h"],
    copts = copts() + cuda_default_copts_without_arch() + [
        "--cuda-gpu-arch=sm_120a",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-nvcc_options=ftz=false",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + select({
        "//:using_cuda13_x86": ["@pip_gpu_cuda13_torch_nvidia_nccl_cu13//:nccl_device"],
        "//:using_cuda13_arm": ["@pip_cuda13_arm_torch_nvidia_nccl_cu13//:nccl_device"],
        "//conditions:default": [],
    }),
    target_compatible_with = select({
        "//:using_cuda13_x86": [],
        "//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//rtp_llm/models_py/bindings/cuda:__pkg__"],
)
