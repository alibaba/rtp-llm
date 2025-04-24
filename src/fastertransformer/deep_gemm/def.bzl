load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
load("//bazel:arch_select.bzl", "torch_deps")
load("//:def.bzl", "copts")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch", "if_cuda")

preloaded_deps = [
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
    "@cutlass//:cutlass",
    ":deepgemm_hdrs",
] + torch_deps()

sm90_cuda_copts = copts() + cuda_default_copts_without_arch() + if_cuda(["-nvcc_options=objdir-as-tempdir"]) + [
    '--cuda-include-ptx=sm_90a', '--cuda-gpu-arch=sm_90a',
    '--compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi'
]

def sub_lib(name, srcs):
    native.cc_library(
        name = name + '_cu',
        hdrs = [
            "deep_gemm_template.h",
            "utils.h",
        ] + native.glob([
            "include/*.cuh"
        ]),
        srcs = [srcs],
        deps = preloaded_deps,
        copts = sm90_cuda_copts,
        visibility = ["//visibility:public"],
    )
    
    cc_shared_library(
        name = name,
        roots = [":" + name + "_cu"],
        preloaded_deps = preloaded_deps,
        visibility = ["//visibility:public"],
    )
    return name
