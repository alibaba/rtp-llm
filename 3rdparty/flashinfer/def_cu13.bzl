load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
load("//bazel:arch_select.bzl", "torch_deps")

preloaded_deps = [
    ":flashinfer_hdrs",
    ":dispatch",
    "@cutlass//:cutlass",
    "@cutlass//:cutlass_utils",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
    "@local_config_cuda//cuda:cublas_headers",
    "@local_config_cuda//cuda:cublas",
    "@local_config_cuda//cuda:cublasLt",
] + torch_deps()


def sub_lib(name, deps, copts):
    native.cc_library(
        name = name + '_cu',
        srcs = native.glob([
            "csrc/*.h",
        ]) + deps,
        deps = [
            ":dispatch",
            ":flashinfer_hdrs",
        ],
        copts = copts,
        visibility = ["//visibility:public"],
    )
    cc_shared_library(
        name = name,
        roots = [":" + name + "_cu"],
        preloaded_deps = preloaded_deps,
        visibility = ["//visibility:public"],
    )
    return name
