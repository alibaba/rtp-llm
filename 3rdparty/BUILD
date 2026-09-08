load("//:def.bzl", "rpm_library", "copts", "cuda_copts")
load("@arch_config//:arch_select.bzl", "torch_deps")
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cuda_driver",
    hdrs = glob([
        "common/*.h",
    ]),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "@local_config_cuda//cuda:cuda_driver",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],  
)

cc_library(
    name = "cub",
    hdrs = glob([
        "cub/**/*.h",
        "cub/**/*.cuh",
    ]) + ["cub_compat.h"],
    srcs = glob([
        "cub/**/*.cc",
        "cub/**/*.cu",
    ]),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

# Standalone header-only compat shim for CUDA 13 CUB symbol removals.
# Included as `#include "3rdparty/cub_compat.h"` from .cu files that used to
# reference cub::Max / cub::Min / cub::Sum / cub::CountingInputIterator /
# cub::TransformInputIterator directly.  No-op on CUDA 12.
cc_library(
    name = "cub_compat",
    hdrs = ["cub_compat.h"],
    deps = ["@local_config_cuda//cuda:cuda_headers"],
    visibility = ["//visibility:public"],
)
