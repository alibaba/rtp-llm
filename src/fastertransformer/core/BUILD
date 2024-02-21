load("//:def.bzl", "rpm_library", "torch_deps", "copts", "cuda_copts",)

cc_library(
    name = "types_hdr",
    hdrs = [
        "Types.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "types",
    srcs = [
        "Types.cc",
    ],
    deps = [
        ":types_hdr",
        "@local_config_cuda//cuda:cuda",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "allocator",
    srcs = [
        "allocator.cc"
    ],
    hdrs = [
        "allocator.h"
    ],
    copts = copts(),
    deps = [
        ":types_hdr",
        "//src/fastertransformer/utils:logger"
    ],
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "buffer_hdr",
    hdrs = [
        "Buffer.h",
    ],
    deps = [
        ":types_hdr",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "buffer",
    srcs = [
        "Buffer.cc",
    ],
    deps = [
        ":buffer_hdr",
        "//src/fastertransformer/utils:logger"
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensor_hdr",
    hdrs = [
        "Tensor.h",
    ],
    deps = [
        ":types_hdr",
    ],
    visibility = ["//visibility:public"],
)


cc_library(
    name = "Tensor",
    srcs = glob([
        "Tensor.cc",
    ]),
    deps = [
        ":tensor_hdr",
        ":types",
        "//src/fastertransformer/cuda:cuda_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)
