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
    ] + select({
        "//:using_cuda": ["@local_config_cuda//cuda:cuda"],
        "//conditions:default": [],
    }),
    copts = cuda_copts(),
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
        ":types",
        "//src/fastertransformer/utils:logger",
        "@havenask//aios/autil:stack_tracer",
    ],
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "memory_tracker",
    hdrs = [
        "MemoryTracker.h",
        "TrackerAllocator.h",
    ],
    srcs = [
        "MemoryTracker.cc",
        "TrackerAllocator.cc",
    ],
    copts = copts(),
    deps = [
        ":allocator",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cpu_allocator",
    srcs = [
        "cpu_allocator.cc"
    ],
    hdrs = [
        "cpu_allocator.h"
    ],
    copts = copts(),
    deps = [
        ":allocator",
    ],
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "buffer_hdr",
    hdrs = [
        "Buffer.h",
        "BufferHelper.h"
    ],
    deps = [
        ":types_hdr",
        "//src/fastertransformer/utils:assert_utils",
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
    name = "buffer_torch",
    hdrs = [
        "torch_utils/BufferTorchUtils.h",
    ],
    deps = [
        ":buffer",
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
        ":types",
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
