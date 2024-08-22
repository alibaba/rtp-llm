load("//:def.bzl", "rpm_library", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

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
        "//:using_cuda": ["@local_config_cuda//cuda:cuda_headers",
                          "@local_config_cuda//cuda:cudart"],
        "//:using_rocm": ["@local_config_rocm//rocm:rocm_headers",
                          "@local_config_rocm//rocm:rocm",
                          "//src/fastertransformer/rocm:rocm_types_hdr"],
        "//:using_arm": ["//src/fastertransformer/devices/arm_impl/type_bf16:hie_bfloat16"],
        "//conditions:default": [],
    }),
    copts = copts(),
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
        "//src/fastertransformer/utils:assert_utils",
    ],
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "buffer_hdr",
    hdrs = [
        "Buffer.h",
        "QBuffer.h",
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
        "QBuffer.cc"
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
    ] + torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)