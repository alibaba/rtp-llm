load("//:def.bzl", "rpm_library", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "types_hdr",
    hdrs = [
        "Types.h",
        "Dispatch.h",
    ],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
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
        "@//:using_cuda": ["@local_config_cuda//cuda:cuda_headers",
                           "@local_config_cuda//cuda:cudart"],
        "@//:using_rocm": ["@local_config_rocm//rocm:rocm_headers",
                           "@local_config_rocm//rocm:rocm",
                           "//rtp_llm/cpp/rocm:rocm_types_hdr"],
        "@//:using_arm": ["//rtp_llm/cpp/devices/arm_impl/type_bf16:hie_bfloat16"],
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
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:stack_trace",
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
        "//rtp_llm/cpp/cuda:cuda_host_utils",
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
        "//rtp_llm/cpp/utils:core_utils",
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
        "//rtp_llm/cpp/utils:core_utils",
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
        "//rtp_llm/cpp/utils:core_utils"
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
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:stack_trace"
    ] + torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "event",
    hdrs = [
        "Event.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "torch_event",
    hdrs = [
        "torch_utils/TorchEvent.h",
    ],
    deps = [
        ":event",
    ] + torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "torch_cuda_allocator",
    srcs = [
        "torch_utils/torch_cuda_allocator.cc"
    ],
    hdrs = [
        "torch_utils/torch_cuda_allocator.h"
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/core:allocator",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/config:model_config",
    ] + torch_deps(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)
