load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "no_block_copy_link_deps", "torch_deps")

cc_library(
    name = "memory_copy_fence",
    hdrs = ["MemoryCopyFence.h"],
    srcs = ["MemoryCopyFence.cc"],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "memory_copy_deadline",
    hdrs = ["MemoryCopyDeadline.h"],
    srcs = ["MemoryCopyDeadline.cc"],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "memory_copy_operation_id",
    hdrs = ["MemoryCopyOperationId.h"],
    srcs = ["MemoryCopyOperationId.cc"],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "memory_copy_task_guard",
    hdrs = ["MemoryCopyTaskGuard.h"],
    srcs = ["MemoryCopyTaskGuard.cc"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":memory_async_context",
        ":memory_copy_deadline",
        "//rtp_llm/cpp/model_rpc:remote_load_lease_retainer",
        "//rtp_llm/cpp/utils:core_utils",
    ],
)

cc_library(
    name = "memory_async_context",
    hdrs = ["MemoryAsyncContext.h"],
    srcs = ["MemoryAsyncContext.cc"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = ["//rtp_llm/cpp/cache/connector:connector_base"],
)

cc_library(
    name = "memory_connector",
    hdrs = glob([
        "*.h",
    ], exclude = [
        "MemoryAsyncContext.h",
        "MemoryCopyDeadline.h",
        "MemoryCopyFence.h",
        "MemoryCopyOperationId.h",
        "MemoryCopyTaskGuard.h",
    ]),
    srcs = glob([
        "*.cc",
    ], exclude = [
        "MemoryAsyncContext.cc",
        "MemoryCopyDeadline.cc",
        "MemoryCopyFence.cc",
        "MemoryCopyOperationId.cc",
        "MemoryCopyTaskGuard.cc",
    ]),
    deps = [
        ":memory_async_context",
        ":memory_copy_deadline",
        ":memory_copy_fence",
        ":memory_copy_operation_id",
        ":memory_copy_task_guard",
        "//rtp_llm/cpp/cache:block_pool",
        "//rtp_llm/cpp/cache/connector:connector_base",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/model_rpc:broadcast_manager",
        "//rtp_llm/cpp/model_rpc:remote_load_lease_retainer",
        "//rtp_llm/cpp/cache:kv_cache_allocator",
        "//rtp_llm/cpp/utils:lru_cache",
        "//rtp_llm/cpp/utils:profiling_scope",
        "@havenask//aios/autil:json",
        "@havenask//aios/autil:lock_free",
    ] + torch_deps() + no_block_copy_link_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)
