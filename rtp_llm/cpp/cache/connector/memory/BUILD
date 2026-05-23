load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "no_block_copy_link_deps", "torch_deps")

cc_library(
    name = "disk_spill_types",
    hdrs = [
        "DiskSpillTypes.h",
    ],
    deps = [
        "//rtp_llm/cpp/cache:cache_group_type",
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "disk_spill_protocol",
    hdrs = [
        "DiskSpillProtocol.h",
    ],
    deps = [
        ":disk_spill_types",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "disk_spill_file_manager",
    srcs = [
        "DiskSpillFileManager.cc",
    ],
    hdrs = [
        "DiskSpillFileManager.h",
    ],
    deps = [
        ":disk_spill_types",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "disk_spill_io_worker",
    srcs = [
        "DiskSpillIoWorker.cc",
    ],
    hdrs = [
        "DiskSpillIoWorker.h",
    ],
    deps = [
        ":disk_spill_file_manager",
        ":disk_spill_types",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "disk_spill_commit_coordinator",
    srcs = [
        "DiskSpillCommitCoordinator.cc",
    ],
    hdrs = [
        "DiskSpillCommitCoordinator.h",
    ],
    deps = [
        ":disk_spill_block_cache",
        ":disk_spill_types",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "disk_spill_block_cache",
    srcs = [
        "DiskSpillBlockCache.cc",
    ],
    hdrs = [
        "DiskSpillBlockCache.h",
        "MemoryBlockCache.h",
    ],
    deps = [
        ":disk_spill_file_manager",
        ":disk_spill_io_worker",
        ":disk_spill_types",
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "//rtp_llm/cpp/cache:cache_group_type",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:lru_cache",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "memory_connector",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        ":disk_spill_block_cache",
        ":disk_spill_commit_coordinator",
        ":disk_spill_file_manager",
        ":disk_spill_io_worker",
        ":disk_spill_protocol",
        ":disk_spill_types",
        "//rtp_llm/cpp/cache/connector:connector_base",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/model_rpc:broadcast_manager",
        "//rtp_llm/cpp/cache:kv_cache_allocator_hdr",
        "//rtp_llm/cpp/utils:lru_cache",
        "//rtp_llm/cpp/utils:profiling_scope",
        "@havenask//aios/autil:json",
        "@havenask//aios/autil:lock_free",
    ] + torch_deps() + no_block_copy_link_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)
