load("//:def.bzl", "copts")

cc_library(
    name = "transfer_plan",
    srcs = [
        "KVCacheTransferPlanner.cc",
    ],
    hdrs = [
        "KVCacheTransferPlanner.h",
        "ShardLayout.h",
        "TransferPlan.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cache:cache_group_type",
        "//rtp_llm/cpp/cache:cp_slot_mapper",
        "//rtp_llm/cpp/cache:kv_cache_specs",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/utils:error_code",
    ],
)

# 与真实 CacheTopology / proto 的接缝。单独一个 target，让 transfer_plan 本身保持
# 无拓扑、无 proto 依赖，从而 planner 单测不必链接这些。
cc_library(
    name = "transfer_plan_glue",
    hdrs = [
        "RouteCodec.h",
        "ShardLayoutFactory.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":transfer_plan",
        "//rtp_llm/cpp/cache:cache_config",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
    ],
)
