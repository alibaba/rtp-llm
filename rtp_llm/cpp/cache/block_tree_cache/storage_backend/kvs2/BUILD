load("//:def.bzl", "copts")

cc_library(
    name = "kvs2",
    srcs = ["Kvs2StorageBackend.cc"],
    hdrs = ["Kvs2StorageBackend.h"],
    copts = copts(),
    tags = ["manual"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cache:cache_config",
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/cache/block_tree_cache:transfer_common",
        "//rtp_llm/cpp/cache/block_tree_cache:transfer_task_registry",
        "//rtp_llm/cpp/model_rpc:broadcast_manager",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "@kvs2_sdk//:client",
    ],
)
