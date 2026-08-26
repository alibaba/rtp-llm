load("//:def.bzl", "copts")

test_copts = [
    "-fno-access-control",
] + copts()

# planner 是纯函数：无 IO、无 GPU 依赖，因此该 target 不需要 device_impl_target()
# 也不需要 exec_properties gpu。这是把绝大部分正确性验证搬到 CPU 单测的关键收益。
cc_test(
    name = "transfer_plan_test",
    srcs = [
        "KVCacheTransferPlannerTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/plan:transfer_plan",
        "@com_google_googletest//:gtest",
    ],
)

# RouteCodec / P2PKeyUtil 只需要 proto，不需要 CacheTopology，因此单独一个 target
# 继续保持 CPU-only。
cc_test(
    name = "route_codec_test",
    srcs = [
        "RouteCodecTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p:p2p_key_util",
        "//rtp_llm/cpp/cache/connector/p2p/plan:transfer_plan",
        "//rtp_llm/cpp/cache/connector/p2p/plan:transfer_plan_glue",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "@com_google_googletest//:gtest",
    ],
)
