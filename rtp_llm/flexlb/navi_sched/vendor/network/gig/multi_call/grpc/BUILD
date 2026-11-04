package(default_visibility = ["//rtp_llm/flexlb/navi_sched/vendor/network/gig:__subpackages__"])

cc_library(
    name = "grpc_headers",
    hdrs = glob([
        "server/*.h",
        "client/*.h",
    ]),
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config:config_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/rpc:rpc_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent:agent_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface:interface_headers",
    ],
)

cc_library(
    name = "grpc",
    srcs = glob([
        "server/*.cpp",
        "client/*.cpp",
    ]),
    deps = [
        ":grpc_headers",
        "@grpc//:grpc++",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream:stream",
    # ],
    # implementation_deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream:stream_inner_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service:service_headers",
    ],
)


