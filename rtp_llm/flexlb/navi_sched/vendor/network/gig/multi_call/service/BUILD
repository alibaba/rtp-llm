package(default_visibility = ["//rtp_llm/flexlb/navi_sched/vendor/network/gig:__subpackages__"])

public_headers = [
    "RetryLimitChecker.h",
    "LatencyTimeWindow.h",
    "LatencyTimeSnapshot.h",
    "CallBack.h",
]

cc_library(
    name = "service_public_headers",
    hdrs = public_headers,
)

cc_library(
    name = "service_headers",
    hdrs = glob(["*.h"], exclude = public_headers),
    deps = [
        ":service_public_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config:config_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller:controller_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc:grpc_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface:interface_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric:metric_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat:new_heartbeat_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/subscribe:subscribe_headers",
    ],
)

cc_library(
    name = "service",
    srcs = glob(["*.cpp"]),
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/rpc:rpc",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat:new_heartbeat",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller:controller",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric:metric",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream:stream_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream:stream_inner_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:agent_proto_cc_proto",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config:config",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/subscribe:subscribe",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:log",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:string_helper",
        ":service_headers",
    ],
)


