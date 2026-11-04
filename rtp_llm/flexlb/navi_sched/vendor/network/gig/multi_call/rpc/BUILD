package(default_visibility = ["//rtp_llm/flexlb/navi_sched/vendor/network/gig:__subpackages__"])

rpc_public_headers = [
    "GigRpcServer.h",
    "GigClosure.h",
    "GigRpcController.h",
    "ServerBizTopoInfo.h",
]

cc_library(
    name = "rpc_public_headers",
    hdrs = rpc_public_headers,
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common:common_headers",
        "@grpc//:grpc++_public_hdrs",
    ],
)

cc_library(
    name = "rpc_headers",
    hdrs = glob(["*.h"], exclude = rpc_public_headers),
    deps = [
        ":rpc_public_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common:common",
        "@grpc//:grpc++_public_hdrs",
    ],
)

cc_library(
    name = "rpc",
    srcs = glob(["*.cpp"]),
    deps = [
        ":rpc_headers",
    # ],
    # implementation_deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface:interface_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service:service_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream:stream_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat:new_heartbeat_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc:grpc",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/arpc:arpc",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent:agent",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config:config",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util:util",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:log",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:net",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:lock_free",
        "//rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric:kmonitor_anet_metric",
        "//rtp_llm/flexlb/navi_sched/vendor/network/http_arpc:http_arpc",
    ],
)

