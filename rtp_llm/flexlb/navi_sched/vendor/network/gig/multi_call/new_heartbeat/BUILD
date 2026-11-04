package(default_visibility = ["//rtp_llm/flexlb/navi_sched/vendor/network/gig:__subpackages__"])

cc_library(
    name = "new_heartbeat_headers",
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:new_heartbeat_proto_cc_proto_headers",
    ],
)

cc_library(
    name = "new_heartbeat",
    srcs = glob([
        "*.cpp",
    ]),
    deps = [
        ":new_heartbeat_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface:interface_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service:service_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/rpc:rpc_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common:common_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream:stream",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:new_heartbeat_proto_cc_proto_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:new_heartbeat_proto_cc_proto",
    ],
)

