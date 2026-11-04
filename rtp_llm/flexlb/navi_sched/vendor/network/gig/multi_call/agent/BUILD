package(default_visibility = ["//rtp_llm/flexlb/navi_sched/vendor/network/gig:__subpackages__"])

agent_public_headers = [
    "GigAgent.h",
    "QueryInfo.h",
    "QueryInfoForMock.h",
]

cc_library(
    name = "agent_public_headers",
    hdrs = agent_public_headers,
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common:common_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:agent_proto_cc_proto_headers",
    ],
)

cc_library(
    name = "agent_headers",
    hdrs = glob(["*.h"], exclude = agent_public_headers),
    deps = [
        ":agent_public_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common:common_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util:util_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:agent_proto_cc_proto",
    ],
)

cc_library(
    name = "agent",
    srcs = glob([
        "*.cpp",
    ]),
    deps = [
        ":agent_headers",
    # ],
    # implementation_deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:agent_proto_cc_proto",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface:interface_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service:service_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream:stream_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config:config",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common:common",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util:util",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:log",
    ],
    alwayslink = True,
)


