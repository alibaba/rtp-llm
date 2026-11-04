package(default_visibility = ["//rtp_llm/flexlb/navi_sched/vendor/network/gig:__subpackages__"])

arpc_public_headers = [
    "CommonClosure.h",
    "ArpcClosure.h",
]

cc_library(
    name = "arpc_public_headers",
    hdrs = arpc_public_headers,
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/rpc:rpc_public_headers",
    ],
)

cc_library(
    name = "arpc_headers",
    hdrs = glob(["*.h"], exclude = arpc_public_headers),
    deps = [
        ":arpc_public_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/rpc:rpc_headers",
    ],
)

cc_library(
    name = "arpc",
    srcs = glob(["*.cpp"]),
    deps = [
        ":arpc_headers",
    # ],
    # implementation_deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface:interface_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service:service_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent:agent_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream:stream_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util:util",
        "//rtp_llm/flexlb/navi_sched/vendor/network/http_arpc:http_arpc",
        "//rtp_llm/flexlb/navi_sched/vendor/network/arpc:arpc",
    ],
)

