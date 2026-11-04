package(default_visibility = ["//rtp_llm/flexlb/navi_sched/vendor/network/gig:__subpackages__"])

closure_headers = [
    "Closure.h",
]

cc_library(
    name = "closure_headers",
    hdrs = closure_headers,
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config:config_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common:common_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util:util_headers",
    ],
)

cc_library(
    name = "interface_headers",
    hdrs = glob([
        "*.h",
    ], exclude = closure_headers),
    deps = [
        ":closure_headers",
	"//rtp_llm/flexlb/navi_sched/vendor/network/gig:common_proto_cc_proto_headers",
	"//rtp_llm/flexlb/navi_sched/vendor/network/gig:jni_proto_cc_proto_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config:config_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common:common_headers",
    ],
)

cc_library(
    name = "interface",
    srcs = glob([
        "*.cpp",
    ]),
    deps = [
        ":interface_headers",
    # ],
    # implementation_deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:agent_proto_cc_proto",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service:service",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream:stream_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream:stream_inner_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common:common",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util:util",
    ],
    alwayslink = True,
)



