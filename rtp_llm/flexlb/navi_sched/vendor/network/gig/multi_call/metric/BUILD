package(default_visibility = ["//rtp_llm/flexlb/navi_sched/vendor/network/gig:__subpackages__"])

cc_library(
    name = "metric_headers",
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common:common_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util:util_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface:interface_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:kmonitor_client_cpp",
    ],
)

cc_library(
    name = "metric",
    srcs = glob([
        "*.cpp",
    ]),
    deps = [
        ":metric_headers",
    # ],
    # implementation_deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common:common",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util:util",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:kmonitor_client_cpp",
    ],
)


