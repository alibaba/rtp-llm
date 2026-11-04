package(default_visibility = ["//rtp_llm/flexlb/navi_sched/vendor/network/gig:__subpackages__"])

cc_library(
    name = "config_headers",
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common:common_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util:util_headers",
    ],
)

cc_library(
    name = "config",
    srcs = glob([
        "*.cpp",
    ]),
    deps = [
        ":config_headers",
    # ],
    # implementation_deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common:common",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util:util",

    ],
    alwayslink = True,
)


