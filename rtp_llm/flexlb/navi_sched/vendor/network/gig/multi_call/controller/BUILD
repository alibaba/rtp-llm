package(default_visibility = ["//rtp_llm/flexlb/navi_sched/vendor/network/gig:__subpackages__"])

cc_library(
    name = "controller_headers",
    hdrs = glob([
        "*.h",
    ]),
    deps = [
    ],
)

cc_library(
    name = "controller",
    srcs = glob([
        "*.cpp",
    ]),
    deps = [
        ":controller_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config:config",
    # ],
    # implementation_deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:agent_proto_cc_proto",
    ],
)

