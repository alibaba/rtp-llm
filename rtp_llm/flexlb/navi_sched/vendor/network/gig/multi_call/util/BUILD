load("//rtp_llm/flexlb/navi_sched/vendor/bazel:defs.bzl", "if_arm")
package(default_visibility = ["//rtp_llm/flexlb/navi_sched/vendor/network/gig:__subpackages__"])

cc_library(
    name = "util_headers",
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common:common_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:kmonitor_client_cpp",
        "//rtp_llm/flexlb/navi_sched/vendor/network/arpc:arpc",
        "//rtp_llm/flexlb/navi_sched/vendor/network/anet:anet",
    ],
)

cc_library(
    name = "util",
    srcs = glob([
        "*.cpp",
    ]),
    deps = [
        ":util_headers",
        "@grpc//:grpc++",
    # ],
    # implementation_deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common:common",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface:closure_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:kmonitor_client_cpp",
        "//rtp_llm/flexlb/navi_sched/vendor/network/arpc:arpc",
    ],
    copts = if_arm([], ["-mrdrnd"]),
)

