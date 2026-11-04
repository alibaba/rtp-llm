package(default_visibility = [
    "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:__subpackages__"
])

load('//rtp_llm/flexlb/navi_sched/vendor/bazel:defs.bzl', 'cc_proto')

cc_proto(
    name = "kmonitor_proto",
    srcs = glob([
        "*.proto",
    ]),
    import_prefix = "kmonitor/protocol",
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/arpc:proto",
    ],
)

cc_library(
    name = "kmonitor_proto_headers",
    hdrs = glob([
        "*.h",
    ]),
    include_prefix = "kmonitor/protocol",
    deps = [
        ":kmonitor_proto_cc_proto_headers",
    ],
)
