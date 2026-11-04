package(default_visibility = ["//visibility:private"])
load("//rtp_llm/flexlb/navi_sched/vendor/bazel:defs.bzl", "if_arm")

cc_library(
    name = "anet",
    visibility = ["//visibility:public"],
    srcs = glob([
        "*.cpp",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/alog:alog",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:env_util",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:scope",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:thread",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:net",
    ],
    copts = ["-Wformat-truncation=0"] + if_arm(["-march=armv8.6-a+crc"], []),
    alwayslink = True,
)



cc_library(
    name = "anet_http_client",
    visibility = ["//visibility:public"],
    hdrs = glob([
        "tools/*.h",
    ]),
    srcs = glob([
        "tools/*.cpp",
    ]),
    deps = [
        ":anet",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:log",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:lock",
    ],
)
























