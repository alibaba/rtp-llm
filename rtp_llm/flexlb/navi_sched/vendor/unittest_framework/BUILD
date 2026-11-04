package(default_visibility = ["//visibility:private"])

cc_library(
    name = "unittest_framework",
    srcs = glob([
        "unittest/dotest.cpp",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":unittest_lib",
        "//rtp_llm/flexlb/navi_sched/vendor/alog",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:backtrace",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:env_util",
        "@com_google_protobuf//:protobuf",
    ],
)


cc_library(
    name = "unittest_lib",
    hdrs = glob([
        "unittest/*.h",
    ]),
    include_prefix = "unittest",
    strip_include_prefix = "unittest",
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_googletest//:gtest",
    ],
)

