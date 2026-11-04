load("//rtp_llm/flexlb/navi_sched/vendor/bazel:defs.bzl", "gen_conf_lib")

cc_library(
    name = "kmonitor_client_cpp",
    srcs = glob(
        [
            "cpp_client/src/kmonitor/client/*.cpp",
            "cpp_client/src/kmonitor/client/common/*.cpp",
            "cpp_client/src/kmonitor/client/core/*.cpp",
            "cpp_client/src/kmonitor/client/metric/*.cpp",
            "cpp_client/src/kmonitor/client/net/*.cpp",
            "cpp_client/src/kmonitor/client/net/thrift/*.cpp",
            "cpp_client/src/kmonitor/client/sink/*.cpp",
        ],
        exclude = ["cpp_client/src/kmonitor/client/MetricsReporterR.cpp"],
    ),
    hdrs = glob(
        [
            "cpp_client/src/kmonitor/client/*.h",
            "cpp_client/src/kmonitor/client/sink/*.h",
            "cpp_client/src/kmonitor/client/core/*.h",
            "cpp_client/src/kmonitor/client/metric/*.h",
            "cpp_client/src/kmonitor/client/net/*.h",
            "cpp_client/src/kmonitor/client/net/thrift/*.h",
            "cpp_client/src/kmonitor/client/common/*.h",
        ],
        exclude = ["cpp_client/src/kmonitor/client/MetricsReporterR.h"],
    ),
    copts = ["-fno-strict-aliasing"],
    implementation_deps = ["//rtp_llm/flexlb/navi_sched/vendor/autil:regex"],
    strip_include_prefix = "cpp_client/src",
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/alog",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:data_buffer",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:env_util",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:json",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:metric",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:thread",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor/cpp_client/src/kmonitor/client/protocol:kmonitor_proto_cc_proto",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor/cpp_client/src/kmonitor/client/protocol:kmonitor_proto_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/anet:anet_http_client",
        "@com_google_googletest//:gtest_prod",
    ],
    alwayslink = 1,
)

cc_library(
    name = "metrics_report_navi_resource",
    srcs = ["cpp_client/src/kmonitor/client/MetricsReporterR.cpp"],
    hdrs = ["cpp_client/src/kmonitor/client/MetricsReporterR.h"],
    strip_include_prefix = "cpp_client/src",
    visibility = ["//visibility:public"],
    deps = [
        ":kmonitor_client_cpp",
        "//rtp_llm/flexlb/navi_sched/vendor/navi",
    ],
    alwayslink = 1,
)



gen_conf_lib(
    name = "kmonitor_cpp_test_h",
    srcs = ["cpp_client/src/kmonitor/test/test.h.in"],
    outs = ["cpp_client/src/kmonitor/test/test.h"],
    args = {"@TEST_DATA_PATH@": "rtp_llm/flexlb/navi_sched/vendor/kmonitor/cpp_client/testdata"},
    include_prefix = "cpp_client/src",
    strip_include_prefix = "cpp_client/src",
)

cc_library(
    name = "kmonitor_cpp_test_helper",
    hdrs = glob(["cpp_client/src/kmonitor/client/test/*.h"]),
    strip_include_prefix = "cpp_client/src",
    visibility = ["//visibility:public"],
    deps = [
        ":kmonitor_cpp_test_h",
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework:unittest_lib",
    ],
)



cc_library(
    name = "kmonitor_cpp_adapter",
    srcs = glob(["adapter/kmonitor_adapter/*.cpp"]),
    hdrs = glob(["adapter/kmonitor_adapter/*.h"]),
    include_prefix = "kmonitor_adapter",
    strip_include_prefix = "adapter/kmonitor_adapter",
    visibility = ["//visibility:public"],
    deps = [
        ":kmonitor_client_cpp",
        "//rtp_llm/flexlb/navi_sched/vendor/alog",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:env_util",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:scope",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:thread",
    ],
    alwayslink = 1,
)






