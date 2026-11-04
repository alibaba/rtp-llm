cc_library(
    name = "anet_metric",
    srcs = [
        "ANetMetricReporter.cpp",
    ],
    hdrs = [
        "ANetMetricReporterConfig.h",
        "ANetMetricReporter.h",
    ],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/anet:anet",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:env_util",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:json",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:thread",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kmonitor_anet_metric",
    srcs = [
        "KMonitorANetMetricReporter.cpp",
    ],
    hdrs = [
        "KMonitorANetMetricReporter.h",
    ],
    deps = [
        ":anet_metric",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:kmonitor_client_cpp",
    ],
    visibility = ["//visibility:public"],
)

