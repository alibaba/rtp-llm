cc_library(
    name = "http_arpc_metric",
    srcs = [
        "HTTPRPCMetricReporterConfig.cpp",
        "HTTPRPCStats.cpp",
        "KMonitorHTTPRPCServerMetricReporter.cpp",
    ],
    hdrs = [
        "HTTPRPCMetricReporterConfig.h",
        "HTTPRPCStats.h",
        "KMonitorHTTPRPCServerMetricReporter.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/autil:env_util",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:json",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:log",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:time",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:kmonitor_client_cpp",
        "//rtp_llm/flexlb/navi_sched/vendor/network/anet/metric:kmonitor_anet_metric",
    ],
)

