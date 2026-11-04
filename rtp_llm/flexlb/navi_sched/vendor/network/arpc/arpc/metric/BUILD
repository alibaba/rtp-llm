cc_library(
    name = "arpc_metric",
    srcs = [
        "MetricReporterConfig.cpp",
        "ClientRPCStats.cpp",
        "ClientMetricReporter.cpp",
        "ServerRPCStats.cpp",
        "ServerMetricReporter.cpp",
    ],
    hdrs = [
        "MetricReporterConfig.h",
        "ClientRPCStats.h",
        "ClientMetricReporter.h",
        "ServerRPCStats.h",
        "ServerMetricReporter.h",
    ],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/autil:log",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:json",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:time",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:env_util",
        "//rtp_llm/flexlb/navi_sched/vendor/network/arpc:proto_cc_proto",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kmonitor_arpc_metric",
    srcs = [
        "KMonitorClientMetricReporter.cpp",
        "KMonitorServerMetricReporter.cpp",
    ],
    hdrs = [
        "KMonitorClientMetricReporter.h",
        "KMonitorServerMetricReporter.h",
    ],
    deps = [
        ":arpc_metric",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:kmonitor_client_cpp",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kmonitor_anet_metric",
    srcs = [
        "KMonitorANetClientMetricReporter.cpp",
        "KMonitorANetServerMetricReporter.cpp",
    ],
    hdrs = [
        "KMonitorANetMetricReporterConfig.h",
        "KMonitorANetClientMetricReporter.h",
        "KMonitorANetServerMetricReporter.h",
    ],
    deps = [
        ":kmonitor_arpc_metric",
        "//rtp_llm/flexlb/navi_sched/vendor/network/anet/metric:kmonitor_anet_metric",
    ],
    visibility = ["//visibility:public"],
)


