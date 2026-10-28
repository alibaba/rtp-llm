load("//bazel:defs.bzl", "if_rdma")

# 仅在 RDMA 构建中提供实现，公开默认 OFF 构建不解析闭源 BareX 依赖。
cc_library(
    name = "rdma_health_monitor",
    srcs = if_rdma([
        "RdmaDeviceHealthMetrics.cc",
        "RdmaDeviceHealthMonitor.cc",
    ]),
    hdrs = if_rdma([
        "RdmaDeviceHealth.h",
        "RdmaDeviceHealthMetrics.h",
        "RdmaDeviceHealthMonitor.h",
    ]),
    deps = if_rdma([
        "//rtp_llm/cpp/config:rdma_device_health_config",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/utils:core_utils",
        "@alibaba_rdma//aios/network/accl-barex:accl-barex",
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
    ]),
    visibility = ["//visibility:public"],
)
