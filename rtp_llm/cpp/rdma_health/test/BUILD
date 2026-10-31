load("//bazel:defs.bzl", "if_rdma")

# 注入 probe / clock / abort 的逻辑单测不需要 RDMA 网卡或 GPU。
# 测试链接闭源 BareX，仅在 RDMA ON 构建中运行；OFF 时明确标记不兼容。
cc_test(
    name = "rdma_device_health_monitor_test",
    srcs = if_rdma(["RdmaDeviceHealthMonitorTest.cc"]),
    target_compatible_with = if_rdma([], ["@platforms//:incompatible"]),
    copts = ["-fno-access-control"],
    deps = if_rdma([
        "//rtp_llm/cpp/rdma_health:rdma_health_monitor",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ]),
)

# BareX probe 单测的 CI suite 留在内源，公开测试包不声明闭源测试目标。
