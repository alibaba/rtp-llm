load("//:def.bzl", "copts")
load("//rtp_llm/cpp/devices:device_defs.bzl", "device_impl_target", "device_test_envs")

package(default_visibility = ["//visibility:public"])

perftest_copts = copts()

cc_library(
    name = "perftest_layer_block_convertor",
    srcs = ["PerfTestLayerBlockConvertor.cc"],
    hdrs = ["PerfTestLayerBlockConvertor.h"],
    copts = perftest_copts,
    deps = [
        "//rtp_llm/cpp/disaggregate/transfer:basic_types",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/core:buffer",
        "@local_config_cuda//cuda:nccl", # dependency from devicebase
    ],
)

cc_library(
    name = "transfer_perftest_client_lib",
    srcs = ["TransferPerfTestClient.cc"],
    hdrs = ["TransferPerfTestClient.h"],
    copts = perftest_copts,
    deps = [
        ":perftest_layer_block_convertor",
        "//rtp_llm/cpp/disaggregate/transfer:transfer_client",
        "//rtp_llm/cpp/disaggregate/transfer:basic_types",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/utils:time_util",
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
    ],
)

cc_library(
    name = "transfer_perftest_server_lib",
    srcs = ["TransferPerfTestServer.cc"],
    hdrs = ["TransferPerfTestServer.h"],
    copts = perftest_copts,
    deps = [
        ":perftest_layer_block_convertor",
        "//rtp_llm/cpp/disaggregate/transfer:transfer_server",
        "//rtp_llm/cpp/disaggregate/transfer:basic_types",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/config:config_modules",
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
    ],
)

cc_binary(
    name = "transfer_perftest_server",
    srcs = ["TransferPerfTestServerMain.cc"],
    copts = perftest_copts,
    deps = [
        ":transfer_perftest_server_lib",
        "//rtp_llm/cpp/metrics:metrics",
        "@havenask//aios/autil:log",
        "@havenask//aios/autil:net",
    ] + device_impl_target(),
)

cc_binary(
    name = "transfer_perftest_client",
    srcs = ["TransferPerfTestClientMain.cc"],
    copts = perftest_copts,
    deps = [
        ":transfer_perftest_client_lib",
        "//rtp_llm/cpp/metrics:metrics",
        "@havenask//aios/autil:log",
        "@havenask//aios/autil:net",
    ] + device_impl_target(),
)
