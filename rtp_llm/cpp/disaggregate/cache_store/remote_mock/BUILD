package(default_visibility = ["//visibility:public"])

cc_library(
    name = "remote_mock_core",
    srcs = ["RemoteMockFormat.cpp", "RemoteMockState.cpp", "RemoteMockTimer.cpp"],
    hdrs = ["RemoteMockFormat.h", "RemoteMockState.h", "RemoteMockTimer.h"],
    deps = [
        "//rtp_llm/cpp/utils:crc32c",
        "@com_google_absl//absl/status:status",
        "@com_google_absl//absl/status:statusor",
    ],
    linkopts = ["-pthread"],
)

cc_library(
    name = "remote_mock_service",
    srcs = ["RemoteMockService.cpp"],
    hdrs = ["RemoteMockService.h"],
    deps = [
        ":remote_mock_core",
        "//rtp_llm/cpp/disaggregate/cache_store:tcp_rpc_transport",
        "//rtp_llm/cpp/disaggregate/cache_store/proto:remote_mock_control_cc_proto",
    ],
)

cc_binary(
    name = "remote_mock_server",
    srcs = ["RemoteMockServerMain.cpp"],
    deps = [":remote_mock_service"],
)
