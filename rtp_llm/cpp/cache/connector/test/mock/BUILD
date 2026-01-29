load("//:def.bzl", "copts")

test_copts = [
    "-fno-access-control",
] + copts()

cc_library(
    name = "connector_mock_files_lib",
    hdrs = [
        "MockAsyncContext.h",
        "MockKVCacheConnector.h",
        "MockKVCacheConnectorCoordinator.h",
        "MockKVCacheConnectorReadWriteContext.h",
    ],
    deps = [
        "//rtp_llm/cpp/cache/connector:connector",
    ],
    copts = test_copts,
    visibility = ["//visibility:public"],
)