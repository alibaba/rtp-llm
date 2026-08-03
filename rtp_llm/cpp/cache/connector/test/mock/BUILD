load("//:def.bzl", "copts")

test_copts = [
    "-fno-access-control",
] + copts()

cc_library(
    name = "connector_mock_files_lib",
    hdrs = [
        "MockAsyncContext.h",
        "MockKVCacheConnector.h",
        "MockKVCacheConnectorReadWriteContext.h",
    ],
    deps = [
        "//rtp_llm/cpp/cache:cache_async_context",
        "//rtp_llm/cpp/cache/connector:connector_base",
        "//rtp_llm/cpp/cache/connector:connector_context",
    ],
    copts = test_copts,
    visibility = ["//visibility:public"],
)
