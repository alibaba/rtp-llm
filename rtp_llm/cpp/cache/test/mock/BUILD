load("//:def.bzl", "copts")

test_copts = [
    "-fno-access-control",
] + copts()

cc_library(
    name = "mock_coordinator_cache_manager_lib",
    hdrs = [
        "MockCoordinatorCacheManager.h",
    ],
    deps = [
        "//rtp_llm/cpp/cache:cache_core",
    ],
    copts = test_copts,
    visibility = ["//visibility:public"],
)
