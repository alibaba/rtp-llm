load("//:def.bzl", "copts")

test_copts = [
    "-fno-access-control",
] + copts()

cc_library(
    name = "mock_allocator_lib",
    hdrs = [
        "MockKVCacheAllocator.h",
    ],
    deps = [
        "//rtp_llm/cpp/cache:cache_core",
    ],
    copts = test_copts,
    visibility = ["//visibility:public"],
)