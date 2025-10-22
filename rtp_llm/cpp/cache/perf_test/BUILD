package(default_visibility = ["//rtp_llm:__subpackages__"])

cc_binary(
    name = "kvcache_perf",
    srcs = [
        "KVCacheOptionBase.h",
        "KVCacheReadPerfTest.h",
        "KVCacheReadWriteMain.cc",
        "KVCacheWritePerfTest.h",
    ],
    linkstatic = False,
    deps = [
        "//rtp_llm/cpp/cache2",
        "//rtp_llm/cpp/devices/cuda_impl",
        "@havenask//aios/autil:option_parser",
    ],
)

cc_binary(
    name = "storage_3fs_perf",
    srcs = [
        "DistStorage3FSTest.cc",
    ],
    linkstatic = False,
    deps = [
        "//rtp_llm/cpp/cache:storage_3fs",
    ],
)
