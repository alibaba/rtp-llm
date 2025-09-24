package(default_visibility = ["//rtp_llm:__subpackages__"])

cc_binary(
    name = "kvcache_perf",
    srcs = [
        "KVCacheOptionBase.h",
        "KVCacheReadPerfTest.h",
        "KVCacheWritePerfTest.h",
        "KVCacheReadWriteMain.cc",
    ],
    deps = [
        "//rtp_llm/cpp/cache",
        "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        "@havenask//aios/autil:option_parser",
    ],
    linkstatic = False,
)

cc_binary(
    name = "storage_3fs_perf",
    srcs = [
        "DistStorage3FSTest.cc",
    ],
    deps = [
        "//rtp_llm/cpp/cache:storage_3fs",
    ],
    linkstatic = False,
)