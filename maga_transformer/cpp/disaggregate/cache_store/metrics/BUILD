package(default_visibility = ["//maga_transformer/cpp/disaggregate/cache_store:__subpackages__"])

cc_library(
    name = "metrics",
    srcs = [
        "CacheStoreMetricsCollector.cpp",
        "CacheStoreMetricsReporter.cpp",
    ],
    hdrs = [
        "CacheStoreMetricsCollector.h",
        "CacheStoreMetricsReporter.h",
    ],
    deps = [
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
        "@havenask//aios/autil:time",
    ],
)