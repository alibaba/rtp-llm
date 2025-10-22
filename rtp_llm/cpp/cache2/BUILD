cc_library(
    name = "cache2",
    srcs = [
        "CacheManagerFactory.cc",
        "ProxyCacheManager.cc",
        "ProxyDistKvCache.cc",
        "ProxyKvCachePlanner.cc",
    ],
    hdrs = [
        "CacheManagerFactory.h",
        "ProxyCacheManager.h",
        "ProxyDistKvCache.h",
        "ProxyKvCachePlanner.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cache",
        "//rtp_llm/cpp/cache2/kv_cache_connector",
    ],
)
