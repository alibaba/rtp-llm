load("@//bazel:arch_select.bzl", "cache_store_deps")

cache_store_deps()

package(default_visibility = ["//maga_transformer/cpp/disaggregate/cache_store:__subpackages__"])

cc_library(
    name = "cache_store",
    deps = [":cache_store_arch_select_impl", ":cache_store_base"],
    visibility=['//visibility:public'],
)

cc_library(
    name = "cache_store_base_impl",
    deps = [":cache_store_base"],
    srcs = ["Impl.cpp"],
    alwayslink = True,
)

cc_library(
    name = "arpc_dep",
    srcs = [],
    hdrs = [],
    deps = [
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
        "@havenask//aios/network/arpc:arpc",
        "@havenask//aios/network/arpc/arpc/metric:kmonitor_anet_metric",
    ],
    visibility=['//visibility:public'],
)
 
cc_library(
    name = "cm_subscribe",
    srcs = [],
    hdrs = [],
    deps = [
        "@havenask//aios/apps/facility/cm2/cm_sub:cm_sub",
    ],
    visibility=['//visibility:public'],
)

cc_library(
    name = "cache_store_base",
    srcs = [
        "RequestBlockBuffer.cpp",
        "RequestBlockBufferStore.cpp",
        "CacheStoreServiceImpl.cpp",
        "MessagerServer.cpp",
        "CacheLoadServiceClosure.cpp",
        "MessagerClient.cpp",
        "NormalCacheStore.cpp",
        "NoRdmaMemoryUtilImpl.cpp"
    ] + select({
        "//:using_cuda": ["MemoryUtil.cpp"],
        "//conditions:default": ["FakeMemoryUtil.cpp"]
    }),
    hdrs = [
        "NoRdmaMemoryUtilImpl.h",
        "CommonDefine.h",
        "Interface.h",
        "MemoryUtil.h",
        "RequestBlockBuffer.h",
        "RequestBlockBufferStore.h",
        "InitParams.h",
        "CacheStoreServiceImpl.h",
        "MessagerServer.h",
        "CacheLoadServiceClosure.h",
        "MessagerClient.h",
        "NormalCacheStore.h",
    ],
    deps = [
        "//maga_transformer/cpp/disaggregate/cache_store/proto:cache_store_service_cc_proto",
        "//maga_transformer/cpp/disaggregate/cache_store/metrics:metrics",
        "@havenask//aios/autil:log",
        "@havenask//aios/autil:thread",
        "@havenask//aios/autil:net",
        ":arpc_dep",
    ] + select({
        "//:using_cuda": [
            "@local_config_cuda//cuda:cuda",
            "@local_config_cuda//cuda:cudart",
        ],
        "//conditions:default": []
    }),
    visibility=['//visibility:public'],
)

cc_library(
    name = "block_buffer_util",
    srcs = [
        "BlockBufferUtil.cpp",
    ],
    hdrs = [
        "BlockBufferUtil.h",
    ],
    deps = [
        "//maga_transformer/cpp/disaggregate/cache_store:cache_store",
    ]
)
