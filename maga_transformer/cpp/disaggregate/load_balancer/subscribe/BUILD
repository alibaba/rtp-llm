package(default_visibility = ["//maga_transformer/cpp/disaggregate/load_balancer:__subpackages__"])

cc_library(
    name = "subscribe",
    srcs = [
        "TopoNode.cpp",
        "SubscribeServiceConfig.cpp",        
        "LocalSubscribeService.cpp",
        "SubscribeServiceManager.cpp",
    ],
    hdrs = [
        "TopoNode.h",
        "SubscribeServiceConfig.h",
        "SubscribeService.h",
        "LocalSubscribeService.h",
        "SubscribeServiceManager.h",
        "SubscribeServiceCreator.h"
    ],
    deps = [
        "@havenask//aios/autil:json",
        "//maga_transformer/cpp:utils",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "subscribe_service_impl",
    deps = [":subscribe"],
    srcs = ["SubscribeServiceCreator.cpp"],
    visibility = ["//visibility:public"],
    alwayslink = True,
)