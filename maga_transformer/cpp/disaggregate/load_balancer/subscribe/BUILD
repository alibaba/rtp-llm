package(default_visibility = ["//maga_transformer/cpp/disaggregate/load_balancer:__subpackages__"])

cc_library(
    name = "subscribe",
    srcs = [
        "TopoNode.cpp",
        "SubscribeServiceConfig.cpp",
        "SubscribeService.cpp",
        "CM2SubscribeService.cpp",
        "LocalSubscribeService.cpp",
        "SubscribeServiceManager.cpp",
    ],
    hdrs = [
        "TopoNode.h",
        "SubscribeServiceConfig.h",
        "SubscribeService.h",
        "CM2SubscribeService.h",
        "LocalSubscribeService.h",
        "SubscribeServiceManager.h",
    ],
    deps = [
        "@havenask//aios/autil:json",
        "@havenask//aios/apps/facility/cm2/cm_sub:cm_sub"
    ],
    
)