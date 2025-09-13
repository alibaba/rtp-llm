package(default_visibility = ["//rtp_llm/cpp/disaggregate/load_balancer:__subpackages__"])

cc_library(
    name = "subscribe",
    srcs = [
        "TopoNode.cpp",
        "SubscribeServiceConfig.cpp",
        "LocalSubscribeService.cpp",
        "SubscribeServiceManager.cpp",
        "NacosSubscribeService.cpp",
        "DomainSubscribeService.cpp",
    ],
    hdrs = [
        "TopoNode.h",
        "SubscribeServiceConfig.h",
        "SubscribeService.h",
        "LocalSubscribeService.h",
        "SubscribeServiceManager.h",
        "SubscribeServiceCreator.h",
        "NacosSubscribeService.h",
        "DomainSubscribeService.h",
    ],
    deps = [
        "@nacos_sdk_cpp//:nacos_sdk",
        "@havenask//aios/autil:json",
        "//rtp_llm/cpp/utils:core_utils",
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