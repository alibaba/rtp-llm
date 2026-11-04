package(default_visibility = ["//rtp_llm/flexlb/navi_sched/vendor/network/gig:__subpackages__"])

cc_library(
    name = "vip_subscription_client",
    hdrs = ["VipSubscriptionClient.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "subscribe_headers",
    hdrs = glob([
        "*.h",
    ], exclude = ["VipSubscriptionClient.h"]),
    deps = [
        ":vip_subscription_client",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config:config_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric:metric_headers",
    ],
)

cc_test(
    name = "vip_subscription_test",
    srcs = ["//rtp_llm/flexlb/navi_sched:test/VipSubscriptionTest.cpp"],
    deps = [":subscribe"],
)

cc_library(
    name = "subscribe",
    srcs = [
        "SubscribeServiceFactory.cpp",
        "SubscribeServiceManager.cpp",
        "VIPSubscribeService.cpp",
    ],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config:config",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric:metric",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:log",
        "@grpc//:grpc++",
        ":subscribe_headers",
    ],
)


