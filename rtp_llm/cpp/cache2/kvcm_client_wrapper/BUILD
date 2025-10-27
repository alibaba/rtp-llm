config_setting(
    name = "kvcm_internal",
    values = {"copt": "-DKVCM_INTERNAL=1"},
)

cc_library(
    name = "kvcm_client_wrapper",
    srcs = [
        "KVCMClientWrapper.cc",
        "KVCMClientWrapperConfig.cc",
    ],
    hdrs = [
        "KVCMClientWrapper.h",
        "KVCMClientWrapperConfig.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":kvcm_subscriber",
        "//3rdparty/kv_cache_manager:kv_cache_manager_client",
        "@havenask//aios/autil:json",
    ],
)

cc_library(
    name = "kvcm_subscriber",
    srcs = [
        "VIPServerSubscriber.cc",
    ],
    hdrs = [
        "KVCMDirectSubscriber.h",
        "KVCMSubscriber.h",
        "VIPServerSubscriber.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
        "@havenask//aios/autil:env_util",
    ] + select({
        ":kvcm_internal": [
            "@vipserver",
        ],
        "//conditions:default": [],
    }),
)
