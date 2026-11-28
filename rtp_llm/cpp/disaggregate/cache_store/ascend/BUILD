package(default_visibility = ["//rtp_llm/cpp/disaggregate/cache_store:__subpackages__"])

cc_library(
    name = "hixl_memory_util",
    srcs = ["HixlMemoryUtil.cc"],
    hdrs = ["HixlMemoryUtil.h"],
    deps = [
        ":hixl_protocol",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/cpp/utils:core_utils",
    ] + select({
        "@//:using_ascend": ["@local_config_ascend//ascend:hixl"],
        "//conditions:default": [],
    }),
    target_compatible_with = select({
        "@//:using_ascend": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

cc_library(
    name = "hixl_manifest_service",
    srcs = ["HixlManifestService.cc"],
    hdrs = ["HixlManifestService.h"],
    deps = [
        ":hixl_protocol",
        ":hixl_manifest",
        ":hixl_timer_queue",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_base",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/cpp/disaggregate/cache_store/ascend/proto:hixl_cache_store_cc_proto",
    ],
)

cc_library(
    name = "hixl_manifest",
    srcs = ["HixlManifest.cc"],
    hdrs = ["HixlManifest.h"],
    deps = [
        ":hixl_protocol",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/cpp/disaggregate/cache_store/ascend/proto:hixl_cache_store_cc_proto",
    ],
)

cc_library(
    name = "hixl_read_worker",
    srcs = ["HixlReadWorker.cc"],
    hdrs = ["HixlReadWorker.h"],
    deps = [":hixl_protocol"],
    linkopts = ["-pthread"],
)

cc_library(
    name = "hixl_messager",
    srcs = ["HixlMessager.cc"],
    hdrs = ["HixlMessager.h"],
    deps = [
        ":hixl_memory_util",
        ":hixl_read_worker",
        ":hixl_timer_queue",
        ":hixl_protocol",
        ":hixl_manifest_service",
        ":hixl_manifest",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_base",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    target_compatible_with = select({
        "@//:using_ascend": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

cc_library(
    name = "hixl_protocol",
    hdrs = ["HixlProtocol.h"],
)

cc_library(
    name = "hixl_timer_queue",
    srcs = ["HixlTimerQueue.cc"],
    hdrs = ["HixlTimerQueue.h"],
    linkopts = ["-pthread"],
)
