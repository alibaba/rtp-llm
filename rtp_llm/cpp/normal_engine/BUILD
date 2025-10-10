load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "normal_engine",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = torch_deps() + [
        "//rtp_llm/cpp/engine_base:engine_base",
        "//rtp_llm/cpp/engine_base/stream:generate_config",
        "//rtp_llm/cpp/engine_base:worker_status_info",
        "//rtp_llm/cpp/engine_base:profiler",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt_constructor",
        "//rtp_llm/cpp/models:eplb",
        "//rtp_llm/cpp/models:eplb_config",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/devices:devices_base_impl",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ] + select({
        "@//:using_cuda": [
            "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        ],
        "@//:using_arm": [
            "//rtp_llm/cpp/devices/arm_impl:arm_cpu_impl",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
