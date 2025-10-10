load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "speculative_engine",
    hdrs = glob([
        "*.h",
        "propose_executor/*.h",
        "score_executor/*.h",
        "speculative_sampler/*.h",
    ]),
    srcs = glob([
        "*.cc",
        "propose_executor/*.cc",
        "score_executor/*.cc",
        "speculative_sampler/*.cc",
    ]),
    deps = torch_deps() + [
        "//rtp_llm/cpp/engine_base:engine_base",
        "//rtp_llm/cpp/engine_base:executor",
        "//rtp_llm/cpp/engine_base/stream:generate_config",
        "//rtp_llm/cpp/engine_base:worker_status_info",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/devices:devices_base_impl",
    ] + select({
        "@//:using_cuda": [
            "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
