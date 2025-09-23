load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")


cc_library(
    name = "engine_base",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = torch_deps() + [
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/dataclass:dataclass",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/models:lora",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:system_utils",
        "//rtp_llm/cpp/utils:data_structures",
        "//rtp_llm/cpp/utils:domain_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    visibility = ["//visibility:public"],
)

