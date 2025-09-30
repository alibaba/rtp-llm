load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

# System prompt - engine level functionality
cc_library(
    name = "system_prompt",
    hdrs = [
        "SystemPrompt.h",
    ],
    deps = torch_deps() + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "system_prompt_constructor",
    hdrs = [
        "SystemPromptConstructor.h",
    ],
    srcs = [
        "SystemPromptConstructor.cc",
    ],
    deps = torch_deps() + [
        "//rtp_llm/cpp/engine_base:engine_base",
        "//rtp_llm/cpp/engine_base/stream:generate_config",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/config:gpt_init_params",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/devices:device_utils",
        "//rtp_llm/cpp/models/models_weight:weights_define",
    ],
    visibility = ["//visibility:public"],
)
