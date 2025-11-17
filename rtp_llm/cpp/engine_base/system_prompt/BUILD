load("//:def.bzl", "copts")

# System prompt - engine level functionality
cc_library(
    name = "system_prompt",
    hdrs = [
        "SystemPrompt.h",
    ],
    deps = [
        "//:rtp_compute_ops",
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
    deps = [
        "//rtp_llm/cpp/engine_base:engine_base",
        "//rtp_llm/cpp/engine_base/stream:generate_config",
    ],
    visibility = ["//visibility:public"],
)
