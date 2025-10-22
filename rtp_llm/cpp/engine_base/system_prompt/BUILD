load("//:def.bzl", "copts")

# System prompt - engine level functionality
cc_library(
    name = "system_prompt",
    hdrs = [
        "SystemPrompt.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
    ],
)

cc_library(
    name = "system_prompt_constructor",
    srcs = [
        "SystemPromptConstructor.cc",
    ],
    hdrs = [
        "SystemPromptConstructor.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cache2",
        "//rtp_llm/cpp/engine_base",
        "//rtp_llm/cpp/engine_base/stream:generate_config",
    ],
)
