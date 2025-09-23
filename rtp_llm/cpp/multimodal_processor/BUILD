load("//:def.bzl", "copts")

cc_library(
    name = "multimodal_types",
    hdrs = [
        "MultimodalTypes.h",
    ],
    deps = [
        "//rtp_llm/cpp/core:buffer_hdr",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "multimodal_processor",
    hdrs = glob([
        "*.h",
    ], exclude = ["MultimodalTypes.h"]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        ":multimodal_types",
        "//rtp_llm/cpp/engine_base/stream:generate_types",
        "//rtp_llm/cpp/embedding_engine:embedding_engine",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/config:config_modules",
    ],
    visibility = ["//visibility:public"],
)