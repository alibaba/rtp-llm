load("//:def.bzl", "copts")

cc_library(
    name = "multimodal_types",
    hdrs = [
        "MultimodalTypes.h",
    ],
    deps = [
        "//:rtp_compute_ops",
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
        "//rtp_llm/cpp/pybind:py_utils",
    ],
    visibility = ["//visibility:public"],
)