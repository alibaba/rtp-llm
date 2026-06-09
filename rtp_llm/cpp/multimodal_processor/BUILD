load("//:def.bzl", "copts")

cc_library(
    name = "multimodal_input_class",
    hdrs = [
        "MultimodalInputClass.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "multimodal_types",
    hdrs = [
        "MultimodalTypes.h",
    ],
    deps = [
        "//:rtp_compute_ops",
        ":multimodal_input_class",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "multimodal_input_utils",
    hdrs = [
        "MultimodalInputUtils.h",
    ],
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "multimodal_processor",
    hdrs = glob([
        "*.h",
    ], exclude = [
        "MultimodalInputUtils.h",
        "MultimodalTypes.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        ":multimodal_types",
        "//rtp_llm/cpp/engine_base/stream:input_embeddings_utils",
        "//rtp_llm/cpp/engine_base/stream:generate_types",
        "//rtp_llm/cpp/embedding_engine:embedding_engine",
        "//rtp_llm/cpp/pybind:py_utils",
    ],
    visibility = ["//visibility:public"],
)
