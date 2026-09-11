load("//:def.bzl", "copts", "cuda_copts")

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
    name = "feature_hash_utils",
    hdrs = ["FeatureHash.h"],
)

cc_library(
    name = "feature_hash_kernel",
    srcs = ["FeatureHashKernel.cu"],
    hdrs = ["FeatureHashKernel.h"],
    copts = cuda_copts(),
    deps = [
        ":feature_hash_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_library(
    name = "multimodal_processor",
    hdrs = glob([
        "*.h",
    ], exclude = ["MultimodalTypes.h", "FeatureHash.h", "FeatureHashKernel.h"]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        ":multimodal_types",
        ":feature_hash_utils",
        "@grpc//:grpc++",
        "//rtp_llm/cpp/engine_base/stream:generate_types",
        "//rtp_llm/cpp/embedding_engine:embedding_engine",
        "//rtp_llm/cpp/pybind:py_utils",
    ] + select({
        "@//:using_cuda": [":feature_hash_kernel"],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
)
