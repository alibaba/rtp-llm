load("//:def.bzl", "copts", "cuda_copts")

cc_library(
    name = "length_predictor_state",
    hdrs = ["LengthPredictorState.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "length_encoder_kernel",
    srcs = ["kernels/length_encoder_kernel.cu"],
    hdrs = ["kernels/length_encoder_kernel.h"],
    copts = cuda_copts(),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "length_predictor",
    hdrs = ["LengthPredictor.h"],
    srcs = ["LengthPredictor.cc"],
    copts = copts(),
    deps = [
        ":length_predictor_state",
        "//rtp_llm/cpp/utils:core_utils",
        "//:rtp_compute_ops",
    ] + select({
        "//:using_cuda": [
            ":length_encoder_kernel",
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
