load("//:def.bzl", "copts")

cc_library(
    name = "mm_grpc_transport",
    srcs = ["MMGrpcTransport.cc"],
    hdrs = ["MMGrpcTransport.h"],
    deps = [
        "//rtp_llm/cpp/model_rpc:multimodal_pb_converter",
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
        "//rtp_llm/cpp/multimodal_processor:multimodal_error",
        "//rtp_llm/cpp/multimodal_processor/transport:mm_remote_output_transport_core",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)
