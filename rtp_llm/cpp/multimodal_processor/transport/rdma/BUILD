load("//:def.bzl", "copts")

cc_library(
    name = "mm_rdma_adapter",
    srcs = ["MMRdmaAdapter.cc"],
    hdrs = ["MMRdmaAdapter.h"],
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:mm_transport_config_extract",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/multimodal_processor:multimodal_types",
        "//rtp_llm/cpp/multimodal_processor/transport:mm_remote_output_transport_core",
        "//rtp_llm/cpp/rdma_transport:rdma_transport_arch_select_impl",
        "//rtp_llm/cpp/rdma_transport:rdma_transport",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)
