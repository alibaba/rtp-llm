load("//:def.bzl", "copts")

cc_library(
    name = "mm_rdma_reader",
    srcs = ["MMRdmaReader.cc"],
    hdrs = ["MMRdmaReader.h"],
    deps = [
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/multimodal_processor:multimodal_types",
        "//rtp_llm/cpp/multimodal_processor/transport:mm_remote_output_transport_core",
        "//rtp_llm/cpp/config:rdma_config",
        "//rtp_llm/cpp/rdma_transport:rdma_transport_arch_select_impl",
        "//rtp_llm/cpp/rdma_transport:rdma_transport",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mm_rdma_exporter",
    srcs = ["MMRdmaExporter.cc"],
    hdrs = ["MMRdmaExporter.h"],
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/config:rdma_config",
        "//rtp_llm/cpp/config:mm_transport_mode",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/pybind:config_extract",
        "//rtp_llm/cpp/rdma_transport:rdma_transport_arch_select_impl",
        "//rtp_llm/cpp/rdma_transport:rdma_transport",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)
