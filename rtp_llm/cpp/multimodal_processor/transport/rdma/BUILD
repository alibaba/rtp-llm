load("//:def.bzl", "copts")

cc_library(
    name = "mm_rdma_transport",
    srcs = ["MMRdmaTransport.cc"],
    hdrs = ["MMRdmaTransport.h"],
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mm_rdma_no_impl",
    deps = [":mm_rdma_transport"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mm_rdma_output_assembler",
    hdrs = ["MMRdmaOutputAssembler.h"],
    deps = [
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/multimodal_processor:multimodal_types",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mm_rdma_receipt_reader",
    srcs = ["MMRdmaReceiptReader.cc"],
    hdrs = ["MMRdmaReceiptReader.h"],
    deps = [
        ":mm_rdma_output_assembler",
        ":mm_rdma_transport",
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/multimodal_processor/transport:mm_remote_output_transport_core",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mm_rdma_encoder_op",
    srcs = ["MMRdmaEncoderOp.cc"],
    hdrs = ["MMRdmaEncoderOp.h"],
    deps = [
        ":mm_rdma_transport",
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:mm_transport_config_extract",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/multimodal_processor:mm_rdma_arch_select_impl",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)
