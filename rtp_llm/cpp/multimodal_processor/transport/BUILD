load("//:def.bzl", "copts")

cc_library(
    name = "mm_remote_output_transport_core",
    srcs = ["MMRemoteOutputTransport.cc"],
    hdrs = ["MMRemoteOutputTransport.h"],
    deps = [
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/multimodal_processor:multimodal_types",
        "//rtp_llm/cpp/utils:error_code",
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mm_remote_output_transport",
    srcs = ["MMRemoteOutputTransportFactory.cc"],
    hdrs = ["MMRemoteOutputTransportFactory.h"],
    deps = [
        ":mm_remote_output_transport_core",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/multimodal_processor:mm_rdma_arch_select_impl",
        "//rtp_llm/cpp/multimodal_processor/transport/grpc:mm_grpc_transport",
        "//rtp_llm/cpp/multimodal_processor/transport/rdma:mm_rdma_receipt_reader",
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)
