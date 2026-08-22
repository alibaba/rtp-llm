load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "mm_rdma_deps")

# Select the registered RDMA implementation or the open-source no-op.
mm_rdma_deps()

cc_library(
    name = "multimodal_input_class",
    hdrs = [
        "MultimodalInputClass.h",
    ],
    visibility = ["//visibility:public"],
)

# Torch-free multimodal ingress decision table.
cc_library(
    name = "mm_processor_config",
    hdrs = [
        "MMProcessorConfig.h",
    ],
    deps = [
        "//rtp_llm/cpp/config:role_types",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
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

# Compatibility target for out-of-tree RDMA implementations.
cc_library(
    name = "mm_rdma_transport",
    hdrs = [
        "MMRdmaTransport.h",
    ],
    deps = [
        "//rtp_llm/cpp/multimodal_processor/transport/rdma:mm_rdma_transport",
    ],
    visibility = ["//visibility:public"],
)

alias(
    name = "mm_rdma_no_impl",
    actual = "//rtp_llm/cpp/multimodal_processor/transport/rdma:mm_rdma_no_impl",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "multimodal_error",
    srcs = ["MultimodalError.cc"],
    hdrs = ["MultimodalError.h"],
    deps = ["//rtp_llm/cpp/utils:error_code"],
    copts = copts(),
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
    hdrs = glob(
        ["*.h"],
        exclude = [
            "MultimodalInputUtils.h",
            "MultimodalTypes.h",
            "MMProcessorConfig.h",
            "MMRdmaTransport.h",
            "MultimodalError.h",
        ],
    ),
    srcs = glob(
        ["*.cc"],
        exclude = ["MultimodalError.cc"],
    ),
    deps = [
        ":multimodal_types",
        ":multimodal_error",
        ":mm_processor_config",
        "//rtp_llm/cpp/multimodal_processor/transport:mm_remote_output_transport",
        # Lean RPC targets avoid a dependency cycle through :model_rpc_server.
        "//rtp_llm/cpp/model_rpc:mm_rpc_codec",
        "//rtp_llm/cpp/engine_base/stream:generate_types",
        "//rtp_llm/cpp/embedding_engine:embedding_engine",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/pybind:py_utils",
    ],
    visibility = ["//visibility:public"],
)
