load("@arch_config//:arch_select.bzl", "torch_deps")
load("//:def.bzl", "copts", "cuda_copts")

cc_library(
    name = "multimodal_input_class",
    hdrs = [
        "MultimodalInputClass.h",
        "MultimodalTokenUtils.h",
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
            "MultimodalError.h",
            "FeatureHashKernel.h",
            "FeatureHash.h",
            "FeatureHashOp.h",
        ],
    ),
    srcs = glob(
        ["*.cc"],
        exclude = ["MultimodalError.cc", "FeatureHashOp.cc"],
    ),
    deps = [
        ":feature_hash_hdr",
        ":feature_hash_op",
        ":multimodal_types",
        ":multimodal_error",
        ":mm_processor_config",
        "//rtp_llm/cpp/config:mm_transport_mode",
        "//rtp_llm/cpp/multimodal_processor/transport:mm_remote_output_transport",
        # Lean RPC targets avoid a dependency cycle through :model_rpc_server.
        "//rtp_llm/cpp/model_rpc:multimodal_pb_converter",
        "//rtp_llm/cpp/engine_base/stream:generate_types",
        "//rtp_llm/cpp/embedding_engine:embedding_engine",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/pybind:py_utils",
    ],
    copts = copts(),
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
    name = "feature_hash_hdr",
    hdrs = ["FeatureHashOp.h"],
    deps = torch_deps(),
    visibility = ["//visibility:public"],
)

# Shared CPU/CUDA implementation lives in compute ops; the ViT worker does not
# need to load the LLM engine merely to produce routing metadata.
cc_library(
    name = "feature_hash_op",
    srcs = ["FeatureHashOp.cc"],
    deps = [
        ":feature_hash_hdr",
        ":feature_hash_utils",
    ] + select({
        "//:using_cuda": [":feature_hash_kernel"],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
)
