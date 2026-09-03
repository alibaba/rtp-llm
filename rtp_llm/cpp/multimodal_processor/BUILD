load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "mm_rdma_deps")

# Produces alias :mm_rdma_arch_select_impl -> internal RDMA impl (internal build, GPU)
# or :mm_rdma_no_impl (open-source / non-GPU). alwayslink on the internal impl ensures
# the MMRdmaTransport factory self-registers.
mm_rdma_deps()

cc_library(
    name = "multimodal_input_class",
    hdrs = [
        "MultimodalInputClass.h",
        "MultimodalTokenUtils.h",
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

# Abstract embedding-RDMA transport interface + factory registry. The concrete
# implementation lives in internal_source and self-registers via the arch_select alias.
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
    name = "mm_rdma_transport",
    srcs = [
        "MMRdmaTransport.cc",
    ],
    hdrs = [
        "MMRdmaTransport.h",
        "MMRdmaVitConfig.h",
    ],
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

# Open-source / non-GPU no-op: only the registry (returns nullptr -> bytes fallback).
cc_library(
    name = "mm_rdma_no_impl",
    deps = [
        ":mm_rdma_transport",
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
    hdrs = glob(
        ["*.h"],
        exclude = [
            "MultimodalTypes.h",
            "MMRdmaTransport.h",
            "MMRdmaVitConfig.h",
            "FeatureHash.h",
            "FeatureHashKernel.h",
        ],
    ),
    srcs = glob(
        ["*.cc"],
        exclude = ["MMRdmaTransport.cc"],
    ),
    deps = [
        ":feature_hash_utils",
        ":multimodal_types",
        ":mm_rdma_transport",
        ":mm_rdma_arch_select_impl",
        "//rtp_llm/cpp/engine_base/stream:generate_types",
        "//rtp_llm/cpp/embedding_engine:embedding_engine",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/pybind:py_utils",
    ] + select({
        "@//:using_cuda": [":feature_hash_kernel"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
