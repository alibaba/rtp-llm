load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "mm_rdma_deps")

mm_rdma_deps()

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
    name = "mm_rdma_transport",
    srcs = ["MMRdmaTransport.cc"],
    hdrs = ["MMRdmaTransport.h", "MMRdmaVitConfig.h"],
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
    ], exclude = ["MultimodalTypes.h", "FeatureHash.h", "FeatureHashKernel.h", "MMRdmaTransport.h", "MMRdmaVitConfig.h"]),
    srcs = glob([
        "*.cc",
    ], exclude = ["MMRdmaTransport.cc"]),
    deps = [
        ":multimodal_types",
        ":mm_rdma_transport",
        ":mm_rdma_arch_select_impl",
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
