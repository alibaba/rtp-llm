load("@arch_config//:arch_select.bzl", "torch_deps")
load("//:def.bzl", "copts")

cc_library(
    name = "mm_kvcm_client",
    srcs = [
        "MMKvcmClient.cc",
    ] + select({
        "//:using_kvcm_emb_storage": ["MMKvcmClientKvcm.cc"],
        "//conditions:default": ["MMKvcmClientStub.cc"],
    }),
    hdrs = ["MMKvcmClient.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/config:mm_kvcm_config",
    ] + select({
        "//:using_kvcm_emb_storage": [
            "//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_client",
            "//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_object_headers",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "mm_kvcm_reader",
    srcs = ["MMKvcmReader.cc"],
    hdrs = ["MMKvcmReader.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = torch_deps() + [
        ":mm_kvcm_client",
        "//rtp_llm/cpp/config:mm_kvcm_config",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/multimodal_processor:multimodal_types",
        "//rtp_llm/cpp/multimodal_processor/transport:mm_remote_output_transport_core",
        "//rtp_llm/cpp/utils:core_utils",
    ],
)

cc_library(
    name = "mm_kvcm_writer",
    srcs = ["MMKvcmWriter.cc"],
    hdrs = ["MMKvcmWriter.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = torch_deps() + [
        ":mm_kvcm_client",
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/config:mm_kvcm_config",
        "//rtp_llm/cpp/pybind:config_extract",
    ],
)
