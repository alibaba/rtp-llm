load("//:def.bzl", "copts", "cuda_copts", "torch_deps")
load("//bazel:arch_select.bzl", "requirement", "whl_deps")

requirement([
    "grpcio"
])

# load("//bazel:tf_proto.bzl", "tf_proto_library")

linkopts = [
    "-lpython3.10",
    "-ltorch",
    "-lc10",
    "-ltorch_cpu",
    "-ltorch_python",
    "-L/usr/local/cuda/lib64",
    "-lcudart",
    "-lcuda",
    "-lnccl",
    "-lnvToolsExt",
    "-ldl",
]

cc_library(
    name = "common",
    hdrs = glob([
        "common/*.h",
    ]),
)

cc_library(
    name = "utils",
    hdrs = glob([
        "utils/*.h",
    ]),
    srcs = glob([
        "utils/*.cc",
    ]),
    deps = torch_deps() + [
        "//:gpt_init_params_hdr",
        "//src/fastertransformer/devices:devices_base",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "dataclass",
    hdrs = glob([
        "dataclass/*.h",
    ]),
    srcs = glob([
        "dataclass/*.cc",
    ]),
    deps = torch_deps() + [
        ":utils",
        ":common",
        ":cache",
        ":system_prompt",
        "//:gpt_init_params_hdr",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@havenask//aios/autil:synchronized_queue",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/devices:devices_base",
        "//src/fastertransformer/devices:device_utils",
        "//src/fastertransformer/models:weights_define",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "system_prompt",
    hdrs = glob([
        "system_prompt/SystemPrompt.h",
    ]),
    srcs = glob([
    ]),
    deps = torch_deps() + [
        ":utils",
        ":common",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "system_prompt_constructor",
    hdrs = glob([
        "system_prompt/SystemPromptConstructor.h",
    ]),
    srcs = glob([
        "system_prompt/SystemPromptConstructor.cc",
    ]),
    deps = torch_deps() + [
        ":utils",
        ":common",
        ":cache",
        ":dataclass",
        ":engine_base",
        "//:gpt_init_params_hdr",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/devices:devices_base",
        "//src/fastertransformer/devices:device_utils",
        "//src/fastertransformer/models:weights_define",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cache",
    hdrs = glob([
        "cache/*.h",
    ]),
    srcs = glob([
        "cache/*.cc",
    ]),
    deps = torch_deps() + [
        ":utils",
        ":metrics",
        ":common",
        "//:gpt_init_params_hdr",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//src/fastertransformer/devices:devices_base",
        "//src/fastertransformer/devices:device_utils",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "models",
    hdrs = glob([
        "models/*.h",
    ]),
    srcs = glob([
        "models/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
        ":utils",
        "//src/fastertransformer/devices:devices_base",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "components",
    hdrs = glob([
        "components/*.h",
    ]),
    srcs = glob([
        "components/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
        ":models",
        "//src/fastertransformer/devices:devices_base",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "schedulers",
    hdrs = glob([
        "schedulers/*.h",
    ]),
    srcs = glob([
        "schedulers/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
        ":metrics",
        ":utils",
        ":common",
        ":components",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "engine_base",
    hdrs = glob([
        "engine_base/*.h",
    ]),
    srcs = glob([
        "engine_base/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
        ":models",
        ":common",
        ":system_prompt",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "normal_engine",
    hdrs = glob([
        "normal_engine/*.h",
    ]),
    srcs = glob([
        "normal_engine/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
        ":metrics",
        ":models",
        ":schedulers",
        ":engine_base",
        ":common",
        ":system_prompt",
        ":system_prompt_constructor",
        ":model_wrapper",
        "//src/fastertransformer/cuda:allocator_cuda",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "embedding_engine",
    hdrs = glob([
        "embedding_engine/*.h",
    ]),
    srcs = glob([
        "embedding_engine/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
        ":models",
        ":schedulers",
        ":engine_base",
        ":common",
        ":system_prompt",
        ":system_prompt_constructor",
        ":model_wrapper",
        "//src/fastertransformer/cuda:allocator_cuda",
        "@com_google_absl//absl/status",
        ":embedding_handler",
        "@com_google_absl//absl/status:statusor",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "embedding_handler",
    hdrs = glob([
        "embedding_engine/handlers/*.h",
    ]),
    srcs = glob([
        "embedding_engine/handlers/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
        ":models",
        "//src/fastertransformer/layers:layers",
        "//src/fastertransformer/models:models",
        "//:th_utils",
        "//:th_op_hdrs",
        "//src/fastertransformer/utils:utils",
        "//3rdparty:cuda_driver",
        "//src/fastertransformer/devices/cuda_impl:cuda_impl",
        "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_impl",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
        "//src/fastertransformer/cuda:allocator_cuda",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"]
)

# cc_library(
#     name = "speculative_engine",
#     hdrs = glob([
#         "speculative_engine/*.h",
#     ]),
#     srcs = glob([
#         "speculative_engine/*.cc",
#     ]),
#     deps = torch_deps() + [
#         ":dataclass",
#         ":models",
#         ":schedulers",
#         ":engine_base",
#         ":normal_engine",
#         ":model_wrapper",
#         ":common",
#         ":system_prompt",
#         ":system_prompt_constructor",
#         "//src/fastertransformer/cuda:allocator_cuda",
#         "@com_google_absl//absl/status",
#         "@com_google_absl//absl/status:statusor",
#     ],
#     visibility = ["//visibility:public"]
# )

cc_library(
    name = "metrics",
    hdrs = glob([
        "metrics/*.h",
    ]),
    srcs = glob([
        "metrics/*.cc",
    ]),
    deps = [
        ":utils",
        "@havenask//:kmonitor_client_cpp",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "model_wrapper",
    hdrs = glob([
        "deprecated/*.h",
    ]),
    srcs = glob([
        "deprecated/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
        ":models",
        "//src/fastertransformer/layers:layers",
        "//src/fastertransformer/models:models",
        "//:th_utils",
        "//:th_op_hdrs",
        "//src/fastertransformer/utils:utils",
        "//3rdparty:cuda_driver",
        "//src/fastertransformer/devices/cuda_impl:cuda_impl",
        "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_impl",
        "//3rdparty/trt_fused_multihead_attention:trt_fused_multihead_attention_impl",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
        "//src/fastertransformer/cuda:allocator_cuda",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"]
)

cc_library(
    name = "model_rpc_server",
    srcs = ["model_rpc/ModelRpcServer.cc", "model_rpc/QueryConverter.cc"],
    hdrs = ["model_rpc/ModelRpcServer.h", "model_rpc/QueryConverter.h"],
    deps = [
        "//maga_transformer/cpp/proto:model_rpc_service_cc_proto",
        "//maga_transformer/cpp:dataclass",
        "//maga_transformer/cpp:normal_engine",
        "//maga_transformer/cpp:embedding_engine",
        # "//maga_transformer/cpp:speculative_engine",
        "//src/fastertransformer/devices:devices_base",
    ],
    visibility = ["//visibility:public"]
)


py_library(
    name = "model_rpc_client",
    srcs = ["model_rpc/model_rpc_client.py"],
    deps = [
        ":grpcio",
        "//maga_transformer/cpp/proto:model_rpc_service_py_proto"
    ],
    visibility = ["//visibility:public"],
)
