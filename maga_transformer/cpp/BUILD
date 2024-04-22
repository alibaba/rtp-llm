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
        "//:gpt_init_params_hdr",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//src/fastertransformer/devices:devices_base",
        "//src/fastertransformer/devices:device_utils",
        "//src/fastertransformer/models:weights_define",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ptuning",
    hdrs = glob([
        "ptuning/*.h",
    ]),
    srcs = glob([
        "ptuning/*.cc",
    ]),
    deps = torch_deps() + [
        ":utils",
        ":common",
        ":cache",
        "//:gpt_init_params_hdr",
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
        ":common",
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
        ":utils",
        ":common",
        ":components",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "batch_stream_processor",
    hdrs = glob([
        "batch_stream_processor/*.h",
    ]),
    srcs = glob([
        "batch_stream_processor/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
        ":models",
        ":common",
        "//src/fastertransformer/devices/cuda_impl:cuda_impl",
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
    deps = torch_deps() + [
        ":dataclass",
        ":models",
        ":common",
        ":ptuning",
        ":batch_stream_processor",
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
        ":models",
        ":schedulers",
        ":engine_base",
        ":common",
        ":ptuning",
        ":model_wrapper",
        ":batch_stream_processor",
        "//src/fastertransformer/cuda:allocator_cuda",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "speculative_engine",
    hdrs = glob([
        "speculative_engine/*.h",
    ]),
    srcs = glob([
        "speculative_engine/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
        ":models",
        ":schedulers",
        ":engine_base",
        ":normal_engine",
        ":model_wrapper",
        ":common",
        ":ptuning",
        ":batch_stream_processor",
        "//src/fastertransformer/cuda:allocator_cuda",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
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
        "//maga_transformer/cpp:speculative_engine",
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

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":model_rpc_server",
        "//3rdparty/flash_attention2:flash_attention2_impl",
    ] + torch_deps(),
    linkopts = linkopts,
    visibility = ["//visibility:public"]
)
