load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "requirement", "whl_deps", "torch_deps")
load("//bazel:arch_select.bzl", "embedding_arpc_deps")
package(default_visibility = ["//maga_transformer:__subpackages__"])


requirement([
    "grpcio"
])


embedding_arpc_deps()
# load("//bazel:tf_proto.bzl", "tf_proto_library")

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
        "//src/fastertransformer/utils:py_utils",
        "@havenask//aios/alog:alog",
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
        "@havenask//aios/autil:json",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/devices:devices_base",
        "//src/fastertransformer/devices:device_utils",
        "//src/fastertransformer/models:weights_define",
    ] + select({
        "//:using_arm": [
            "//src/fastertransformer/devices/arm_impl:arm_cpu_impl",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "lora",
    hdrs = glob([
        "lora/*.h",
    ]),
    srcs = glob([
        "lora/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
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
    ] + select({
        "//:using_cuda": [],
        "//:using_rocm": [
            "//src/fastertransformer/devices/rocm_impl:rocm_impl",
        ],
        "//:using_arm": [
            "//src/fastertransformer/devices/arm_impl:arm_cpu_impl",
        ],
        "//conditions:default": [
            "//src/fastertransformer/devices/cpu_impl:cpu_impl",
        ],
    }),
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
        ":lora",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//src/fastertransformer/devices:devices_base_impl",
    ] + select({
        "//:using_cuda": [
            "//src/fastertransformer/devices/cuda_impl:cuda_impl",
        ],
        "//:using_arm": [
            "//src/fastertransformer/devices/arm_impl:arm_cpu_impl",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"]
)

cc_library(
    name = "speculative_engine",
    hdrs = glob([
        "speculative_engine/*.h",
        "speculative_engine/propose_executor/*.h",
        "speculative_engine/score_executor/*.h",
        "speculative_engine/speculative_sampler/*.h",
        "speculative_engine/speculative_updater/*.h"
    ]),
    srcs = glob([
        "speculative_engine/*.cc",
        "speculative_engine/propose_executor/*.cc",
        "speculative_engine/score_executor/*.cc",
        "speculative_engine/speculative_sampler/*.cc",
        "speculative_engine/speculative_updater/*.cc"
    ]),
    deps = torch_deps() + [
        ":dataclass",
        ":metrics",
        ":models",
        ":schedulers",
        ":engine_base",
        ":normal_engine",
        ":common",
        ":system_prompt",
        ":system_prompt_constructor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//src/fastertransformer/devices:devices_base_impl",
    ] + select({
        "//:using_cuda": [
            "//src/fastertransformer/devices/cuda_impl:cuda_impl",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"]
)

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
    name = "multimodal_processor",
    hdrs = glob([
        "multimodal_processor/*.h",
    ]),
    srcs = glob([
        "multimodal_processor/*.cc",
    ]),
    deps = [
        "//src/fastertransformer/devices:devices_base",
        "//src/fastertransformer/utils:py_utils",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "http_api_server",
    srcs = ["HttpApiServer.cc"],
    hdrs = ["HttpApiServer.h"],
    deps = [
        "//maga_transformer/cpp/http_server:http_server",
        "//maga_transformer/cpp:dataclass",
        "//maga_transformer/cpp:normal_engine",
        "//maga_transformer/cpp/embedding_engine:embedding_engine",
        ":embedding_arpc_deps",
        # "//maga_transformer/cpp:speculative_engine",
        "//src/fastertransformer/devices:devices_base",
        "//maga_transformer/cpp:multimodal_processor",
    ],
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
        "//maga_transformer/cpp/embedding_engine:embedding_engine",
        ":embedding_arpc_deps",
        "//maga_transformer/cpp:speculative_engine",
        "//src/fastertransformer/devices:devices_base",
        "//maga_transformer/cpp:multimodal_processor",
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
