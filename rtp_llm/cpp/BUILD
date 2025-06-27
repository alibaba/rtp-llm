load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "requirement", "whl_deps", "torch_deps")
load("//bazel:arch_select.bzl", "embedding_arpc_deps")
package(default_visibility = ["//rtp_llm:__subpackages__"])


requirement([
    "grpcio"
])


embedding_arpc_deps()
# load("//bazel:tf_proto.bzl", "tf_proto_library")

cc_library(
    name = "utils",
    hdrs = glob([
        "utils/*.h",
    ]),
    srcs = glob([
        "utils/*.cc",
    ]),
    deps = torch_deps() + [
        "@havenask//aios/alog:alog",
        "@havenask//aios/autil:env_util",
        "@havenask//aios/autil:net",
        "@com_google_absl//absl/debugging:stacktrace",
        "@com_google_absl//absl/debugging:symbolize",
        "//rtp_llm/cpp/proto:model_rpc_service_cc_proto",
        "//:config_modules",
    ],
    copts = copts(),
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
        ":metrics",
        ":tokenizer",
        ":system_prompt",
        ":position_ids_generator",
        "//:gpt_init_params",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/devices:device_utils",
        "//rtp_llm/cpp/models_weight:weights_define",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@havenask//aios/autil:synchronized_queue",
        "@havenask//aios/autil:json",

    ] + select({
        "@//:using_arm": [
            "//rtp_llm/cpp/devices/arm_impl:arm_cpu_impl",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)


cc_library(
    name = "logits_processor",
    hdrs = glob([
        "logits_processor/*.h",
    ]),
    srcs = glob([
        "logits_processor/*.cc",
    ]),
    deps = torch_deps() + [
        ":cache",
        ":dataclass",
        ":utils",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "stream",
    hdrs = glob([
        "stream/*.h",
    ]),
    srcs = glob([
        "stream/*.cc",
    ]),
    deps = torch_deps() + [
        ":models",
        ":cache",
        ":dataclass",
        ":utils",
    ],
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
        ":stream",
        ":utils",
        ":cache",
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
        ":cache",
        ":dataclass",
        ":engine_base",
        "//:gpt_init_params",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/devices:device_utils",
        "//rtp_llm/cpp/models_weight:weights_define",
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
        ":dataclass",
        "//:gpt_init_params",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/devices:device_utils",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
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
        ":logits_processor",
        ":dataclass",
        ":utils",
        ":cache",
        "//rtp_llm/cpp/devices:devices_base",
    ] + select({
        "@//:using_cuda": [],
        "@//:using_rocm": [
            "//rtp_llm/cpp/devices/rocm_impl:rocm_impl",
        ],
        "@//:using_arm": [
            "//rtp_llm/cpp/devices/arm_impl:arm_cpu_impl",
        ],
        "//conditions:default": [
            "//rtp_llm/cpp/devices/cpu_impl:cpu_impl",
        ],
    }),
    visibility = ["//visibility:public"],
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
        "//rtp_llm/cpp/devices:devices_base",
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
        ":stream",
        ":dataclass",
        ":metrics",
        ":utils",
        ":cache",
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
        ":schedulers",
        ":stream",
        ":dataclass",
        ":models",
        ":utils",
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
        "eplb/*.h",
    ]),
    srcs = glob([
        "normal_engine/*.cc",
        "eplb/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
        ":metrics",
        ":models",
        ":schedulers",
        ":engine_base",
        ":utils",
        ":system_prompt",
        ":system_prompt_constructor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//rtp_llm/cpp/devices:devices_base_impl",
    ] + select({
        "@//:using_cuda": [
            "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        ],
        "@//:using_arm": [
            "//rtp_llm/cpp/devices/arm_impl:arm_cpu_impl",
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
    ]),
    srcs = glob([
        "speculative_engine/*.cc",
        "speculative_engine/propose_executor/*.cc",
        "speculative_engine/score_executor/*.cc",
        "speculative_engine/speculative_sampler/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
        ":metrics",
        ":models",
        ":schedulers",
        ":engine_base",
        ":normal_engine",
        ":utils",
        ":system_prompt",
        ":system_prompt_constructor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//rtp_llm/cpp/devices:devices_base_impl",
    ] + select({
        "@//:using_cuda": [
            "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
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
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
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
        ":dataclass",
        "//rtp_llm/cpp/devices:devices_base",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "position_ids_generator",
    hdrs = glob([
        "position_ids_generator/*.h",
    ]),
    srcs = glob([
        "position_ids_generator/*.cc",
    ]),
    deps = [
        "//rtp_llm/cpp/devices:devices_base",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "tokenizer",
    hdrs = glob([
        "tokenizer/*.h",
    ]),
    srcs = glob([
        "tokenizer/*.cc",
    ]),
    deps = [
        "//rtp_llm/cpp/devices:devices_base", # for pybind11
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "openai",
    hdrs = glob([
        "openai/*.h",
    ]),
    srcs = glob([
        "openai/*.cc",
    ]),
    deps = [
        "@havenask//aios/autil:json",
        "//rtp_llm/cpp:utils",
        "//rtp_llm/cpp:dataclass",
        "//rtp_llm/cpp:normal_engine",
        "//rtp_llm/cpp/embedding_engine:embedding_engine",
        ":embedding_arpc_deps",
        # "//rtp_llm/cpp:speculative_engine",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp:multimodal_processor",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "http_api_server",
    srcs = glob([
        "api_server/*.cc",
    ]),
    hdrs = glob([
        "api_server/*.h",
    ]),
    deps = [
        "//rtp_llm/cpp:utils",
        "//rtp_llm/cpp:openai",
        "//rtp_llm/cpp:dataclass",
        "//rtp_llm/cpp:normal_engine",
        "//rtp_llm/cpp:multimodal_processor",
        "//rtp_llm/cpp/http_server:http_server",
        "//rtp_llm/cpp/http_server:http_client",
        ":embedding_arpc_deps",
        "@havenask//aios/autil:net",
        "@havenask//aios/autil:time",
        "//rtp_llm/cpp/api_server/common:common_service",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "model_rpc_server",
    srcs =  glob(["model_rpc/*.cc"]),
    hdrs =  glob(["model_rpc/*.h"]),
    deps = [
        ":embedding_arpc_deps",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp:dataclass",
        "//rtp_llm/cpp:normal_engine",
        "//rtp_llm/cpp/embedding_engine:embedding_engine",
        "//rtp_llm/cpp:speculative_engine",
        "//rtp_llm/cpp:multimodal_processor",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
        "//rtp_llm/cpp/disaggregate/load_balancer:load_balancer",
        "@havenask//aios/autil:net",
    ],
    visibility = ["//visibility:public"]
)

py_library(
    name = "model_rpc_client",
    srcs = ["model_rpc/model_rpc_client.py"],
    deps = [
        ":grpcio",
        "//rtp_llm/cpp/proto:model_rpc_service_py_proto"
    ],
    visibility = ["//visibility:public"],
)

exports_files(glob(["th_op/**/*.cc"]))
exports_files(glob(["th_op/**/*.h"]))