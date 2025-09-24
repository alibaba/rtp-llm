load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "dataclass",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = torch_deps() + [
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/api_server:tokenizer",
        "//rtp_llm/cpp/models:position_ids_generator",
        "//rtp_llm/cpp/config:gpt_init_params",
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/devices:device_utils",
        "//rtp_llm/cpp/models/models_weight:weights_define",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@havenask//aios/autil:synchronized_queue",
        "@havenask//aios/autil:json",
    ],
    visibility = ["//visibility:public"],
)