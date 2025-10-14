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
    deps = [
        "//rtp_llm/cpp/api_server:tokenizer",
        "//rtp_llm/cpp/models:position_ids_generator",
        "//rtp_llm/cpp/pybind:py_utils",
        "@havenask//aios/autil:synchronized_queue",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
    alwayslink = True,
)