load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "cache_new",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
        "//rtp_llm/cpp/model_rpc:rpc_error_code",
        "//rtp_llm/cpp/utils:lru_cache",
        "//rtp_llm/cpp/metrics:metrics",
        "//:rtp_compute_ops",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ] + select({
        "//:enable_3fs": [
            ":storage_3fs",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "storage_3fs",
    hdrs = [
    ],
    srcs = [
    ],
    deps = [
        "//3rdparty/3fs:hf3fs",
    ],
    visibility = ["//visibility:public"],
)
