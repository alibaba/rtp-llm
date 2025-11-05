package(default_visibility = ["//rtp_llm/cpp/disaggregate/cache_store_new:__subpackages__"])

load("//bazel:tf_proto.bzl", "cc_proto")

cc_proto(
    name = "service",
    srcs = [
        "service.proto",
    ],
    deps = [
        "@havenask//aios/network/arpc:proto",
    ],
)