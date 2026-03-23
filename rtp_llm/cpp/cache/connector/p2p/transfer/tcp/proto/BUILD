package(default_visibility = ["//visibility:public"])

load("//bazel:tf_proto.bzl", "cc_proto")

cc_proto(
    name = "tcp_service",
    srcs = [
        "tcp_service.proto",
    ],
    deps = [
        "@havenask//aios/network/arpc:proto",
    ],
    visibility = ["//visibility:public"],
)
