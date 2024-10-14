package(default_visibility = ["//maga_transformer/cpp/disaggregate/cache_store:__subpackages__"])

load("//bazel:tf_proto.bzl", "cc_proto")

cc_proto(
    name = "cache_store_service",
    srcs = [
        "cache_store_service.proto",
    ],   
    deps = [
        "@havenask//aios/network/arpc:proto",
    ], 
)
