package(default_visibility = ["//rtp_llm/cpp/disaggregate/cache_store:__subpackages__"])

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

cc_proto(
    name = "remote_mock_control",
    visibility = ["//rtp_llm/cpp/cache/block_tree_cache:__subpackages__", "//rtp_llm/cpp/disaggregate/cache_store:__subpackages__"],
    srcs = ["remote_mock_control.proto"],
    deps = [
        "//rtp_llm/cpp/disaggregate/cache_store/proto:cache_store_service",
        "@havenask//aios/network/arpc:proto",
    ],
)

# tf_proto dependency code generation expects the unsuffixed provider name.
alias(
    name = "cache_store_service_genproto",
    actual = ":cache_store_service_cc_genproto",
)
