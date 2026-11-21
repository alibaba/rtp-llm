load("//bazel:tf_proto.bzl", "cc_proto")

package(default_visibility = ["//rtp_llm/cpp/disaggregate/cache_store:__subpackages__"])

cc_proto(
    name = "hixl_cache_store",
    srcs = ["hixl_cache_store.proto"],
    deps = ["@havenask//aios/network/arpc:proto"],
)
