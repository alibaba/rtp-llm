package(default_visibility = ["//visibility:public"])

load("//bazel:tf_proto.bzl", "cc_proto")

# accl-free: only havenask arpc protos. Generated headers are included as
# rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/proto/rdma_service.pb.h.
cc_proto(
    name = "rdma_service",
    srcs = [
        "rdma_service.proto",
    ],
    deps = [
        "@havenask//aios/network/arpc:proto",
    ],
    import_prefix = "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/proto",
    visibility = ["//visibility:public"],
)
