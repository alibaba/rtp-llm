package(default_visibility = ["//aios/network/rdma:__subpackages__"])
load("//bazel:defs.bzl", "if_rdma")

# Public aggregator over the accl wrapper's ON/OFF impls: the accl package
# keeps subpackage visibility, so consumers (transfer backend, cache_store,
# barex transport) depend on this public entry point instead. Replaces the
# arpc_rdma aggregator that was removed together with the unused arpc RPC
# layer.
cc_library(
    name = "accl_rdma",
    srcs = [],
    hdrs = [],
    deps = if_rdma(
        ["//aios/network/rdma/accl:accl_impl",],
        ["//aios/network/rdma/accl:no_accl_impl",]
    ),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rdma_mocks",
    srcs = [],
    hdrs = [],
    deps = if_rdma(
        ["//aios/network/rdma/accl:mocks_accl_impl",],
        ["//aios/network/rdma/accl:mocks_no_accl_impl",]
    ),
    visibility = ["//visibility:public"],
)
