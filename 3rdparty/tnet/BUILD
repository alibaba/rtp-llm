# Wraps the @tnet rpm (declared as http_file in `internal_source/deps/http.bzl`)
# into a cc_library `//3rdparty/tnet:tnet`, consumed by the RDMA cache-store
# transport at
# `internal_source/rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma:transfer_rdma_impl`.
#
# CUDA + ROCm Bazel analysis of `//:th_transformer` requires this BUILD.
# Platform overlays may skip the barex_rdma target, but this package should
# remain available unless the transport is removed everywhere.
# Do NOT delete during deps cleanup unless every consumer in
# `internal_source/rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/**`
# (and the @tnet http_file in `internal_source/deps/http.bzl`) is also being removed.
load("//bazel:defs.bzl", "rpm_library")

rpm_library(
    name = "tnet" ,
    rpms = ["@tnet//file:file"],
    lib_path = "usr/lib64",
    include_path = "usr/include",
    static_lib = "liblibtnet.a",
    shared_lib = "liblibtnet.so",
    hdrs = ["libtnet.h",],
)
