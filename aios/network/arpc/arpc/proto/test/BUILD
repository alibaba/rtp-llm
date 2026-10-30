package(default_visibility = ["//visibility:public"])

load("//bazel:defs.bzl", "cc_proto")

# Test-support proto for the open-sourced RDMA accl wrapper tests
# (//aios/network/rdma/accl): the channel and server tests include the
# generated header as
# "aios/network/arpc/arpc/proto/test/TestMessage.pb.h" — the path the closed
# @alibaba_rdma world expected @havenask's aios/network/arpc:arpc_test_proto
# target to provide. Neither the public github havenask pin nor the internal
# gitlab fork declares that target or carries the file (their
# aios/network/arpc/BUILD retains only the dangling test_deps entry), so the
# moved tests carry the proto locally at the include-compatible path; the
# moved-tree test BUILDs depend on this label instead of @havenask.
#
# Deliberately import-free (no arpc global_service_id/local_method_id options
# from @havenask's rpc_extensions.proto): PacketArg.h reads those extensions
# with default values when unset, so client and server still derive the same
# rpcCode for the single Echo method without any cross-repo proto dep.
cc_proto(
    name = "arpc_test_proto",
    srcs = [
        "TestMessage.proto",
    ],
    deps = [],
    visibility = ["//visibility:public"],
)
