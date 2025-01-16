load("//bazel:tf_proto.bzl", "tf_proto_library_cc")
load("//bazel:py_proto.bzl", "generate_grpc_proto")

tf_proto_library_cc(
    name = "embedding_grpc_service",
    srcs = ["all_embedding_rpc_service.proto"],
    has_services = True,
    cc_grpc_version = True,
    default_header = True,
)

cc_library(
    name = "embedding_grpc_service_cc_proto",
    hdrs = glob(["*.h"]),
    srcs = glob(["*.cc"]),
    deps = [":embedding_grpc_service_cc_impl"],
    # include_prefix = import_prefix,
    # strip_include_prefix = strip_import_prefix,
    visibility = ["//visibility:public"],
)
