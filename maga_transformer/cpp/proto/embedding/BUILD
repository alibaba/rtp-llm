load("//bazel:tf_proto.bzl", "tf_proto_library")

tf_proto_library(
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
    deps = [":embedding_rpc_service_cc_impl"],
    # include_prefix = import_prefix,
    # strip_include_prefix = strip_import_prefix,
    visibility = ["//visibility:public"],
)

py_library(
    name = "all_embedding_grpc_service_py_proto",
    srcs = glob(["*.py"]),
    deps = [":embedding_grpc_service_py"],
    # include_prefix = import_prefix,
    # strip_include_prefix = strip_import_prefix,
    visibility = ["//visibility:public"],
)
