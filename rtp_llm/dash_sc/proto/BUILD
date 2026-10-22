# BUILD for dash_sc inference proto (predict_v2 + model_config).
# predict_v2.proto imports model_config.proto; use include="." so protoc finds it.
# Python pb: same pattern as rtp_llm/cpp/model_rpc/proto (py_binary + genrule + py_library).
load("//bazel:tf_proto.bzl", "cc_proto_library")

# ---------- C++ protos (unchanged) ----------
cc_proto_library(
    name = "model_config_cc",
    srcs = ["model_config.proto"],
    include = ".",
    default_header = True,
    cc_libs = ["@com_google_protobuf//:protobuf"],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "predict_v2_cc",
    srcs = ["predict_v2.proto"],
    use_grpc_plugin = True,
    protolib_deps = [":model_config_cc"],
    include = ".",
    default_header = True,
    deps = [":model_config_cc_impl"],
    cc_libs = ["@com_google_protobuf//:protobuf"],
    visibility = ["//visibility:public"],
)

# ---------- Python protos (like model_rpc) ----------
py_binary(
    name = "create_grpc_proto",
    srcs = ["create_grpc_proto.py"],
    data = ["model_config.proto", "predict_v2.proto"],
    deps = ["//rtp_llm:grpcio-tools", "//rtp_llm:setuptools"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "predict_v2_py",
    srcs = [],
    outs = [
        "model_config_pb2.py",
        "predict_v2_pb2.py",
        "predict_v2_pb2_grpc.py",
        "__init__.py",
    ],
    # Use the configured execution Python for the real py_binary launcher.
    cmd = """set -eu
: "$${PYTHON_BIN_PATH:?PYTHON_BIN_PATH must identify the configured execution Python}"
case "$${PYTHON_BIN_PATH}" in
    */*) export PATH="$${PYTHON_BIN_PATH%/*}:$${PATH:-/usr/bin:/bin}" ;;
esac
exec "$${PYTHON_BIN_PATH}" "$(execpath :create_grpc_proto)" "$(@D)"
""",
    tools = [":create_grpc_proto"],
)

py_library(
    name = "predict_v2_py_proto",
    srcs = [":predict_v2_py"],
    visibility = ["//visibility:public"],
)
