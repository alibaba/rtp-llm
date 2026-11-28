load("@arch_config//:arch_select.bzl", "requirement")

requirement(["grpcio"])

py_library(
    name = "rl_inference_client",
    srcs = ["rl_inference_client.py"],
    deps = [
        ":grpcio",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_py_proto",
    ],
    visibility = ["//visibility:public"],
)
