package(default_visibility = ["//rtp_llm:__subpackages__"])

py_library(
    name = "request_headers",
    srcs = ["request_headers.py"],
)

py_library(
    name = "vit_rpc_control",
    srcs = ["vit_rpc_control.py"],
    deps = ["//rtp_llm:grpcio"],
)

py_library(
    name = "server",
    srcs = glob([
        "*.py",
        "server_args/*.py"
    ], exclude = [
        "request_headers.py",
        "vit_rpc_control.py",
    ]),
    deps = [
        ":request_headers",
        ":vit_rpc_control",
        "//rtp_llm:telemetry",
        "//rtp_llm:warmup",
        "//rtp_llm:vipserver",
        "//rtp_llm/cpp/model_rpc/proto:flexlb_schedule_service_py_proto",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_py_proto",
    ]
)
