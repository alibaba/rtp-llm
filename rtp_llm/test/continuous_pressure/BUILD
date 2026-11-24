package(default_visibility = ["//visibility:public"])

py_library(
    name = "continuous_pressure",
    srcs = [
        "__init__.py",
        "config.py",
        "diagnostics.py",
        "phases.py",
        "ports.py",
        "results.py",
        "seed_plan.py",
        "sender.py",
        "sequence.py",
        "session.py",
    ],
)

# Planning and injected transports remain importable without a native runtime.
py_library(
    name = "rtp_adapter",
    srcs = glob(["adapters/*.py", "analysis/*.py"]),
    deps = [":continuous_pressure"],
)

# Standalone scripts using connect() need the installed DashSC client and both wire APIs.
py_library(
    name = "rtp_sdk",
    deps = [
        ":rtp_adapter",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_py_proto",
        "//rtp_llm/dash_sc:dash_sc_grpc_server",
        "//rtp_llm/dash_sc/proto:predict_v2_py_proto",
    ],
)
