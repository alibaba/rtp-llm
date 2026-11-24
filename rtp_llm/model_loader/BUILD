package(default_visibility = ["//:__subpackages__"])

py_library(
    name = "loader",
    srcs = glob([
        "*.py",
    ]),
    deps = [
        "//rtp_llm:utils",
        "//rtp_llm/models_py:int8_export_helpers",
        "//rtp_llm/model_loader/tipc:tipc"
    ]
)
