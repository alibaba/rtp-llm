
# TOOD(wangyin.yx): layers and modules should be organized in a more structred way.

py_library(
    name = "models",
    srcs = glob([
        "model_desc/*.py",
        "modules/*.py",
    ]),
    deps = [
        "//rtp_llm/model_loader:loader",
    ],
    visibility = ["//visibility:public"],
)
