py_library(
    name = "ascend_base",
    srcs = glob([
        "*.py",
        "test/*.py",
    ]),
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm/models_py/distributed:collective_torch",
        "//rtp_llm:config",
        "//rtp_llm:ops",
    ],
    visibility = ["//visibility:public"],
)
