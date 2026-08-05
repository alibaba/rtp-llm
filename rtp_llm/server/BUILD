package(default_visibility = ["//rtp_llm:__subpackages__"])

py_library(
    name = "server",
    srcs = glob([
        "*.py",
        "server_args/*.py"
    ]),
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:vipserver"
    ]
)
