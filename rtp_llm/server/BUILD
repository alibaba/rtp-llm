package(default_visibility = ["//rtp_llm:__subpackages__"])

py_library(
    name = "server",
    srcs = glob([
        "*.py",
        "server_args/*.py"
    ]),
    deps = [
        "//internal_source/prompt_generator/service:server",
        "//rtp_llm:vipserver"
    ]
)
