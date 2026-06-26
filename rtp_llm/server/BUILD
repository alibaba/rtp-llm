package(default_visibility = ["//rtp_llm:__subpackages__"])

py_library(
    name = "request_headers",
    srcs = ["request_headers.py"],
)

py_library(
    name = "server",
    srcs = glob([
        "*.py",
        "server_args/*.py"
    ]),
    deps = [
        "//rtp_llm:vipserver"
    ]
)
