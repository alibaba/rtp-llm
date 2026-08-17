package(default_visibility = ["//rtp_llm:__subpackages__"])

py_library(
    name = "request_headers",
    srcs = ["request_headers.py"],
)

py_library(
    name = "server_args_util",
    srcs = ["server_args/util.py"],
)

py_library(
    name = "server",
    srcs = glob([
        "*.py",
        "server_args/*.py"
    ], exclude = [
        "request_headers.py",
        "server_args/util.py",
    ]),
    deps = [
        ":request_headers",
        ":server_args_util",
        "//rtp_llm:vipserver",
    ]
)
