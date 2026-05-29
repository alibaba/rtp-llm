package(default_visibility = ["//rtp_llm:__subpackages__"])

exports_files(["server_args/pd_separation_group_args.py"])

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
