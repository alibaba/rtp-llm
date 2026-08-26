package(default_visibility = ["//rtp_llm:__subpackages__"])

py_library(
    name = "server",
    srcs = glob([
        "*.py",
        "server_args/*.py",
        "kvcm_proto/*.py",
    ]),
    deps = [
        "//rtp_llm:grpcio",
        "//rtp_llm:protobuf",
        "//rtp_llm:vipserver",
    ]
)
