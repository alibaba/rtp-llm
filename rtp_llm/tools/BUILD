package(default_visibility = ["//:__subpackages__"])

py_library(
    name = "model_assistant_api",
    srcs = glob([
        "api/*.py",
    ]),
    deps = ["//rtp_llm:model_factory"],
)

py_library(
    name = "model_assistant",
    srcs = [
           "model_assistant_server.py",
           ":model_assistant_api"
    ]
)

