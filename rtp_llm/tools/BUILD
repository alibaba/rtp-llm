package(default_visibility = ["//:__subpackages__"])

py_library(
    name = "model_assistant_api",
    srcs = glob([
        "api/*.py",
    ]),
    deps = ["//rtp_llm:frontend_model_factory"],
)

py_library(
    name = "model_assistant",
    srcs = ["model_assistant_server.py"],
    deps = [":model_assistant_api"],
)

