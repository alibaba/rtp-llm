package(default_visibility = ["//:__subpackages__"])

py_library(
    name = "convert",
    srcs = glob([
        "*.py",
    ]),
    deps = [
        "//rtp_llm:utils",
        "//rtp_llm:config",
        "//rtp_llm:structure",
        "//rtp_llm:models",
        "//rtp_llm:openai_api",
        "//rtp_llm/server:server",
        "//rtp_llm/tools:model_assistant_api"
    ]
)


py_test(
    name = "weights_convert_test",
    srcs = ["weights_convert_test.py"],
    data = ["weights_convert.py"],
)
