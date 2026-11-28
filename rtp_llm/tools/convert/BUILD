package(default_visibility = ["//:__subpackages__"])

py_library(
    name = "convert",
    srcs = glob([
        "*.py",
    ]),
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:models",
        "//rtp_llm:openai_api",
        "//rtp_llm:structure",
        "//rtp_llm:utils",
        "//rtp_llm/server",
        "//rtp_llm/tools:model_assistant_api",
    ],
)

py_test(
    name = "weights_convert_test",
    srcs = [
        "weights_convert_test.py",
        "weights_convert_utils.py",
    ],
    data = [
        "//rtp_llm/libs:cache_config_creator_test_runfile",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)
