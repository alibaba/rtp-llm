package(default_visibility = ["//:__subpackages__"])

py_library(
    name = "ft_export_engram",
    srcs = ["ft_export_engram.py"],
    deps = ["//rtp_llm:safetensors_torch"],
)

py_library(
    name = "ft_export_metadata",
    srcs = ["ft_export_metadata.py"],
    deps = ["//rtp_llm:safetensors_torch"],
)

py_library(
    name = "ft_export_writer",
    srcs = ["ft_export_writer.py"],
    deps = ["//rtp_llm:safetensors_torch"],
)

py_test(
    name = "ft_export_metadata_test",
    srcs = ["ft_export_metadata_test.py"],
    deps = [":ft_export_metadata", "//rtp_llm:torch"],
)

py_library(
    name = "convert",
    srcs = glob([
        "*.py",
    ]),
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:model_factory",
        "//rtp_llm:models",
        "//rtp_llm:openai_api",
        "//rtp_llm:structure",
        "//rtp_llm:utils",
        "//rtp_llm/models_py/distributed:rank_layout",
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
        "//rtp_llm/cpp/cache/test:cache_config_creator_py_test",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)
