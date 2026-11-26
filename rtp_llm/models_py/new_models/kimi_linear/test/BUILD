load("//rtp_llm/models_py:checkpoint_test_platforms.bzl", "checkpoint_reader_test_env")

py_library(
    name = "hybrid_ft_fixture",
    srcs = ["test_kimi_owner.py"],
    testonly = True,
    deps = ["//rtp_llm/models_py:checkpoint_reader_test_support", "//rtp_llm/models_py:new_weight_loader", "//rtp_llm:models"],
    visibility = ["//rtp_llm/models_py/new_models/qwen3_next/test:__pkg__"],
)

py_test(
    name = "test_kda_parameters",
    srcs = ["test_kda_parameters.py"],
    env = checkpoint_reader_test_env(),
    deps = ["//rtp_llm/models_py:checkpoint_reader_test_support", "//rtp_llm/models_py:new_weight_loader", "//rtp_llm:fastsafetensors", "//rtp_llm:safetensors", "//rtp_llm:torch"],
)

py_test(
    name = "test_kimi_owner",
    srcs = ["test_kimi_owner.py"],
    env = checkpoint_reader_test_env(),
    deps = ["//rtp_llm/models_py:checkpoint_reader_test_support", "//rtp_llm/models_py:new_weight_loader", "//rtp_llm:models", "//rtp_llm:fastsafetensors", "//rtp_llm:safetensors", "//rtp_llm:torch"],
)
