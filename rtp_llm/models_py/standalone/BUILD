load("//bazel:arch_select.bzl", get_triton_deps = "triton_deps")

py_library(
    name = "auto_model",
    srcs = ["auto_model.py"],
    deps = [
        ":py_standalone_testlib",

        # auto_model extra deps
        "//rtp_llm:async_model",           # async_decoder_engine
        "//rtp_llm:embedding",             # embedding
        "//rtp_llm:models",                # rtp_llm.models
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "py_standalone_testlib",
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:_ft_pickler",
        "//rtp_llm:ops",
        "//rtp_llm:utils",
        "//rtp_llm/models_py:models",
        "//rtp_llm/model_loader:loader",
        "//rtp_llm:device",
        "//rtp_llm:rtp_llm_frontend_lib",
        "//rtp_llm:uvicorn",
        "//rtp_llm:fastapi",
        "//rtp_llm:psutil",
        "//rtp_llm:aiohttp",
        "//rtp_llm:pydantic",
        "//rtp_llm:json5",
        "//rtp_llm:dashscope",
        "//rtp_llm:jieba",
        "//rtp_llm:partial_json_parser",
        "//rtp_llm:openai",
        "//rtp_llm:torch",
        "//rtp_llm:transformers",
        "//rtp_llm/test/utils:test_util",
    ] + select({
        "@//:using_rocm": [
            "//rtp_llm:aiter",
            "//rtp_llm:pytorch-triton-rocm",
            "//rtp_llm:amdsmi",
        ],
        "@//:using_cuda12": [
            "//rtp_llm:deep_gemm",
            "//rtp_llm:flashinfer-python",
        ],
        "//conditions:default": [],
    }),
    data = [
        "//:th_transformer_config",
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)
