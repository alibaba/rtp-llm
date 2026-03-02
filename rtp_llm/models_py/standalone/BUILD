load("//bazel:arch_select.bzl", get_triton_deps = "triton_deps", "internal_deps", "requirement")

# Create aliases for fast-safetensors packages from the main rtp_llm BUILD
requirement(["fast-safetensors", "fastsafetensors"])

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

flashinfer = [
    "//rtp_llm:flashinfer-python",
]

flashinfer_with_cache = [
    "//rtp_llm:flashinfer-python",
    "//rtp_llm:flashinfer-cubin",
    "//rtp_llm:flashinfer-jit-cache",
]

flashmla = [
    "//rtp_llm:fast-hadamard-transform",
    "//rtp_llm:flash-mla",
    "//rtp_llm:tilelang"
]

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
            "//rtp_llm:amdsmi",
        ],
        "@//:using_cuda12": [
            "//rtp_llm:deep_gemm",
        ],
        "//conditions:default": [],
    }) + select({
        "@//:using_cuda12_9_x86": flashinfer_with_cache + flashmla,
        "@//:using_cuda12_arm": flashinfer_with_cache,
        "@//:cuda_pre_12_9": flashinfer,
        "//conditions:default": []
    }) + internal_deps(),
    data = [
        "//:th_transformer_config",
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)
