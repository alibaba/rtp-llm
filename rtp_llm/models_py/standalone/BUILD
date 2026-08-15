load("@arch_config//:arch_select.bzl", "internal_deps", "requirement")

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
        "//rtp_llm/models_py:utils",
    ],
    visibility = ["//visibility:public"],
)

# ninja = the executable invoked by flashinfer's JIT backend (see the note on the
# variable of the same name in //rtp_llm:BUILD)
flashinfer = [
    "//rtp_llm:flashinfer-python",
    "//rtp_llm:ninja",
]

flashinfer_with_cache = [
    "//rtp_llm:flashinfer-cubin",
    "//rtp_llm:flashinfer-jit-cache",
    "//rtp_llm:nvidia-cutlass-dsl",
    "//rtp_llm:ninja",
]

flashmla = [
    "//rtp_llm:fast-hadamard-transform",
    "//rtp_llm:flash-mla",
    "//rtp_llm:tilelang"
]

rtp_kernel = [
    "//rtp_llm:rtp-kernel"
]

flashattn = [
    "//rtp_llm:flash_attn",
    "//rtp_llm:flash-attn-3"
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
            "//rtp_llm:triton",
        ],
        # deep-gemm: x86 uses 2.2.0/2.1.1, cuda12_9_arm uses the 2.1.1+local aarch64
        # wheel. That version once shipped as both linux_aarch64 and
        # manylinux_2_28_aarch64 (same build re-tagged and repacked; the 838 entries
        # differ only in RECORD/WHEEL); on arm both matched, making the alias fail
        # with "Illegal ambiguous match". The non-canonical-tag copy is now in
        # pip_repo's QUARANTINED_KEYS and off the index page.
        "@//:using_cuda12_9_x86": [
            "//rtp_llm:deep_gemm",
        ],
        "@//:cuda_pre_12_9": [
            "//rtp_llm:deep_gemm",
        ],
        "@//:using_cuda12_arm": [
            "//rtp_llm:deep_gemm",
        ],
        "@//:using_cuda12": [
            "//rtp_llm:deep_gemm",
        ],
        "@//:using_ppu": [
            "//rtp_llm:deep_gemm",
        ],
        "//conditions:default": [],
    }) + select({
        # cuda13 has no pip supply in this tree, so these branches only keep the failure
        # explicit (the //rtp_llm shims route cuda13 to the absence stub); without them
        # cuda13_arm would take the using_cuda12_arm wheels below.
        "@//:using_cuda13_x86": ["//rtp_llm:flash-mla"],
        "@//:using_cuda13_arm": ["//rtp_llm:flash-mla"],
        # The cuda12_9_arm lock contains flashinfer-python 0.6.9 + rtp-kernel aarch64
        # wheels; jit-cache/cutlass-dsl truly only have x86_64 wheels (absent), and
        # the flash-attn family is carried by the main package, not referenced by
        # standalone. cubin must be included: it is py3-none-any, the same artifact
        # as on x86, and it supplies the fmha_gen JIT's flashInferMetaInfo.h; without
        # it compilation is guaranteed to fail in CI (no internet) -- same criteria
        # as the arm branch in //rtp_llm:BUILD.
        "@//:using_cuda12_9_x86": flashinfer_with_cache + flashmla,
        "@//:using_cuda12_arm": flashinfer + ["//rtp_llm:flashinfer-cubin"],
        "@//:cuda_pre_12_9": flashinfer,
        "//conditions:default": []
    }) + select({
        "@//:using_cuda13_x86": rtp_kernel,
        "@//:using_cuda13_arm": rtp_kernel,
        "@//:using_cuda12_9_x86": rtp_kernel,
        "@//:using_cuda12_arm": rtp_kernel,
        "@//:cuda_pre_12_9": [],
        "//conditions:default": []
    }) + select({
        "@//:using_cuda13_x86": [],
        "@//:using_cuda13_arm": [],
        "@//:using_cuda12_9_x86": flashattn,
        "@//:cuda_pre_12_9": [],
        "//conditions:default": []
    }) + internal_deps(),
    data = [
        "//:th_transformer_config",
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)
