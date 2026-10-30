load("@arch_config//:arch_select.bzl", "requirement")
load("@pip_gpu_cuda13_torch//:requirements.bzl", cuda13_requirement = "requirement")
load("@pip_cuda13_arm_torch//:requirements.bzl", cuda13_arm_requirement = "requirement")

requirement([
    "triton",
    "torch",
    "numpy",
    "einops",
    "packaging",
])

py_test(
    name = "test_kda_ops",
    srcs = [
        "test_kda_ops.py",
    ],
    deps = [
        "//rtp_llm/models_py:utils",
        "//rtp_llm/models_py/triton_kernels:fla",
        "//rtp_llm/models_py/triton_kernels:kda",
    ] + [
        ":einops",
        ":numpy",
        ":packaging",
        ":torch",
        ":triton",
    ] + select({
        "@//:using_cuda13_x86": [
            cuda13_requirement("cuda-linear-attention"),
            "//rtp_llm:nvidia-cutlass-dsl",
            "//rtp_llm:tilelang",
        ],
        "@//:using_cuda13_arm": [
            cuda13_arm_requirement("cuda-linear-attention"),
            "//rtp_llm:nvidia-cutlass-dsl",
            "//rtp_llm:tilelang",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    # cuLA prefill requires SM100/SM103. The ARM lane supplies the matching
    # aarch64 wheel and runs both prefill and prefill/decode consistency.
    exec_properties = {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
    tags = ["SM100_ARM_CU13"],
)

py_test(
    name = "test_kda_decode",
    srcs = ["test_kda_ops.py"],
    main = "test_kda_ops.py",
    args = ["TestKdaOps.test_decode_vs_naive"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
        "//rtp_llm/models_py/triton_kernels:kda",
        ":einops",
        ":numpy",
        ":packaging",
        ":torch",
        ":triton",
    ],
    visibility = ["//visibility:public"],
    # Recurrent Triton decode does not import cuLA; retain its L20 coverage.
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
    tags = ["L20_CU13"],
)

py_test(
    name = "test_chunk_prefill",
    srcs = ["test_chunk_prefill.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
        "//rtp_llm/models_py/triton_kernels:kda",
        ":torch",
        ":triton",
        ":einops",
        ":numpy",
        ":packaging",
    ],
    visibility = ["//visibility:public"],
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "test_chunk_prefill_h20_cached",
    srcs = ["test_chunk_prefill.py"],
    main = "test_chunk_prefill.py",
    env = {"TRITON_AUTOTUNE_CACHE_MODE": "cached"},
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
        "//rtp_llm/models_py/triton_kernels:kda",
        ":torch",
        ":triton",
        ":einops",
        ":numpy",
        ":packaging",
    ],
    visibility = ["//visibility:public"],
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    tags = ["H20_CU13"],
)
