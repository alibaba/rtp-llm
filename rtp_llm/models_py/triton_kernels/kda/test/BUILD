load("@arch_config//:arch_select.bzl", "requirement")

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
        "//rtp_llm/models_py/triton_kernels:fla",
        "//rtp_llm/models_py/triton_kernels:kda",
    ] + [
        ":einops",
        ":numpy",
        ":packaging",
        ":torch",
        ":triton",
    ],
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
