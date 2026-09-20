load("//bazel:arch_select.bzl", "requirement")

requirement(["torch", "triton"])

py_test(
    name = "cached_launch_compat_test",
    srcs = ["cached_launch_compat_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "cached_launch_test",
    srcs = ["cached_launch_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":torch",
        ":triton",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20", "gpu_count": "1"},
    tags = ["manual", "H20"],
)

[
    py_test(
        name = "projection_ktp_nccl_%d_test" % ranks,
        srcs = ["projection_ktp_nccl_test.py"],
        main = "projection_ktp_nccl_test.py",
        deps = [
            "//rtp_llm/models_py:modules",
            "//rtp_llm/models_py/triton_kernels:kimi_kda",
            ":torch",
            ":triton",
        ],
        env = {"GPU_COUNT": str(ranks)},
        exec_properties = {"gpu": "H20", "gpu_count": str(ranks)},
        tags = ["manual", "H20"],
        timeout = "long",
    )
    for ranks in [1, 2, 4, 8]
]

py_test(
    name = "moe_decode_small_kernel_test",
    srcs = ["moe_decode_small_kernel_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":torch",
        ":triton",
    ],
    env = {"GPU_COUNT": "1", "DSV4_MOE_STRICT_FUSED": "0"},
    exec_properties = {"gpu": "H20", "gpu_count": "1"},
    tags = ["manual", "H20"],
)

py_test(
    name = "decode_bf16_producers_test",
    srcs = ["decode_bf16_producers_test.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":torch",
        ":triton",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20", "gpu_count": "1"},
    tags = ["manual", "H20"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "projection_ktp_small_kernel_test",
    srcs = ["projection_ktp_small_kernel_test.py"],
    deps = [
        "//rtp_llm/model_loader:loader",
        "//rtp_llm/models_py:kernels",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":torch",
        ":triton",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20", "gpu_count": "1"},
    tags = ["manual", "H20"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "attn_res_test",
    srcs = ["attn_res_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":torch",
        ":triton",
    ],
    exec_properties = {"gpu": "B300"},
    tags = ["B300"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "rms_norm_gate_test",
    srcs = ["rms_norm_gate_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":torch",
        ":triton",
    ],
    exec_properties = {"gpu": "H20"},
    tags = ["H20"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "short_conv_paged_decode_test",
    srcs = ["short_conv_paged_decode_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":torch",
        ":triton",
    ],
    exec_properties = {"gpu": "H20"},
    tags = ["H20"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "paged_recurrent_layout_test",
    srcs = ["paged_recurrent_layout_test.py"],
    deps = [
        "//rtp_llm:models",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":torch",
    ],
    exec_properties = {"gpu": "H20"},
    tags = ["H20"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "fused_paged_prefill_test",
    srcs = ["fused_paged_prefill_test.py"],
    deps = [
        "//rtp_llm:nvidia-cutlass-dsl",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":torch",
        ":triton",
    ],
    exec_properties = {"gpu": "B300"},
    tags = ["B300"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "fp8_quant_test",
    srcs = ["fp8_quant_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":torch",
        ":triton",
    ],
    exec_properties = {"gpu": "B300"},
    tags = ["B300"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "producer_fp8_test",
    srcs = ["producer_fp8_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["manual", "B300"],
    exec_properties = {"gpu": "B300", "gpu_count": "1"},
)

py_test(
    name = "collective_fp8_test",
    srcs = ["collective_fp8_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib", "//rtp_llm/models_py/modules/hybrid/test:collective_gemm_reference"],
    tags = ["manual", "B300"],
    exec_properties = {"gpu": "B300"},
)

py_binary(
    name = "producer_fp8_benchmark",
    srcs = ["producer_fp8_benchmark.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual", "B300"],
)

py_test(
    name = "dynamic_length_fp8_test",
    srcs = ["dynamic_length_fp8_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual", "B300"],
    exec_properties = {"gpu": "B300"},
)
