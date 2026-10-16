load("//bazel:arch_select.bzl", "requirement")

requirement(["numpy", "packaging", "torch", "triton"])



py_test(
    name = "stable_tp_reduce_test",
    srcs = ["stable_tp_reduce_test.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu_count": "1"},
    tags = ["manual"],
    deps = ["//rtp_llm/models_py/triton_kernels:common", ":torch", ":triton"],
)

py_test(
    name = "address_overflow_test",
    srcs = ["address_overflow_test.py"],
    env = {"TRITON_INTERPRET": "1"},
    deps = [
        "//rtp_llm/models_py/triton_kernels:sparse_mla",
        "//rtp_llm/models_py:dsv4_fused_kernel_sources",
        "//rtp_llm/models_py/triton_kernels:causal_conv1d",
        "//rtp_llm/models_py/triton_kernels:common",
        "//rtp_llm/models_py/triton_kernels:fla",
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":numpy",
        ":packaging",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "address_overflow_gpu_test",
    srcs = ["address_overflow_gpu_test.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu_count": "1"},
    tags = ["manual"],  # Actual >2^31-element storage needs at least 9 GiB free.
    deps = [
        "//rtp_llm/models_py/triton_kernels:sparse_mla",
        "//rtp_llm/models_py/triton_kernels:causal_conv1d",
        "//rtp_llm/models_py/triton_kernels:common",
        "//rtp_llm/models_py/triton_kernels:fla",
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        ":packaging",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "address_overflow_gpu_test",
    srcs = ["address_overflow_gpu_test.py"],
    main = "address_overflow_gpu_test.py",
    env = {"GPU_COUNT": "4"},
    exec_properties = {"gpu_count": "4"},
    tags = ["manual"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:causal_conv1d",
        "//rtp_llm/models_py/triton_kernels:common",
        "//rtp_llm/models_py/triton_kernels:fla",
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        "//rtp_llm/models_py/triton_kernels:sparse_mla",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        ":packaging",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "canonicalize_topk_gpu_test",
    srcs = ["canonicalize_topk_test.py"],
    main = "canonicalize_topk_test.py",
    env = {"GPU_COUNT": "4"},
    exec_properties = {"gpu_count": "4"},
    tags = ["manual"],
    deps = ["//rtp_llm/models_py/triton_kernels:common", ":torch", ":triton"],
)

py_test(
    name = "stable_tp_reduce_gpu_test",
    srcs = ["stable_tp_reduce_test.py"],
    main = "stable_tp_reduce_test.py",
    env = {"GPU_COUNT": "4"},
    exec_properties = {"gpu_count": "4"},
    tags = ["manual"],
    deps = ["//rtp_llm/models_py/triton_kernels:common", ":torch", ":triton"],
)

py_test(
    name = "layernorm_gated_stats_gpu_test",
    srcs = ["layernorm_gated_stats_test.py"],
    main = "layernorm_gated_stats_test.py",
    env = {"GPU_COUNT": "4"},
    exec_properties = {"gpu_count": "4"},
    tags = ["manual"],
    deps = ["//rtp_llm/models_py/triton_kernels:common", ":torch", ":triton"],
)
