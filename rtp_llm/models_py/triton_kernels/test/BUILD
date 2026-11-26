load("@arch_config//:arch_select.bzl", "requirement")

requirement(["numpy", "packaging", "torch", "triton"])

py_test(
    name = "block_index_to_global_test",
    srcs = ["block_index_to_global_test.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
    tags = ["SM103_CU13", "glm53_contract"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:sparse_mla",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "address_overflow_gpu_test",
    srcs = ["address_overflow_gpu_test.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
    tags = ["manual", "SM103_CU13"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:common",
        "//rtp_llm/models_py/triton_kernels:fla",
        "//rtp_llm/models_py/triton_kernels:kda_fp32_state",
        "//rtp_llm/models_py/triton_kernels:sparse_mla",
        ":numpy",
        ":packaging",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "address_overflow_test",
    srcs = ["address_overflow_test.py"],
    env = {"GPU_COUNT": "0", "TRITON_INTERPRET": "1"},
    deps = [
        "//rtp_llm/models_py:dsv4_fused_kernel_sources",
        "//rtp_llm/models_py/triton_kernels:causal_conv1d",
        "//rtp_llm/models_py/triton_kernels:common",
        "//rtp_llm/models_py/triton_kernels:fla",
        "//rtp_llm/models_py/triton_kernels:kda_fp32_state",
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        "//rtp_llm/models_py/triton_kernels:sparse_mla",
        ":numpy",
        ":packaging",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "legacy_fp8_indexer_score_test",
    srcs = ["legacy_fp8_indexer_score_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
    tags = ["SM100_ARM_CU13"],
)
py_test(
    name = "linear_replay_test",
    srcs = ["linear_replay_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:triton_kernels",
        "//rtp_llm:torch",
        "//rtp_llm:triton",
    ],
    exec_properties = {"gpu": "H20_CU13"},
    tags = ["H20_CU13"],
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//visibility:public"],
)
