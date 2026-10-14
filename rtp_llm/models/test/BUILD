_DSV4_CUDA13_ONLY = select({
    "@//:using_cuda13_x86": [],
    "@//:using_cuda13_arm": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

py_test(
    name = "dsv4_kv_cache_test",
    srcs = ["dsv4_kv_cache_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "deepseek_v4_dspark_commit_only_test",
    srcs = ["deepseek_v4_dspark_commit_only_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "moe_config_propagation_test",
    srcs = ["moe_config_propagation_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "moe_cuda13_config_propagation_test",
    srcs = ["moe_cuda13_config_propagation_test.py"],
    deps = ["//rtp_llm:testlib"],
    target_compatible_with = _DSV4_CUDA13_ONLY,
    # Initializes CUDA at import: pin to a GPU test pool now that unpinned
    # cuda13 actions default to the CPU worker pool. Every branch carries a
    # real pool value — an empty-dict default is NOT a GPU request and lands
    # the test on a CPU worker (see 3rdparty/trt_beam_search/tests/BUILD).
    exec_properties = select({
        "@//:using_cuda13_x86": {"gpu": "L20_CU13", "gpu_count": "1"},
        "@//:using_cuda13_arm": {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
        "//conditions:default": {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
    }),
)

py_test(
    name = "gpt_oss_config_test",
    srcs = ["gpt_oss_config_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "gpt_oss_math_test",
    srcs = ["gpt_oss_math_test.py"],
    deps = ["//rtp_llm:testlib"],
    target_compatible_with = _DSV4_CUDA13_ONLY,
    # triton kernel test: pin to a real GPU pool
    exec_properties = select({
        "@//:using_cuda13_x86": {"gpu": "SM103_CU13", "gpu_count": "1"},
        "@//:using_cuda13_arm": {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
        "//conditions:default": {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
    }),
)

py_test(
    name = "gpt_oss_weights_compare_test",
    srcs = ["gpt_oss_weights_compare_test.py"],
    deps = ["//rtp_llm:testlib"],
    target_compatible_with = _DSV4_CUDA13_ONLY,
    # loads the real 13.8GB checkpoint; skips when GPT_OSS_CKPT is absent
    exec_properties = select({
        "@//:using_cuda13_x86": {"gpu": "SM103_CU13", "gpu_count": "1"},
        "@//:using_cuda13_arm": {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
        "//conditions:default": {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
    }),
)

py_test(
    name = "gpt_oss_moe_synth_test",
    srcs = ["gpt_oss_moe_synth_test.py"],
    deps = ["//rtp_llm:testlib"],
    target_compatible_with = _DSV4_CUDA13_ONLY,
    exec_properties = select({
        "@//:using_cuda13_x86": {"gpu": "SM103_CU13", "gpu_count": "1"},
        "@//:using_cuda13_arm": {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
        "//conditions:default": {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
    }),
)

py_test(
    name = "gpt_oss_bf16_moe_test",
    srcs = ["gpt_oss_bf16_moe_test.py"],
    deps = ["//rtp_llm:testlib"],
    target_compatible_with = _DSV4_CUDA13_ONLY,
    exec_properties = select({
        "@//:using_cuda13_x86": {"gpu": "SM103_CU13", "gpu_count": "1"},
        "@//:using_cuda13_arm": {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
        "//conditions:default": {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
    }),
)

py_test(
    name = "gpt_oss_sink_fallback_test",
    srcs = ["gpt_oss_sink_fallback_test.py"],
    deps = ["//rtp_llm:testlib"],
)
