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
    name = "decision_module_test",
    srcs = ["decision_module_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "decision_sources_test",
    srcs = ["decision_sources_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "ffn_lora_wrap_test",
    srcs = ["ffn_lora_wrap_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "qkv_lora_wrap_test",
    srcs = ["qkv_lora_wrap_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "linear_attn_lora_wrap_test",
    srcs = ["linear_attn_lora_wrap_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "decider_label_cache_test",
    srcs = ["decider_label_cache_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "clef_decision_module_test",
    srcs = ["clef_decision_module_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "decider_decision_test",
    srcs = ["decider_decision_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "qwen35_dense_config_test",
    srcs = ["qwen35_dense_config_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "qwen35_moe_flat_config_test",
    srcs = ["qwen35_moe_flat_config_test.py"],
    deps = ["//rtp_llm:testlib"],
)
