package(default_visibility = ["//visibility:public"])

py_test(
    name = "qwen4_exp_test",
    srcs = ["qwen4_exp_test.py"],
    env = {"GPU_COUNT": "0"},
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "moe_fp8_test",
    srcs = ["moe_fp8_test.py"],
    env = {"GPU_COUNT": "0"},
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "qwen4_exp_mtp_component_test",
    srcs = ["qwen4_exp_mtp_component_test.py"],
    env = {"GPU_COUNT": "0"},
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "real_ckpt_mapping_test",
    srcs = ["real_ckpt_mapping_test.py"],
    data = [
        "reference/config.json",
        "reference/model.safetensors.index.json",
    ],
    env = {"GPU_COUNT": "0"},
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "kv_cache_test",
    srcs = ["kv_cache_test.py"],
    env = {"GPU_COUNT": "0"},
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "ple_weight_loading_test",
    srcs = ["ple_weight_loading_test.py"],
    env = {"GPU_COUNT": "0"},
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "moe_fp4_test",
    srcs = ["moe_fp4_test.py"],
    env = {"GPU_COUNT": "0"},
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "full_fp8_test",
    srcs = ["full_fp8_test.py"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
    tags = ["RTX_5000_PRO_CU13"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "runtime_snapshot_test",
    srcs = ["runtime_snapshot_test.py"],
    env = {"GPU_COUNT": "0"},
    deps = ["//rtp_llm:testlib"],
)

test_suite(
    name = "qwen4_config",
    tests = [
        ":kv_cache_test",
        ":moe_fp4_test",
        ":moe_fp8_test",
        ":ple_weight_loading_test",
        ":qwen4_exp_mtp_component_test",
        ":qwen4_exp_test",
        ":real_ckpt_mapping_test",
        ":runtime_snapshot_test",
    ],
)
