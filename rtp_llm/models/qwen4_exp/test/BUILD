py_test(
    name = "qwen4_exp_test",
    srcs = ["qwen4_exp_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "qwen4_exp_mtp_component_test",
    srcs = ["qwen4_exp_mtp_component_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "real_ckpt_mapping_test",
    srcs = ["real_ckpt_mapping_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "kv_cache_test",
    srcs = ["kv_cache_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "ple_weight_loading_test",
    srcs = ["ple_weight_loading_test.py"],
    deps = ["//rtp_llm:testlib"],
)

# Opt-in real-checkpoint test.  The released PLE table is 102.4 GB, so this
# target requires all eight H20s and skips unless QWEN4_EXP_CKPT_DIR points at
# the complete 131-shard checkpoint.
py_test(
    name = "ple_real_tp8_loading_test",
    srcs = ["ple_real_tp8_loading_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "8"},
    timeout = "eternal",
    tags = [
        "H20",
        "open_skip",
    ],
    exec_properties = {
        "gpu": "H20",
        "gpu_count": "8",
    },
)
