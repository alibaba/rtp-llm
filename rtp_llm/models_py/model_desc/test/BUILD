py_test(
    name = "qwen3_next_qkvz_ba_fusion_test",
    srcs = ["qwen3_next_qkvz_ba_fusion_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)

# G1 aux-hidden-state capture for the dspark/dflash draft (tiny random
# weights, self-contained).
py_test(
    name = "qwen3_aux_capture_test",
    srcs = ["qwen3_aux_capture_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)

# DSpark draft model vs dense reference golden.  Needs the converted draft
# ckpt + golden dump on local disk (skips gracefully when absent), so it is
# a local/manual gate rather than a CI-portable test.
py_test(
    name = "qwen3_dspark_test",
    srcs = ["qwen3_dspark_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:models",
        "//rtp_llm:async_model",  # rtp_llm.models package import pulls it in
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20", "manual"],
    size = "large",
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)
