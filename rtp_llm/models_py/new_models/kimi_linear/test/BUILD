# The CUDA build imports librtp_compute_ops during model construction, which
# requires the NVML runtime even though this test keeps tensors on CPU.
py_test(
    name = "test_kimi_linear_load",
    srcs = ["test_kimi_linear_load.py"],
    exec_properties = {
        "gpu": "H20",
        "gpu_count": "1",
    },
    tags = ["H20"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)
