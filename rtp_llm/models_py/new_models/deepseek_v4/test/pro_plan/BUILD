py_test(
    name = "pro_main_release_test",
    srcs = ["pro_main_release_test.py"],
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
    },
    env_inherit = ["DSV4_PRO_SOURCE"],
    tags = ["manual"],
    timeout = "long",
    deps = ["//rtp_llm:models", "//rtp_llm/models_py:new_weight_loader"],
)
