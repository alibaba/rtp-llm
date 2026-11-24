py_test(
    name = "hc_sconv_ple_test",
    srcs = ["test_hc_sconv_ple.py"],
    main = "test_hc_sconv_ple.py",
    deps = [
        "//rtp_llm/models_py:modules",
    ],
)

py_test(
    name = "latent_moe_mxfp4_test",
    srcs = ["test_latent_moe_mxfp4.py"],
    main = "test_latent_moe_mxfp4.py",
    deps = [
        "//rtp_llm/models_py:modules",
    ],
)

py_test(
    name = "attn_gdn_yoco_test",
    srcs = ["test_attn_gdn_yoco.py"],
    main = "test_attn_gdn_yoco.py",
    deps = [
        "//rtp_llm/models_py:modules",
    ],
)

py_test(
    name = "e2e_qwen4_5_model_test",
    srcs = ["test_e2e_qwen4_5_model.py"],
    main = "test_e2e_qwen4_5_model.py",
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm/models_py:modules",
    ],
)

py_test(
    name = "qwen4_5_pd_serving_test",
    srcs = ["test_qwen4_5_pd_serving.py", "test_e2e_qwen4_5_model.py"],
    main = "test_qwen4_5_pd_serving.py",
    env = {"GPU_COUNT": "2"},
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:models",
        "//rtp_llm/models_py:modules",
    ],
)

py_test(
    name = "qwen4_5_accuracy_bench",
    srcs = ["bench_qwen4_5_accuracy.py", "test_e2e_qwen4_5_model.py"],
    main = "bench_qwen4_5_accuracy.py",
    env = {"GPU_COUNT": "2"},
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:models",
        "//rtp_llm/models_py:modules",
    ],
)
