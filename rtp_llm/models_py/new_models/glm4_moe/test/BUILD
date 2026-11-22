py_library(
    name = "checkpoint_test_support",
    testonly = True,
    srcs = ["checkpoint_fixture.py", "load_config_fixture.py"],
    deps = ["//rtp_llm/models_py:new_weight_loader"],
    visibility = ["//rtp_llm/models_py/new_models/deepseek_vl2/test:__pkg__"],
)

py_test(
    name = "test_glm4_moe_load",
    srcs = ["test_glm4_moe_load.py"],
    main = "test_glm4_moe_load.py",
    data = ["//:rtp_compute_ops", "//:th_transformer_config"],
    deps = [
        ":checkpoint_test_support",
        "//rtp_llm:config",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

[
    py_test(
        name = test_name,
        srcs = [test_name + ".py", "checkpoint_fixture.py"],
        main = test_name + ".py",
        env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""},
        deps = ["//rtp_llm/models_py:new_weight_loader"],
    )
    for test_name in ["test_main_decoder", "test_main_experts"]
]
