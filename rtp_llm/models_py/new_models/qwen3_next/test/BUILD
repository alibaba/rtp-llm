load("//rtp_llm/models_py:checkpoint_test_platforms.bzl", "checkpoint_reader_test_env")

py_test(
    name = "hybrid_ft_test",
    srcs = ["hybrid_ft_test.py", "main_checkpoint_test.py"],
    main = "hybrid_ft_test.py",
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm/models_py/new_models/kimi_linear/test:hybrid_ft_fixture",
        "//rtp_llm/models_py/new_models/kimi_k25/test:hybrid_ft_fixture",
        "//rtp_llm/tools/convert:convert",
    ],
)

py_test(
    name = "gdn_parameters_test",
    srcs = ["gdn_parameters_test.py"],
    env = checkpoint_reader_test_env({}),
    deps = ["//rtp_llm/models_py:new_weight_loader"],
)

py_test(
    name = "main_checkpoint_test",
    srcs = ["main_checkpoint_test.py"],
    data = ["//:rtp_compute_ops", "//:th_transformer_config"],
    deps = ["//rtp_llm/models_py:new_weight_loader"],
)

py_test(
    name = "native_service_test",
    srcs = ["native_service_test.py", "main_checkpoint_test.py"],
    main = "native_service_test.py",
    data = ["//:rtp_compute_ops", "//:th_transformer_config"],
    deps = ["//rtp_llm:models", "//rtp_llm/models_py:new_weight_loader"],
)
