py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "fused_moe_test",
    srcs = ["fused_moe_test.py"],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = py_test_deps,
)

py_test(
    name = "moe_ep_reorder_test",
    srcs = ["moe_ep_reorder_test.py"],
    exec_properties = {"gpu": "A10_CU13"},
    tags = [
        "A10_CU13",
        "open_skip",
    ],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm:utils",
        "//rtp_llm/models_py:models",
        "//rtp_llm/test/model_test/test_util",
    ],
)
