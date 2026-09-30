
py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test (
    name = "embeding_test",
    srcs = ["embeding_test.py"],
    deps = py_test_deps,
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "embedding_tp_test",
    srcs = ["embedding_tp_test.py"],
    deps = py_test_deps,
    tags = ["multi_device"],
    exec_properties = {"gpu": "H20"},
)
