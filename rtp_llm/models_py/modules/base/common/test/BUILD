
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
    name = "kvcache_store_test",
    srcs = ["kvcache_store_test.py"],
    deps = py_test_deps,
    exec_properties = {"gpu": "L20D_DEV"},
)
