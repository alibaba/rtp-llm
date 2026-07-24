# Dynamic decode attention backend dispatch tests. The aggregate deliberately
# owns the production attention import closure and exercises real CUDA benchmark
# behavior; pure policy and mocked orchestration regressions run inside the same
# established native boundary rather than creating GPU-pinned CPU-only targets.

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "test_backend_bench",
    srcs = [
        "test_backend_bench.py",
        "test_backend_selector.py",
        "test_selector.py",
    ],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)
