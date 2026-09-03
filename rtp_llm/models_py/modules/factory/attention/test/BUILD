py_test(
    name = "attn_factory_prefill_cuda_graph_test",
    srcs = ["attn_factory_prefill_cuda_graph_test.py"],
    tags = ["H20"],
    exec_properties = {
        "gpu": "H20",
        "gpu_count": "1",
    },
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "attn_factory_prefill_cuda_graph_test_rocm",
    main = "attn_factory_prefill_cuda_graph_test.py",
    srcs = ["attn_factory_prefill_cuda_graph_test.py"],
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)
