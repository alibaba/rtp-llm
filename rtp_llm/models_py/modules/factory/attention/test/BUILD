py_test(
    name = "attn_factory_generation_prefill_cuda_graph_test",
    srcs = ["attn_factory_generation_prefill_cuda_graph_test.py"],
    exec_properties = {
        "gpu": "H20_CU13",
        "gpu_count": "1",
    },
    tags = ["H20_CU13"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "attn_factory_generation_prefill_cuda_graph_test_rocm",
    srcs = ["attn_factory_generation_prefill_cuda_graph_test.py"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
    main = "attn_factory_generation_prefill_cuda_graph_test.py",
    tags = ["rocm"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)
