py_test(
    name = "test_dsv4_kv_cache",
    srcs = ["test_dsv4_kv_cache.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "test_cp_async",
    srcs = ["test_cp_async.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_metadata_triton",
    srcs = ["test_metadata_triton.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "test_compressor_hot_paths",
    srcs = ["test_compressor_hot_paths.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_hc_impl",
    srcs = ["test_hc_impl.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py:tile_kernels_mhc",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_strategy_select",
    srcs = ["test_strategy_select.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_indexer_topk_backend",
    srcs = ["test_indexer_topk_backend.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "test_shared_expert_executor",
    srcs = ["test_shared_expert_executor.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "test_mega_moe_input_packer",
    srcs = ["test_mega_moe_input_packer.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)
