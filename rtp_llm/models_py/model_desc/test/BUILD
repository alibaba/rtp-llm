py_test(
    name = "attention_input_routing_test",
    srcs = ["attention_input_routing_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)

py_test(
    name = "attention_input_routing_test_rocm",
    srcs = ["attention_input_routing_test.py"],
    main = "attention_input_routing_test.py",
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["rocm"],
    exec_properties = {'gpu': 'MI308X-ROCM7', 'gpu_count': '1'},
)

py_test(
    name = "qwen3_next_gdn_decode_routing_test",
    srcs = ["qwen3_next_gdn_decode_routing_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm/models_py/triton_kernels:aiter_flydsl_decode",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)

py_test(
    name = "qwen3_next_gdn_decode_routing_test_rocm",
    srcs = ["qwen3_next_gdn_decode_routing_test.py"],
    main = "qwen3_next_gdn_decode_routing_test.py",
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm/models_py/triton_kernels:aiter_flydsl_decode",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["rocm"],
    exec_properties = {'gpu': 'MI308X-ROCM7', 'gpu_count': '1'},
)

py_test(
    name = "qwen3_next_cache_store_test",
    srcs = ["qwen3_next_cache_store_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)

py_test(
    name = "qwen3_next_cache_store_test_rocm",
    srcs = ["qwen3_next_cache_store_test.py"],
    main = "qwen3_next_cache_store_test.py",
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["rocm"],
    exec_properties = {'gpu': 'MI308X-ROCM7', 'gpu_count': '1'},
)

py_test(
    name = "qwen3_next_qkvz_ba_fusion_test",
    srcs = ["qwen3_next_qkvz_ba_fusion_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)

py_test(
    name = "qwen3_next_qkvz_ba_fusion_test_rocm",
    srcs = ["qwen3_next_qkvz_ba_fusion_test.py"],
    main = "qwen3_next_qkvz_ba_fusion_test.py",
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["rocm"],
    exec_properties = {'gpu': 'MI308X-ROCM7', 'gpu_count': '1'},
)
