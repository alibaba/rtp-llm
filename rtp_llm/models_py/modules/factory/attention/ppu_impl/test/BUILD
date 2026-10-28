py_test(
    name = "test_cascade_candidate_comparison",
    srcs = [
        "test_cascade_candidate_comparison.py",
        "test_cascade_kernel_benchmark.py",
    ],
    main = "test_cascade_candidate_comparison.py",
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual", "PPU-ZW810E"],
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
)

py_test(
    name = "test_cascade_acu_profile",
    srcs = [
        "test_cascade_acu_profile.py",
        "test_cascade_kernel_benchmark.py",
    ],
    main = "test_cascade_acu_profile.py",
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual", "PPU-ZW810E"],
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
)

py_test(
    name = "test_cascade_kernel_benchmark",
    srcs = ["test_cascade_kernel_benchmark.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual", "PPU-ZW810E"],
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
)

py_test(
    name = "test_beam_search_cascade",
    srcs = ["test_beam_search_cascade.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual", "PPU-ZW810E"],
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
)

py_test(
    name = "test_full_prefill_graph",
    srcs = ["test_full_prefill_graph.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual", "PPU-ZW810E"],
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
)
