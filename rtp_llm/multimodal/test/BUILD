filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/*",
        "testdata/**",
    ]),
)

py_test(
    name = "multimodal_image_load_test",
    srcs = [
        "multimodal_image_load_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "multimodal_process_engine_test",
    srcs = [
        "multimodal_process_engine_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    data = [
        ":testdata",
    ],
    env = {
        "MM_CACHE_ITEM_NUM": "0",
        "MM_CACHE_GPU_MAX_BYTES": "0",
        "MM_CACHE_CPU_MAX_BYTES": "0",
    },
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "mm_embedding_cache_test",
    srcs = ["mm_embedding_cache_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "multimodal_util_test",
    srcs = [
        "multimodal_util_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    data = [
        ":testdata",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "mm_profiler_test",
    srcs = [
        "mm_profiler_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "mm_scheduler_test",
    srcs = [
        "mm_scheduler_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
)
py_test(
    name = "mm_cancellation_test",
    srcs = ["mm_cancellation_test.py", "multimodal_process_engine_test.py"],
    deps = ["//rtp_llm:testlib"],
    data = [":testdata"],
)

py_test(
    name = "cuda_graph_gate_test",
    srcs = ["cuda_graph_gate_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {"gpu": "A10"},
)
