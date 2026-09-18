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
)

py_test(
    name = "minimax_m3_vl_vit_test",
    srcs = [
        "minimax_m3_vl_vit_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "mm_kvcm_emb_config_test",
    srcs = [
        "mm_kvcm_emb_config_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "mm_kvcm_emb_client_test",
    srcs = [
        "mm_kvcm_emb_client_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
)

# Opt-in cross-repo test. It starts a live KVCM KVMeta service and uses
# the packaged generic client through RTP's public Python facade. KVCM remains
# outside RTP's default dependency graph.
py_test(
    name = "mm_kvcm_emb_client_integration_test",
    timeout = "long",
    srcs = [
        "mm_kvcm_emb_client_integration_test.py",
    ],
    tags = ["manual"],
    deps = [
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "minimax_m3_vl_rope_test",
    srcs = [
        "minimax_m3_vl_rope_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "cuda_graph_gate_test",
    srcs = ["cuda_graph_gate_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "mm_kvcm_tensor_object_test",
    srcs = ["mm_kvcm_tensor_object_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "mm_remote_embedding_cache_test",
    srcs = ["mm_remote_embedding_cache_test.py", "multimodal_process_engine_test.py"],
    data = [":testdata"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {"gpu": "A10"},
)
