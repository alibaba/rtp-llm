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
        "MM_CACHE_ITEM_NUM": "0"
    },
    exec_properties = {'gpu':'A10'},
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
    name = "vit_metrics_test",
    srcs = [
        "vit_metrics_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "qwen35_prefix_detection_test",
    srcs = [
        "qwen35_prefix_detection_test.py",
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
    name = "mm_output_transport_test",
    srcs = [
        "mm_output_transport_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
)

# This target covers the production shared-library and Python/C++ config boundary.
# The real Barex multi-GPU export -> read -> release smoke lives in internal_source.
py_test(
    name = "mm_rdma_exporter_library_test",
    srcs = [
        "mm_rdma_exporter_library_test.py",
    ],
    data = [
        "//rtp_llm/libs:libmm_rdma_exporter_so",
        "//rtp_llm/libs:librtp_compute_ops_so",
        "//rtp_llm/libs:libth_transformer_config_so",
    ],
    deps = [
        "//rtp_llm:config_ops",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "qwen3_video_preprocess_test",
    srcs = ["qwen3_video_preprocess_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "qwen35_video_batch_test",
    srcs = ["qwen35_video_batch_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "qwen35_gpu_video_test",
    srcs = ["qwen35_gpu_video_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "qwen35_vision_kernels_test",
    srcs = ["qwen35_vision_kernels_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "mm_embedding_cache_test",
    srcs = ["mm_embedding_cache_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "qwen35_vision_graph_test",
    srcs = ["qwen35_vision_graph_test.py"],
    deps = ["//rtp_llm:testlib"],
)



py_test(
    name = "mm_feature_hash_test",
    srcs = ["mm_feature_hash_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "qwen35_vit_test",
    srcs = ["qwen35_vit_test.py"],
    deps = ["//rtp_llm:testlib"],
)
