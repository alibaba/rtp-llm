py_test(
    name = "test_prefill",
    srcs = ["test_prefill.py", "test_attention.py", "test_transformer.py", "fixture.py", "official_compressor.py", "test_compact_writer.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {
        "DSV41_ATTENTION": "1",
        "DSV41_OWNER_COMPRESSOR": "1",
        "DSV41_SPARSE_INDEXER": "1",
        "DSV41_NATIVE_COMPACT_READER": "1",
        "DSV41_NATIVE_COMPACT_WRITER": "1",
    },
)

py_test(
    name = "test_flashmla",
    srcs = ["test_flashmla.py", "official_flashmla.py", "test_compact_reader.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {"DSV41_FLASHMLA": "1"},
)

py_test(
    name = "test_deepselect",
    srcs = ["test_deepselect.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {"DSV41_DEEPSELECT": "1"},
)

py_test(
    name = "test_attention",
    srcs = ["test_attention.py", "official_compressor.py", "test_compact_writer.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {
        "DSV41_ATTENTION": "1",
        "DSV41_OWNER_COMPRESSOR": "1",
        "DSV41_SPARSE_INDEXER": "1",
        "DSV41_NATIVE_COMPACT_READER": "1",
        "DSV41_NATIVE_COMPACT_WRITER": "1",
    },
)

py_test(
    name = "test_indexer",
    srcs = ["test_indexer.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {
        "DSV41_SPARSE_INDEXER": "1",
        "DSV41_NATIVE_COMPACT_WRITER": "1",
    },
)

py_test(
    name = "test_compressor",
    srcs = ["test_compressor.py", "official_compressor.py", "test_compact_writer.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {
        "DSV41_OWNER_COMPRESSOR": "1",
        "DSV41_NATIVE_COMPACT_READER": "1",
        "DSV41_NATIVE_COMPACT_WRITER": "1",
    },
)

py_library(
    name = "host_cuda_probe_support",
    srcs = ["host_cuda_test_support.py", "standalone_load.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_host_shared_cuda",
    srcs = ["test_host_shared_cuda.py", "host_cuda_test_support.py", "standalone_load.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["requires-two-gpus"],
    env = {"GPU_COUNT": "2"},
)

py_test(
    name = "test_compact_writer",
    srcs = ["test_compact_writer.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {
        "DSV41_NATIVE_COMPACT_READER": "1",
        "DSV41_NATIVE_COMPACT_WRITER": "1",
    },
)

py_test(
    name = "test_compact_reader",
    srcs = ["test_compact_reader.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {"DSV41_NATIVE_COMPACT_READER": "1"},
)

py_test(
    name = "test_cache_layout",
    srcs = ["test_cache_layout.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_ced",
    srcs = ["test_ced.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_config_manifest",
    srcs = ["test_config_manifest.py", "fixture.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_math",
    srcs = ["test_math.py", "standalone_load.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_linear",
    srcs = ["test_linear.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {"DSV41_BLOCK32_LINEAR": "1"},
)

py_test(
    name = "test_transformer",
    srcs = ["test_transformer.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_draft",
    srcs = ["test_draft.py", "fixture.py", "official_compressor.py", "test_compact_writer.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {
        "DSV41_BLOCK32_LINEAR": "1",
        "DSV41_NATIVE_COMPACT_WRITER": "1",
    },
)

py_test(
    name = "test_inputs",
    srcs = ["test_inputs.py", "standalone_load.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_engram_history",
    srcs = ["test_engram_history.py", "standalone_load.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_host_shared_weights",
    srcs = ["test_host_shared_weights.py", "standalone_load.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_host_shared_lifecycle",
    srcs = ["test_host_shared_lifecycle.py", "host_cuda_test_support.py", "standalone_load.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["requires-two-gpus"],
    env = {"GPU_COUNT": "2"},
)
