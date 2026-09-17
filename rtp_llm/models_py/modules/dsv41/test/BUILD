py_test(
    name = "test_decode_draft",
    srcs = ["test_decode_draft.py", "test_draft.py", "fixture.py", "official_compressor.py", "test_compact_writer.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:rtp_llm_lib",
    ],
    data = ["//:rtp_compute_ops"],
)

py_test(
    name = "test_decode_fmha_impl",
    srcs = ["test_decode_fmha_impl.py", "test_attention.py", "official_compressor.py", "test_compact_writer.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:rtp_llm_lib",
    ],
    data = ["//:rtp_compute_ops"],
)

py_test(
    name = "test_decode_fmha_impl_flashmla",
    main = "test_decode_fmha_impl.py",
    srcs = ["test_decode_fmha_impl.py", "test_attention.py", "official_compressor.py", "test_compact_writer.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:rtp_llm_lib",
    ],
    data = ["//:rtp_compute_ops"],
    env = {
        "DSV41_ATTENTION_BACKEND": "flashmla",
    },
)

py_test(
    name = "test_decode_attention",
    srcs = ["test_decode_attention.py", "test_attention.py", "official_compressor.py", "test_compact_writer.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_decode_compressor",
    srcs = ["test_decode_compressor.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_cp_prefill",
    srcs = ["test_cp_prefill.py", "test_cp_attention.py", "fixture.py"],
    deps = ["//rtp_llm:rtp_llm_lib"],
    data = ["//:rtp_compute_ops"],
    env = {"GPU_COUNT": "8"},
    tags = ["manual", "requires-gpu-8"],
    timeout = "long",
)

py_test(
    name = "test_cp_attention",
    srcs = ["test_cp_attention.py"],
    deps = ["//rtp_llm:rtp_llm_lib"],
    data = ["//:rtp_compute_ops"],
    env = {"GPU_COUNT": "8"},
    tags = ["manual", "requires-gpu-8"],
    timeout = "long",
)

py_test(
    name = "test_cp_selected",
    srcs = ["test_cp_selected.py"],
    deps = ["//rtp_llm:rtp_llm_lib"],
    data = ["//:rtp_compute_ops"],
)

py_test(
    name = "test_engine_adapter",
    srcs = ["test_engine_adapter.py", "fixture.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:rtp_llm_lib",
    ],
)

py_test(
    name = "test_flashmla_compact",
    srcs = ["test_flashmla_compact.py", "test_flashmla.py", "official_flashmla.py", "test_compact_reader.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_host_shared_deadline",
    srcs = ["test_host_shared_deadline.py", "test_host_shared_lifecycle.py", "host_cuda_test_support.py", "standalone_load.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_cprr_reader_native",
    main = "test_cprr_reader.py",
    srcs = ["test_cprr_reader.py", "test_compact_reader.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {
        "DSV41_CPRR_READER_BACKEND": "native",
    },
)

py_test(
    name = "test_cprr_reader_flashmla",
    main = "test_cprr_reader.py",
    srcs = ["test_cprr_reader.py", "test_compact_reader.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {
        "DSV41_CPRR_READER_BACKEND": "flashmla",
    },
)

py_test(
    name = "test_generation_lifecycle",
    srcs = ["test_generation_lifecycle.py", "test_generation.py", "test_prefill.py", "test_attention.py", "test_transformer.py", "fixture.py", "official_compressor.py", "test_compact_writer.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:models",
    ],
)

py_test(
    name = "test_generation",
    srcs = ["test_generation.py", "test_prefill.py", "test_attention.py", "test_transformer.py", "fixture.py", "official_compressor.py", "test_compact_writer.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:models",
    ],
)

py_test(
    name = "test_snapshot_metadata_tail",
    srcs = ["test_snapshot_metadata_tail.py", "test_prefill.py", "test_attention.py", "test_transformer.py", "fixture.py", "official_compressor.py", "test_compact_writer.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:models",
    ],
)

py_test(
    name = "test_prefill_short_checkpoint",
    srcs = ["test_prefill_short_checkpoint.py", "test_prefill.py", "test_attention.py", "test_transformer.py", "fixture.py", "official_compressor.py", "test_compact_writer.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:models",
    ],
)

py_test(
    name = "test_compact_writer_alias",
    srcs = ["test_compact_writer_alias.py", "test_compact_writer.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_l20_tail_storage",
    srcs = ["test_l20_tail_storage.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_prefill",
    srcs = ["test_prefill.py", "test_attention.py", "test_transformer.py", "fixture.py", "official_compressor.py", "test_compact_writer.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:models",
    ],
)

py_test(
    name = "test_flashmla_multikey",
    srcs = ["test_flashmla_multikey.py", "test_flashmla.py", "official_flashmla.py", "test_compact_reader.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_flashmla",
    srcs = ["test_flashmla.py", "official_flashmla.py", "test_compact_reader.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_deepselect",
    srcs = ["test_deepselect.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_attention",
    srcs = ["test_attention.py", "official_compressor.py", "test_compact_writer.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_indexer",
    srcs = ["test_indexer.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_compressor",
    srcs = ["test_compressor.py", "official_compressor.py", "test_compact_writer.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
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
    name = "test_engram_gather_quant",
    srcs = ["test_engram_gather_quant.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_compact_writer",
    srcs = ["test_compact_writer.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_compact_reader",
    srcs = ["test_compact_reader.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
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
    name = "test_contract_lint",
    srcs = ["test_contract_lint.py", "standalone_load.py"],
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
)

py_test(
    name = "test_moe",
    srcs = ["test_moe.py", "fixture.py", "official_compressor.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["requires-dsv41-component-inputs"],
)

py_test(
    name = "test_engram_binding",
    srcs = ["test_engram_binding.py", "host_cuda_test_support.py", "standalone_load.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:models",
    ],
    tags = ["requires-full-engram-host-memory"],
    timeout = "long",
)

py_test(
    name = "test_target_binding",
    srcs = ["test_target_binding.py", "test_transformer.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
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
