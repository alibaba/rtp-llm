load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")

py_test(
    name = "dsv4_offload_gpu_test",
    srcs = ["dsv4_offload_gpu_test.py"],
    data = [
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_compressor_fp8_per_token.py",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_flash_mla_decode_kernel_layout.py",
    ],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13"},
)

cuda13_sm100_py_test(
    name = "dsv4_offload_mega_gpu_test",
    srcs = ["dsv4_offload_mega_gpu_test.py", "dsv4_offload_gpu_test.py"],
    data = [
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_compressor_fp8_per_token.py",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_flash_mla_decode_kernel_layout.py",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_mega_csa_rtp_eager.py",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:mega_attention_test_utils.py",
    ],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:deep_gemm",
    ],
    timeout = "long",
)

py_test(
    name = "glm_indexshare_test",
    srcs = ["glm_indexshare_test.py"],
    deps = ["//rtp_llm:testlib"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

py_test(
    name = "sparse_kv_offload_gpu_test",
    srcs = ["sparse_kv_offload_gpu_test.py"],
    data = ["//rtp_llm/cpp/cache/test:cache_config_creator_py_test"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13"},
)

test_suite(
    name = "sparse_kv_offload_unit_tests",
    tests = [
        ":dsv4_offload_gpu_test",
        ":dsv4_offload_mega_gpu_test",
        ":dsv4_offload_mega_gpu_test_x86",
        ":glm_indexshare_test",
        ":sparse_kv_offload_gpu_test",
        "//rtp_llm/config/test:kv_cache_config_pickle_test",
        "//rtp_llm/cpp/cache/block_tree_cache/block_pool/test:device_block_pool_test",
        "//rtp_llm/cpp/cache/test:kv_cache_manager_test",
        "//rtp_llm/cpp/cache/test:tiered_cache_config_test",
        "//rtp_llm/cpp/models/test:model_data_test",
        "//rtp_llm/cpp/normal_engine/test:engine_test",
        "//rtp_llm/models_py/model_desc/test:attention_input_routing_test",
        "//rtp_llm/models_py/model_desc/test:hidden_state_capture_test",
        "//rtp_llm/models_py/modules/hybrid/test:mla_attention_test",
    ],
)
