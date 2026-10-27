py_test(
    name = "glm_indexshare_test",
    srcs = ["glm_indexshare_test.py"],
    deps = ["//rtp_llm:testlib"],
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
