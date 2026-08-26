bert_newloader_test_deps = [
    "//rtp_llm:_ft_pickler",
    "//rtp_llm:config",
    "//rtp_llm:numpy",
    "//rtp_llm:safetensors",
    "//rtp_llm:torch",
    "//rtp_llm/models_py:new_weight_loader",
]

py_test(
    name = "bert_newloader_integration_test",
    srcs = ["bert_newloader_test.py"],
    main = "bert_newloader_test.py",
    args = [
        "BertNewloaderTest.test_direct_model_tree_forward_matches_torch_reference",
        "BertNewloaderTest.test_linear_layers_consume_native_checkpoint_orientation",
        "BertNewloaderTest.test_missing_token_type_embedding_uses_device_local_zeros",
        "BertNewloaderTest.test_real_bert_load_owns_checkpoint_parameters_in_model_tree",
        "BertNewloaderTest.test_registry_resolves_bert_and_roberta",
        "BertNewloaderTest.test_roberta_and_layernorm_gamma_beta_are_supported",
        "BertNewloaderTest.test_tp2_loads_rank_local_attention_and_ffn_shards",
    ],
    deps = bert_newloader_test_deps,
)

py_test(
    name = "bert_newloader_integrity_test",
    srcs = ["bert_newloader_test.py"],
    main = "bert_newloader_test.py",
    args = [
        "BertNewloaderTest.test_attention_config_invariants_fail_fast",
        "BertNewloaderTest.test_custom_weights_use_exact_declared_contract",
        "BertNewloaderTest.test_duplicate_layernorm_alias_fails",
        "BertNewloaderTest.test_invalid_runtime_config_values_fail_fast",
        "BertNewloaderTest.test_missing_required_weight_fails_before_postprocess",
        "BertNewloaderTest.test_quantized_checkpoints_are_rejected_until_adapted",
        "BertNewloaderTest.test_ep_is_rejected",
        "BertNewloaderTest.test_unknown_floating_tensor_is_not_silently_accepted",
        "BertNewloaderTest.test_wrong_model_prefix_is_rejected",
    ],
    deps = bert_newloader_test_deps,
)

test_suite(
    name = "bert_newloader_test",
    tests = [
        ":bert_newloader_integration_test",
        ":bert_newloader_integrity_test",
    ],
)
