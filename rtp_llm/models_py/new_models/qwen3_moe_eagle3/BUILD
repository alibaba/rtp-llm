load("//rtp_llm/models_py:checkpoint_test_platforms.bzl", "checkpoint_reader_test_env")

py_library(
    name = "draft_body",
    srcs = ["__init__.py", "body.py", "ft_checkpoint.py", "proposal.py", "service.py"],
    deps = ["//rtp_llm/models_py:new_loader_core", "//rtp_llm:torch", "//rtp_llm/tools/convert:ft_export_writer"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "draft_body_test",
    srcs = ["draft_body_test.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    deps = [":draft_body", "//rtp_llm:safetensors"],
)

py_test(
    name = "proposal_test",
    srcs = ["draft_body_test.py", "proposal_test.py"],
    main = "proposal_test.py",
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    deps = [":draft_body", "//rtp_llm:safetensors"],
)

# Rank-local checkpoint/loan assertions need no compute binding; the full
# output contract deliberately constructs real C++ request and result objects.
[
    py_test(
        name = name,
        srcs = ["draft_body_test.py", "service_test.py"],
        main = "service_test.py",
        args = cases,
        env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}),
        tags = tags,
        exec_properties = properties,
        deps = [":draft_body", "//rtp_llm:model_factory", "//rtp_llm:models", "//rtp_llm:sdk", "//rtp_llm:transformers", "//rtp_llm/models_py:new_weight_loader", "//rtp_llm:safetensors", "//rtp_llm/tools/convert:convert"],
    )
    for name, cases, tags, properties in [
        ("service_test", [
            "ServiceTest.test_tp2_scratch_shards",
            "ServiceTest.test_tp2_fst_shards",
            "ServiceTest.test_tp2_ft_roundtrip",
            "ServiceTest.test_tp2_complete_descriptor_ft",
            "ServiceTest.test_tp2_ft_roundtrip_fst",
            "ServiceTest.test_tp2_ep2_scratch_shards",
            "ServiceTest.test_tp2_ep2_fst_shards",
            "ServiceTest.test_zero_experts_fail_for_routed_owners_and_configs",
        ], [], {}),
        ("native_service_output_test", ["ServiceTest.test_scratch", "ServiceTest.test_native_fst"], ["L20_CU13"], {"gpu": "L20_CU13", "gpu_count": "1"}),
    ]
]
