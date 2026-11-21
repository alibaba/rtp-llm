py_library(
    name = "draft_body",
    srcs = ["__init__.py", "body.py", "proposal.py", "service.py"],
    deps = ["//rtp_llm/models_py:new_loader_core", "//rtp_llm:torch"],
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
        env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NEWLOADER_TEST_NATIVE_FST": "1"},
        tags = tags,
        exec_properties = properties,
        deps = [":draft_body", "//rtp_llm:model_factory", "//rtp_llm:models", "//rtp_llm:sdk", "//rtp_llm:transformers", "//rtp_llm/models_py:new_weight_loader", "//rtp_llm:safetensors"],
    )
    for name, cases, tags, properties in [
        ("service_test", ["ServiceTest.test_tp2_scratch_shards", "ServiceTest.test_tp2_fst_shards"], [], {}),
        ("native_service_output_test", ["ServiceTest.test_scratch", "ServiceTest.test_native_fst"], ["L20_CU13"], {"gpu": "L20_CU13", "gpu_count": "1"}),
    ]
]
