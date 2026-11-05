py_test(
    name = "test_embedding_contract",
    size = "small",
    srcs = ["test_embedding_contract.py"],
    main = "test_embedding_contract.py",
    deps = ["//rtp_llm/models_py:engram_model_cpu"],
    env = {"GPU_COUNT": "0"},
    legacy_create_init = 0,
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
