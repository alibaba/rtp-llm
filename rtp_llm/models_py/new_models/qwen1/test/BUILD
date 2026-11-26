load("//rtp_llm/models_py:checkpoint_test_platforms.bzl", "checkpoint_reader_test_env")

py_test(
    name = "qwen7b_pd_checkpoint_test",
    srcs = ["qwen7b_pd_checkpoint_test.py"],
    env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}),
    deps = [
        "//rtp_llm/models_py:checkpoint_reader_test_support",
        "//rtp_llm:model_factory",
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:safetensors_torch",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

py_test(
    name = "qwen7b_ppu_scratch_storage_test",
    srcs = ["qwen7b_pd_checkpoint_test.py"],
    main = "qwen7b_pd_checkpoint_test.py",
    args = ["Qwen1CheckpointTest.test_ppu_scratch_populates_final_parameter_storage"],
    env = checkpoint_reader_test_env({"PPU_QWEN_SCRATCH_TEST": "1", "OMP_NUM_THREADS": "1"}),
    tags = ["PPU-ZW810E"],
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    visibility = ["//rtp_llm/test/smoke:__pkg__"],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm/models_py:checkpoint_reader_test_support",
        "//rtp_llm:model_factory",
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:safetensors_torch",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)
