py_test(
    name = "arm_streamed_scratch_test",
    srcs = ["arm_streamed_scratch_test.py"],
    deps = [
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:new_loader_core",
    ],
)

# On this lane the CPU copy source and CUDA parameter destination must both be
# exercised. The standard CPU target never claims accelerator coverage.
py_test(
    name = "arm_streamed_scratch_sm100_test",
    srcs = ["arm_streamed_scratch_test.py"],
    main = "arm_streamed_scratch_test.py",
    env = {"ARM_CUDA_SCRATCH_TEST": "1"},
    exec_properties = {"gpu": "SM100_ARM_CU13"},
    tags = ["SM100_ARM_CU13"],
    target_compatible_with = select({
        "//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:new_loader_core",
    ],
)
