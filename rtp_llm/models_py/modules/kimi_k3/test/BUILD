load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")
load("@pip_gpu_cuda13_torch//:requirements.bzl", cuda13_requirement = "requirement")
load("@pip_cuda13_arm_torch//:requirements.bzl", cuda13_arm_requirement = "requirement")

cuda13_sm100_py_test(
    name = "native_kda_checkpoint_test",
    srcs = ["native_kda_checkpoint_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ] + select({
        "@//:using_cuda13_x86": [cuda13_requirement("cuda-linear-attention")],
        "@//:using_cuda13_arm": [cuda13_arm_requirement("cuda-linear-attention")],
        "//conditions:default": [],
    }),
)

py_test(
    name = "bounded_projection_jit_warmup_test",
    srcs = ["test_bounded_projection_jit_warmup.py"],
    main = "test_bounded_projection_jit_warmup.py",
    data = [
        "//rtp_llm/models_py:bounded_projection_jit_test_sources",
    ],
)

py_test(
    name = "flash_mla_prefill_api_test",
    srcs = ["flash_mla_prefill_api_test.py"],
    data = ["//rtp_llm/models_py:flash_mla_prefill_api_test_sources"],
)

py_test(
    name = "bounded_projection_jit_gpu_test",
    srcs = [
        "test_bounded_projection_jit_gpu.py",
    ],
    main = "test_bounded_projection_jit_gpu.py",
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:einops",
        "//rtp_llm:transformers",
        "//rtp_llm:triton",
        cuda13_requirement("flash-linear-attention"),
        cuda13_requirement("nvidia-cutlass-dsl"),
        cuda13_requirement("backports-strenum"),
    ],
    tags = ["manual", "SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)
