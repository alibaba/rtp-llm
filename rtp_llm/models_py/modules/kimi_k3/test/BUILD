load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")
load("@pip_gpu_cuda13_torch//:requirements.bzl", cuda13_requirement = "requirement")
load("@pip_cuda13_arm_torch//:requirements.bzl", cuda13_arm_requirement = "requirement")

cuda13_sm100_py_test(
    name = "flashinfer_prefill_test",
    srcs = ["flashinfer_prefill_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:flashinfer-python",
        "//rtp_llm:flashinfer-cubin",
        "//rtp_llm:triton",
    ],
)

py_test(
    name = "mla_fia2a_integration_test_x86",
    srcs = ["mla_fia2a_integration_test.py"],
    main = "mla_fia2a_integration_test.py",
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:nvidia-cutlass-dsl",
        "//rtp_llm:triton",
    ],
    env = {"GPU_COUNT": "4", "FIA2A_EXPECTED_CC": "10.3"},
    tags = ["SM103_CU13", "multi_device"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "4"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "mla_fia2a_integration_test",
    srcs = ["mla_fia2a_integration_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:nvidia-cutlass-dsl",
        "//rtp_llm:triton",
    ],
    env = {"GPU_COUNT": "4", "FIA2A_EXPECTED_CC": "10.0"},
    tags = ["SM100_ARM_CU13", "multi_device"],
    exec_properties = {"gpu": "SM100_ARM_CU13", "gpu_count": "4"},
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

cuda13_sm100_py_test(
    name = "shared_modeling_test",
    srcs = ["shared_modeling_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        # The decoder and MTP model import K3 checkpoint weight contracts.
        "//rtp_llm:models",
        "//rtp_llm:flashinfer-python",
    ],
)

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
    name = "kv_up_projection_test",
    srcs = ["kv_up_projection_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:deep_gemm",
    ],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
    tags = ["SM103_CU13"],
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
