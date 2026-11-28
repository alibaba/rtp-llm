load("//rtp_llm/models_py:checkpoint_test_platforms.bzl", "checkpoint_reader_test_env")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "linear_bias_cpu_test",
    main = "linear_bias_contract_test.py",
    srcs = ["linear_bias_contract_test.py"],
    args = ["CPUCheckpointBiasTest"],
    env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""}),
    deps = ["//rtp_llm/models_py:checkpoint_reader_test_support", "//rtp_llm/models_py:new_weight_loader", "//rtp_llm:fastsafetensors"],
)

py_test(
    name = "linear_bias_hipblas_test",
    main = "linear_bias_contract_test.py",
    srcs = ["linear_bias_contract_test.py"],
    args = ["ROCmCheckpointBiasTest"],
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7", "gpu_count": "1"},
    env = checkpoint_reader_test_env({}),
    deps = ["//rtp_llm/models_py:checkpoint_reader_test_support", "//rtp_llm/models_py:new_weight_loader"],
)

py_library(
    name = "fp4_test_fixtures",
    srcs = ["fp4_linear_test.py", "fp4_moe_test.py"],
    deps = ["//rtp_llm/models_py:new_weight_loader"],
)

py_test(
    name = "fp4_cpu_test",
    srcs = [
        "fp4_cpu_test.py", "fp4_read_spec_test.py", "nvfp4_cache_test.py",
        "nvfp4_backend_test.py", "fp4_accelerator_checkpoint_test.py",
    ],
    main = "fp4_cpu_test.py",
    deps = [
        ":fp4_test_fixtures",
        "//rtp_llm/models_py/quant_methods/test:mxfp4_test_fixtures",
    ],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
)

py_test(
    name = "fp4_native_checkpoint_test",
    srcs = ["fp4_native_checkpoint_test.py"],
    deps = [
        "//rtp_llm/models_py:checkpoint_reader_test_support",
        ":fp4_test_fixtures",
        "//rtp_llm/models_py/quant_methods/test:mxfp4_test_fixtures",
        "//rtp_llm:fastsafetensors",
    ],
    env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}),
)

py_test(
    name = "fp4_sm100_checkpoint_test",
    srcs = ["fp4_accelerator_checkpoint_test.py"],
    main = "fp4_accelerator_checkpoint_test.py",
    deps = ["//rtp_llm/models_py:new_weight_loader"],
    env = {"FP4_TEST_PLATFORM": "sm100", "GPU_COUNT": "1"},
    tags = ["SM100_ARM_CU13", "manual"],
    exec_properties = {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
)

py_test(
    name = "mxfp4_gfx950_checkpoint_test",
    srcs = ["fp4_accelerator_checkpoint_test.py"],
    main = "fp4_accelerator_checkpoint_test.py",
    deps = ["//rtp_llm/models_py:new_weight_loader"],
    env = {"FP4_TEST_PLATFORM": "gfx950", "GPU_COUNT": "1"},
    tags = ["rocm_gfx950", "manual"],
    exec_properties = {"gpu": "MI355X-ROCM7", "gpu_count": "1"},
)
