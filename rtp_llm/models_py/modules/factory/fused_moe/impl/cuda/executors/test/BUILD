# SM100_ARM executor tests run only on the ARM SM100 pools (ut-gb200 /
# ut-gb200-cu13). This makes that a structural, analysis-time guardrail rather
# than relying solely on each CI lane carrying a -SM100_ARM tag filter: any
# non-ARM config cannot build these targets, so an accidental/mis-scheduled
# lane fails fast instead of running them against the wrong (or no) GPU pool.
# The GPU pool value itself still comes from the root .bazelrc (single source);
# this only gates buildability, it does not duplicate the pool or its tag.
_SM100_ARM_ONLY = select({
    "@//:using_cuda_arm": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

py_library(
    name = "fused_moe_executor_test_util",
    srcs = ["fused_moe_executor_test_util.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
    ]
)

py_test (
    name = "deepgemm_masked_executor_test",
    srcs = ["deepgemm_masked_executor_test.py"],
    deps = [
        ":fused_moe_executor_test_util",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'1'},
)

# Selected by the SM100_ARM tag in ut-gb200/ut-gb200-cu13. The root .bazelrc
# injects gpu=SM100_ARM for --config=cuda12_9_arm and gpu=SM100_ARM_CU13 for
# --config=cuda13_arm; this repository-level config is intentionally the single
# source for the versioned GPU pool, while gpu_count remains target-specific.
# The ARM-only tag is also deliberate: CUDA 13 x86 numerical coverage waits for
# an x86 SM100 CI pool and qualified wheel revisions. Do not duplicate the ARM
# pool value or reuse its tag at target level.
py_test (
    name = "deepgemm_masked_executor_sm100_test",
    srcs = ["deepgemm_masked_executor_test.py",
            "deepgemm_masked_executor_sm100_test.py"],
    deps = [
        ":fused_moe_executor_test_util",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    env = {
        "GPU_COUNT": "1",
    },
    tags = ["SM100_ARM"],
    exec_properties = {'gpu_count':'1'},
    target_compatible_with = _SM100_ARM_ONLY,
)

py_test (
    name = "deepep_normal_executor_test",
    srcs = ["deepep_normal_executor_test.py"],
    deps = [
        ":fused_moe_executor_test_util",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'1'},
)

py_test (
    name = "deepep_normal_executor_sm100_test",
    srcs = ["deepep_normal_executor_sm100_test.py",
            "deepep_normal_executor_test.py"],
    deps = [
        ":fused_moe_executor_test_util",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    env = {
        "GPU_COUNT": "1",
    },
    tags = ["SM100_ARM"],
    exec_properties = {'gpu_count':'1'},
    target_compatible_with = _SM100_ARM_ONLY,
)

py_test (
    name = "trtllm_fp4_executor_test",
    srcs = ["trtllm_fp4_executor_test.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm:flashinfer-python",
        "//rtp_llm:nvidia-cutlass-dsl",
    ],
    env = {
        "GPU_COUNT": "1",
    },
    tags = ["SM100_ARM"],
    exec_properties = {'gpu_count':'1'},
    target_compatible_with = _SM100_ARM_ONLY,
)

# x86 CUDA 13 hardware-free counterpart to the ARM-only CuteDSL FP4 numerical
# regression: verifies the FP4 import/export path resolves cleanly on x86
# (real impl or _fp4_unavailable stub) without needing an SM100 device.
py_test(
    name = "cutedsl_fp4_x86_import_test",
    srcs = ["cutedsl_fp4_x86_import_test.py"],
    deps = [
        "//rtp_llm/models_py:kernels",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test (
    name = "cutedsl_fp4_executor_test",
    srcs = ["cutedsl_fp4_executor_test.py"],
    deps = [
        "//rtp_llm/models_py:kernels",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm:flashinfer-python",
    ],
    env = {
        "GPU_COUNT": "1",
    },
    tags = ["SM100_ARM"],
    exec_properties = {'gpu_count':'1'},
    target_compatible_with = _SM100_ARM_ONLY,
)
