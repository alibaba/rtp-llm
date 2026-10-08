load(":qwen35_e1pd4_perf_smoke.bzl", "qwen35_e1pd4_perf_suite")
load(":qwen35_epd_fusion_4gpu_smoke.bzl", "qwen35_epd_fusion_4gpu_suite")
load(":qwen35_e2p4d2_repro.bzl", "qwen35_e2p4d2_repro_suite")
package(default_visibility = ["//visibility:public"])

# Framework source files for cross-package smoke tests.
# Used as data dep (not srcs/deps) so internal smoke targets can merge their
# own smoke/ children through smoke.__path__ without duplicating runner srcs.
filegroup(
    name = "smoke_framework_srcs",
    srcs = glob(["*.py"]),
)

py_test(
    name = "dash_cancel_state_test",
    srcs = [
        "dash_cancel_state.py",
        "dash_cancel_state_test.py",
    ],
)

exports_files([
    "entry.py",
    "data/prompt_candidates.json",
])

filegroup(
    name = "smoke_model_configs",
    srcs = glob(["data/model/**/*.json"]),
)

filegroup(
    name = "deepseek_v2_lite_task_info",
    srcs = ["data/model/deepseek_v2/q_r_mla_pymodel.json"],
)

filegroup(
    name = "qwen3_8b_task_info",
    srcs = ["data/model/qwen3/q_r_new_model_py.json"],
)

# ============================================================================
# Open-source smoke test suites (platform-grouped)
# ============================================================================

load("//rtp_llm/test/smoke:suites_h20_oss.bzl", "h20_oss_suites")
load("//rtp_llm/test/smoke:suites_sm8x.bzl", "sm8x_suites")
load("//rtp_llm/test/smoke:suites_sm100.bzl", "sm100_suites")
load("//rtp_llm/test/smoke:suites_sm120.bzl", "sm120_suites")
load("//rtp_llm/test/smoke:suites_rocm_oss.bzl", "rocm_oss_suites")
load("//rtp_llm/test/smoke:suites_remote_cache.bzl", "remote_cache_suites")

h20_oss_suites()
sm8x_suites()
sm100_suites()
sm120_suites()
rocm_oss_suites()
remote_cache_suites()

# ============================================================================
# Top-level CI suites (OSS tests only)
# Internal-only tests live in //internal_source/rtp_llm/test/smoke:maga_model_smoke_internal{_light}
# ============================================================================

test_suite(
    name = "maga_model_smoke_light",
    tests = [
        ":smoke_h20_dense",
        ":smoke_h20_eagle",
        ":smoke_sm8x_basic",
        ":smoke_sm120_basic",
    ],
)

test_suite(
    name = "maga_model_smoke",
    tests = [
        ":smoke_h20_mla",
        ":smoke_h20_moe",
        ":smoke_h20_next",
        ":smoke_h20_kimi_linear",
        ":smoke_h20_grammar_heavy",
        ":smoke_cuda_remote_cache",
        ":smoke_h20_jit_remote_cache",
        ":smoke_rocm_basic",
        ":smoke_rocm_dense",
        ":smoke_rocm_moe",
        ":smoke_rocm_qwen35_mrope_cg",
        ":smoke_rocm_eagle",
        ":smoke_rocm_pd",
        ":smoke_rocm_embedding",
        ":smoke_rocm_jit_remote_cache",
        ":smoke_sm100_dense",
        ":smoke_sm100_moe",
    ],
)

qwen35_e2p4d2_repro_suite()

qwen35_epd_fusion_4gpu_suite()


py_test(
    name = "qwen35_epd_fusion_4gpu_harness_test",
    srcs = [
        "qwen35_epd_fusion_4gpu_smoke.py",
        "qwen35_epd_fusion_4gpu_smoke_test.py",
    ],
    main = "qwen35_epd_fusion_4gpu_smoke_test.py",
    data = [
        "qwen35_epd_fusion_4gpu_smoke.bzl",
        "//rtp_llm:start_server.py",
        "qwen35_e2p4d2_data/manifest.json",
        "qwen35_e2p4d2_data/messages.json",
        "qwen35_e2p4d2_data/video.mp4",
    ],
    deps = [
        "//rtp_llm:requests",
        "//rtp_llm/test/utils:device_resource",
    ],
    legacy_create_init = 0,
    tags = ["manual"],
    timeout = "moderate",
)

py_test(
    name = "pdfusion_schedule_trace_test",
    srcs = ["pdfusion_schedule_trace.py", "pdfusion_schedule_trace_test.py"],
    main = "pdfusion_schedule_trace_test.py",
)

qwen35_e1pd4_perf_suite()

py_test(
    name = "qwen35_e1pd4_perf_preset_test",
    srcs = [
        "qwen35_e1pd4_perf_smoke.py",
        "qwen35_e1pd4_perf_smoke_test.py",
        "qwen35_epd_fusion_4gpu_smoke.py",
    ],
    main = "qwen35_e1pd4_perf_smoke_test.py",
    data = ["qwen35_e1pd4_perf_reference.json"],
    deps = ["//rtp_llm:requests"],
    legacy_create_init = 0,
    tags = ["manual"],
    timeout = "moderate",
)
