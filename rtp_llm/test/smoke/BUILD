package(default_visibility = ["//visibility:public"])

# Framework source files for cross-package smoke tests.
# Used as data dep (not srcs/deps) so internal smoke targets can merge their
# own smoke/ children through smoke.__path__ without duplicating runner srcs.
filegroup(
    name = "smoke_framework_srcs",
    srcs = glob(["*.py"]),
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
load("//rtp_llm/test/smoke:defs.bzl", "custom_smoke_test")

custom_smoke_test(
    name = "qwen35_dense_mtp_pp_test",
    main = "mtp_pp_test.py",
    smoke_args = "--tp_size 2 --pp_size 2 --dp_size 1 --ep_size 1 --act_type BF16 --max_seq_len 4096 --seq_size_per_block 2048 --ssm_state_dtype fp32 --kv_cache_mem_mb 8192 --warm_up 0 --model_warm_up 0 --enable_cuda_graph 0 --frontend_server_count 1 --concurrency_limit 4 --load_method scratch --shutdown_timeout 10",
    gpu_type = ["H20"],
)

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
