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

# Keep the original MTP regression target and its launch defaults.
custom_smoke_test(
    name = "qwen35_dense_mtp_pp_test",
    main = "pp_test.py",
    args = ["--cases=pdfusion_mtp_regression"],
    smoke_args = "--tp_size 2 --pp_size 2 --dp_size 1 --ep_size 1 --act_type BF16 --max_seq_len 4096 --seq_size_per_block 2048 --ssm_state_dtype fp32 --kv_cache_mem_mb 8192 --warm_up 0 --model_warm_up 0 --enable_cuda_graph 0 --frontend_server_count 1 --concurrency_limit 4 --load_method scratch --shutdown_timeout 10",
    gpu_type = ["H20"],
)

# PD separation across pp x tp variants, each compared against a PDFUSION
# baseline of matching TP. world_size only feeds the GPU reservation (max 8
# across variants); each server sets its own parallelism.
custom_smoke_test(
    name = "qwen3_pd_pp_test",
    main = "pp_test.py",
    smoke_args = "--tp_size 1 --dp_size 1 --world_size 8 --act_type BF16 --max_seq_len 4096 --seq_size_per_block 2048 --warm_up 0 --enable_cuda_graph 0 --frontend_server_count 1 --concurrency_limit 4 --load_method scratch --shutdown_timeout 10",
    gpu_type = ["H20"],
)

# PP multi-task system prompt: startup builds resident KV through the pipeline and
# requests reuse it via task_id. Compared against a PP=1 baseline of the same config.
custom_smoke_test(
    name = "qwen3_multi_task_prompt_pp_test",
    main = "pp_test.py",
    args = ["--cases=multi_task_prompt_pp2"],
    smoke_args = "--tp_size 1 --dp_size 1 --world_size 2 --act_type BF16 --max_seq_len 4096 --seq_size_per_block 16 --warm_up 0 --enable_cuda_graph 0 --frontend_server_count 1 --concurrency_limit 4 --load_method scratch --shutdown_timeout 10",
    gpu_type = ["H20"],
)

# PP=2 TP=2 multi-task system prompt: verifies tpSyncModelInputs broadcast alignment
# during the startup build across TP ranks within each stage.
custom_smoke_test(
    name = "qwen3_multi_task_prompt_pp2_tp2_test",
    main = "pp_test.py",
    args = ["--cases=multi_task_prompt_pp2_tp2"],
    smoke_args = "--tp_size 2 --dp_size 1 --world_size 4 --act_type BF16 --max_seq_len 4096 --seq_size_per_block 16 --warm_up 0 --enable_cuda_graph 0 --frontend_server_count 1 --concurrency_limit 4 --load_method scratch --shutdown_timeout 10",
    gpu_type = ["H20"],
)

# PP=2 DP=2 multi-task system prompt: verifies mayAddFakeStream alignment across DP lanes
# during the startup build (all lanes build the same tasks in lockstep).
custom_smoke_test(
    name = "qwen3_multi_task_prompt_pp2_dp2_test",
    main = "pp_test.py",
    args = ["--cases=multi_task_prompt_pp2_dp2"],
    smoke_args = "--tp_size 1 --dp_size 2 --world_size 4 --act_type BF16 --max_seq_len 4096 --seq_size_per_block 16 --warm_up 0 --enable_cuda_graph 0 --frontend_server_count 1 --concurrency_limit 4 --load_method scratch --shutdown_timeout 10",
    gpu_type = ["H20"],
)

# PP=2 PD separation with multi-task system prompt: P and D each build resident KV
# independently through their own PP pipelines; D-side prefix reuse skips KV transfer.
custom_smoke_test(
    name = "qwen3_multi_task_prompt_pp2_pd_test",
    main = "pp_test.py",
    args = ["--cases=multi_task_prompt_pp2_pd"],
    smoke_args = "--tp_size 1 --dp_size 1 --world_size 4 --act_type BF16 --max_seq_len 4096 --seq_size_per_block 16 --warm_up 0 --enable_cuda_graph 0 --frontend_server_count 1 --concurrency_limit 4 --load_method scratch --shutdown_timeout 10",
    gpu_type = ["H20"],
)

# PP=2 CP=2 multi-task system prompt: CP reuses the TP group (tp_size=2, prefill_cp_size=2),
# each CP rank stores a sharded portion of the resident KV.
custom_smoke_test(
    name = "qwen3_multi_task_prompt_pp2_cp2_test",
    main = "pp_test.py",
    args = ["--cases=multi_task_prompt_pp2_cp2"],
    smoke_args = "--tp_size 2 --dp_size 1 --world_size 8 --act_type BF16 --max_seq_len 4096 --seq_size_per_block 16 --warm_up 0 --enable_cuda_graph 0 --frontend_server_count 1 --concurrency_limit 4 --load_method scratch --shutdown_timeout 10",
    gpu_type = ["H20"],
)

# PP=2 MTP multi-task system prompt: build stream is a pure prefill (max_new_tokens=1),
# SP dispatch enforces accepted_length==1 for context streams, pump loop exits normally.
custom_smoke_test(
    name = "qwen35_multi_task_prompt_pp2_mtp_test",
    main = "pp_test.py",
    args = ["--cases=multi_task_prompt_pp2_mtp"],
    smoke_args = "--tp_size 1 --dp_size 1 --world_size 2 --act_type BF16 --max_seq_len 8192 --seq_size_per_block 2048 --warm_up 0 --enable_cuda_graph 0 --frontend_server_count 1 --concurrency_limit 4 --load_method scratch --shutdown_timeout 10",
    gpu_type = ["H20"],
)

# PP=2 PD separation + MTP: P and D each build resident KV through their own PP pipelines
# with speculative decoding enabled; D-side prefix reuse skips KV transfer for resident blocks.
custom_smoke_test(
    name = "qwen35_multi_task_prompt_pp2_pd_mtp_test",
    main = "pp_test.py",
    args = ["--cases=multi_task_prompt_pp2_pd_mtp"],
    smoke_args = "--tp_size 1 --dp_size 1 --world_size 4 --act_type BF16 --max_seq_len 8192 --seq_size_per_block 2048 --warm_up 0 --enable_cuda_graph 0 --frontend_server_count 1 --concurrency_limit 4 --load_method scratch --shutdown_timeout 10",
    gpu_type = ["H20"],
)

# The same entry supports model overrides and --cases / PP_TEST_CASES.
# Keep the original target available for existing PD smoke commands.
test_suite(
    name = "pp_topology_test",
    tags = ["manual"],
    tests = [":qwen3_pd_pp_test"],
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
