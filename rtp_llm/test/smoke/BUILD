load(":defs.bzl", "custom_smoke_test")

package(default_visibility = ["//visibility:public"])

# Framework source files for cross-package smoke tests.
# Used as data dep (not srcs/deps) so internal smoke targets can merge their
# own smoke/ children through smoke.__path__ without duplicating runner srcs.
filegroup(
    name = "smoke_framework_srcs",
    srcs = glob(
        ["*.py"],
        exclude = ["*_test.py"],
    ),
)

py_test(
    name = "disk_path_preparation_test",
    srcs = glob(
        ["*.py"],
        exclude = ["entry.py"],
    ),
    main = "disk_path_preparation_test.py",
    imports = [".."],
    deps = [
        "//rtp_llm/test/utils:device_resource",
        "//rtp_llm/test/utils:maga_server_manager",
        "//rtp_llm/test/utils:test_util",
        "//rtp_llm:dashscope",
        "//rtp_llm:fastapi",
        "//rtp_llm:jieba",
        "//rtp_llm:json5",
        "//rtp_llm:openai",
        "//rtp_llm:partial_json_parser",
        "//rtp_llm:psutil",
        "//rtp_llm:pydantic",
        "//rtp_llm:testlib",
        "//rtp_llm:tiktoken",
        "//rtp_llm:uvicorn",
    ],
)

py_test(
    name = "dash_cancel_state_test",
    srcs = [
        "dash_cancel_state.py",
        "dash_cancel_state_test.py",
    ],
)

py_test(
    name = "comparison_utils_test",
    srcs = [
        "comparison_utils.py",
        "comparison_utils_test.py",
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
load("//rtp_llm/test/smoke:suites_cuda13.bzl", "cuda13_suites")
load("//rtp_llm/test/smoke:suites_sm8x.bzl", "sm8x_suites")
load("//rtp_llm/test/smoke:suites_sm100.bzl", "sm100_suites")
load("//rtp_llm/test/smoke:suites_sm120.bzl", "sm120_suites")
load("//rtp_llm/test/smoke:suites_rocm_oss.bzl", "rocm_oss_suites")
load("//rtp_llm/test/smoke:suites_remote_cache.bzl", "remote_cache_suites")

h20_oss_suites()
cuda13_suites()
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
        ":smoke_sm120_dense",
        ":smoke_sm120_moe",
    ],
)

custom_smoke_test(
    name = "deepseek_v41_flash_tp4_ep4_graph_sm100_arm",
    main = "deepseek_v41_contract_test.py",
    smoke_args = "--tp_size 4 --ep_size 4 --dp_size 1 --world_size 4 --act_type BF16 --fp8_kv_cache 1 --seq_size_per_block 128 --max_seq_len 4096 --max_context_batch_size 1 --concurrency_limit 4 --enable_cuda_graph 1 --reuse_cache 1 --load_method fastsafetensors --reserver_runtime_mem_mb 32768",
    gpu_type = ["SM100_ARM_CU13"],
    tags = ["dsv41_cuda13", "model_contract"],
    data = ["//rtp_llm:sdk"],
)

custom_smoke_test(
    name = "deepseek_v41_flash_ced_bounded_pd_cp2_tp2_sm100_arm",
    main = "deepseek_v41_pd_contract_test.py",
    smoke_args = "--tp_size 2 --ep_size 2 --dp_size 1 --world_size 4 --act_type BF16 --fp8_kv_cache 1 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --max_seq_len 4096 --max_context_batch_size 1 --concurrency_limit 4 --reuse_cache 1 --load_method fastsafetensors --reserver_runtime_mem_mb 32768 --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 120000 --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 2",
    gpu_type = ["SM100_ARM_CU13"],
    tags = ["dsv41_cuda13", "model_contract", "pd_contract"],
    data = ["//rtp_llm:sdk", "deepseek_v41_contract_test.py"],
)

custom_smoke_test(
    name = "deepseek_v41_flash_dspark_tp4_ep4_graph_sm100_arm",
    main = "deepseek_v41_dspark_contract_test.py",
    smoke_args = "--tp_size 4 --ep_size 4 --dp_size 1 --world_size 4 --act_type BF16 --fp8_kv_cache 1 --seq_size_per_block 128 --max_seq_len 4096 --max_context_batch_size 1 --concurrency_limit 4 --enable_cuda_graph 1 --reuse_cache 1 --load_method fastsafetensors --reserver_runtime_mem_mb 32768 --sp_type dspark --sp_model_type deepseek_v41_dspark --sp_act_type BF16 --gen_num_per_cycle 3 --decode_capture_config 1,2,4",
    gpu_type = ["SM100_ARM_CU13"],
    tags = ["dsv41_cuda13", "model_contract", "dspark_contract"],
    data = ["//rtp_llm:sdk", "deepseek_v41_contract_test.py"],
)
