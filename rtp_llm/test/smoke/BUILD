load("//rtp_llm/test/smoke:defs.bzl", "SMOKE_FRAMEWORK_DEPS", "smoke_test")
package(default_visibility = ["//visibility:public"])

py_test(
    name = "kev_family_acceptance_test",
    srcs = ["kev_family_acceptance_test.py"],
    data = ["//rtp_llm:sdk"],
    deps = SMOKE_FRAMEWORK_DEPS + ["//rtp_llm:requests", "//rtp_llm:transformers"],
    env = {"GPU_COUNT": "1", "PYTHONNOUSERSITE": "1"},
    exec_properties = {"gpu": "MI308X-ROCM7", "gpu_count": "1"},
    tags = ["manual", "rocm"],
    timeout = "eternal",
    legacy_create_init = 0,
)

# d3 code-readout acceptance: one test and one full-modal corpus, three thin
# platform variants (H20 / RTX PRO 5000 / MI308X). The staged model snapshot
# and the precision-specific official reference are supplied at run time via
# D3_ACCEPTANCE_MODEL_PATH / D3_ACCEPTANCE_REFERENCE and are never committed;
# the corpus fixture data/d3/corpus.json is the shared request source.
_D3_ACCEPTANCE_DEPS = SMOKE_FRAMEWORK_DEPS + [
    "//rtp_llm:requests",
    "//rtp_llm:transformers",
]

py_test(
    name = "d3_acceptance_h20",
    srcs = ["d3_acceptance_test.py"],
    main = "d3_acceptance_test.py",
    data = [
        "//rtp_llm:sdk",
        "data/d3/corpus.json",
    ],
    deps = _D3_ACCEPTANCE_DEPS,
    env = {
        "D3_ACCEPTANCE_USE_FLYDSL": "0",
        "GPU_COUNT": "1",
        "PYTHONNOUSERSITE": "1",
    },
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    tags = ["manual", "cuda13"],
    timeout = "eternal",
    legacy_create_init = 0,
)

py_test(
    name = "d3_acceptance_pro5000",
    srcs = ["d3_acceptance_test.py"],
    main = "d3_acceptance_test.py",
    data = [
        "//rtp_llm:sdk",
        "data/d3/corpus.json",
    ],
    deps = _D3_ACCEPTANCE_DEPS,
    env = {
        "D3_ACCEPTANCE_USE_FLYDSL": "0",
        "GPU_COUNT": "1",
        "PYTHONNOUSERSITE": "1",
    },
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
    tags = ["manual", "cuda13"],
    timeout = "eternal",
    legacy_create_init = 0,
)

py_test(
    name = "d3_acceptance_mi308x",
    srcs = ["d3_acceptance_test.py"],
    main = "d3_acceptance_test.py",
    data = [
        "//rtp_llm:sdk",
        "data/d3/corpus.json",
    ],
    deps = _D3_ACCEPTANCE_DEPS,
    env = {
        "D3_ACCEPTANCE_USE_FLYDSL": "1",
        "GPU_COUNT": "1",
        "PYTHONNOUSERSITE": "1",
    },
    exec_properties = {"gpu": "MI308X-ROCM7", "gpu_count": "1"},
    tags = ["manual", "rocm"],
    timeout = "eternal",
    legacy_create_init = 0,
)


# Official reference generation: runs the staged snapshot's pinned d3_runtime
# with the frozen official deps (transformers 5.17.0 venv supplied via
# D3_REFERENCE_VENV) over the shared corpus; writes reference + raw outputs to
# D3_REFERENCE_OUTPUT. GPU ownership via the canonical gpu_lock wrapper.
py_test(
    name = "d3_reference_gen_pro5000",
    srcs = [
        "d3_reference_gen_test.py",
        "d3_acceptance_test.py",
    ],
    main = "d3_reference_gen_test.py",
    data = ["data/d3/corpus.json"],
    deps = _D3_ACCEPTANCE_DEPS,
    env = {"GPU_COUNT": "1", "PYTHONNOUSERSITE": "1"},
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
    tags = ["manual", "cuda13"],
    timeout = "eternal",
    legacy_create_init = 0,
)

py_test(
    name = "d3_reference_gen_mi308x",
    srcs = [
        "d3_reference_gen_test.py",
        "d3_acceptance_test.py",
    ],
    main = "d3_reference_gen_test.py",
    data = ["data/d3/corpus.json"],
    deps = _D3_ACCEPTANCE_DEPS,
    env = {"GPU_COUNT": "1", "PYTHONNOUSERSITE": "1"},
    exec_properties = {"gpu": "MI308X-ROCM7", "gpu_count": "1"},
    tags = ["manual", "rocm"],
    timeout = "eternal",
    legacy_create_init = 0,
)

py_test(
    name = "d3_reference_gen_h20",
    srcs = [
        "d3_reference_gen_test.py",
        "d3_acceptance_test.py",
    ],
    main = "d3_reference_gen_test.py",
    data = ["data/d3/corpus.json"],
    deps = _D3_ACCEPTANCE_DEPS,
    env = {"GPU_COUNT": "1", "PYTHONNOUSERSITE": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    tags = ["manual", "cuda13"],
    timeout = "eternal",
    legacy_create_init = 0,
)

# Exact token-sequence audit: official d3_runtime.prepare vs RTP
# D3Executor._prompt_text per question, using the actual checkpoint tokenizer.
# CPU only (no GPU lock); text cases while the video transport fix is pending.
py_test(
    name = "d3_token_audit",
    srcs = [
        "d3_token_audit_test.py",
        "d3_acceptance_test.py",
    ],
    main = "d3_token_audit_test.py",
    data = ["data/d3/corpus.json"],
    deps = _D3_ACCEPTANCE_DEPS + [
        "//rtp_llm:frontend",
    ],
    env = {"GPU_COUNT": "0", "PYTHONNOUSERSITE": "1"},
    tags = ["manual"],
    timeout = "eternal",
    legacy_create_init = 0,
)

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

# Frontend, a real FlexLB master, separate prefill/decode, and two ViT workers.
# Five real requests with the same image check that FlexLB keeps sending
# image computation to the ViT that already owns the cached embeddings.
smoke_test(
    name = "qwen3_vl_pd_multi_vit_cache_affinity",
    task_info = "data/model/qwen_vl/q_r_pd_multi_vit_cache_affinity.json",
    smoke_args = {
        "frontend": "--act_type BF16 --role_type FRONTEND --max_seq_len 4096 --use_local 1 --warm_up 0",
        "prefill": "--act_type BF16 --role_type PREFILL --tp_size 1 --world_size 1 --max_seq_len 4096 --kv_cache_mem_mb 1024 --load_cache_timeout_ms 120000 --cache_store_rdma_mode 0 --use_local 1 --warm_up 0 --enable_cuda_graph 0",
        "decode": "--act_type BF16 --role_type DECODE --tp_size 1 --world_size 1 --max_seq_len 4096 --kv_cache_mem_mb 1024 --load_cache_timeout_ms 120000 --cache_store_rdma_mode 0 --use_local 1 --warm_up 0 --enable_cuda_graph 0",
        "vit_0": "--act_type BF16 --tp_size 1 --world_size 1 --use_local 1 --use_local_preprocess 1 --warm_up 0",
        "vit_1": "--act_type BF16 --tp_size 1 --world_size 1 --use_local 1 --use_local_preprocess 1 --warm_up 0",
    },
    envs = {
        "vit_0": ["MM_CACHE_GPU_MAX_BYTES=2147483648", "MM_CACHE_CPU_MAX_BYTES=2147483648"],
        "vit_1": ["MM_CACHE_GPU_MAX_BYTES=2147483648", "MM_CACHE_CPU_MAX_BYTES=2147483648"],
    },
    data = [
        ":flexlb_runtime_bundle",
    ],
    gpu_type = ["L20D_TEST"],
    sleep_time_qr = 2,
    test_envs = {
        "SMOKE_START_FLEXLB_MASTER": "1",
        "SMOKE_ASSERT_VIT_CACHE_AFFINITY": "1",
        "SMOKE_AFFINITY_RECHECK_DELAY_SECONDS": "125",
    },
)
