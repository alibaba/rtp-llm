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

# This helper links only the selected SDK RPM, not the RTP model or CUDA stack.
cc_binary(
    name = "pace_sdk_smoke",
    testonly = True,
    srcs = ["pace_sdk_smoke.cc"],
    # The paired SDK RPM uses the C++11 string ABI; --config=cpu uses ABI 0.
    copts = ["-std=c++17", "-U_GLIBCXX_USE_CXX11_ABI", "-D_GLIBCXX_USE_CXX11_ABI=1"],
    deps = ["//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_client"],
)

cc_binary(
    name = "pace_publisher_smoke",
    testonly = True,
    srcs = ["pace_publisher_smoke.cc"],
    copts = ["-std=c++17"],
    deps = ["//rtp_llm/cpp/cache/events:kv_cache_event_publisher"],
)

py_library(
    name = "pace_smoke_fixture",
    srcs = [
        "pace_fixture.py", "remote_kvcm_server.py", "//rtp_llm:utils/util.py",
    ],
    deps = ["//rtp_llm:psutil", "//rtp_llm:requests", "//rtp_llm/test/utils:test_util"],
)

load("//deps:kvcm.bzl", "KVCM_SOURCE_LOCK")

_KVCM_SMOKE_SOURCE_ID = ":".join([KVCM_SOURCE_LOCK[key] for key in [
    "internal_commit", "opensource_commit", "pace_commit",
]])

_PACE_SMOKE_DATA = [
    ":pace_sdk_smoke",
    ":pace_publisher_smoke",
    "@remote_kv_cache_manager_server//:bin/kv_cache_manager_bin",
    "@remote_kv_cache_manager_server//:KVCM_SOURCE_ID",
    "@remote_kv_cache_manager_server//:KVCM_ARTIFACT_SHA256",
    "@remote_kv_cache_manager_server//:etc/default_startup_config.json",
    "@remote_kv_cache_manager_client_rpm//:KVCM_SOURCE_ID",
    "@remote_kv_cache_manager_client_rpm//:KVCM_ARTIFACT_SHA256",
    "@remote_kv_cache_manager_client_rpm//:KVCM_CLIENT_VARIANT",
]

py_test(
    name = "remote_cache_pace_contract",
    srcs = ["pace_contract_smoke.py"],
    main = "pace_contract_smoke.py",
    deps = [":pace_smoke_fixture"],
    data = _PACE_SMOKE_DATA,
    args = ["--sdk", "$(location :pace_sdk_smoke)", "--publisher", "$(location :pace_publisher_smoke)", "--backend", "pace"],
    env = {"KVCM_EXPECTED_SOURCE_ID": _KVCM_SMOKE_SOURCE_ID, "KVCM_SMOKE_CLIENT_VARIANT": "cpu"},
    env_inherit = ["KVCM_PACE_FIXTURE"],
    tags = ["manual", "smoke_case", "no-remote", "requires-pace"],
    timeout = "long",
    legacy_create_init = 0,
)

py_test(
    name = "remote_cache_pace_ssd_contract",
    srcs = ["pace_contract_smoke.py"],
    main = "pace_contract_smoke.py",
    deps = [":pace_smoke_fixture"],
    data = _PACE_SMOKE_DATA,
    args = ["--sdk", "$(location :pace_sdk_smoke)", "--publisher", "$(location :pace_publisher_smoke)", "--backend", "pace_ssd"],
    env = {"KVCM_EXPECTED_SOURCE_ID": _KVCM_SMOKE_SOURCE_ID, "KVCM_SMOKE_CLIENT_VARIANT": "cpu"},
    env_inherit = ["KVCM_PACE_FIXTURE"],
    tags = ["manual", "smoke_case", "no-remote", "requires-pace-ssd"],
    timeout = "long",
    legacy_create_init = 0,
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
