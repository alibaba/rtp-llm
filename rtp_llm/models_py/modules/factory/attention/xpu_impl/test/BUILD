# Tests for XPU paged flash attention helpers.
#
# - test_kv_cache_layout: verifies KVCache::getLayerCache() (OpDefs.h XPU branch)
#   produces the NSHD KV layout the vllm_flash_attn consumer reads/writes.
# - test_no_rope: verifies RoPE is skipped for RopeStyle.No models.
# - test_multi_kv_group: verifies hybrid/multi-KV-group models are rejected.
# - test_import_guard: focused unit test for the skip-vs-fail import policy
#   used by the three tests above (see _import_guard.py). Pure stdlib, no
#   package-level preflight import required, so it always runs.
#
# Pure Python tests; use py_standalone_testlib (no compiled C++ .so required
# at analysis time), matching the ROCm test convention.
#
# `env = device_test_envs()` injects TEST_USING_DEVICE (XPU/CUDA/ROCM) per
# //bazel:device_defs.bzl so the XPU-tagged tests below can tell a real XPU
# test run apart from a CPU/non-XPU one, and fail (rather than skip) on a
# guarded import failure when actually targeting XPU.

load("//bazel:device_defs.bzl", "device_test_envs")

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_library(
    name = "import_guard",
    srcs = ["_import_guard.py"],
)

py_test(
    name = "test_kv_cache_layout",
    srcs = ["test_kv_cache_layout.py"],
    deps = py_test_deps + [":import_guard"],
    timeout = "short",
    tags = ["xpu"],
    env = device_test_envs(),
)

py_test(
    name = "test_no_rope",
    srcs = ["test_no_rope.py"],
    deps = py_test_deps + [":import_guard"],
    timeout = "short",
    tags = ["xpu"],
    env = device_test_envs(),
)

py_test(
    name = "test_multi_kv_group",
    srcs = ["test_multi_kv_group.py"],
    deps = py_test_deps + [":import_guard"],
    timeout = "short",
    tags = ["xpu"],
    env = device_test_envs(),
)

py_test(
    name = "test_import_guard",
    srcs = ["test_import_guard.py"],
    deps = [":import_guard"],
    timeout = "short",
)
