# Tests for XPU paged flash attention helpers.
#
# - test_kv_cache_layout: verifies KVCache::getLayerCache() (OpDefs.h XPU branch)
#   produces the NSHD KV layout the vllm_flash_attn consumer reads/writes.
# - test_no_rope: verifies RoPE is skipped for RopeStyle.No models.
#
# Pure Python tests; use py_standalone_testlib (no compiled C++ .so required
# at analysis time), matching the ROCm test convention.

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "test_kv_cache_layout",
    srcs = ["test_kv_cache_layout.py"],
    deps = py_test_deps,
    timeout = "short",
    tags = ["xpu"],
)

py_test(
    name = "test_no_rope",
    srcs = ["test_no_rope.py"],
    deps = py_test_deps,
    timeout = "short",
    tags = ["xpu"],
)
