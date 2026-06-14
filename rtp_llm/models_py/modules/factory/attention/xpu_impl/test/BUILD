# Tests for XPU paged flash attention helpers.
#
# - test_kv_cache_layout: verifies KVCache::getLayerCache() (OpDefs.h XPU branch)
#   produces the NSHD KV layout the vllm_flash_attn consumer reads/writes.
#   Plain torch on CPU; needs only the compiled rtp_llm.ops bindings.

py_test(
    name = "test_kv_cache_layout",
    srcs = ["test_kv_cache_layout.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    tags = ["xpu"],
)
