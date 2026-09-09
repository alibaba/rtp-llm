# Test configuration for PyHeadWise

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

# FlashInfer dependencies (包括 apache-tvm-ffi 用于 JIT 编译)
flashinfer_deps = [
    "//rtp_llm:apache-tvm-ffi",
    "//rtp_llm:flashinfer-python",
]

py_test(
    name = "test_py_headwise",
    srcs = [
        "test_py_headwise.py",
    ],
    deps = py_test_deps + flashinfer_deps + select({
        "@//:using_cuda12_9_x86": ["@pip_gpu_cuda12_9_torch//rtp_kernel:pkg"],
        "//conditions:default": [],
    }),
    tags = ["H20", "cuda12_9"],
    # Stays on H20 for VRAM, not compute capability: op.support() returns True on
    # sm86 and flashinfer + rtp_kernel.sparse_attention both work there, but
    # test_long_context_prefill's flex-attention reference mask needs one 32 GiB
    # allocation, which OOMs on any 24 GB card (A10 included).
    exec_properties = {
        "gpu": "H20",
    },
)
