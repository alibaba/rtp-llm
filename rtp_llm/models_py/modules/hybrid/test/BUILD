
test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "512000000",  # 512MB
}

flashinfer_test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "512000000",  # 512MB
    "FLASHINFER_JIT_VERBOSE": "1",  # Enable verbose JIT compilation logs
}

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

# FlashInfer dependencies (包括 apache-tvm-ffi 用于 JIT 编译)
flashinfer_deps = [
    "//rtp_llm:apache-tvm-ffi",
    "//rtp_llm:flashinfer-python",
]

py_test (
    name = "mla_attention_test",
    srcs = ["mla_attention_test.py", "mla_attention_ref.py"],
    deps = py_test_deps + [
        "//rtp_llm:testlib",
    ] + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "mla_reuse_cache_test",
    srcs = ["mla_reuse_cache_test.py", "mla_attention_ref.py"],
    deps = py_test_deps + [
        "//rtp_llm:testlib",
    ] + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "mla_perf_test",
    srcs = ["mla_perf_test.py", "mla_attention_ref.py"],
    deps = py_test_deps + [
        "//rtp_llm:testlib",
    ],
    env = test_envs,
    tags = ["open_skip", "H20", "manual"],
    exec_properties = {'gpu':'H20'},
)

py_test (
   name = "mlp_test",
   srcs = ["mlp_test.py", "dense_mlp_ref.py"],
   deps = [
       "//rtp_llm/models_py:models",
       "//rtp_llm:config",
       "//rtp_llm:utils",
      "//rtp_llm:testlib",
       "//rtp_llm/test/model_test/test_util:test_util"
   ],
   env = test_envs,
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20'},
)

py_test (
   name = "flashinfer_jit_test",
   srcs = ["flashinfer_jit_test.py"],
   deps = [
       "//rtp_llm/models_py:models",
       "//rtp_llm:config",
       "//rtp_llm:utils",
      "//rtp_llm:testlib",
       "//rtp_llm/test/model_test/test_util:test_util"
   ],
   env = flashinfer_test_envs,
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20'},
)

py_test (
   name = "indexer_test",
   srcs = ["indexer_test.py", "indexer_ref.py"],
   deps = py_test_deps + [
        "//rtp_llm:testlib",
   ],
   env = test_envs,
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20'},
)