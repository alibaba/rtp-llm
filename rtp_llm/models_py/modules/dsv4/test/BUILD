py_test(
    name = "mhc_test",
    srcs = ["mhc_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm/models_py:tile_kernels_mhc",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "test_dsv4_kv_cache",
    srcs = ["test_dsv4_kv_cache.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

# ── DSV4 SWA prefill Triton kernel UTs ────────────────────────────────────
# Lightweight: only need the triton kernel files + torch / triton runtime.
# Avoid `py_standalone_testlib` because it pulls model_loader / flash-mla
# which break under --config=cuda13.
load("//bazel:arch_select.bzl", "requirement")
requirement(["torch", "triton"])

_swa_triton_ut_deps = [
    "//rtp_llm/models_py:modules",
    ":torch",
    ":triton",
]

py_test(
    name = "test_swa_fp8_kv_roundtrip",
    srcs = ["test_swa_fp8_kv_roundtrip.py"],
    deps = _swa_triton_ut_deps,
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "test_swa_slot_mapping",
    srcs = ["test_swa_slot_mapping.py"],
    deps = _swa_triton_ut_deps,
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "test_swa_prefill_gather_lens",
    srcs = ["test_swa_prefill_gather_lens.py"],
    deps = _swa_triton_ut_deps,
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "test_swa_combine_topk",
    srcs = ["test_swa_combine_topk.py"],
    deps = _swa_triton_ut_deps,
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)
