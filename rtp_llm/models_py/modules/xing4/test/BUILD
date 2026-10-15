package(default_visibility = ["//visibility:public"])

# Xing4 mHC fusion-lane equivalence UTs (one py_test per kernel module,
# main = srcs). Manual + SM103_CU13 routing follows the lane contract; the
# kernel math authority is Xing4HCUnit in
# rtp_llm/models_py/model_desc/xing4_0_model.py.

py_test(
    name = "test_hc_norm_triton",
    size = "large",
    srcs = ["test_hc_norm_triton.py"],
    main = "test_hc_norm_triton.py",
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = [
        "manual",
        "SM103_CU13",
    ],
    exec_properties = {
        "gpu": "SM103_CU13",
        "gpu_count": "1",
    },
)

py_test(
    name = "test_hc_post_triton",
    srcs = ["test_hc_post_triton.py"],
    main = "test_hc_post_triton.py",
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = [
        "manual",
        "SM103_CU13",
    ],
    exec_properties = {
        "gpu": "SM103_CU13",
        "gpu_count": "1",
    },
)

py_test(
    name = "test_hc_collapse_triton",
    srcs = ["test_hc_collapse_triton.py"],
    main = "test_hc_collapse_triton.py",
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = [
        "manual",
        "SM103_CU13",
    ],
    exec_properties = {
        "gpu": "SM103_CU13",
        "gpu_count": "1",
    },
)

# Fusion round-2 (K1+GEMM): fused row RMSNorm + hc_fn skinny GEMV. Imports
# norm_triton for the chained-reference comparison, hence :modules.
py_test(
    name = "test_hc_fn_gemv_triton",
    size = "large",
    srcs = ["test_hc_fn_gemv_triton.py"],
    main = "test_hc_fn_gemv_triton.py",
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = [
        "manual",
        "SM103_CU13",
    ],
    exec_properties = {
        "gpu": "SM103_CU13",
        "gpu_count": "1",
    },
)

py_test(
    name = "test_hc_mix_triton",
    srcs = ["test_hc_mix_triton.py"],
    main = "test_hc_mix_triton.py",
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = [
        "manual",
        "SM103_CU13",
    ],
    exec_properties = {
        "gpu": "SM103_CU13",
        "gpu_count": "1",
    },
)

# K5 integration lane: Xing4HCUnit fused-vs-torch aggregate UT plus the
# XING4_HC_IMPL factory/fallback contract. Imports model_desc (Xing4HCUnit),
# hence the extra :models/:config deps on top of the lane contract set.
py_test(
    name = "test_hc_unit_fused",
    size = "large",
    srcs = ["test_hc_unit_fused.py"],
    main = "test_hc_unit_fused.py",
    deps = [
        "//rtp_llm:config",
        "//rtp_llm/models_py:models",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = [
        "manual",
        "SM103_CU13",
    ],
    exec_properties = {
        "gpu": "SM103_CU13",
        "gpu_count": "1",
    },
)

py_test(
    name = "test_mean_norm_triton",
    srcs = ["test_mean_norm_triton.py"],
    main = "test_mean_norm_triton.py",
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = [
        "manual",
        "SM103_CU13",
    ],
    exec_properties = {
        "gpu": "SM103_CU13",
        "gpu_count": "1",
    },
)
