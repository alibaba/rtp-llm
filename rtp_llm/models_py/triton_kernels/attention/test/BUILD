py_test(
    name = "test_beam_search_cascade",
    srcs = ["test_beam_search_cascade.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:attention",
        "//rtp_llm:torch",
    ],
    timeout = "long",
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

py_test(
    name = "test_beam_search_cascade_mkt",
    srcs = ["test_beam_search_cascade_mkt.py"],
    args = ["MktCascadeAccuracyTest"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm/models_py/triton_kernels:attention",
        "//rtp_llm:torch",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    timeout = "long",
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

py_test(
    name = "perf_beam_search_cascade_mkt",
    srcs = ["test_beam_search_cascade_mkt.py"],
    main = "test_beam_search_cascade_mkt.py",
    args = ["MktCascadePerformanceTest"],
    env = {"MKT_CASCADE_RUN_PERF": "1"},
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm/models_py/triton_kernels:attention",
        "//rtp_llm:torch",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    timeout = "long",
    tags = ["manual", "perf", "rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

py_test(
    name = "test_beam_search_cascade_mkt_address_boundary",
    srcs = ["test_beam_search_cascade_mkt.py"],
    main = "test_beam_search_cascade_mkt.py",
    args = ["MktCascadeAddressBoundaryTest"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm/models_py/triton_kernels:attention",
        "//rtp_llm:torch",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    timeout = "long",
    tags = ["manual", "rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

# Successful execution collects a controlled numerical experiment; it does not
# constitute kernel or end-to-end model accuracy acceptance.
py_test(
    name = "diagnose_beam_search_cascade_mkt_precision",
    srcs = [
        "diagnose_beam_search_cascade_mkt_precision.py",
        "test_beam_search_cascade_mkt.py",
    ],
    main = "diagnose_beam_search_cascade_mkt_precision.py",
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm/models_py/triton_kernels:attention",
        "//rtp_llm:torch",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    timeout = "long",
    tags = ["diagnostic", "manual", "rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

py_test(
    name = "test_beam_search_cascade_mkt_tail_dispatch",
    srcs = ["test_beam_search_cascade_mkt.py"],
    main = "test_beam_search_cascade_mkt.py",
    args = ["MktCascadeTailDispatchTest"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm/models_py/triton_kernels:attention",
        "//rtp_llm:torch",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    timeout = "long",
    tags = ["manual", "rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

py_test(
    name = "test_beam_search_cascade_mkt_generic_regression",
    srcs = [
        "test_beam_search_cascade.py",
        "test_beam_search_cascade_mkt.py",
    ],
    main = "test_beam_search_cascade_mkt.py",
    args = ["MktCascadeGenericRegressionTest"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm/models_py/triton_kernels:attention",
        "//rtp_llm:torch",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    timeout = "long",
    tags = ["manual", "rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

py_test(
    name = "profile_beam_search_cascade_mkt",
    srcs = ["test_beam_search_cascade_mkt.py"],
    main = "test_beam_search_cascade_mkt.py",
    args = ["MktCascadeProfileTest"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm/models_py/triton_kernels:attention",
        "//rtp_llm:torch",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    timeout = "long",
    tags = ["manual", "profile", "rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

py_test(
    name = "diagnose_beam_search_cascade_mkt_nan_poison",
    srcs = ["test_beam_search_cascade_mkt.py"],
    main = "test_beam_search_cascade_mkt.py",
    args = ["MktCascadeNanPoisonDiagnosisTest"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm/models_py/triton_kernels:attention",
        "//rtp_llm:torch",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    timeout = "long",
    tags = ["diagnostic", "manual", "rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

py_test(
    name = "test_beam_search_cascade_mkt_tail_safety",
    srcs = ["test_beam_search_cascade_mkt.py"],
    main = "test_beam_search_cascade_mkt.py",
    args = ["MktCascadeTailSafetyTest"],
    env = {"MKT_CASCADE_HIGH_ADDRESS_GIB": "4"},
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm/models_py/triton_kernels:attention",
        "//rtp_llm:torch",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    timeout = "long",
    tags = ["manual", "rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

# Builds only inside an explicitly selected private AITER_JIT_DIR; the helper
# rejects installed directories and previously built native target modules.
py_test(
    name = "build_beam_search_cascade_mkt_native_rn",
    srcs = ["build_beam_search_cascade_mkt_native_rn.py"],
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm:aiter",
    ],
    timeout = "long",
    tags = ["diagnostic", "manual", "rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

# Private diagnostic instance: BF16 Q/K/V/P, RN conversions, FP32 prefix O.
py_test(
    name = "build_beam_search_cascade_mkt_native_fp32o",
    srcs = ["build_beam_search_cascade_mkt_native_fp32o.py"],
    data = [
        "native_partial_safe/LICENSE",
        "native_partial_safe/patch_receipt.json",
        "native_partial_safe/include/ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async.hpp",
    ],
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm:aiter",
    ],
    timeout = "long",
    tags = ["diagnostic", "manual", "rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

py_test(
    name = "test_beam_search_cascade_mkt_real_fixture",
    data = [
        "off_precast_candidate/LICENSE",
        "off_precast_candidate/provenance.json",
        "off_probability_candidate/LICENSE",
        "off_probability_candidate/provenance.json",
    ],
    srcs = [
        "diagnose_beam_search_cascade_mkt_off_boundary.py",
        "diagnose_beam_search_cascade_mkt_partial_prefix.py",
        "diagnose_beam_search_cascade_mkt_partial_graph.py",
        "off_precast_candidate/pa_decode_gluon_precast.py",
        "off_probability_candidate/pa_decode_gluon_probabilities.py",
        "test_beam_search_cascade_mkt.py",
        "test_beam_search_cascade_mkt_real_fixture.py",
    ],
    args = ["MktCascadeRealFixtureTest"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm/models_py/triton_kernels:attention",
        "//rtp_llm:torch",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    timeout = "long",
    tags = ["manual", "rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)
