exports_files([
    "cp_readme.md",
    "kimi_k3_full_model_two_host_pd_smoke.sh",
    "start_kimi_k3_pd.sh",
])

py_library(
    name = "kimi_k3_full_model_pd_cases_lib",
    srcs = [
        "kimi_k3_full_model_pd_cases.py",
        "kimi_k3_long_prefix_case.py",
    ],
    data = ["//rtp_llm/multimodal/test:testdata/qwen2_vl/1.jpg"],
)

py_library(
    name = "kimi_k3_cache_pressure_lib",
    srcs = ["kimi_k3_long_prefix_case.py"],
)

py_binary(
    name = "kimi_k3_cache_pressure",
    srcs = ["kimi_k3_cache_pressure.py"],
    deps = [":kimi_k3_cache_pressure_lib"],
    data = ["//rtp_llm/multimodal/test:testdata/qwen2_vl/1.jpg"],
)

py_binary(
    name = "kimi_k3_perf",
    srcs = ["kimi_k3_perf.py"],
    deps = ["//rtp_llm/test/perf_test:perf_test_lib"],
    data = ["//rtp_llm:sdk"],
)

py_test(
    name = "kimi_k3_perf_test",
    srcs = ["kimi_k3_perf_test.py", "kimi_k3_perf.py"],
    deps = ["//rtp_llm/test/perf_test:perf_test_lib"],
)

py_test(
    name = "kimi_k3_full_model_pd_cases_test",
    srcs = ["kimi_k3_full_model_pd_cases_test.py"],
    deps = [":kimi_k3_full_model_pd_cases_lib"],
)

py_library(
    name = "kimi_k3_full_model_pd_nic_selection_lib",
    srcs = ["kimi_k3_full_model_pd_nic_selection.py"],
)

py_test(
    name = "kimi_k3_full_model_pd_nic_selection_test",
    srcs = ["kimi_k3_full_model_pd_nic_selection_test.py"],
    deps = [":kimi_k3_full_model_pd_nic_selection_lib"],
)

py_test(
    name = "kimi_k3_rdma_readiness_test",
    srcs = ["kimi_k3_rdma_readiness_test.py", "kimi_k3_rdma_readiness.py"],
)

py_library(
    name = "kimi_k3_cache_evidence_lib",
    srcs = ["kimi_k3_cache_evidence.py"],
)

py_test(
    name = "kimi_k3_cache_evidence_test",
    srcs = ["kimi_k3_cache_evidence_test.py"],
    deps = [":kimi_k3_cache_evidence_lib"],
)

py_library(
    name = "kimi_k3_full_model_two_host_pd_smoke_driver_lib",
    srcs = ["kimi_k3_full_model_two_host_pd_smoke_driver.py"],
    deps = ["//rtp_llm/test/utils:test_util"],
)

py_test(
    name = "kimi_k3_full_model_two_host_pd_smoke_driver_test",
    srcs = ["kimi_k3_full_model_two_host_pd_smoke_driver_test.py"],
    data = [
        "//rtp_llm/multimodal/test:testdata/qwen2_vl/1.jpg",
        "kimi_k3_full_model_two_host_pd_smoke.sh",
        "start_kimi_k3_pd.sh",
        "kimi_k3_full_model_pd_cases.py",
        "kimi_k3_long_prefix_case.py",
        "kimi_k3_logits_compare.py",
        "kimi_k3_full_model_pd_nic_selection.py",
        "kimi_k3_rdma_readiness.py",
        "kimi_k3_cache_evidence.py",
    ],
    deps = [":kimi_k3_full_model_two_host_pd_smoke_driver_lib"],
)

py_test(
    name = "start_kimi_k3_pd_test",
    srcs = ["start_kimi_k3_pd_test.py"],
    data = ["start_kimi_k3_pd.sh"],
)


sh_binary(
    name = "kimi_k3_full_model_two_host_pd_smoke",
    srcs = ["kimi_k3_full_model_two_host_pd_smoke.sh"],
    data = [
        ":kimi_k3_full_model_pd_cases_lib",
        ":kimi_k3_cache_pressure_lib",
        ":kimi_k3_full_model_two_host_pd_smoke_driver_lib",
        "kimi_k3_full_model_pd_cases.py",
        "kimi_k3_long_prefix_case.py",
        "kimi_k3_logits_compare.py",
        "kimi_k3_full_model_pd_nic_selection.py",
        "kimi_k3_rdma_readiness.py",
        "kimi_k3_cache_evidence.py",
        "kimi_k3_cache_pressure.py",
        "kimi_k3_perf.py",
        "start_kimi_k3_pd.sh",
        "//rtp_llm:rtp_llm_server",
    ],
)

# Build probes with the same operators as the server; execution still requires
# the GPU host-selector gate. None of these probes loads a checkpoint.
py_binary(
    name = "fp8_shape_probe",
    srcs = ["fp8_shape_probe.py"],
    deps = ["//rtp_llm:rtp_llm_lib"],
    data = ["//:rtp_compute_ops", "//:th_transformer", "//:th_transformer_config"],
)

py_binary(
    name = "fp8_runtime_probe",
    srcs = ["fp8_runtime_probe.py"],
    deps = ["//rtp_llm:rtp_llm_lib"],
    data = ["//:rtp_compute_ops", "//:th_transformer", "//:th_transformer_config"],
)

py_binary(
    name = "fp8_collective_probe",
    srcs = ["fp8_collective_probe.py", "fp8_shape_probe.py"],
    deps = ["//rtp_llm:rtp_llm_lib"],
    data = ["//:rtp_compute_ops", "//:th_transformer", "//:th_transformer_config"],
)

py_library(
    name = "kimi_k3_logits_compare_lib",
    srcs = ["kimi_k3_logits_compare.py"],
)

py_test(
    name = "kimi_k3_logits_compare_test",
    srcs = ["kimi_k3_logits_compare_test.py"],
    deps = [":kimi_k3_logits_compare_lib"],
)

py_test(
    name = "kimi_k3_long_prefix_case_test",
    srcs = ["kimi_k3_long_prefix_case_test.py"],
    deps = [":kimi_k3_full_model_pd_cases_lib"],
)
