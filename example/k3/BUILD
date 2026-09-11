exports_files([
    "kimi_k3_full_model_two_host_pd_smoke.sh",
    "start_kimi_k3_pd.sh",
])

py_library(
    name = "kimi_k3_full_model_pd_cases_lib",
    srcs = [
        "kimi_k3_full_model_pd_cases.py",
        "kimi_k3_long_prefix_case.py",
    ],
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

py_library(
    name = "kimi_k3_full_model_two_host_pd_smoke_driver_lib",
    srcs = ["kimi_k3_full_model_two_host_pd_smoke_driver.py"],
)

py_test(
    name = "kimi_k3_full_model_two_host_pd_smoke_driver_test",
    srcs = ["kimi_k3_full_model_two_host_pd_smoke_driver_test.py"],
    data = ["kimi_k3_full_model_two_host_pd_smoke.sh"],
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
        "kimi_k3_full_model_pd_cases.py",
        "kimi_k3_long_prefix_case.py",
        "kimi_k3_full_model_pd_nic_selection.py",
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
