exports_files([
    "kimi_k3_full_model_two_host_pd_smoke.sh",
    "start_kimi_k3_pd.sh",
])

py_library(
    name = "kimi_k3_full_model_pd_cases_lib",
    srcs = ["kimi_k3_full_model_pd_cases.py"],
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
    deps = [":kimi_k3_full_model_two_host_pd_smoke_driver_lib"],
)

sh_binary(
    name = "kimi_k3_full_model_two_host_pd_smoke",
    srcs = ["kimi_k3_full_model_two_host_pd_smoke.sh"],
    data = [
        "kimi_k3_full_model_pd_nic_selection.py",
        "start_kimi_k3_pd.sh",
        "//rtp_llm:rtp_llm_server",
    ],
)
