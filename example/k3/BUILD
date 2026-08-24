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

sh_binary(
    name = "kimi_k3_full_model_two_host_pd_smoke",
    srcs = ["kimi_k3_full_model_two_host_pd_smoke.sh"],
    data = [
        "start_kimi_k3_pd.sh",
        "//rtp_llm:rtp_llm_server",
    ],
)
