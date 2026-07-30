package(default_visibility = ["//visibility:public"])

py_binary(
    name = "kimi_k3_prefill_server",
    srcs = ["server_main.py"],
    main = "server_main.py",
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer",
        "//:th_transformer_config",
    ],
    deps = [
        "//rtp_llm:rtp_llm_lib",
        "//rtp_llm:sdk",
    ],
)
