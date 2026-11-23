load("//rtp_llm/test/smoke:defs.bzl", "QWEN4_M11_SMOKE_ARGS", "custom_smoke_test")

py_library(
    name = "service_test_lib",
    srcs = ["service_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "config_test",
    srcs = ["config_test.py"],
    env = {"GPU_COUNT": "0"},
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "m11_gdn_test",
    srcs = ["gdn_test.py"],
    main = "gdn_test.py",
    args = ["--m11"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "m11_ops_test",
    srcs = ["gdn_test.py", "m11_ops_test.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
    deps = ["//rtp_llm:testlib"],
)

custom_smoke_test(
    name = "qwen4_m11_mxfp4_sm10x",
    main = "m11_service_test.py",
    smoke_args = QWEN4_M11_SMOKE_ARGS + " --reuse_cache 0 --ssm_state_dtype fp32",
    gpu_type = ["SM103_CU13"],
    data = ["//rtp_llm:sdk"],
    deps = [
        ":service_test_lib",
        "//rtp_llm/test/smoke:qwen4_m11_smoke",
    ],
    test_envs = {"RTP_TEST_WATCHDOG_SECONDS": "5400"},
)

custom_smoke_test(
    name = "qwen4_m11_continuous",
    main = "m11_continuous_server.py",
    smoke_args = QWEN4_M11_SMOKE_ARGS,
    gpu_type = ["SM103_CU13"],
    data = ["//rtp_llm:sdk"],
    deps = [":service_test_lib"],
    tags = ["manual"],
    test_envs = {"RTP_TEST_WATCHDOG_SECONDS": "14400"},
)
