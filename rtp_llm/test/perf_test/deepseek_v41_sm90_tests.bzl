"""Fixed standalone CP8 prefill and DP8 decode matrices on eight H20 GPUs."""

def deepseek_v41_sm90_role_test(name, role, mode):
    native.py_test(
        name = name,
        main = "//rtp_llm/test/perf_test:deepseek_v41_sm90_role_test.py",
        srcs = ["//rtp_llm/test/perf_test:deepseek_v41_sm90_role_test.py"],
        args = ["--role", role, "--mode", mode],
        deps = ["//rtp_llm:testlib", "//rtp_llm/test/perf_test:perf_test_lib"],
        data = ["//rtp_llm:sdk"],
        env = {"WORLD_SIZE": "8", "GPU_COUNT": "8"},
        timeout = "eternal",
        tags = ["manual", "dsv41_cuda13", "H20_CU13", "standalone_pd"],
        exec_properties = {"gpu": "H20_CU13", "gpu_count": "8"},
        target_compatible_with = select({
            "@//:using_cuda13_x86": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
