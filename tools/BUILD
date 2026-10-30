sh_binary(
    name = "nvidia_wheel_env",
    srcs = ["nvidia_wheel_env.sh"],
    data = ["//rtp_llm/test/utils:gpu_lock"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "runtime_library_path_test",
    size = "small",
    srcs = ["runtime_library_path_test.py"],
    data = ["runtime_library_path.sh"],
)

py_test(
    name = "collect_bazel_testlogs_test",
    size = "small",
    srcs = ["collect_bazel_testlogs.py", "collect_bazel_testlogs_test.py"],
    env = {"GPU_COUNT": "0"},
)
