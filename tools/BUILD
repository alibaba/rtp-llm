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
