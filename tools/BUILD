load("@pip_cuda13_arm_torch//:requirements.bzl", requirement_cuda13_arm = "requirement")
load("@pip_gpu_cuda12_9_torch//:requirements.bzl", requirement_gpu_cuda12_9 = "requirement")
load("@pip_gpu_cuda12_torch//:requirements.bzl", requirement_gpu_cuda12 = "requirement")
load("@pip_gpu_cuda13_torch//:requirements.bzl", requirement_gpu_cuda13 = "requirement")

exports_files(
    ["perf_test"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "perf_test_script",
    srcs = ["perf_test"],
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "nvidia_wheel_env",
    srcs = ["nvidia_wheel_env.sh"],
    data = ["//rtp_llm/test/utils:gpu_lock"] + select({
        "@rtp_llm//:cuda_pre_12_9": [requirement_gpu_cuda12("nvidia-cudnn-cu12")],
        "@rtp_llm//:using_cuda12_9_x86": [requirement_gpu_cuda12_9("nvidia-cudnn-cu12")],
        "@rtp_llm//:using_cuda13_x86": [requirement_gpu_cuda13("nvidia-cudnn-cu13")],
        "@rtp_llm//:using_cuda13_arm": [requirement_cuda13_arm("nvidia-cudnn-cu13")],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

sh_test(
    name = "nvidia_wheel_env_test",
    srcs = ["nvidia_wheel_env_test.sh"],
    data = ["nvidia_wheel_env.sh"],
    env = {"GPU_COUNT": "0"},
)
