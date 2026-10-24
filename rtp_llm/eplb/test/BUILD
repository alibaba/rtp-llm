load("@arch_config//:arch_select.bzl", "requirement")
load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")

requirement(["torch"])

py_test(
    name = "policy_test",
    srcs = ["policy_test.py"],
    deps = ["//rtp_llm:eplb", ":torch"],
)

cuda13_sm100_py_test(
    name = "runtime_distributed_test",
    srcs = ["runtime_distributed_test.py", "runtime_test.py"],
    gpu_count = 2,
    timeout = "moderate",
    deps = ["//rtp_llm:eplb", ":torch"],
)

py_test(
    name = "runtime_test",
    srcs = ["runtime_test.py"],
    deps = ["//rtp_llm:eplb", ":torch"],
)
