load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")
load("@pip_gpu_cuda13_torch//:requirements.bzl", cuda13_requirement = "requirement")
load("@pip_cuda13_arm_torch//:requirements.bzl", cuda13_arm_requirement = "requirement")

cuda13_sm100_py_test(
    name = "native_kda_checkpoint_test",
    srcs = ["native_kda_checkpoint_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ] + select({
        "@//:using_cuda13_x86": [cuda13_requirement("cuda-linear-attention")],
        "@//:using_cuda13_arm": [cuda13_arm_requirement("cuda-linear-attention")],
        "//conditions:default": [],
    }),
)
