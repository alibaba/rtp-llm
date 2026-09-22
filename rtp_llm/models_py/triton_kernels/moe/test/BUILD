load("//bazel:arch_select.bzl", "requirement")

requirement(["torch", "triton"])

py_test(
    name = "output_add_test",
    srcs = ["output_add_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:moe",
        ":torch",
        ":triton",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20", "gpu_count": "1"},
    tags = ["manual", "H20"],
)
