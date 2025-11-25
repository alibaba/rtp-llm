load("//bazel:arch_select.bzl", "requirement")

py_test(
    name = "test_registry",
    srcs = ["test_registry.py"],
    deps = [
        "//rtp_llm/models_py:modules_rocm",
        "//rtp_llm:testlib",
    ],
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

