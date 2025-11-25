load("//bazel:arch_select.bzl", "requirement")

py_test(
    name = "test_registry",
    srcs = ["test_registry.py"],
    deps = [
        "//rtp_llm/models_py:modules_cuda",
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'H20'},
)

