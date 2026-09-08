load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "projection_ktp_test",
    srcs = ["test_projection_ktp.py"],
    main = "test_projection_ktp.py",
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:testlib",
    ],
    timeout = "short",
)
