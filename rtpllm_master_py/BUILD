package(default_visibility = ["//visibility:public"])
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")

py_library(
    name = "rtpllm_master_lib",
    srcs = glob(["*.py"]),
    data = [
        "//rtpllm_master_py/stub:stubs"
    ],
)

py_package(
    name = "rtpllm_master_package",
    deps = [
        ":rtpllm_master_lib"
    ],
    packages = [
    ],
)

py_wheel(
    name = "rtpllm_master_whl",
    distribution = "rtpllm_master",
    python_tag = "py3",
    tags = ["manual", "local"],
    version = "0.0.1",
    deps = [
        ":rtpllm_master_package"
    ],
    requires = [],
)
