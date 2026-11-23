load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = [
    "//3rdparty/blackwell:__pkg__",
    "//rtp_llm/libs:__pkg__",
])

exports_files([
    "format",
    "mhc_binding.cpp",
    "sparse_mla_registration.cpp",
])

py_binary(
    name = "build_extension",
    srcs = ["build_extension.py"],
    deps = [
        "//rtp_llm:numpy",
        "//rtp_llm:setuptools",
        "//rtp_llm:torch",
    ],
)
