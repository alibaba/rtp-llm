load("//rtp_llm/models_py/modules/dsv4/test:defs.bzl", "dsv4_sm100_py_test")

package(default_visibility = ["//visibility:public"])

dsv4_sm100_py_test(
    name = "test_decode_smallops_fusion",
    args = ["-s"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
        "//rtp_llm:deep_gemm",
    ],
    timeout = "long",
)

dsv4_sm100_py_test(
    name = "test_dspark_decode_smallops",
    args = ["-s"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
        "//rtp_llm:deep_gemm",
    ],
    timeout = "long",
)
