load("//rtp_llm/models_py/test:defs.bzl", "models_py_test")

package(default_visibility = ["//visibility:public"])

models_py_test(
    name = "marlin_large_decode_test",
    src = "marlin_large_decode_test.py",
    gpu = "RTX_5000_PRO_CU13",
    testlib = "//rtp_llm/models_py/standalone:auto_model",
)

models_py_test(
    name = "marlin_prefill_test",
    src = "marlin_prefill_test.py",
    gpu = "RTX_5000_PRO_CU13",
    testlib = "//rtp_llm/models_py/standalone:auto_model",
)

exports_files([
    "marlin_large_decode_test.py",
    "marlin_nvfp4_test.py",
])

py_library(
    testonly = True,
    name = "marlin_test_support",
    srcs = ["marlin_fixture.py"],
    visibility = ["//visibility:public"],
    deps = ["//rtp_llm:testlib"],
)

models_py_test(
    name = "marlin_nvfp4_test",
    src = "marlin_nvfp4_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [":marlin_test_support"],
)

models_py_test(
    name = "marlin_nvfp4_hopper_test",
    src = "marlin_nvfp4_test.py",
    args = ["MarlinNvfp4Test.test_precompiled_library_in_fresh_process_without_toolchain"],
    gpu = "H20_CU13",
    deps = [":marlin_test_support"],
)
