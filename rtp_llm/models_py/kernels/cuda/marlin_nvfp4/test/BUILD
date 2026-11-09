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

exports_files(["marlin_large_decode_test.py"])
