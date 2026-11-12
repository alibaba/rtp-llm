load("//rtp_llm/models_py/test:defs.bzl", "models_py_test")

package(default_visibility = ["//visibility:public"])

models_py_test(
    name = "norm_test",
    src = "norm_test.py",
    gpu = "RTX_5000_PRO_CU13",
)

models_py_test(
    name = "norm_quant_test",
    src = "norm_quant_test.py",
    gpu = "RTX_5000_PRO_CU13",
)

py_library(
    testonly = True,
    name = "prefill_reference",
    srcs = ["reference.py"],
    deps = ["//rtp_llm:testlib"],
)

models_py_test(
    name = "prefill_test",
    src = "prefill_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":prefill_reference",
    ],
)
