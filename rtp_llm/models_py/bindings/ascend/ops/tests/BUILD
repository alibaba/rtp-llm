load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

# ============================================================
# C++ compile-time check (does NOT run on NPU without Python)
# ============================================================
cc_test(
    name = "ascend_sampler_test",
    srcs = [
        "SamplerTest.cc",
    ],
    data = [],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings/core:exec_ctx_state",
        "//rtp_llm/models_py/bindings/core:exec_ctx_ops",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    # Only buildable on Ascend platform
    target_compatible_with = select({
        "@//:using_ascend": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    tags = ["npu", "exclusive", "manual"],
)

# ============================================================
# Pybind11 module for Python-based NPU integration tests.
#
# IMPORTANT: this .so MUST be loaded AFTER Python `import torch_npu`.
# The DT_NEEDED libtorch_npu.so is harmlessly reused (already loaded
# by Python), avoiding the "Two accelerators" conflict.
# ============================================================
cc_binary(
    name = "sampler_test_module.so",
    srcs = ["SamplerPybind.cc"],
    deps = [
        "//rtp_llm/models_py/bindings/core:exec_ctx_state",
        "//rtp_llm/models_py/bindings/core:exec_ctx_ops",
    ] + torch_deps(),
    linkshared = True,
    linkstatic = True,
    copts = copts(),
    # Only buildable on Ascend platform
    target_compatible_with = select({
        "@//:using_ascend": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
