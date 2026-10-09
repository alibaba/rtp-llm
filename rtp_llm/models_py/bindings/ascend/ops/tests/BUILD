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
    data = ["//3rdparty/aclnn_custom_ops:aclnn_custom_ops"],
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

# Python NPU integration test for the Ascend sampler. The .so is pulled into
# the test's runfiles via data and imported as a module from next to the
# test script (see the import-fallback logic in test_ascend_sampler.py).
py_test(
    name = "test_ascend_sampler",
    srcs = ["test_ascend_sampler.py"],
    data = [
        ":sampler_test_module.so",
        "//3rdparty/aclnn_custom_ops:aclnn_custom_ops",
    ],
    deps = [
        "//rtp_llm:torch",
    ],
    # 与同文件 cc_test / cc_binary 的 Ascend 平台门控一致
    target_compatible_with = select({
        "@//:using_ascend": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    tags = ["npu", "exclusive", "manual"],
)
