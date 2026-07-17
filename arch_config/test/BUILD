# Consistency test for the XPU wheel version overrides.
#
# test_xpu_version_overrides: verifies _XPU_VERSION_OVERRIDES in
#   arch_config/arch_select.bzl stays in sync with the versions locked in
#   deps/requirements_lock_xpu.txt.
#
# Pure Python test; the .bzl source and lockfile are provided as data so the
# test can read them inside the bazel sandbox.

py_test(
    name = "test_xpu_version_overrides",
    srcs = ["test_xpu_version_overrides.py"],
    data = [
        "//arch_config:arch_select.bzl",
        "//deps:requirements_lock_xpu.txt",
    ],
    timeout = "short",
    tags = ["xpu"],
)
