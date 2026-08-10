package(default_visibility = ["//visibility:public"])
exports_files(["upload_package.py"])
exports_files(["pyc_wheel.py"])
exports_files(["cuda13_packages.bzl"])

# Static, lane-agnostic consistency check for the duplicated CUDA 13 wheel
# declarations (see the module docstring). Tagged manual so it never enters
# wildcard //... builds — @rtp_deps / @arch_config are override-swapped in
# internal builds. Run it explicitly (e.g. in the opensource / cuda13 lane):
#   bazel test //bazel:cuda13_wheel_consistency_test
py_test(
    name = "cuda13_wheel_consistency_test",
    srcs = ["check_cuda13_wheel_consistency.py"],
    main = "check_cuda13_wheel_consistency.py",
    args = [
        "$(location @arch_config//:arch_select.bzl)",
        "$(location :cuda13_packages.bzl)",
        "$(location @rtp_deps//:requirements_torch_gpu_cuda13.txt)",
        "$(location @rtp_deps//:requirements_cuda13_arm.txt)",
    ],
    data = [
        ":cuda13_packages.bzl",
        "@arch_config//:arch_select.bzl",
        "@rtp_deps//:requirements_cuda13_arm.txt",
        "@rtp_deps//:requirements_torch_gpu_cuda13.txt",
    ],
    tags = ["manual"],
)
