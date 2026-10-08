package(default_visibility = ["//visibility:public"])
exports_files(["upload_package.py"])
exports_files(["pyc_wheel.py"])

py_binary(
    name = "merge_cutlass_wheels",
    srcs = ["merge_cutlass_wheels.py"],
)

py_library(
    name = "cutlass_wheel_merger",
    srcs = ["merge_cutlass_wheels.py"],
)

py_test(
    name = "merge_cutlass_wheels_test",
    srcs = ["merge_cutlass_wheels_test.py"],
    deps = [":cutlass_wheel_merger"],
)
