exports_files(["cbb-error-child.patch"])

cc_test(
    name = "cbb_error_test",
    srcs = ["CbbErrorTest.cc"],
    deps = [
        "@boringssl//:crypto",
        "@com_google_googletest//:gtest_main",
    ],
)
