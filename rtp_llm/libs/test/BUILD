py_test(
    name = "packaged_cuda13_libs_test",
    srcs = ["packaged_cuda13_libs_test.py"],
    data = [
        "//rtp_llm/libs:libs",
        "//rtp_llm/libs:whl_package_libs",
    ],
    deps = ["//rtp_llm:torch"],
    target_compatible_with = select({
        "//:using_cuda13_x86": [],
        "//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
