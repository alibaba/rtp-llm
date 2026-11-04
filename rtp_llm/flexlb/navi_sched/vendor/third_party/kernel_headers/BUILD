load("//rtp_llm/flexlb/navi_sched/vendor/bazel:defs.bzl", "rpm_library")

rpm_library(
    name = "kernel_headers",
    rpms = [
        "@kernel_headers//file:file",
    ],
    hdrs = [
        "perf_event.h",
    ],
    include_path = "usr/include/linux",
    lib_path = "usr/lib64",
    include_prefix = "linux",
    header_only = True,
)
