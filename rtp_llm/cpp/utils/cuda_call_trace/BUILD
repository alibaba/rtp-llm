load("//:def.bzl", "copts")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cuda_call_trace",
    srcs = ["CudaCallTrace.cc"],
    hdrs = ["CudaCallTrace.h", "RecordQueue.h", "IncidentRetention.h", "ActiveRecords.h"],
    textual_hdrs = ["ApiParams.inc"],
    copts = copts(),
    linkopts = ["-ldl", "-pthread"],
    deps = select({
        "//:using_cuda": ["@local_config_cuda//cuda:cupti_headers"],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "active_records_test",
    srcs = ["ActiveRecordsTest.cc"],
    copts = copts(),
    linkopts = ["-pthread"],
    deps = [":cuda_call_trace"],
)

cc_test(
    name = "record_queue_test",
    srcs = ["RecordQueueTest.cc"],
    copts = copts(),
    linkopts = ["-pthread"],
    deps = [":cuda_call_trace"],
)

cc_test(
    name = "incident_retention_test",
    srcs = ["IncidentRetentionTest.cc"],
    copts = copts(),
    deps = [":cuda_call_trace"],
)

# Uses CUDA headers but no device/context; validates an unreturned API and a
# blocked CUPTI flush without hanging either the CPU writer or the test runner.
cc_binary(
    name = "fake_cupti_test_library.so",
    srcs = ["FakeCuptiTestLibrary.cc"],
    copts = copts(),
    linkshared = True,
    linkopts = ["-pthread"],
    deps = ["@local_config_cuda//cuda:cupti_headers"],
    target_compatible_with = select({
        "//:using_cuda13_x86": [],
        "//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

cc_test(
    name = "recorder_host_test",
    srcs = ["RecorderHostTest.cc"],
    args = ["$(location :fake_cupti_test_library.so)"],
    data = [":fake_cupti_test_library.so"],
    copts = copts(),
    linkopts = ["-ldl", "-pthread"],
    deps = [":cuda_call_trace"],
    target_compatible_with = select({
        "//:using_cuda13_x86": [],
        "//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
