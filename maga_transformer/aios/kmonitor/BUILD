
py_library(
    name = "kmon_flume",
    srcs = glob(["python_client/flume/*.py"]),
    visibility = ["//visibility:public"],
    deps = ["//maga_transformer:thrift"],
)

py_library(
    name = "kmonitor_py",
    srcs = glob([
        "python_client/kmonitor/*.py",
        "python_client/kmonitor/**/*.py",
    ]),
    imports = ["./python_client/"],
    visibility = ["//visibility:public"],
    deps = [":kmon_flume"],
)

py_test(
    name = "kmon_py_test",
    srcs = glob(["python_client/test/*.py"]),
    main = "python_client/test/kmon_test.py",
    deps = [":kmonitor_py"],
)
