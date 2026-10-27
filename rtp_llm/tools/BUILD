package(default_visibility = ["//:__subpackages__"])

py_library(
    name = "collect_host_stall",
    srcs = ["collect_host_stall.py"],
)

py_test(
    name = "collect_host_stall_test",
    srcs = ["test_collect_host_stall.py"],
    main = "test_collect_host_stall.py",
    deps = [":collect_host_stall"],
    tags = ["cpu"],
)

py_library(
    name = "model_assistant_api",
    srcs = glob([
        "api/*.py",
    ]),
)

py_library(
    name = "model_assistant",
    srcs = [
           "model_assistant_server.py",
           ":model_assistant_api"
    ]
)

