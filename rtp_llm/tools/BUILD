package(default_visibility = ["//:__subpackages__"])

py_binary(
    name = "analyze_cuda_call_trace",
    srcs = ["analyze_cuda_call_trace.py"],
    main = "analyze_cuda_call_trace.py",
    python_version = "PY3",
)

py_test(
    name = "analyze_cuda_call_trace_test",
    srcs = ["analyze_cuda_call_trace.py", "analyze_cuda_call_trace_test.py"],
    main = "analyze_cuda_call_trace_test.py",
    python_version = "PY3",
    tags = ["no_gpu"],
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

