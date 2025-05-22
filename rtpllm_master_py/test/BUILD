load("//bazel:arch_select.bzl", "requirement", "whl_deps")

requirement(["fastapi", "uvicorn", "torch", "orjson"])

filegroup(
    name = "testdata",    
    data = glob(["testdata/**/*"]),
)

py_test(
    name = "rtpllm_master_test",
    srcs = [
       "rtpllm_master_test.py",
       "fake_server.py"
    ],
    deps = [
        "//rtpllm_master_py:rtpllm_master_lib",
        "//rtp_llm/test/utils:port_util",
        ":fastapi",
        ":uvicorn",
        ":torch",
        ":orjson"
    ],
    data = [
        "//rtpllm_master_py/test:testdata"
    ]
)