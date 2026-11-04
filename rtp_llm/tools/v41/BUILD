py_library(
    name = "tooling",
    srcs = ["__init__.py", "create_rdma_container.py", "launch.py", "native_build_precheck.py", "native_smoke.py", "pd_joint.py", "preflight.py", "prepare_native_runtime.py"],
    deps = [
        "//rtp_llm:v41_runtime_contract",
        "//rtp_llm/models_py:v41_ppu_contract",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "integration_test",
    srcs = ["test_integration.py"],
    data = ["//rtp_llm/models_py:v41_dispatch_sources"],
    deps = [":tooling"],
    tags = ["cpu"],
)
