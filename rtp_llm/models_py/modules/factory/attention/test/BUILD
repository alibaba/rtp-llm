# Factory-selection logic tests (registry monkeypatched, no kernels run).
# The gpu exec_properties is only for the import chain (rtp_llm.ops loads
# the compiled .so from the GPU image), not for actual GPU compute.
py_test(
    name = "attn_factory_test",
    srcs = ["attn_factory_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)
