py_test(
    name = "hidden_state_capture_test",
    srcs = ["hidden_state_capture_test.py"],
    # Use the same Python/native runtime as the neighboring model/operator tests.
    # Test tensors stay on CPU; no accelerator allocation or kernel execution.
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)
