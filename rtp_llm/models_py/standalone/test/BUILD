
test_envs = {}

py_test (
    name = "qwen3_test",
    srcs = ["qwen3_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:auto_model",
    ],
    env = test_envs,
    tags = ["H20"],
    # Back on H20. This test asserts the generated text equals a hardcoded
    # expected string, so it is sensitive to any change in reduction order. On
    # A10 in run 69714248 the model produced different but perfectly coherent
    # Chinese and the assertEqual failed.
    #
    # It did pass on the 8x RTX 3090 sm86 proxy used to vet these moves, which is
    # where that proxy falls short: equal compute capability is not equal
    # hardware. The 3090 has 82 SMs and the A10 has 72, and kernels that
    # partition work by SM count accumulate in a different order -- enough to
    # flip a greedy token. Exact-output tests need the real target GPU, not a
    # same-capability stand-in. Staying on H20 by decision.
    exec_properties = {'gpu':'H20'},
)

