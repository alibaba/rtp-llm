py_test(
    name = "test_embedding_profile_config",
    srcs = [
        "test_embedding_profile_config.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    # Pure dict-parsing test; the gpu slot is only for the import chain
    # (//rtp_llm:testlib transitively loads the compiled .so from the GPU
    # image), matching the repo's other py_test targets.
    exec_properties = {'gpu':'H20'},
)
