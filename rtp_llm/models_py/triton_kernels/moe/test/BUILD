load("@arch_config//:arch_select.bzl", "requirement")

requirement(["triton", "torch", "numpy"])

py_test(
    name = "test_remap_local_ids",
    srcs = ["test_remap_local_ids.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:moe",
    ] + [":triton", ":torch"],
    env = {"GPU_COUNT": "1"},
    tags = ["rocm"],
    exec_properties = {'gpu': 'MI308X-ROCM7', 'gpu_count': '1'},
)

py_test(
    name = "test_ep_kernels",
    srcs = ["test_ep_kernels.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:moe",
    ] + [":triton", ":torch", ":numpy"],
    visibility = ["//visibility:public"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "test_ep_scatter",
    srcs = ["test_ep_scatter.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:moe",
    ] + [":triton", ":torch"],
    visibility = ["//visibility:public"],
    tags = ["H20"],
    # The crash source is gone: test_old_kernel_reads_poison is now opt-in behind
    # RTP_LLM_EP_SCATTER_POISON_DIAG. It asserted nothing and only printed a
    # mismatch count, yet drove a knowingly racy kernel 100 times, and when that
    # raised an illegal memory access the poisoned CUDA context took every sibling
    # test down with it -- which is what made this target order-dependent (2 of 3
    # sm86 runs failed, passing when run alone). test_new_kernel_immune_to_poison
    # is the real gate and still runs by default.
    #
    # Still pinned to H20, where it passes today, because the first test to fail on
    # sm86 was never isolated -- it could have been the 200-round stress case in
    # TestEpScatter1Correctness rather than the poison class. Candidate for A10 once
    # someone confirms a clean sm86 run with the diagnostic gated.
    exec_properties = {'gpu':'H20'},
)
