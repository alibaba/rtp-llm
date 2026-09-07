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
    # Held on H20 because its sm86 behaviour is not understood, not because it
    # needs sm90. On sm86 TestEpScatter1PoisonRegression raised
    # "CUDA error: an illegal memory access" in 2 of 3 runs and passed once in
    # isolation; no H20 baseline exists in any CI log, so neither "regression"
    # nor "fine" is supportable yet.
    #
    # Worth knowing before touching it: test_old_kernel_reads_poison
    # deliberately drives a kernel known to be buggy (256 experts x 100 rounds),
    # and an IMA poisons the CUDA context for every test after it in the same
    # process. That makes the whole target order-dependent regardless of GPU.
    # Root-cause that first; pin it afterwards.
    exec_properties = {'gpu':'H20'},
)
