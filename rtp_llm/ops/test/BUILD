py_test(
    name = "compute_ops_backend_test",
    size = "small",
    srcs = ["compute_ops_backend_test.py"],
    data = ["//rtp_llm:ops/compute_ops.py"],
    # Pure stdlib import isolation; avoid GPU locks and remote JIT bootstrap.
    env = {"GPU_COUNT": "0"},
)
