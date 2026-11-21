load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")

# Separate processes: an asynchronous CUDA error poisons its context. The
# control intentionally waits before FST can release each source tensor.
[
    cuda13_sm100_py_test(
        name = "v4_fst_startup_" + mode,
        srcs = ["v4_fst_startup.py"],
        main = "v4_fst_startup.py",
        args = [mode],
        gpu_count = 2,
        # In V4 prefill the nonzero device's file buffer uses expandable
        # segments. Keep the default allocator as a separate control.
        env = {
            "FASTSAFETENSORS_NOGDS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        } if mode == "fenced" else {"FASTSAFETENSORS_NOGDS": "1"},
        tags = ["manual"],
        deps = [
            "//rtp_llm:fastsafetensors",
            "//rtp_llm:testlib",
            "//rtp_llm/models_py:new_loader_core",
            "//rtp_llm/models_py:new_weight_loader",
        ],
    )
    for mode in ["ordinary", "fenced"]
]
