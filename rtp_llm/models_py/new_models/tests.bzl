"""Native-runtime regression targets for the first dense NewLoader batch."""

def dense_loader_test(name, src, gpu = False, deps = []):
    for suffix, tag, device in [
        ("", "H20_CU13", "H20_CU13"),
        ("_rocm", "rocm", "MI308X-ROCM7"),
    ]:
        native.py_test(
            name = name + suffix,
            srcs = [src],
            main = src,
            deps = deps + [
                "//rtp_llm/models_py:new_loader_models",
                "//rtp_llm/models_py/standalone:py_standalone_testlib",
                "//rtp_llm:numpy",
                "//rtp_llm:safetensors",
                "//rtp_llm:torch",
            ],
            data = ["//:th_transformer_config", "//:rtp_compute_ops", "//:th_transformer"],
            env = {"GPU_COUNT": "1"} if gpu else {
                "CUDA_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
                "ROCR_VISIBLE_DEVICES": "",
            },
            # CPU tensor tests still import the native runtime extension.
            # Use accelerator images, not a CPU-only wheel with absent ops.
            tags = [tag],
            exec_properties = {"gpu": device, "gpu_count": "1"},
        )
