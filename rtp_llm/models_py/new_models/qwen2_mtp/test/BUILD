py_test(
    name = "test_qwen2_mtp_load",
    args = ["Qwen2MTPLoadTest"],
    srcs = ["test_qwen2_mtp_load.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    deps = [
        "//rtp_llm:safetensors_torch",
        "//rtp_llm:sdk",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

# These tests construct the real C++ result and service boundary. Tensors
# stay on CPU; a CUDA-linked build of the binding requires a driver to dlopen.
py_test(
    name = "qwen2_mtp_native_output_test",
    main = "test_qwen2_mtp_load.py",
    args = ["Qwen2MTPNativeBoundaryTest"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
    srcs = ["test_qwen2_mtp_load.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    deps = [
        "//rtp_llm:safetensors_torch",
        "//rtp_llm:sdk",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)
