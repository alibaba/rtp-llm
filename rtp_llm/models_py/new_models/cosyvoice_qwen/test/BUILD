py_test(
    name = "test_inputs",
    srcs = ["test_inputs.py"],
    deps = ["//rtp_llm/models_py:new_weight_loader"],
)

py_test(
    name = "test_language",
    args = ["MultimodalLanguageLoadTest"],
    srcs = ["test_language.py"],
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "HIP_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
    },
    deps = [
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:safetensors",
        "//rtp_llm:sdk",
        "//rtp_llm:torch",
        "//rtp_llm:transformers",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

# These tests construct the real C++ result and service boundary. Tensors
# stay on CPU; a CUDA-linked build of the binding requires a driver to dlopen.
py_test(
    name = "native_language_output_test",
    main = "test_language.py",
    args = ["MultimodalLanguageNativeBoundaryTest"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
    srcs = ["test_language.py"],
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "HIP_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
    },
    deps = [
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:safetensors",
        "//rtp_llm:sdk",
        "//rtp_llm:torch",
        "//rtp_llm:transformers",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)
