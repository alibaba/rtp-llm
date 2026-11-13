# Infrastructure-only tests do not instantiate any accelerator backend.
py_test(
    name = "foundation_test",
    srcs = ["foundation_test.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""},
    deps = [
        "//rtp_llm/models_py:new_loader_core",
        "//rtp_llm:new_loader_policy",
        # safetensors.save_file imports NumPy when serializing CPU tensors.
        "//rtp_llm:numpy",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
    ],
)

py_test(
    name = "quantization_exclusion_test",
    srcs = ["quantization_exclusion_test.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""},
    deps = ["//rtp_llm/models_py:quantization_exclusion"],
)

py_test(
    name = "core_boundary_test",
    srcs = ["core_boundary_test.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""},
    deps = ["//rtp_llm/models_py:new_loader_core"],
)
