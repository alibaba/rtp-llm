# Run each binding against only its own dependencies. The script starts the
# test body unconditionally so unsupported devices and missing kernels fail.
[
    py_test(
        name = "test_" + op + "_binding",
        srcs = ["test_" + op + "_binding.py"],
        main = "test_" + op + "_binding.py",
        deps = [
            "//rtp_llm:package_initializer",
            "//rtp_llm:torch",
            "//rtp_llm/models_py/kernels/cuda/sm120/" + op,
        ],
        env = {"GPU_COUNT": "1"},
        tags = ["dsv4_cuda13", "RTX_5000_PRO_CU13"],
        exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
        target_compatible_with = select({
            "@//:using_cuda13_x86": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
        legacy_create_init = 0,
        timeout = "long",
    )
    for op in ["graph_moe", "grouped_decode", "sparse"]
]
