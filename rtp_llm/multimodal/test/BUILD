filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/*",
        "testdata/**",
    ]),
)

py_test(
    name = "multimodal_image_load_test",
    srcs = [
        "multimodal_image_load_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "multimodal_process_engine_test",
    srcs = [
        "multimodal_process_engine_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    data = [
        ":testdata",
    ],
    env = {
        "MM_CACHE_ITEM_NUM": "0"
    },
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "multimodal_util_test",
    srcs = [
        "multimodal_util_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    data = [
        ":testdata",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "mm_profiler_test",
    srcs = [
        "mm_profiler_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "vit_metrics_test",
    srcs = [
        "vit_metrics_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "qwen35_prefix_detection_test",
    srcs = [
        "qwen35_prefix_detection_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "qwen2_vl_newloader_routing_test",
    srcs = ["qwen2_vl_newloader_routing_test.py"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:multimodal",
        "//rtp_llm:safetensors",
        "//rtp_llm:testlib",
        "//rtp_llm:torch",
        "//rtp_llm/model_loader:loader",
        "//rtp_llm/models_py:qwen2_vl_vision_loader",
    ],
)

py_test(
    name = "qwen3_vl_newloader_routing_test",
    srcs = ["qwen3_vl_newloader_routing_test.py"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:multimodal",
        "//rtp_llm:models",
        "//rtp_llm:safetensors",
        "//rtp_llm:testlib",
        "//rtp_llm:torch",
        "//rtp_llm/model_loader:loader",
        "//rtp_llm/models_py:qwen3_vl_vision_loader",
    ],
)
