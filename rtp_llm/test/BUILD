py_test(
    name = "generate_config_test",
    srcs = [
        "generate_config_test.py",
    ],
    data = [
        "//rtp_llm:sdk",
        "//rtp_llm/test/model_test/fake_test/testdata",
    ],
    exec_properties = {"gpu": "A10"},
    deps = [
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "slice_stop_word_list_test",
    srcs = [
        "slice_stop_word_list_test.py",
    ],
    data = [
        "//rtp_llm:sdk",
        "//rtp_llm/test/model_test/fake_test/testdata",
    ],
    exec_properties = {"gpu": "A10"},
    imports = ["aios/kmonitor/python_client"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util",
    ],
)

py_test(
    name = "stop_words_test",
    srcs = [
        "stop_words_test.py",
    ],
    data = [],
    exec_properties = {"gpu": "A10"},
    imports = [],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util",
    ],
)

py_test(
    name = "renderer_stop_words_test",
    srcs = [
        "renderer_stop_words_test.py",
    ],
    data = [],
    exec_properties = {"gpu": "A10"},
    imports = [],
    deps = [
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "reasoning_tool_parsing_test",
    srcs = [
        "reasoning_tool_parsing_test.py",
    ],
    data = [],
    exec_properties = {"gpu": "A10"},
    imports = [],
    deps = [
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "stop_words_pipeline_test",
    srcs = [
        "stop_words_pipeline_test.py",
    ],
    data = [],
    exec_properties = {"gpu": "A10"},
    imports = [],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util",
    ],
)

py_test(
    name = "openai_response_test",
    srcs = [
        "openai_response_test.py",
    ],
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata",
        "//rtp_llm/test/tokenizer_test:testdata",
    ],
    exec_properties = {"gpu": "A10"},
    deps = [
        "//rtp_llm:sdk",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "server_test",
    timeout = "eternal",
    srcs = [
        "server_test.py",
    ],
    data = [
        "//rtp_llm:sdk",
    ],
    imports = ["aios/kmonitor/python_client"],
    tags = ["manual"],
    deps = [
        "//rtp_llm:fastapi",
        "//rtp_llm:psutil",
        "//rtp_llm:testlib",
        "//rtp_llm:tiktoken",
        "//rtp_llm:uvicorn",
    ],
)

py_test(
    name = "yemu_pd_sep_test",
    timeout = "eternal",
    srcs = [
        "yemu_pd_sep_test.py",
    ],
    data = [
        "//rtp_llm:sdk",
    ],
    imports = ["aios/kmonitor/python_client"],
    tags = ["manual"],
    deps = [
        "//rtp_llm:fastapi",
        "//rtp_llm:psutil",
        "//rtp_llm:testlib",
        "//rtp_llm:tiktoken",
        "//rtp_llm:uvicorn",
    ],
)

py_test(
    name = "concurrency_limit_test",
    timeout = "short",
    srcs = [
        "concurrency_limit_test.py",
    ],
    exec_properties = {"gpu": "A10"},
    shard_count = 1,
    deps = [
        "//rtp_llm:sdk",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util",
    ],
)

py_test(
    name = "template_test",
    srcs = [
        "template_test.py",
    ],
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata",
        "//rtp_llm/test/tokenizer_test:testdata",
    ],
    exec_properties = {"gpu": "A10"},
    deps = [
        "//rtp_llm:sdk",
        "//rtp_llm:testlib",
        # qwen agent
        "//rtp_llm:pydantic",
        "//rtp_llm:json5",
        "//rtp_llm:dashscope",
        "//rtp_llm:jieba",
        "//rtp_llm:openai",
        "//rtp_llm:Pillow",
    ],
)

py_test(
    name = "pipeline_test",
    srcs = [
        "pipeline_test.py",
    ],
    exec_properties = {"gpu": "A10"},
    deps = [
        "//rtp_llm:sdk",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "prefill_capture_seq_lens_test",
    srcs = [
        "prefill_capture_seq_lens_test.py",
    ],
    exec_properties = {"gpu": "A10"},
    deps = [
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "auto_configure_deepep_test",
    srcs = [
        "auto_configure_deepep_test.py",
    ],
    exec_properties = {"gpu": "A10"},
    deps = [
        "//rtp_llm:sdk",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "glm4_moe_detector_test",
    srcs = [
        "glm4_moe_detector_test.py",
    ],
    data = [],
    exec_properties = {"gpu": "A10"},
    imports = [],
    deps = [
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "mtp_streaming_tool_parsing_test",
    srcs = [
        "mtp_streaming_tool_parsing_test.py",
    ],
    data = [],
    exec_properties = {"gpu": "A10"},
    imports = [],
    deps = [
        "//rtp_llm:testlib",
    ],
)
