
py_test(
    name = "generate_config_test",
    srcs = [
        "generate_config_test.py",
    ],
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata:testdata",
        "//rtp_llm:sdk"
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "slice_stop_word_list_test",
    srcs = [
        "slice_stop_word_list_test.py",
    ],
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata:testdata",
        "//rtp_llm:sdk"
    ],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util",
    ],
    imports = ["aios/kmonitor/python_client"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "stop_words_test",
    srcs = [
        "stop_words_test.py",
    ],
    data = [],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util",
    ],
    imports = [],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "stop_words_pipeline_test",
    srcs = [
        "stop_words_pipeline_test.py",
    ],
    data = [],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util",
    ],
    imports = [],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "openai_response_test",
    srcs = [
        "openai_response_test.py",
    ],
    deps = [
        "//rtp_llm:sdk",
        "//rtp_llm:testlib",
    ],
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata:testdata",
        "//rtp_llm/test/tokenizer_test:testdata",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "server_test",
    srcs = [
        "server_test.py",
    ],
    data = [
        "//rtp_llm:sdk",
    ],
    deps = [
        "//rtp_llm:uvicorn",
        "//rtp_llm:fastapi",
        "//rtp_llm:psutil",
        "//rtp_llm:tiktoken",
        "//rtp_llm:testlib",
        "//deps:extension_lib",
    ],
    imports = ["aios/kmonitor/python_client"],
    tags = ["manual"],
    timeout = 'eternal',
)

py_test(
    name = "concurrency_limit_test",
    srcs = [
        "concurrency_limit_test.py"
    ],
    deps = [
        "//rtp_llm:sdk",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util",
    ],
    timeout = 'short',
    exec_properties = {'gpu':'A10'},
    shard_count = 1,
)

py_test(
    name = "template_test",
    srcs = [
        "template_test.py",
    ],
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
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata:testdata",
        "//rtp_llm/test/tokenizer_test:testdata",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "prefill_capture_seq_lens_test",
    srcs = [
        "prefill_capture_seq_lens_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "auto_configure_deepep_test",
    srcs = [
        "auto_configure_deepep_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm:sdk",
    ],
    exec_properties = {'gpu':'A10'},
)
