
py_test(
    name = "generate_config_test",
    srcs = [
        "generate_config_test.py",
    ],
    data = [
        "//maga_transformer/test/model_test/fake_test/testdata:testdata",
        "//maga_transformer:sdk"
    ],
    deps = [
        "//maga_transformer:testlib",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "default_plugin_test",
    srcs = [
        "default_plugin_test.py",
    ],
    data = [
        "//maga_transformer/test/model_test/fake_test/testdata:testdata",
        "//maga_transformer:sdk"
    ],
    deps = [
        "//maga_transformer:testlib",
    ],
    imports = ["aios/kmonitor/python_client"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "slice_stop_word_list_test",
    srcs = [
        "slice_stop_word_list_test.py",
    ],
    data = [
        "//maga_transformer/test/model_test/fake_test/testdata:testdata",
        "//maga_transformer:sdk"
    ],
    deps = [
        "//maga_transformer:testlib",
        "//maga_transformer/test/model_test/test_util",
    ],
    imports = ["aios/kmonitor/python_client"],
    exec_properties = {'gpu':'A10'},
)


py_test(
    name = "chatapi_test",
    srcs = [
        "chatapi_test.py",
    ],
    deps = [
        "//maga_transformer:sdk",
        "//maga_transformer:testlib",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "template_test",
    srcs = [
        "template_test.py",
    ],
    deps = [
        "//maga_transformer:sdk",
        "//maga_transformer:testlib",
        # qwen agent
        "//maga_transformer:pydantic",
        "//maga_transformer:json5",
        "//maga_transformer:dashscope",
        "//maga_transformer:jieba",
        "//maga_transformer:openai",
        "//maga_transformer:Pillow",
    ],
    data = [
        "//maga_transformer/test/model_test/fake_test/testdata:testdata",
        "//maga_transformer/test/tokenizer_test:testdata",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "openai_response_test",
    srcs = [
        "openai_response_test.py",
    ],
    deps = [
        "//maga_transformer:sdk",
        "//maga_transformer:testlib",
    ],
    data = [
        "//maga_transformer/test/model_test/fake_test/testdata:testdata",
        "//maga_transformer/test/tokenizer_test:testdata",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "server_test",
    srcs = [
        "server_test.py",
    ],
    data = [
        "//maga_transformer:sdk",
    ],
    deps = [
        "//maga_transformer:uvicorn",
        "//maga_transformer:fastapi",
        "//maga_transformer:psutil",
        "//maga_transformer:tiktoken",
        "//maga_transformer:testlib",
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
        "//maga_transformer:sdk",
        "//maga_transformer:testlib",
        "//maga_transformer/test/model_test/test_util",
    ],
    timeout = 'short',
    exec_properties = {'gpu':'A10'},
    shard_count = 1,
)
