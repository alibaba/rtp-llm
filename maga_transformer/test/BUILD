
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
    ]
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
)

py_test(
    name = "template_test",
    srcs = [
        "template_test.py",
    ],
    deps = [
        "//maga_transformer:sdk",
        "//maga_transformer:testlib",
    ],
    data = [
        "//maga_transformer/test/model_test/fake_test/testdata:testdata",
        "//maga_transformer/test/tokenizer_test:testdata",
    ],
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
    name = "async_gather_batch_test",
    srcs = [
        "async_gather_batch_test.py"
    ],
    deps = [
        "//maga_transformer:sdk",
        "//maga_transformer:testlib",
        "//maga_transformer/test/model_test/test_util",
    ]
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
    timeout = 'short'
)