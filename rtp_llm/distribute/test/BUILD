filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/*",
        "testdata/**",
        "testdata/tokenizer",
        "testdata/cpt",
    ]),
)

# py_test(
#     name = "gang_test",
#     srcs = [
#         "gang_test.py",
#         "fake_model.py"
#     ],
#     data = [
#         ":testdata",
#         "//rtp_llm:sdk"
#     ],
#     deps = [
#         "//rtp_llm:testlib",
#     ],
#     timeout = 'short',
#     flaky = True,
#     imports = ["aios/kmonitor/python_client"],
#     exec_properties = {'gpu':'A10','gpu_count':'2'},
#     env = {"GPU_COUNT": "2"},
# )

py_test (
    name = "deep_ep_test",
    srcs = ["deep_ep_test.py"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/distribute:deep_ep",
        "//rtp_llm/distribute:process_group_state",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    env = {"GPU_COUNT": "2"},
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'2'},
)
