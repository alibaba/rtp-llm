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
