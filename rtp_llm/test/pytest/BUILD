exports_files([
    "pytest_main.py",
    "pytest_main_sm120.py",
    "requirements_lock.txt",
])

py_test(
    name = "pytest_main_test",
    srcs = ["pytest_main_test.py"],
    data = ["pytest_main.py"],
    deps = ["//rtp_llm:jit_cache_env", "@pip_dsv4_test_pytest//:pkg"],
)

# SM120 gate cases split out of pytest_main_test.py.
# The sibling launcher carries the --cuda-major/--cuda-minor gate so
# pytest_main.py is never edited.
py_test(
    name = "pytest_main_sm120_test",
    srcs = ["pytest_main_sm120_test.py"],
    data = ["pytest_main.py", "pytest_main_sm120.py"],
    deps = ["//rtp_llm:jit_cache_env", "@pip_dsv4_test_pytest//:pkg"],
)
