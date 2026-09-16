exports_files([
    "pytest_main.py",
    "requirements_lock.txt",
])

py_test(
    name = "pytest_main_test",
    srcs = ["pytest_main_test.py"],
    data = ["pytest_main.py"],
    deps = ["@pip_dsv4_test_pytest//:pkg"],
)
