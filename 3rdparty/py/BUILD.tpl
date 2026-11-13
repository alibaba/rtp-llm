licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "python_lib",
    srcs = [":python_import_lib"],
)

cc_library(
    name = "python_headers",
    hdrs = [":python_include"],
    deps = [],
    includes = ["python_include"],
)

%{PYTHON_INCLUDE_GENRULE}

# The interpreter this repository was configured with (PYTHON_BIN_PATH, else
# `python` on PATH). --config=xpu uses it as --python_top instead of
# //:python310, so Bazel's Python targets run on the active environment.
py_runtime(
    name = "python_runtime",
    interpreter_path = "%{PYTHON_BIN_PATH}",
    python_version = "PY3",
    stub_shebang = "#!%{PYTHON_BIN_PATH}",
)

genrule(
    name = "python_import_lib",
    outs = [
        "%{PYTHON_LIB_NAME}",
    ],
    tags = ["local"],
    cmd = """
cp -f "%{PYTHON_LIB_PATH}" "$(@D)/%{PYTHON_LIB_NAME}"
   """,
)