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

genrule(
    name = "python_import_lib",
    outs = [
        "%{PYTHON_IMPORT_LIB_NAME}",
    ],
    cmd = """
cp -f "%{PYTHON_IMPORT_LIB_PATH}" "$(@D)/%{PYTHON_IMPORT_LIB_NAME}"
   """,
)
