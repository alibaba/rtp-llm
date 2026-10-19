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
        "libpython3.10.so",
    ],
    cmd = """
cp -f "/opt/conda310/lib/libpython3.10.so" "$(@D)/libpython3.10.so"
   """,
)