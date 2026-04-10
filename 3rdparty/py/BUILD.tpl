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

%{PYTHON_IMPORT_LIB_GENRULE}
