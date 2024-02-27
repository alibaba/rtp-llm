licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

cc_import(
    name = "python_lib",
    interface_library = "not-existing.lib",
    system_provided = 1,
)

cc_library(
    name = "python_headers",
    hdrs = [":python_include"],
    deps = [],
    includes = ["python_include"],
)

%{PYTHON_INCLUDE_GENRULE}
%{PYTHON_IMPORT_LIB_GENRULE}
