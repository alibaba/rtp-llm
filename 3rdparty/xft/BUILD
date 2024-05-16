package(default_visibility = ["//visibility:public"])

cc_import(
    name = "libxfastertransformer",
    shared_library = "libxfastertransformer.so",
)

cc_library(
    name = "xft_header",
    hdrs = glob([
        "include/*.h"
    ]),
    visibility = ["//visibility:public"],    
)
