genrule(
    name = "cpp_libraries",
    srcs = glob([
        "**/*"
    ]),
    outs = [
        "aiter/jit/libmodule_custom_all_reduce.so",
        # "aiter/jit/libmodule_attention.so",
        # "aiter/jit/libmodule_norm.so",
        # "aiter/jit/libmodule_cache.so",
        # "aiter/jit/libmodule_mha_fwd.so"
    ],
    cmd = """
        cd external/aiter;
        /opt/conda310/bin/python -m pip install -r requirements.txt;
        /opt/conda310/bin/python -m pip install ninja;
        /opt/conda310/bin/python -m pip install packaging;
        ROCM_HOME=/opt/rocm LD_LIBRARY_PATH=/opt/amdgpu/lib64 PATH=/opt/conda310/bin:$$PATH /opt/conda310/bin/python build_aiter_module.py > tmp_aiter.log 2>&1;
        cd ../..;
        cp external/aiter/aiter/jit/module_custom_all_reduce.so $(location aiter/jit/libmodule_custom_all_reduce.so)
    """
)

cc_library(
    name = "module_custom_all_reduce",
    srcs = ["aiter/jit/libmodule_custom_all_reduce.so"],
    hdrs = ["csrc/include/custom_all_reduce.h"],
    deps = [":cpp_libraries"],
    copts = [],
    linkopts = [],
    strip_include_prefix = "csrc/include/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

# cc_library(
#     name = "module_mha_fwd",
#     srcs = ["aiter/jit/libmodule_mha_fwd.so"],
#     hdrs = ["csrc/include/mha_fwd.h"],
#     deps = [":cpp_libraries"],
#     copts = [],
#     linkopts = [],
#     strip_include_prefix = "csrc/include/",
#     visibility = ["//visibility:public"],
# )
