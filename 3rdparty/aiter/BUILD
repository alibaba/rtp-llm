genrule(
    name = "cpp_libraries",
    srcs = glob([
        "**/*"
    ]),
    outs = [
        "aiter/jit/libmodule_custom_all_reduce.so",
        "csrc/cpp_itfs/mla/libasm_mla_decode_fwd_torch.so"
        # "aiter/jit/libmodule_attention.so",
        # "aiter/jit/libmodule_norm.so",
        # "aiter/jit/libmodule_cache.so",
        # "aiter/jit/libmodule_mha_fwd.so"
    ],
    cmd = """
        cd external/aiter;
        yum install -y fmt fmt-devel;
        /opt/conda310/bin/python -m pip install -r requirements.txt;
        /opt/conda310/bin/python -m pip install ninja;
        /opt/conda310/bin/python -m pip install packaging;
        /opt/conda310/bin/python setup.py develop;
        ROCM_HOME=/opt/rocm LD_LIBRARY_PATH=/opt/amdgpu/lib64 PATH=/opt/conda310/bin:$$PATH /opt/conda310/bin/python build_aiter_module.py > tmp_aiter.log 2>&1;
        bash build_mla_kernel.sh >> tmp_aiter.log 2>&1;
        cd ../..;
        cp external/aiter/aiter/jit/module_custom_all_reduce.so $(location aiter/jit/libmodule_custom_all_reduce.so);
        cp external/aiter/csrc/cpp_itfs/mla/asm_mla_decode_fwd_torch_lib.so $(location csrc/cpp_itfs/mla/libasm_mla_decode_fwd_torch.so)
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

cc_library(
    name = "decode_mla",
    srcs = ["csrc/cpp_itfs/mla/libasm_mla_decode_fwd_torch.so"],
    hdrs = ["csrc/cpp_itfs/mla/asm_mla_decode_fwd_torch.h"],
    deps = [":cpp_libraries"],
    copts = [],
    linkopts = [],
    strip_include_prefix = "csrc/cpp_itfs/",
    visibility = ["//visibility:public"],
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
