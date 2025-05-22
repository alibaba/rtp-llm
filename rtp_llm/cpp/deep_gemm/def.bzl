load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
load("//bazel:arch_select.bzl", "torch_deps")
load("//:def.bzl", "copts", "cuda_copts", "gen_cpp_code")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch", "if_cuda")

preloaded_deps = [
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
    "@cutlass//:cutlass",
    ":deepgemm_hdrs",
] + torch_deps()

sm90_cuda_copts = ["-x", "cuda", "-std=c++17", "-shared", "--cuda-include-ptx=sm_90a", "--cuda-gpu-arch=sm_90a", "--compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi"]

def sub_lib(name, srcs):
    native.cc_library(
        name = name + "_cu",
        srcs = [srcs],
        deps = preloaded_deps,
        copts = sm90_cuda_copts,
        visibility = ["//visibility:public"],
    )
    
    cc_shared_library(
        name = name,
        roots = [":" + name + "_cu"],
        preloaded_deps = preloaded_deps,
        visibility = ["//visibility:public"],
    )

def sub_lib_and_interleave(name, srcs):
    native.cc_library(
        name = name + "_cu",
        srcs = [srcs],
        deps = preloaded_deps,
        copts = sm90_cuda_copts,
        visibility = ["//visibility:public"],
    )
    
    cc_shared_library(
        name = name + "_so",
        roots = [":" + name + "_cu"],
        preloaded_deps = preloaded_deps,
        visibility = ["//visibility:public"],
    )

    native.genrule(
        name = name,
        srcs = [":" + name + "_so"],
        tools = ["interleave_ffma.py"],
        outs = ["lib" + name + ".so"],
        cmd = select({
            "@//:using_cuda12": """
                cp "$(location :{lib_name})" "$@"
                chmod 777 $@
                /opt/conda310/bin/python "$(location interleave_ffma.py)" --so "$@"
            """.format(lib_name = name + "_so"),
            "@//conditions:default": """
                cp "$(location :{lib_name})" "$@"
            """.format(lib_name = name + "_so")
        }),
        visibility = ["//visibility:public"],
    )

def gen_cu_and_lib(name, params_list, split_num, template_header, template, template_tail, element_per_file):
    for i in range((len(params_list) + split_num - 1) // split_num):
        gen_cpp_code(name + "_" + str(i), [params_list[i * split_num: (i + 1) * split_num]], template_header, template, template_tail, element_per_file, suffix=".cu")
        sub_lib_and_interleave(name + "_" + str(i) + "_inst", ":" + name + "_" + str(i))

    native.cc_library(
        name = name + "_inst",
        hdrs = ["DeepGemmPlugin.h", "utils.h"],
        srcs = ["DeepGemmPlugin.cpp"] + [
            ":" + name + "_" + str(i) + "_inst" for i in range((len(params_list) + split_num - 1) // split_num)
        ],
        copts = copts(),
        deps = [
            "//rtp_llm/cpp/core:buffer_torch",
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
            "//rtp_llm/cpp/cuda:nvtx",
        ] + torch_deps(),
        visibility = ["//visibility:public"],
    )

def gen_dispatch_code(name, params_list, template_header, template, template_tail):
    all_dispatch_code = ""
    for params in params_list:
        all_dispatch_code += template.format(*list(params))
    
    native.genrule(
        name = name,
        srcs = [],
        outs = [name + ".cc"],
        cmd = "cat > $@ << 'EOF'\n" + template_header + all_dispatch_code + template_tail + "EOF"
    )