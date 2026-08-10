load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
load("@arch_config//:arch_select.bzl", "torch_deps")

preloaded_deps = [
    ":flashinfer_hdrs",
    ":dispatch",
    "@cutlass//:cutlass",
    "@cutlass//:cutlass_utils",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
    "@local_config_cuda//cuda:cublas_headers",
    "@local_config_cuda//cuda:cublas",
    "@local_config_cuda//cuda:cublasLt",
] + torch_deps()


def flashinfer_dispatch_genrule(repo_name):
    # Single source for the dispatch.inc generation shared by flashinfer.BUILD
    # (@flashinfer_cpp) and flashinfer_cu13.BUILD (@flashinfer_cpp_cu13). The
    # only per-variant input is the repository name; the interpreter path and
    # generate arguments live here once so the two variants cannot drift.
    native.genrule(
        name = "generate_dispatch",
        outs = ["dispatch.inc"],
        tools = [":dispatch_generate_py"],
        cmd = (
            "loc=$(locations @" + repo_name + "//:dispatch_generate_py); " +
            "loc=$${loc%/*}; loc=$${loc%/*}; " +
            "PYTHONPATH=$$loc /opt/conda310/bin/python -m aot_build_utils.generate_dispatch_inc " +
            "--use_fp16_qk_reductions false --mask_modes 1 --path $(RULEDIR)/dispatch.inc " +
            "--head_dims_sm90 64,64 128,128 --head_dims 64 128 256 --pos_encoding_modes 0"
        ),
        tags = ["local"],
    )

def sub_lib(name, deps, copts):
    native.cc_library(
        name = name + '_cu',
        srcs = native.glob([
            "csrc/*.h",
        ]) + deps,
        deps = [
            ":dispatch",
            ":flashinfer_hdrs",
        ],
        copts = copts,
        visibility = ["//visibility:public"],
    )
    cc_shared_library(
        name = name,
        roots = [":" + name + "_cu"],
        preloaded_deps = preloaded_deps,
        visibility = ["//visibility:public"],
    )
    return name
