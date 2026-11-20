"""Build the private binding with a separate compiler and the locked Torch ABI."""

def _find(files, suffix):
    return [file for file in files if file.path.endswith(suffix)][0]

def _wheel_files(target, path):
    return [
        file
        for file in target[DefaultInfo].default_runfiles.files.to_list()
        if path in file.path
    ]

def _host_binding_impl(ctx):
    torch = _wheel_files(ctx.attr.torch, "/site-packages/torch/include/")
    deep_gemm = _wheel_files(ctx.attr.deep_gemm, "/site-packages/deep_gemm/include/")
    torch_libs = [
        _find(_wheel_files(ctx.attr.torch, "/site-packages/torch/lib/"), "/" + name)
        for name in ["libc10.so", "libc10_cuda.so", "libtorch.so", "libtorch_cpu.so", "libtorch_python.so"]
    ]
    cudart = [
        lib.resolved_symlink_dynamic_library or lib.dynamic_library
        for linker_input in ctx.attr.cudart[CcInfo].linking_context.linker_inputs.to_list()
        for lib in linker_input.libraries
        if lib.dynamic_library
    ]
    cuda = ctx.files.cuda_headers
    python = ctx.files.python_headers
    root = ctx.file.source.dirname
    cuda_include = _find(cuda, "/include/cuda_runtime.h").dirname
    torch_include = _find(torch, "/include/torch/extension.h").dirname[:-len("/torch")]
    includes = [
        cuda_include,
        cuda_include + "/cccl",
        _find(python, "/Python.h").dirname,
        torch_include,
        torch_include + "/torch/csrc/api/include",
    ] + [root + "/" + path for path in [
        "deep_gemm_sm120/shim/include",
        "deep_gemm_sm120/shim/csrc/jit_kernels/impls",
        "deep_gemm_sm120/vendor/deep_gemm/include",
        "deep_gemm_sm120/adapter/include",
        "deep_gemm/include",
        "csrc",
        "csrc/jit_kernels/impls",
        "csrc/jit_kernels/heuristics",
        "third-party/deep_jit/include",
    ]] + [
        # The private DeepGEMM headers take precedence over the locked wheel.
        _find(deep_gemm, "/cutlass/version.h").dirname[:-len("/cutlass")],
    ]
    args = ctx.actions.args()
    args.add(ctx.file.compiler_identity)
    args.add_all([
        "-std=c++20", "-O3", "-shared", "-fPIC", "-fvisibility=hidden",
        "-fvisibility-inlines-hidden", "-fno-gnu-unique", "-Wno-deprecated-declarations",
        "-Wno-psabi", "-D_GLIBCXX_USE_CXX11_ABI=1",
        "-DFMT_HEADER_ONLY=1", "-DRTP_DG120_ENABLE_LIBDW=0",
    ])
    args.add_all(includes, before_each = "-isystem")
    args.add(ctx.file.source)
    args.add(ctx.file.exports, format = "-Wl,--version-script=%s")
    args.add_all(torch_libs)
    args.add_all(cudart)
    args.add("-ldl")
    args.add("-o", ctx.outputs.out)
    ctx.actions.run(
        executable = ctx.executable.compiler,
        arguments = [args],
        inputs = depset(
            [ctx.file.source, ctx.file.exports, ctx.file.compiler_identity] + ctx.files.native_sources +
            torch + deep_gemm + torch_libs + cuda + python + cudart,
        ),
        outputs = [ctx.outputs.out],
        mnemonic = "Sm120GroupedHostBinding",
        progress_message = "Compiling private SM120 grouped GEMM binding %{label}",
    )

host_binding = rule(
    implementation = _host_binding_impl,
    attrs = {
        "out": attr.output(mandatory = True),
        "source": attr.label(allow_single_file = [".cpp"], mandatory = True),
        "exports": attr.label(allow_single_file = True, mandatory = True),
        "native_sources": attr.label(mandatory = True),
        "compiler": attr.label(executable = True, allow_single_file = True, mandatory = True, cfg = "exec"),
        "compiler_identity": attr.label(allow_single_file = True, mandatory = True),
        "torch": attr.label(mandatory = True),
        "deep_gemm": attr.label(mandatory = True),
        "python_headers": attr.label(mandatory = True),
        "cuda_headers": attr.label(mandatory = True),
        "cudart": attr.label(mandatory = True, providers = [CcInfo]),
    },
)
