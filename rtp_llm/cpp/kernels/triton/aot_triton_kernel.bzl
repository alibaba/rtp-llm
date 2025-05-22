load("@bazel_skylib//lib:paths.bzl", "paths")
load("//bazel:arch_select.bzl", get_triton_deps = "triton_deps")
load("//:def.bzl", "copts")


def _get_python_interpreter_path(ctx):
    python_interpreter = ctx.attr._python[PyRuntimeInfo].interpreter_path
    return python_interpreter

def _get_output_dir(ctx):
    output_dir = paths.join(ctx.bin_dir.path, ctx.label.package)
    return output_dir

def _get_output_name(ctx):
    output_name = ctx.attr.output_name
    if output_name == None or len(output_name) == 0:
        output_name = ctx.attr.name
    return output_name

def _hash_spec(spec):
    hash_value = 0
    hash_value = hash(spec)
    return hash_value

def _format_unique_name(ctx, ty_idx, num_warps, spec):
    s = "id_"
    s += ctx.attr.name + "_"
    s += "n" + str(num_warps) + "_"
    s += "s" + str(_hash_spec(spec)) + "_"
    s += str(ty_idx)
    return s

def _compile_kernel(ctx, num_warps, ty_idx, output_name, extra_input, py_imports, kwargs):
    spec = ctx.attr.spec.format(**kwargs)
    python_interpreter = _get_python_interpreter_path(ctx)
    bazel_id = _format_unique_name(ctx, ty_idx, num_warps, spec)
    hdr = ctx.actions.declare_file('aot/' + bazel_id + '/' +output_name + '__' + bazel_id + ".h")
    src = ctx.actions.declare_file('aot/' + bazel_id + '/' +output_name + '__' + bazel_id + ".c")
    args = ctx.actions.args()
    args.add(ctx.file._compiler)
    args.add("--interpreter", python_interpreter)
    args.add("--script_name", ctx.file.triton_script)
    args.add("--output_dir", _get_output_dir(ctx))
    args.add("--output_name", output_name)
    args.add("--kernel_name", ctx.attr.kernel_name)
    args.add("--name", ctx.attr.name)
    args.add("--grid", ctx.attr.grid)
    args.add_all("--imports", py_imports)
    if num_warps != None:
        args.add("--num_warps", num_warps)
    args.add("--spec", spec)
    args.add("--bazel_id", bazel_id)
    ctx.actions.run(
        outputs = [hdr, src],
        inputs = depset([ctx.file._compiler, ctx.file.triton_script], transitive=extra_input),
        executable = python_interpreter,
        arguments = [args],
        use_default_shell_env = True,
    )
    return hdr, src

def _link_kernels(ctx, output_name, ty_hdrs, py_imports, extra_input):
    python_interpreter = _get_python_interpreter_path(ctx)
    hdr = ctx.actions.declare_file('aot/'+output_name + ".h")
    src = ctx.actions.declare_file('aot/'+output_name + ".c")
    args = ctx.actions.args()
    args.add(ctx.file._linker)
    args.add("--interpreter", python_interpreter)
    args.add("--output_dir", _get_output_dir(ctx))
    args.add("--output_name", output_name)
    args.add_all("--hdrs", ty_hdrs)
    args.add_all("--imports", py_imports)

    ctx.actions.run(
        outputs = [hdr, src],
        inputs = depset([ctx.file._linker] + ty_hdrs, transitive=extra_input),
        executable = python_interpreter,
        arguments = [args],
        use_default_shell_env = True,
    )
    return hdr, src

def _get_extra_input_and_imports(ctx):
    py_deps = []
    py_imports = []
    cc_hdrs = []
    runfiles = []
    deps = ctx.attr.deps
    for dep in deps:
        if PyInfo in dep:
            runfiles.append(dep.data_runfiles.files)
            py_deps.append(dep[PyInfo].transitive_sources)
            # print(dep[PyInfo].transitive_sources)
            py_imports.append(dep[PyInfo].imports)
        if CcInfo in dep:
            cc_hdrs.append(dep[CcInfo].compilation_context.headers)
    py_imports = depset([],transitive=py_imports).to_list()

    return py_deps + cc_hdrs + runfiles, py_imports

def _handle_num_wraps_and_compile(ctx, ty_idx, output_name, extra_input, py_imports, kwargs):
    ty_hdrs = []
    ty_srcs = []
    _ty_idx = ty_idx
    if len(ctx.attr.num_warps) > 0:
        for num_warps in ctx.attr.num_warps:
            _ty_idx += 1
            ty_hdr, ty_src = _compile_kernel(ctx, num_warps, _ty_idx, output_name, extra_input, py_imports, kwargs)
            ty_hdrs.append(ty_hdr)
            ty_srcs.append(ty_src)
    else:
        _ty_idx += 1
        num_warps = None
        ty_hdr, ty_src = _compile_kernel(ctx, num_warps, _ty_idx, output_name, extra_input, py_imports, kwargs)
        ty_hdrs.append(ty_hdr)
        ty_srcs.append(ty_src)
    return ty_hdrs, ty_srcs

def _dispatch2(ctx, var_map):
    if len(var_map) != 2:
        fail("internal ERROR, script dispatch wrong")

    extra_input, py_imports = _get_extra_input_and_imports(ctx)

    keys = var_map.keys()
    key1 = keys[0]
    key2 = keys[1]
    spec_list = []
    ty_hdrs = []
    ty_srcs = []
    ty_idx = 0
    output_name = _get_output_name(ctx)
    common_args = {
        "ctx": ctx,
        "output_name": output_name,
        "extra_input": extra_input,
        "py_imports": py_imports
    }
    for value1 in var_map[key1]:
        for value2 in var_map[key2]:
            kwargs = {
                key1: value1,
                key2: value2,
            }
            sub_ty_hdrs, sub_ty_srcs = _handle_num_wraps_and_compile(ty_idx = ty_idx, kwargs = kwargs, **common_args)
            ty_idx += len(sub_ty_hdrs)
            ty_hdrs += sub_ty_hdrs
            ty_srcs += sub_ty_srcs
    hdr, src = _link_kernels(ctx, output_name, ty_hdrs, py_imports, extra_input)
    return [DefaultInfo(files=depset([hdr, src] + ty_srcs))]


def _dispatch3(ctx, var_map):
    if len(var_map) != 3:
        fail("internal ERROR, script dispatch wrong")

    extra_input, py_imports = _get_extra_input_and_imports(ctx)

    keys = var_map.keys()
    key1 = keys[0]
    key2 = keys[1]
    key3 = keys[2]
    spec_list = []
    ty_hdrs = []
    ty_srcs = []
    ty_idx = 0
    output_name = _get_output_name(ctx)

    common_args = {
        "ctx": ctx,
        "output_name": output_name,
        "extra_input": extra_input,
        "py_imports": py_imports
    }
    for value1 in var_map[key1]:
        for value2 in var_map[key2]:
            for value3 in var_map[key3]:
                kwargs = {
                    key1: value1,
                    key2: value2,
                    key3: value3,
                }
                sub_ty_hdrs, sub_ty_srcs = _handle_num_wraps_and_compile(ty_idx = ty_idx, kwargs = kwargs, **common_args)
                ty_idx += len(sub_ty_hdrs)
                ty_hdrs += sub_ty_hdrs
                ty_srcs += sub_ty_srcs
    hdr, src = _link_kernels(ctx, output_name, ty_hdrs, py_imports, extra_input)
    return [DefaultInfo(files=depset([hdr, src] + ty_srcs))]


def _impl(ctx):
    loop_nested_count = len(ctx.attr.var_map)
    fn = None
    if loop_nested_count == 2:
        fn = _dispatch2
    if loop_nested_count == 3:
        fn = _dispatch3
    if fn == None:
        fail("NYI, out of range. var map keys is ", ctx.attr.var_map.keys())
    return fn(ctx, ctx.attr.var_map)
aot_triton_kernel = rule(
    implementation = _impl,
    attrs = {
        "kernel_name": attr.string(),
        "triton_script": attr.label(allow_single_file = True),
        "deps": attr.label_list(),
        "spec": attr.string(),
        "grid": attr.string(),
        "output_name": attr.string(),
        "num_warps": attr.int_list(),
        "var_map": attr.string_list_dict(),
        "_python": attr.label(
            # TODO: get value from --python_top=//:python310
            default = Label("//:python310")
        ),
        "_compiler": attr.label(
            default = Label(":aot_triton_kernel_compiler.py"),
            allow_single_file = True
        ),
        "_linker": attr.label(
            default = Label(":aot_triton_kernels_linker.py"),
            allow_single_file = True
        )
    },
)

def aot_triton_kernel_library(
    name,
    output_name_tpl,
    kernel_name,
    triton_script,
    num_warps,
    grid,
    spec,
    groupby,
    var_map
):
    if groupby not in var_map:
        fail("groupby must in var_map keys, groupby is ", groupby, ", var_map keys is ", var_map.keys())
    groupby_value_list = var_map[groupby]
    var_map.pop(groupby)
    srcs_list = []
    for groupby_value in groupby_value_list:
        output_name = output_name_tpl.format(**{groupby:groupby_value})
        aot_triton_kernel(
            name = name+"_"+groupby_value,
            output_name = output_name,
            kernel_name = kernel_name,
            triton_script = triton_script,
            grid = grid,
            spec = spec,
            num_warps = num_warps,
            var_map = var_map | {groupby:[groupby_value]},
            tags = ["manual"],
            deps = [
                "@local_config_cuda//cuda:cuda_headers"
            ] + get_triton_deps(["triton", "torch", "numpy"])
        )
        srcs_list.append(":" + name+"_"+groupby_value)
    native.filegroup(
        name = name,
        srcs = srcs_list,
    )
    native.cc_library(
        name = name+"_lib",
        srcs = select({
            "//:enable_triton": [":"+name],
            "//conditions:default": []
        }),
        deps = [
            "@local_config_cuda//cuda:cuda_headers",
        ],
        tags = ["manual"],
        copts = copts(),
    )
