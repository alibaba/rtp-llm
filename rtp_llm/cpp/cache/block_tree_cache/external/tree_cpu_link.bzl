"""Link the scheduler C ABI without the engine's build-time Torch/Python runtime."""

def _as_list(values):
    return values.to_list() if type(values) == "depset" else list(values)

def _engine_runtime(name):
    return name in ["torch", "torch_python", "torch_cpu", "shm", "c10"] or name.startswith("python3.")

def _engine_library(lib):
    # The solib symlink may have a mangled name; inspect its original as well.
    for so in [lib.resolved_symlink_dynamic_library, lib.dynamic_library]:
        if so and so.basename.startswith("lib") and ".so" in so.basename:
            if _engine_runtime(so.basename[3:].split(".so")[0]):
                return True
    return False

def _cpu_tree_link_impl(ctx):
    engine = cc_common.merge_cc_infos(cc_infos = [dep[CcInfo] for dep in ctx.attr.deps])
    inputs = []
    for item in engine.linking_context.linker_inputs.to_list():
        libs = _as_list(item.libraries)
        flags = _as_list(item.user_link_flags)
        keep = [lib for lib in libs if not _engine_library(lib)]
        opts = [flag for flag in flags if not (flag.startswith("-l") and _engine_runtime(flag[2:]))]
        if len(keep) == len(libs) and len(opts) == len(flags):
            inputs.append(item)
        elif keep or opts or item.additional_inputs:
            inputs.append(cc_common.create_linker_input(
                owner = item.owner,
                libraries = depset(keep),
                user_link_flags = depset(opts),
                additional_inputs = depset(_as_list(item.additional_inputs)),
            ))
    # Headers and native static objects are the same ones as the RTP engine.
    # --no-undefined at the consuming link rejects actual runtime references.
    return [CcInfo(
        compilation_context = engine.compilation_context,
        linking_context = cc_common.create_linking_context(
            linker_inputs = depset(inputs, order = "topological"),
        ),
    )]

cpu_tree_link = rule(
    implementation = _cpu_tree_link_impl,
    attrs = {"deps": attr.label_list(providers = [CcInfo], mandatory = True)},
)
