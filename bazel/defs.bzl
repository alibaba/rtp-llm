# py_package, py_wheel, rename_wheel 已清理
# Python 打包通过 pyproject.toml 和 setup.py 管理

def copy_so(target):
    name = 'lib' + target.split(':')[1] + '_so'
    so_name = 'lib' + target.split(':')[1] + '.so'
    native.genrule(
        name = name,
        srcs = [target],
        outs = [so_name],
        cmd = "cp $(SRCS) $(@D)",
        tags = ["no-remote"],
    )

def copy_so_inst(target, inst_num):
    for i in range(inst_num):
        name = 'lib' + target.split(':')[1] + '_' + str(i) + '_inst_so'
        so_name = 'lib' + target.split(':')[1] + '_' + str(i) + '_inst.so'
        native.genrule(
            name = name,
            srcs = [target + '_' + str(i) + '_inst'],
            outs = [so_name],
            cmd = "cp $(SRCS) $@",
            tags = ["no-remote"],
        )

    native.filegroup(
        name = 'lib' + target.split(':')[1] + '_inst_so',
        srcs = [':lib' + target.split(':')[1] + '_' + str(i) + '_inst_so' for i in range(inst_num)],
    )

def copy_target_to(name, to_copy, copy_name, dests = [], **kwargs):
    if dests:
        outs = [path + copy_name for path in dests]
        cmds = ["mkdir -p %s" % (dest) for dest in dests]
        cmd = "&&".join(cmds) + " && "
    else:
        outs = [copy_name]
        cmd = ""
    cmd += "for out in $(OUTS); do cp $(location %s) $$out; done" % to_copy
    native.genrule(
        name = name,
        srcs = [to_copy],
        outs = outs,
        cmd = cmd,
        **kwargs
    )

def _upload_pkg_impl(ctx):
    ctx.actions.expand_template(
        template = ctx.file._deploy_script,
        substitutions = {
            "{oss_prefix}" : ctx.attr.oss_prefix,
            "{pkg_prefix}" : ctx.attr.pkg_prefix,
        },
        output = ctx.outputs.executable,
        is_executable = True
    )
    return DefaultInfo(
        executable = ctx.outputs.executable,
        runfiles = ctx.runfiles(
            files = [ctx.file.target],
            symlinks = {"pkg.tar": ctx.file.target}))

_upload_pkg = rule(
    attrs = {
        "target": attr.label(
            allow_single_file = True,
        ),
        "oss_prefix": attr.string(
            mandatory = True,
            doc = "upload to, ex. rtp_pkg",
        ),
        "pkg_prefix": attr.string(
            mandatory = True,
            doc = "ex. rtp_",
        ),
        "_deploy_script": attr.label(
            allow_single_file = True,
            default = "//bazel:upload_package.py",
        ),
    },
    implementation = _upload_pkg_impl,
    executable = True,
)

def upload_pkg(name, **kwargs):
    key = "tags"
    tags = kwargs.get(key) or []
    tags.extend(["manual"])
    kwargs.setdefault(key, tags)

    _upload_pkg( name = name, **kwargs)

def rpm_library(
        name,
        hdrs,
        include_path=None,
        lib_path=None,
        rpms=None,
        static_lib=None,
        static_libs=[], # multi static libs, do not add to cc_library, provide .a filegroup
        shared_lib=None,
        shared_libs=[],
        bins=[],
        include_prefix=None,
        static_link=False,
        deps=[],
        header_only=False,
        tags={},
        **kwargs):
    hdrs = [ "include/" + hdr for hdr in hdrs ]
    outs = [] + hdrs
    if static_lib:
        outs.append(static_lib)
    if shared_lib :
        outs.append(shared_lib)
    if not rpms:
        rpms = ["@" + name + "//file:file"]
    bash_cmd = "mkdir " + name + " && cd " + name
    bash_cmd += " && for e in $(SRCS); do rpm2cpio ../$$e | cpio -idm; done"
    if include_path != None:
        if header_only:
            bash_cmd += "&& cp -rf " + include_path + "/* ../$(@D)/"
        else:
            bash_cmd += "&& cp -rf " + include_path + "/* ../$(@D)/include"
    if len(static_libs) > 0:
        # extract all .a files to its own directory in case .o file conflict, and ar them together to target .a file.
        bash_cmd += "&& for a in " + " ".join(static_libs) + "; do d=$${a%.a} && mkdir $$d && cd $$d && ar x ../" + lib_path + "$$a && cd -; done && ar rc ../$(@D)/" + static_lib + " */*.o"
    elif static_lib:
        bash_cmd += "&& cp -L " + lib_path + "/*.a" + " ../$(@D)/"
    if shared_lib:
        bash_cmd += "&& echo $$PATH && which patchelf && patchelf --version && cp -L " + lib_path + "/" + shared_lib + " ../$(@D) && patchelf --set-soname " + shared_lib + " ../$(@D)/" + shared_lib
    for share_lib in shared_libs:
        outs.append(share_lib)
        bash_cmd += "&& cp -L " + lib_path + "/" + share_lib + " ../$(@D) && patchelf --set-soname " + share_lib + " ../$(@D)/" + share_lib
    for path in bins:
        outs.append(path)
        bash_cmd += "&& cp -rL " + path + " ../$(@D)"
    bash_cmd += " && cd -"

    native.genrule(
        name = name + "_files",
        srcs = rpms,
        outs = outs,
        cmd = bash_cmd,
        visibility = ["//visibility:public"],
        tags=tags,
    )
    hdrs_fg_target = name + "_hdrs_fg"
    native.filegroup(
        name = hdrs_fg_target,
        srcs = hdrs,
    )
    if static_lib:
        native.filegroup(
            name = name + "_static",
            srcs = [static_lib],
            visibility = ["//visibility:public"],
        )
    srcs = []
    shared_files = shared_libs + (shared_lib and [shared_lib] or [])
    if shared_files:
        shared_filegroup = name + "_shared"
        native.filegroup(
            name = shared_filegroup,
            srcs = shared_files,
            visibility = ["//visibility:public"],
        )
        if shared_libs:
            srcs.append(shared_filegroup)

    if bins:
        bins_filegroup = name + "_bins"
        native.filegroup(
            name = bins_filegroup,
            srcs = bins,
            visibility = ["//visibility:public"],
            tags=tags,
        )

    if static_lib == None:
        native.cc_library(
            name = name,
            hdrs = [hdrs_fg_target],
            srcs = shared_files,
            deps = deps,
            strip_include_prefix = "include",
            include_prefix = include_prefix,
            visibility = ["//visibility:public"],
            **kwargs
        )
    else:
        import_target = name + "_import"
        alwayslink = static_lib!=None
        native.cc_import(
            name = import_target,
            static_library = static_lib,
            shared_library = shared_lib,
            alwayslink=alwayslink,
            visibility = ["//visibility:public"],
        )
        native.cc_library(
            name = name,
            hdrs = [hdrs_fg_target],
            srcs = srcs,
            deps = deps + [import_target],
            visibility = ["//visibility:public"],
            strip_include_prefix = "include",
            include_prefix = include_prefix,
            **kwargs
        )

def _copy_files_impl(ctx):
    if not ctx.files.srcs:
        return [DefaultInfo(files = depset())]

    output_files = []
    for f in ctx.files.srcs:
        output_files.append(ctx.actions.declare_file(f.basename))

    commands = []
    for i in range(len(output_files)):
        f_in = ctx.files.srcs[i]
        f_out = output_files[i]
        commands.append("cp {input} {output}".format(input = f_in.path, output = f_out.path))

    ctx.actions.run_shell(
        inputs = ctx.files.srcs,
        outputs = output_files,
        command = " && ".join(commands),
        progress_message = "Copying %d files for %s" % (len(output_files), ctx.label),
    )

    return [DefaultInfo(files = depset(output_files))]

copy_files = rule(
    implementation = _copy_files_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True, mandatory = True),
    },
)
