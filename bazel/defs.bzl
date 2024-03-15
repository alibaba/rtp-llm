load("@pip_gpu_torch//:requirements.bzl", requirement_gpu="requirement")
load("@pip_gpu_cuda12_torch//:requirements.bzl", requirement_gpu_cuda12="requirement")

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

def upload_wheel(name, src, dir, wheel_prefix):
    oss_path = "oss://search-ad/%s/%s" % (dir, wheel_prefix) + "_$$(date '+%Y-%m-%d_%H_%M_%S')"
    native.genrule(
        name = name,
        srcs = [src],
        outs = ["tmp_wheel.whl"],
        cmd = "bash -c 'set -xe;" +
            "mkdir tmp;" +
            "cp $(locations %s) tmp; " % (src) +
            "osscmd put $(locations %s) %s/$$(basename $(locations %s));" % (src, oss_path, src) +
            "mv tmp/$$(basename $(locations %s)) $(OUTS);" % (src) +
            "rm tmp -rf;" +
            "'",
        tags = [
            "local",
            "manual",
        ],
        visibility = ["//visibility:public"],
    )

def pyc_wheel(name, package_name, src):
    native.genrule(
        name = name,
        srcs = [src],
        outs = [package_name + "-cp310-cp310-manylinux1_x86_64.whl"],
        exec_tools = ["//bazel:pyc_wheel.py"],
        cmd = "bash -c 'set -xe;" +
            "cp $(locations %s) $(OUTS);" % (src) +
            "chmod a+w $(OUTS);" +
            "/opt/conda310/bin/python $(location //bazel:pyc_wheel.py) $(OUTS);" +
            "'",
        tags = [
            "local",
            "manual",
        ],
        visibility = ["//visibility:public"],
    )

def rename_wheel(name, package_name, src):
    native.genrule(
        name = name,
        srcs = [src],
        outs = [package_name + "-cp310-cp310-manylinux1_x86_64.whl"],
        cmd = "bash -c 'set -xe;" +
            "cp $(locations %s) $(OUTS);" % (src) +
            "chmod a+w $(OUTS);" +      
            "'",
        tags = [
            "local",
            "manual",
        ],
        visibility = ["//visibility:public"],
    )