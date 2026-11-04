
def _gen_git_source_impl(ctx):
    out = ctx.actions.declare_file(ctx.attr.out_filename)
    info_file = ctx.info_file
    cmd = """
    # 提取 SHA，默认值为 unknown
    SHA=$(grep "^STABLE_GIT_COMMIT" {info_path} | cut -d ' ' -f 2)
    if [ -z "$SHA" ]; then SHA="unknown"; fi

    cat <<EOF > {out_path}
#include "{header_path}"

namespace {ns} {{
    const char* GIT_COMMIT_SHA = "$SHA";
}}
EOF
    """.format(
        info_path = info_file.path,
        header_path = ctx.attr.header_path,
        out_path = out.path,
        ns = ctx.attr.namespace
    )

    ctx.actions.run_shell(
        inputs = [info_file], # 关键：依赖 status 文件
        outputs = [out],
        command = cmd,
        mnemonic = "GenGitVersionSource"
    )
    return [DefaultInfo(files = depset([out]))]

_gen_git_source = rule(
    implementation = _gen_git_source_impl,
    attrs = {
        "out_filename": attr.string(mandatory=True),
        "header_path": attr.string(mandatory=True),
        "namespace": attr.string(mandatory=True),
    },
)

def git_version_library(name, namespace, **kwargs):
    header_file = name + ".h"
    source_file = name + ".cpp"
    native.genrule(
        name = name + "_gen_header",
        outs = [header_file],
        cmd = """echo '#pragma once
namespace {ns} {{ extern const char* GIT_COMMIT_SHA; }}' > $@
""".format(ns = namespace),
    )
    _gen_git_source(
        name = name + "_gen_source",
        out_filename = source_file,
        header_path = header_file,
        namespace = namespace,
    )
    native.cc_library(
        name = name,
        hdrs = [header_file],
#        linkstamp =  ":" + name + "_gen_source",
        srcs = [":" + name + "_gen_source"],
        **kwargs
    )
