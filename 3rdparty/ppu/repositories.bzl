"""SDK source inputs for PPU. No private repository or arch overlay is required."""

# Use pristine SDK source trees. Patches and BUILD ownership stay in RTP-LLM.
def _sdk_source_impl(ctx):
    directory = ctx.os.environ.get(ctx.attr.source_env)
    if not directory:
        fail("Set --repo_env=%s=/path/to/pristine/sdk/source (revision %s)" % (ctx.attr.source_env, ctx.attr.revision))
    source = ctx.path(directory)
    if not source.exists:
        fail("PPU SDK source directory does not exist: " + directory)
    if source.get_child(".git").exists:
        result = ctx.execute(["git", "-C", directory, "rev-parse", "HEAD"])
        if result.return_code or result.stdout.strip() != ctx.attr.revision:
            fail("PPU SDK revision mismatch for " + ctx.attr.source_env)
        dirty = ctx.execute(["git", "-C", directory, "status", "--porcelain", "--untracked-files=all"])
        if dirty.return_code or dirty.stdout.strip():
            fail("PPU SDK source must be pristine: " + directory)
    else:
        stamp = source.get_child("REVISION")
        if not stamp.exists or ctx.read(stamp).strip() != ctx.attr.revision:
            fail("SDK source archives must contain REVISION=" + ctx.attr.revision)
    # Copy into this repository before patching, never modify the supplied SDK.
    result = ctx.execute(["cp", "-R", directory + "/.", str(ctx.path("."))])
    if result.return_code:
        fail("Copying PPU SDK sources failed: " + result.stderr)
    # Source archives may carry BUILD symlinks; remove them before writing our
    # definitions so ctx.file cannot follow a link into the supplier's tree.
    for name in ["BUILD", "BUILD.bazel", "WORKSPACE", "WORKSPACE.bazel", ".git"]:
        ctx.delete(name)
    for patch in ctx.attr.patches:
        ctx.patch(patch, strip = 0)
    ctx.file("BUILD.bazel", ctx.read(ctx.attr.build_file))
    ctx.file("WORKSPACE", "workspace(name = %r)\n" % ctx.name)

_sdk_source = repository_rule(
    implementation = _sdk_source_impl,
    attrs = {
        "source_env": attr.string(mandatory = True),
        "revision": attr.string(mandatory = True),
        "build_file": attr.label(mandatory = True),
        "patches": attr.label_list(),
    },
    environ = ["PPU_CUTLASS_FLASHINFER_SOURCE", "PPU_CUTLASS_FLASHMLA_SOURCE", "PPU_FLASHINFER_SOURCE", "PPU_FLASHMLA_SOURCE"],
    local = True,
)

def ppu_sdk_repositories():
    _sdk_source(
        name = "cutlass3_ppu_flashinfer",
        source_env = "PPU_CUTLASS_FLASHINFER_SOURCE",
        revision = "587809382ab90a68e74a42835ff6ddc1d835a830",
        build_file = str(Label("@rtp_llm//3rdparty/ppu/cutlass_ppu:cutlass.BUILD")),
    )

    _sdk_source(
        name = "cutlass3_ppu_flashmla",
        source_env = "PPU_CUTLASS_FLASHMLA_SOURCE",
        revision = "b1fb3d1f5d404387b3dd475d9bc5007af4cf8644",
        build_file = str(Label("@rtp_llm//3rdparty/ppu/cutlass_ppu:cutlass.BUILD")),
    )

    _sdk_source(
        name = "flashmla_ppu",
        source_env = "PPU_FLASHMLA_SOURCE",
        revision = "cb87b7763e5de1ca3c6254a0995656c38af6963c",
        build_file = str(Label("@rtp_llm//3rdparty/ppu/flashmla:flashmla.BUILD")),
        # header-only on PPU; 0001 adds the interface header (kernels run via pip wheel)
        patches = [
            "@rtp_llm//3rdparty/ppu/flashmla:0001-add-interface.patch",
        ],
    )

    _sdk_source(
        name = "flashinfer_ppu",
        source_env = "PPU_FLASHINFER_SOURCE",
        revision = "10c67278e15565a3fe1d7dfd401377ee26dbad28",
        build_file = str(Label("@rtp_llm//3rdparty/ppu/flashinfer:flashinfer.BUILD")),
        patches = [
            "@rtp_llm//3rdparty/flashinfer:0002-dispatch-group-size.patch",
            "@rtp_llm//3rdparty/flashinfer:0005-update-add-mla-attn-test-impl-mla-write-kvcache.patch",
            "@rtp_llm//3rdparty/flashinfer:0006-add-mla-dispatch-inc.patch",
            "@rtp_llm//3rdparty/flashinfer:0009-sp-sample.patch",
            "@rtp_llm//3rdparty/ppu/flashinfer:0001-cuda13-cub-reduce-ops.patch",
        ],
    )
