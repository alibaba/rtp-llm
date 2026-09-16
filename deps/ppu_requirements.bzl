"""Use the PPU pip inputs and lock for distribution metadata as well as Bazel."""

def _ppu_requirements_impl(ctx):
    locked = {}
    for line in ctx.read(ctx.attr.lock).replace("\\\n", " ").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = line.split(" --hash=")[0].strip()
        locked[requirement] = line

    requirements = []
    for line in ctx.read(ctx.attr.requirements).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line not in locked:
            fail("PPU requirement is missing or differs from its lock: " + line)
        if " @ " in line:
            url = line.split(" @ ")[1]
            if not url.startswith("https://") or "#sha256=" not in url:
                fail("PPU wheel URLs must use HTTPS and SHA256: " + line)
            digest = url.split("#sha256=")[1]
            if len(digest) != 64 or "--hash=sha256:" + digest not in locked[line]:
                fail("PPU wheel URL hash differs from its pip lock: " + line)
        requirements.append(line)

    ctx.file("BUILD.bazel", 'exports_files(["requirements.bzl"])\n')
    ctx.file("requirements.bzl", "PPU_WHEEL_REQUIREMENTS = " + repr(requirements) + "\n")

ppu_requirements = repository_rule(
    implementation = _ppu_requirements_impl,
    attrs = {
        "requirements": attr.label(default = Label("//:requirements_torch_ppu.txt"), allow_single_file = True),
        "lock": attr.label(default = Label("//:requirements_lock_torch_ppu.txt"), allow_single_file = True),
    },
)
