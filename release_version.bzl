# Independent of def.bzl: this file is loaded by the module extension and must have zero
# external loads. def.bzl loads @rules_cc/@rules_python at the top, which forms a main-repo
# mapping cycle during evaluation (observed: cycles detected + @@rules_python_internal not
# visible).

def _read_release_version_impl(repository_ctx):
    release_version_content = repository_ctx.read(repository_ctx.path(Label("//rtp_llm:release_version.py")))

    pattern = 'RELEASE_VERSION = "'
    start_index = release_version_content.find(pattern)
    if start_index == -1:
        fail('rtp_llm/release_version.py: no \'RELEASE_VERSION = "<version>"\' assignment found; refusing to fabricate a release version')
    start_index += len(pattern)
    end_index = release_version_content.find('"', start_index)
    if end_index == -1:
        fail("rtp_llm/release_version.py: RELEASE_VERSION string literal is unterminated (missing closing quote)")
    release_version = release_version_content[start_index:end_index]

    repository_ctx.file("BUILD", "")
    repository_ctx.file("defs.bzl", "RELEASE_VERSION = '{}'".format(release_version))

read_release_version = repository_rule(
    implementation = _read_release_version_impl,
    attrs = {},
)
