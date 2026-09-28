"""Pinned KVCM/PACE sources and the paired binary artifact gate."""

# The SDK changes virtual interfaces. The April RPM cannot provide this ABI.
# Populate both artifact records only after packaging these source commits.
KVCM_SOURCE_LOCK = {
    "internal_commit": "bf6de8ff3c27b7543c489d420c367830f5ff5cb2",
    "opensource_commit": "6015fca48a091dc18ea9497518138cb58959c3f2",
    "pace_commit": "d6cec4a4adfb8d46624b9014560a4f15ebb91296",
}

KVCM_CLIENT_ARTIFACT = {"urls": [], "sha256": "", "source_id": ""}
KVCM_SERVER_ARTIFACT = {"urls": [], "sha256": "", "source_id": ""}

def _source_id():
    return ":".join([KVCM_SOURCE_LOCK[key] for key in [
        "internal_commit", "opensource_commit", "pace_commit",
    ]])

def _kvcm_artifact_impl(ctx):
    if not ctx.attr.urls or len(ctx.attr.sha256) != 64 or ctx.attr.source_id != ctx.attr.expected_source_id:
        fail("KVCM P1 requires a newly packaged, paired SDK RPM and server archive. " +
             "Record their verified URLs, SHA256 and source_id in deps/kvcm.bzl; " +
             "see docs/kvcm_remote_cache.md. The legacy RPM is ABI-incompatible.")
    ctx.file("WORKSPACE", "workspace(name = %r)\n" % ctx.name)
    if ctx.attr.kind == "client":
        ctx.download(url = ctx.attr.urls, output = "file/kv-cache-manager-client.rpm", sha256 = ctx.attr.sha256)
        ctx.file("file/BUILD.bazel", """
filegroup(
    name = "file",
    srcs = ["kv-cache-manager-client.rpm"],
    visibility = ["//visibility:public"],
)
""")
    else:
        ctx.download_and_extract(url = ctx.attr.urls, sha256 = ctx.attr.sha256, type = "tar.gz")
        ctx.file("BUILD.bazel", 'exports_files(["bin/kv_cache_manager_bin"])\n')
    ctx.file("KVCM_SOURCE_ID", ctx.attr.source_id + "\n")

_kvcm_artifact = repository_rule(
    implementation = _kvcm_artifact_impl,
    attrs = {
        "kind": attr.string(mandatory = True),
        "urls": attr.string_list(),
        "sha256": attr.string(),
        "source_id": attr.string(),
        "expected_source_id": attr.string(mandatory = True),
    },
)

def kvcm_deps():
    for name, kind, artifact in [
        ("remote_kv_cache_manager_client_rpm", "client", KVCM_CLIENT_ARTIFACT),
        ("remote_kv_cache_manager_server", "server", KVCM_SERVER_ARTIFACT),
    ]:
        _kvcm_artifact(
            name = name,
            kind = kind,
            urls = artifact["urls"],
            sha256 = artifact["sha256"],
            source_id = artifact["source_id"],
            expected_source_id = _source_id(),
        )
