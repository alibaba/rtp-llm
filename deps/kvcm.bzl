"""Pinned KVCM/PACE sources and the paired binary artifact gate."""

# The SDK changes virtual interfaces. The April RPM cannot provide this ABI.
# Populate both artifact records only after packaging these source commits.
KVCM_SOURCE_LOCK = {
    "internal_commit": "32dc3162ec4f9f981617f8d82f9696a4faa2fe5b",
    "opensource_commit": "6015fca48a091dc18ea9497518138cb58959c3f2",
    "pace_commit": "770bd4df361f86cd937f9144e910d202e1a7401f",
}

KVCM_CLIENT_ARTIFACT = {"urls": [], "sha256": "", "source_id": ""}
KVCM_SERVER_ARTIFACT = {"urls": [], "sha256": "", "source_id": ""}

def _source_id():
    return ":".join([KVCM_SOURCE_LOCK[key] for key in [
        "internal_commit", "opensource_commit", "pace_commit",
    ]])

def _artifact_from_manifest(ctx):
    manifest_path = ctx.os.environ.get("KVCM_ARTIFACT_MANIFEST", "")
    if not manifest_path:
        return {"urls": ctx.attr.urls, "sha256": ctx.attr.sha256, "source_id": ctx.attr.source_id}
    manifest = json.decode(ctx.read(ctx.path(manifest_path)))
    if manifest.get("source_id") != ctx.attr.expected_source_id:
        fail("KVCM_ARTIFACT_MANIFEST does not match KVCM_SOURCE_LOCK")
    variant = ctx.os.environ.get("KVCM_CLIENT_VARIANT", "cuda")
    if variant not in ["cpu", "cuda"]:
        fail("KVCM_CLIENT_VARIANT must be cpu or cuda")
    # Validate the pair even when only the client repository is requested.
    selected = {}
    for kind, wanted in [("client", variant), ("server", "server")]:
        matches = [item for item in manifest.get("artifacts", []) if item.get("variant") == wanted]
        if len(matches) != 1:
            fail("KVCM manifest must contain exactly one %s artifact" % wanted)
        item = matches[0]
        digest = item.get("sha256", "")
        if (item.get("source_id") != ctx.attr.expected_source_id or
            len(digest) != 64 or [char for char in digest.elems() if char not in "0123456789abcdef"] or
            not item.get("url")):
            fail("KVCM manifest artifact requires a matching source_id, SHA256 and URL")
        selected[kind] = {"urls": [item["url"]], "sha256": digest, "source_id": item["source_id"]}
    return selected[ctx.attr.kind]

def _kvcm_artifact_impl(ctx):
    artifact = _artifact_from_manifest(ctx)
    if not artifact["urls"] or len(artifact["sha256"]) != 64 or artifact["source_id"] != ctx.attr.expected_source_id:
        fail("KVCM P1 requires a newly packaged, paired SDK RPM and server archive. " +
             "Record their verified URLs, SHA256 and source_id in deps/kvcm.bzl or pass " +
             "--repo_env=KVCM_ARTIFACT_MANIFEST=/absolute/path/MANIFEST.json; " +
             "see docs/kvcm_remote_cache.md. The legacy RPM is ABI-incompatible.")
    ctx.file("WORKSPACE", "workspace(name = %r)\n" % ctx.name)
    if ctx.attr.kind == "client":
        ctx.download(url = artifact["urls"], output = "file/kv-cache-manager-client.rpm", sha256 = artifact["sha256"])
        ctx.file("BUILD.bazel", 'exports_files(["KVCM_SOURCE_ID", "KVCM_CLIENT_VARIANT", "KVCM_ARTIFACT_SHA256"])\n')
        ctx.file("KVCM_CLIENT_VARIANT", ctx.os.environ.get("KVCM_CLIENT_VARIANT", "cuda") + "\n")
        ctx.file("file/BUILD.bazel", """
filegroup(
    name = "file",
    srcs = ["kv-cache-manager-client.rpm"],
    visibility = ["//visibility:public"],
)
""")
    else:
        ctx.download_and_extract(url = artifact["urls"], sha256 = artifact["sha256"], type = "tar.gz")
        if ctx.read("KVCM_SOURCE_ID").strip() != artifact["source_id"]:
            fail("KVCM server archive source marker does not match the manifest")
        ctx.file("BUILD.bazel", 'exports_files(["bin/kv_cache_manager_bin", "KVCM_SOURCE_ID", "KVCM_ARTIFACT_SHA256", "etc/default_startup_config.json"])\n')
    ctx.file("KVCM_SOURCE_ID", artifact["source_id"] + "\n")
    ctx.file("KVCM_ARTIFACT_SHA256", artifact["sha256"] + "\n")

_kvcm_artifact = repository_rule(
    implementation = _kvcm_artifact_impl,
    environ = ["KVCM_ARTIFACT_MANIFEST", "KVCM_CLIENT_VARIANT"],
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
