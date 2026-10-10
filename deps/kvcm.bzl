"""Pinned KVCM/PACE sources and the paired binary artifact gate."""

# The SDK changes virtual interfaces. The April RPM cannot provide this ABI.
# Published SDKs and Manager share the source tuple below.
KVCM_SOURCE_LOCK = {
    "internal_commit": "1c24aeac35c819c544316e9753eec0186ea53bd3",
    "opensource_commit": "a71117d9745d7228f92aaf64c5414737878bd153",
    "pace_commit": "770bd4df361f86cd937f9144e910d202e1a7401f",
}

KVCM_SOURCE_ID = ":".join([KVCM_SOURCE_LOCK[key] for key in [
    "internal_commit", "opensource_commit", "pace_commit",
]])

KVCM_CLIENT_ARTIFACT = {
    "urls": ["http://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/kv_cache_manager/client/kv-cache-manager-client-2026_10_08_00_51-77672840-cuda12.x86_64.rpm"],
    "sha256": "f49ce9a3e2bc16053d558a0d976f955ab9d25a06c34dac84aa4b09a28bc04538",
    "source_id": KVCM_SOURCE_ID,
}
KVCM_CLIENT_CUDA129_X86_ARTIFACT = {
    "urls": ["http://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/kv_cache_manager/client/kv-cache-manager-client-2026_10_08_00_52-77672840-cuda129.x86_64.rpm"],
    "sha256": "2c463314a91a73d98248bf7866a55808e3761f0914597d4012204fd04a098fa5",
    "source_id": KVCM_SOURCE_ID,
}
KVCM_CLIENT_CUDA130_X86_ARTIFACT = {
    "urls": ["http://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/kv_cache_manager/client/kv-cache-manager-client-2026_10_08_00_52-77672840-cuda130.x86_64.rpm"],
    "sha256": "14aec756d95c64a9abf5899f14f3f3d5ebfa9ee131264f3900871ad01edd0054",
    "source_id": KVCM_SOURCE_ID,
}
KVCM_CLIENT_CUDA130_ARM_ARTIFACT = {
    "urls": ["http://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/kv_cache_manager/client/kv-cache-manager-client-2026_10_08_00_52-77672840-cuda130.aarch64.rpm"],
    "sha256": "6612b00830ea9a97acb01e2d7ecb96c5f6b8254a70011f4de15a42083ef2948c",
    "source_id": KVCM_SOURCE_ID,
}
KVCM_SERVER_ARTIFACT = {
    "urls": ["http://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/kv_cache_manager/server/kv-cache-manager-server-77672840.x86_64.tar.gz"],
    "sha256": "d0b2f7d8ba6b0f38ec1be9d9b5d0c14cd386308723fbfbb5af4941f467528f0d",
    "source_id": KVCM_SOURCE_ID,
}

def _kvcm_manifest_impl(ctx):
    ctx.file("WORKSPACE", "")
    ctx.file("BUILD.bazel", 'exports_files(["manifest.json"])\n')
    manifest_path = ctx.os.environ.get("KVCM_ARTIFACT_MANIFEST", "")
    if manifest_path:
        ctx.symlink(manifest_path, "manifest.json")
    else:
        ctx.file("manifest.json", "", executable = False)

# Bazel 6 tracks file content through labels, not reads of absolute paths.
# Only this small local repository is refreshed; binary artifacts stay cached.
_kvcm_manifest = repository_rule(
    implementation = _kvcm_manifest_impl,
    local = True,
    environ = ["KVCM_ARTIFACT_MANIFEST"],
)

def _client_variant(ctx):
    if ctx.attr.client_variant:
        return ctx.attr.client_variant
    variant = ctx.os.environ.get("KVCM_CLIENT_VARIANT", "cuda")
    if variant not in ["cpu", "cuda"]:
        fail("KVCM_CLIENT_VARIANT must be cpu or cuda")
    return variant

def _artifact_from_manifest(ctx):
    manifest_path = ctx.os.environ.get("KVCM_ARTIFACT_MANIFEST", "")
    if not manifest_path:
        if ctx.attr.kind == "client" and _client_variant(ctx) == "cpu":
            fail("No CPU-only SDK is published in the source lock. Supply a paired cpu artifact via KVCM_ARTIFACT_MANIFEST.")
        return {"urls": ctx.attr.urls, "sha256": ctx.attr.sha256, "source_id": ctx.attr.source_id}
    manifest = json.decode(ctx.read(ctx.attr.manifest))
    if manifest.get("source_id") != ctx.attr.expected_source_id:
        fail("KVCM_ARTIFACT_MANIFEST does not match KVCM_SOURCE_LOCK")
    # A client validates its server pair; platform-specific variants come from BUILD selects.
    wanted_variants = [ctx.attr.server_variant]
    if ctx.attr.kind == "client":
        wanted_variants.append(_client_variant(ctx))
    selected = {}
    for wanted in wanted_variants:
        matches = [item for item in manifest.get("artifacts", []) if item.get("variant") == wanted]
        if len(matches) != 1:
            fail("KVCM manifest must contain exactly one %s artifact" % wanted)
        item = matches[0]
        digest = item.get("sha256", "")
        if (item.get("source_id") != ctx.attr.expected_source_id or
            len(digest) != 64 or [char for char in digest.elems() if char not in "0123456789abcdef"] or
            not item.get("url")):
            fail("KVCM manifest artifact requires a matching source_id, SHA256 and URL")
        selected[wanted] = {"urls": [item["url"]], "sha256": digest, "source_id": item["source_id"]}
    return selected[_client_variant(ctx) if ctx.attr.kind == "client" else ctx.attr.server_variant]

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
        variant = _client_variant(ctx)
        # Keep the legacy manifest key while recording the concrete SDK variant.
        ctx.file("KVCM_CLIENT_VARIANT", ("cuda12_x86" if variant == "cuda" else variant) + "\n")
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
        "manifest": attr.label(default = "@kvcm_artifact_manifest//:manifest.json", allow_single_file = True),
        "kind": attr.string(mandatory = True),
        "client_variant": attr.string(),
        "server_variant": attr.string(default = "server"),
        "urls": attr.string_list(),
        "sha256": attr.string(),
        "source_id": attr.string(),
        "expected_source_id": attr.string(mandatory = True),
    },
)

def kvcm_deps():
    _kvcm_manifest(name = "kvcm_artifact_manifest")
    for name, kind, client_variant, server_variant, artifact in [
        ("remote_kv_cache_manager_client_rpm", "client", "", "server", KVCM_CLIENT_ARTIFACT),
        ("remote_kv_cache_manager_client_rpm_cuda129_x86", "client", "cuda129_x86", "server", KVCM_CLIENT_CUDA129_X86_ARTIFACT),
        ("remote_kv_cache_manager_client_rpm_cuda130_x86", "client", "cuda130_x86", "server", KVCM_CLIENT_CUDA130_X86_ARTIFACT),
        ("remote_kv_cache_manager_client_rpm_cuda130_arm", "client", "cuda130_arm", "server", KVCM_CLIENT_CUDA130_ARM_ARTIFACT),
        ("remote_kv_cache_manager_server", "server", "", "server", KVCM_SERVER_ARTIFACT),
    ]:
        _kvcm_artifact(
            name = name,
            kind = kind,
            client_variant = client_variant,
            server_variant = server_variant,
            urls = artifact["urls"],
            sha256 = artifact["sha256"],
            source_id = artifact["source_id"],
            expected_source_id = KVCM_SOURCE_ID,
        )
