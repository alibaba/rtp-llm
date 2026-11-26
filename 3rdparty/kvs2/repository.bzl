def _kvs2_sdk_repository(repository_ctx):
    prefix = repository_ctx.os.environ.get("RTP_LLM_KVS2_SDK")
    if not prefix or not prefix.startswith("/"):
        fail("Set --repo_env=RTP_LLM_KVS2_SDK to an absolute KVS2 SDK installation prefix")
    sdk = repository_ctx.path(prefix)
    for relative in ["include/kvs/client.h", "lib/libkvs_client.so"]:
        if not sdk.get_child(relative).exists:
            fail("Incomplete KVS2 SDK installation: missing " + relative)
    repository_ctx.symlink(sdk.get_child("include"), "include")
    repository_ctx.symlink(sdk.get_child("lib"), "lib")
    runtime_libs = sdk.get_child("runtime-libs")
    if runtime_libs.exists:
        repository_ctx.symlink(runtime_libs, "runtime-libs")
    repository_ctx.file("BUILD.bazel", """
cc_library(
    name = "client",
    srcs = ["lib/libkvs_client.so"],
    hdrs = glob(["include/kvs/*.h"]),
    includes = ["include"],
    additional_linker_inputs = glob(["runtime-libs/*.so*"]),
    linkopts = ["-Wl,-rpath-link,external/kvs2_sdk/runtime-libs"],
    data = glob(["runtime-libs/*.so*"]),
    visibility = ["//visibility:public"],
)
""")

kvs2_sdk_repository = repository_rule(
    implementation = _kvs2_sdk_repository,
    environ = ["RTP_LLM_KVS2_SDK"],
    local = True,
)
