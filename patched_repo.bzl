def _patched_pip_repository_impl(repository_ctx):
    # 获取原始pip仓库路径：注意，这里我们使用Label来定位原始仓库的BUILD文件，然后取它的目录
    original_path = repository_ctx.path(Label("@pip_gpu_rocm_torch_aiter//:BUILD.bazel")).dirname

    original_path_str = str(original_path)

    # 复制原始仓库内容到当前仓库根目录
    repository_ctx.execute([
        "cp", "-r", original_path_str + "/.", repository_ctx.path("")
    ])

    patch_path = repository_ctx.path(Label("//3rdparty/aiter:refine-aiter-asm-dir.patch"))
    patch_path_aiter = repository_ctx.path(Label("//3rdparty/aiter:aiter-flash_attn.patch"))
    patch_path_str = str(patch_path)

    result = repository_ctx.execute([
        "sh", "-c",
        "cd site-packages && patch -p1 -i " + patch_path_str + " && patch -p1 -i " + str(patch_path_aiter)
    ])

    if result.return_code != 0:
        fail("Patch failed: %s" % result.stderr)

patched_pip_repository = repository_rule(
    implementation = _patched_pip_repository_impl,
    attrs = {},
)