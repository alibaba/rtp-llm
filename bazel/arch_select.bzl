load(
    "@arch_config//:arch_select.bzl",
    _no_block_copy_link_deps = "no_block_copy_link_deps",
    _torch_deps = "torch_deps",
    _transfer_backend_deps = "transfer_backend_deps",
)

torch_deps = _torch_deps
transfer_backend_deps = _transfer_backend_deps

def no_block_copy_link_deps():
    return _no_block_copy_link_deps()
