load(
    "@arch_config//:arch_select.bzl",
    _no_block_copy_link_deps = "no_block_copy_link_deps",
    _requirement = "requirement",
    _torch_deps = "torch_deps",
)

requirement = _requirement
torch_deps = _torch_deps

def no_block_copy_link_deps():
    return _no_block_copy_link_deps()
