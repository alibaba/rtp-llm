load(
    "@arch_config//:arch_select.bzl",
    _requirement = "requirement",
    _torch_deps = "torch_deps",
    _transfer_backend_deps = "transfer_backend_deps",
)

requirement = _requirement
torch_deps = _torch_deps
transfer_backend_deps = _transfer_backend_deps
