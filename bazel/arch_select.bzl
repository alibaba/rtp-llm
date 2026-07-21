load(
    "@arch_config//:arch_select.bzl",
    _no_block_copy_link_deps = "no_block_copy_link_deps",
    _torch_deps = "torch_deps",
)

torch_deps = _torch_deps

def no_block_copy_link_deps():
    # Keep downstream device backends source-compatible when the generic
    # NoBlockCopy interface gains optional fast paths. Device implementations
    # override these weak fallbacks when they support the new operations.
    return _no_block_copy_link_deps() + [
        "//rtp_llm/models_py/bindings:no_block_copy_fallback",
    ]
