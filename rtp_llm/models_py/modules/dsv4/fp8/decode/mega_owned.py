"""Bind native TP1 Mega decode to the loaded V4 parameter tree."""

from .mega_policy import mega_decode_enabled


def initialize_owned_mega_decode(backbone, args) -> None:
    """Initialize once after parameter postprocessing, before warmup or graph capture."""
    if backbone.commit_only or not mega_decode_enabled(
        args, backbone.embed.weight.device
    ):
        return
    if backbone._mega_csa_runtime is not None:
        raise RuntimeError("V4 Mega decode was already initialized")
    from .mega_csa_runtime import MegaCSARuntime

    runtime = MegaCSARuntime()
    for layer in backbone.layers:
        # None selects owned parameters in the adapters. Block still dispatches
        # the original CSA/HCA constructors according to its compression ratio.
        layer.enable_mega_csa(runtime, None)
        layer.enable_mega_hca(runtime, None)
    backbone._mega_csa_runtime = runtime
    backbone._mega_decode_enabled = True
