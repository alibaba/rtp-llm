"""Public PPU entrypoint; SDK and kernels load only when a consumer selects PPU.

Reusable SDK adapters/operators live in runtime.py, kernels/ and modules/.
Model-specific composition and capability predicates live in models/<model>/.
Extend the existing factories and ModuleRegistry, retaining per-instance weight,
state and collective contracts; registration does not qualify new chips/models.
"""


def register_backend_hooks():
    from rtp_llm.utils.backend_registry import register_backend_hook

    from .attention import register_attention
    from .models.dsv4.register import install

    register_backend_hook("attention", register_attention)
    install()


def register_modules(registry):
    # A model adapter registers its full module contract before platform
    # discovery. Other model registries must not acquire V4 implementations.
    if not registry.has_module("rtp.dsv4.model"):
        return
    from .models.dsv4.manifest import register_modules as register_dsv4

    register_dsv4(registry)
