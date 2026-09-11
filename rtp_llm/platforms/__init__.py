"""Built-in platform manifests. Loading hooks must not import a device runtime.

Platforms supply deferred adapters to the existing factories and lightweight
model descriptors to ModuleRegistry; execution uses the selected real modules.
"""

from importlib import import_module

_BUILTIN_PLATFORMS = ("rtp_llm.platforms.ppu",)


def register_backend_hooks():
    for name in _BUILTIN_PLATFORMS:
        import_module(name).register_backend_hooks()


def register_modules(registry):
    for name in _BUILTIN_PLATFORMS:
        import_module(name).register_modules(registry)
