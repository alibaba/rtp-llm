"""Registry-level NAME invariants for FMHA / MLA implementations.

These tests guard the explicit-dispatch contract of attn_factory.get_fmha_impl:
the user types `--attn_backend=<name>` and the factory looks up impls via NAME.
A missing NAME on a registered impl turns into a generic "can not find mha
type" error at request time; a duplicate NAME within one registry silently
shadows whichever impl appears later in the list.

Both tests are import-only — no GPU, no kernel — so they run on every
arch's CI worker.
"""

import pytest

from rtp_llm.models_py.modules.factory.attention import (  # noqa: F401  (registers impls)
    AttnImplFactory,
)
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    DECODE_MHA_IMPS,
    DECODE_MLA_IMPS,
    PREFILL_MHA_IMPS,
    PREFILL_MLA_IMPS,
)


_REGISTRIES = (
    ("PREFILL_MHA_IMPS", PREFILL_MHA_IMPS),
    ("DECODE_MHA_IMPS", DECODE_MHA_IMPS),
    ("PREFILL_MLA_IMPS", PREFILL_MLA_IMPS),
    ("DECODE_MLA_IMPS", DECODE_MLA_IMPS),
)


@pytest.mark.parametrize(
    "registry_name,registry",
    _REGISTRIES,
    ids=[name for name, _ in _REGISTRIES],
)
def test_all_registered_impls_have_name(registry_name, registry):
    """Every impl in every registry must have a non-empty NAME ClassVar.

    The attn_factory `_validate_impl_names()` startup hook enforces the same
    invariant — this test re-asserts it from the test runner's POV so a
    regression shows up as a normal pytest FAILED rather than as an obscure
    ImportError during collection.
    """
    missing = [
        f"{cls.__module__}.{cls.__name__}"
        for cls in registry
        if not getattr(cls, "NAME", "")
    ]
    assert not missing, (
        f"Registry {registry_name} contains impl(s) with empty NAME: {missing}. "
        f"Set NAME on each class to its --attn_backend value (see "
        f"rtp_llm/server/server_args/fmha_group_args.py help text)."
    )


@pytest.mark.parametrize(
    "registry_name,registry",
    _REGISTRIES,
    ids=[name for name, _ in _REGISTRIES],
)
def test_no_duplicate_class_objects_within_registry(registry_name, registry):
    """Within a single registry, no class object is registered more than once.

    Multiple DIFFERENT impl classes legitimately share a NAME within one
    registry when they're differentiated by support_parallelism_config()
    (e.g. SparseMlaImpl + SparseMlaCpImpl both expose "sparse_mla" in
    PREFILL_MLA_IMPS — non-CP vs CP).  attn_factory iterates ALL impls
    matching a NAME and tries each, so same-NAME-different-class is
    expected and intentional.

    What IS a bug: registering the SAME class object twice (e.g. an
    accidental second `.append(PyFlashinferPrefillImpl)` in __init__.py).
    The second copy is dead — the dispatcher would always return the first
    one's instance — but it slows iteration and signals lost edits during
    refactors.  This test catches that shape.
    """
    seen: set[type] = set()
    duplicates: list[str] = []
    for cls in registry:
        if cls in seen:
            duplicates.append(f"{cls.__module__}.{cls.__name__}")
        else:
            seen.add(cls)
    assert not duplicates, (
        f"Registry {registry_name} has duplicate class registrations: "
        + ", ".join(duplicates)
        + ". Look for a stray `.append(...)` / `.extend([...])` in "
        + "rtp_llm/models_py/modules/factory/attention/__init__.py."
    )
