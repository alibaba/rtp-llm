"""Registry-level NAME invariants for FMHA / MLA implementations.

These tests guard the explicit-dispatch contract of attn_factory.get_fmha_impl:
the user types `--attn_backend=<name>` and the factory looks up impls via NAME.
A missing NAME on a registered impl turns into a generic "can not find mha
type" error at request time; a duplicate NAME within one registry silently
shadows whichever impl appears later in the list.

Both tests are import-only — no GPU, no kernel — so they run on every
arch's CI worker (CUDA / ROCm / PPU / CPU).
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
def test_no_duplicate_names_within_registry(registry_name, registry):
    """Within a single registry, NAME → impl is 1:1.

    Multiple impls CAN share a NAME across PREFILL_MHA_IMPS / DECODE_MHA_IMPS
    (e.g. AiterPrefillImplAsm vs AiterDecodeImplAsm both expose "aiter_asm")
    — that's the natural mental model: one user-facing name covers both
    stages. But within ONE registry, two impls with the same NAME means the
    second one silently shadows the first in attn_factory's name_to_impls
    lookup, which is a footgun.
    """
    seen: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []
    for cls in registry:
        name = getattr(cls, "NAME", "")
        if not name:
            continue
        prior = seen.get(name)
        cls_id = f"{cls.__module__}.{cls.__name__}"
        if prior is None:
            seen[name] = cls_id
        else:
            duplicates.append((name, prior, cls_id))
    assert not duplicates, (
        f"Registry {registry_name} has duplicate NAMEs: "
        + ", ".join(
            f"NAME={n!r} on both {a} and {b}" for n, a, b in duplicates
        )
    )
