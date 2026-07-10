"""Backend-selection contract for attn_factory._select_attn_impls (shared by MHA/MLA).

Pure-Python: no GPU / kernels. Uses fake impl registries so the behavior of
attn_backend / prefill_/decode_ overrides, attn_backend=none, the global
disable_attn_backends blocklist, and name validation can be asserted directly.

Crucially this covers the MLA path: get_mla_impl previously ignored the backend
config entirely and returned the first supported impl. _select_attn_impls is the
shared resolver both MHA and MLA now call, so testing it guards the MLA wiring.
"""

from types import SimpleNamespace

import pytest

import rtp_llm.models_py.modules.factory.attention.attn_factory as af
from rtp_llm.models_py.modules.factory.attention.attn_factory import _select_attn_impls


class _FlashinferMla:
    NAME = "flashinfer_mla"


class _SparseMla:
    NAME = "sparse_mla"


MLA_IMPLS = [_FlashinferMla, _SparseMla]


@pytest.fixture(autouse=True)
def _fake_registries(monkeypatch):
    """Point every registry at the fakes so the global blocklist validation
    (_blocklist_known_names reads these module globals) is consistent with the
    impls we pass in — hermetic regardless of what the platform registered."""
    monkeypatch.setattr(af, "PREFILL_MHA_IMPS", [])
    monkeypatch.setattr(af, "DECODE_MHA_IMPS", [])
    monkeypatch.setattr(af, "PREFILL_MLA_IMPS", MLA_IMPLS)
    monkeypatch.setattr(af, "DECODE_MLA_IMPS", MLA_IMPLS)


def _cfg(attn_backend="auto", prefill_attn_backend="", decode_attn_backend="", disable_attn_backends=""):
    # attn_backend defaults to "auto" to match the real FMHAConfig default (see
    # server_args/fmha_group_args.py), so a blocklist-only config resolves to auto mode.
    return SimpleNamespace(
        attn_backend=attn_backend,
        prefill_attn_backend=prefill_attn_backend,
        decode_attn_backend=decode_attn_backend,
        disable_attn_backends=disable_attn_backends,
    )


def test_none_disables_attention():
    with pytest.raises(Exception, match="none"):
        _select_attn_impls(MLA_IMPLS, _cfg(attn_backend="none"), True)


def test_auto_returns_all_in_registration_order():
    assert _select_attn_impls(MLA_IMPLS, None, True) == MLA_IMPLS
    assert _select_attn_impls(MLA_IMPLS, _cfg(attn_backend="auto"), True) == MLA_IMPLS


def test_explicit_mla_backend_selection():
    # The core MLA fix: an explicit backend selects exactly that impl instead of
    # blindly returning the first supported one.
    assert _select_attn_impls(MLA_IMPLS, _cfg(attn_backend="sparse_mla"), True) == [_SparseMla]
    assert _select_attn_impls(MLA_IMPLS, _cfg(attn_backend="flashinfer_mla"), True) == [_FlashinferMla]


def test_explicit_ordered_list_preserves_priority():
    got = _select_attn_impls(MLA_IMPLS, _cfg(attn_backend="sparse_mla,flashinfer_mla"), True)
    assert got == [_SparseMla, _FlashinferMla]


def test_blocklist_excludes_impl():
    got = _select_attn_impls(MLA_IMPLS, _cfg(disable_attn_backends="flashinfer_mla"), True)
    assert got == [_SparseMla]


def test_prefill_decode_override():
    cfg = _cfg(attn_backend="flashinfer_mla", prefill_attn_backend="sparse_mla")
    # prefill uses the prefill override; decode falls back to the global attn_backend.
    assert _select_attn_impls(MLA_IMPLS, cfg, True) == [_SparseMla]
    assert _select_attn_impls(MLA_IMPLS, cfg, False) == [_FlashinferMla]


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown attention backend"):
        _select_attn_impls(MLA_IMPLS, _cfg(attn_backend="does_not_exist"), True)


def test_unknown_blocklist_name_raises():
    with pytest.raises(ValueError, match="disable_attn_backends"):
        _select_attn_impls(MLA_IMPLS, _cfg(disable_attn_backends="bogus_backend"), True)
