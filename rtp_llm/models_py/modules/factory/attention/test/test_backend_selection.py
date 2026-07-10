"""Backend-selection coverage for the MLA attention path.

Two layers, both pure-Python (no GPU / kernels):

* ``TestSelectAttnImpls`` unit-tests the shared resolver ``_select_attn_impls``
  (attn_backend / overrides / none / blocklist / name validation).
* ``TestGetMlaImplEntry`` drives the REAL production entry ``get_mla_impl`` with
  minimal MlaImplBase fakes, so the tests fail if the resolver wiring is removed
  from the factory (i.e. if MLA goes back to ignoring the backend config). The
  resolver unit test alone would keep passing in that regression, so this layer
  is what actually guards the fix.
"""

from types import SimpleNamespace

import pytest

import rtp_llm.models_py.modules.factory.attention.attn_factory as af
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    _select_attn_impls,
    get_mla_impl,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import MlaImplBase


def _cfg(attn_backend="auto", prefill_attn_backend="", decode_attn_backend="", disable_attn_backends=""):
    # attn_backend defaults to "auto" to match the real FMHAConfig default (see
    # server_args/fmha_group_args.py), so a blocklist-only config resolves to auto mode.
    return SimpleNamespace(
        attn_backend=attn_backend,
        prefill_attn_backend=prefill_attn_backend,
        decode_attn_backend=decode_attn_backend,
        disable_attn_backends=disable_attn_backends,
    )


class TestSelectAttnImpls:
    """Unit tests for the shared resolver with NAME-only fakes."""

    class _FlashinferMla:
        NAME = "flashinfer_mla"

    class _SparseMla:
        NAME = "sparse_mla"

    @pytest.fixture(autouse=True)
    def _fake_registries(self, monkeypatch):
        # Point every registry at the fakes so the global blocklist validation
        # (_blocklist_known_names reads these module globals) stays consistent
        # with the impls we pass in — hermetic regardless of platform registration.
        self.impls = [self._FlashinferMla, self._SparseMla]
        monkeypatch.setattr(af, "PREFILL_MHA_IMPS", [])
        monkeypatch.setattr(af, "DECODE_MHA_IMPS", [])
        monkeypatch.setattr(af, "PREFILL_MLA_IMPS", self.impls)
        monkeypatch.setattr(af, "DECODE_MLA_IMPS", self.impls)

    def test_none_disables_attention(self):
        with pytest.raises(Exception, match="none"):
            _select_attn_impls(self.impls, _cfg(attn_backend="none"), True)

    def test_auto_returns_all_in_registration_order(self):
        assert _select_attn_impls(self.impls, None, True) == self.impls
        assert _select_attn_impls(self.impls, _cfg(attn_backend="auto"), True) == self.impls

    def test_explicit_backend_selection(self):
        assert _select_attn_impls(self.impls, _cfg(attn_backend="sparse_mla"), True) == [self._SparseMla]
        assert _select_attn_impls(self.impls, _cfg(attn_backend="flashinfer_mla"), True) == [self._FlashinferMla]

    def test_explicit_ordered_list_preserves_priority(self):
        got = _select_attn_impls(self.impls, _cfg(attn_backend="sparse_mla,flashinfer_mla"), True)
        assert got == [self._SparseMla, self._FlashinferMla]

    def test_blocklist_excludes_impl(self):
        got = _select_attn_impls(self.impls, _cfg(disable_attn_backends="flashinfer_mla"), True)
        assert got == [self._SparseMla]

    def test_prefill_decode_override(self):
        cfg = _cfg(attn_backend="flashinfer_mla", prefill_attn_backend="sparse_mla")
        assert _select_attn_impls(self.impls, cfg, True) == [self._SparseMla]
        assert _select_attn_impls(self.impls, cfg, False) == [self._FlashinferMla]

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown attention backend"):
            _select_attn_impls(self.impls, _cfg(attn_backend="does_not_exist"), True)

    def test_unknown_blocklist_name_raises(self):
        with pytest.raises(ValueError, match="disable_attn_backends"):
            _select_attn_impls(self.impls, _cfg(disable_attn_backends="bogus_backend"), True)


# --- Minimal MlaImplBase fakes that satisfy the get_mla_impl contract ----------
# They implement support()/is_sparse()/support_parallelism_config() (inherited)
# and the base constructor, so get_mla_impl can filter, instantiate and return
# them exactly as it does for real impls.
class _FakeFlashinferMla(MlaImplBase):
    NAME = "flashinfer_mla"

    @staticmethod
    def support(attn_configs, attn_inputs):
        return True


class _FakeSparseMla(MlaImplBase):
    NAME = "sparse_mla"

    @staticmethod
    def support(attn_configs, attn_inputs):
        return True


class _FakeMax:
    def item(self):
        return 0


class _FakeCuKvSeqlens:
    def max(self):
        return _FakeMax()


class TestGetMlaImplEntry:
    """End-to-end through the real get_mla_impl production entry."""

    MLA_IMPLS = [_FakeFlashinferMla, _FakeSparseMla]

    @pytest.fixture(autouse=True)
    def _fake_registries(self, monkeypatch):
        monkeypatch.setattr(af, "PREFILL_MHA_IMPS", [])
        monkeypatch.setattr(af, "DECODE_MHA_IMPS", [])
        monkeypatch.setattr(af, "PREFILL_MLA_IMPS", self.MLA_IMPLS)
        monkeypatch.setattr(af, "DECODE_MLA_IMPS", self.MLA_IMPLS)
        # is_sparse=False on both fakes + attn_configs.is_sparse=False keeps the
        # sparse/fast-path skip branches inert so routing is driven purely by NAME.
        self.attn_configs = SimpleNamespace(indexer_topk=0, is_sparse=False)
        self.weight = SimpleNamespace(get_global_weight_or_none=lambda key: None, weights=[])

    def _decode_inputs(self):
        return SimpleNamespace(is_prefill=False)

    def _prefill_inputs(self):
        return SimpleNamespace(is_prefill=True, cu_kv_seqlens_device=_FakeCuKvSeqlens())

    def _run(self, attn_inputs, fmha_config):
        return get_mla_impl(self.attn_configs, self.weight, attn_inputs, fmha_config=fmha_config)

    def test_explicit_backend_routes_through_factory(self):
        # The whole point: an explicit MLA backend selects that impl via get_mla_impl,
        # not just the first registered one.
        inst = self._run(self._decode_inputs(), _cfg(attn_backend="sparse_mla"))
        assert type(inst) is _FakeSparseMla
        inst = self._run(self._decode_inputs(), _cfg(attn_backend="flashinfer_mla"))
        assert type(inst) is _FakeFlashinferMla

    def test_none_raises_through_factory(self):
        with pytest.raises(Exception, match="none"):
            self._run(self._decode_inputs(), _cfg(attn_backend="none"))

    def test_blocklist_applied_through_factory(self):
        # flashinfer_mla is blocked -> auto falls through to sparse_mla.
        inst = self._run(self._decode_inputs(), _cfg(disable_attn_backends="flashinfer_mla"))
        assert type(inst) is _FakeSparseMla

    def test_stage_override_through_factory(self):
        cfg = _cfg(attn_backend="flashinfer_mla", prefill_attn_backend="sparse_mla")
        # prefill uses the prefill override; decode falls back to the global attn_backend.
        assert type(self._run(self._prefill_inputs(), cfg)) is _FakeSparseMla
        assert type(self._run(self._decode_inputs(), cfg)) is _FakeFlashinferMla

    def test_default_auto_returns_first_supported(self):
        # Regression baseline: default auto keeps returning the first supported impl.
        inst = self._run(self._decode_inputs(), _cfg())
        assert type(inst) is _FakeFlashinferMla
