"""backend_selector unit tests: dynamic decode selection honors fmha_config disable flags.

Pure-CPU: DECODE_MHA_IMPS is monkeypatched with fake impls; the *real*
_is_fmha_impl_disabled (class-name matching) is exercised. No GPU / torch compute.
These guard the fix that _eligible() and instantiate_decode_impl() must drop backends
the user disabled (disable_flash_infer / enable_xqa=False), matching the fixed-priority
get_fmha_impl path -- otherwise dynamic selection could benchmark & pick a disabled backend.
"""

import types
import unittest
from unittest import mock

from rtp_llm.models_py.modules.factory.attention.dispatch import backend_selector

_ATTN_FACTORY = "rtp_llm.models_py.modules.factory.attention.attn_factory"


def _make_fake_impl(name):
    """A minimal decode impl stub that always 'supports' everything."""

    class _Fake:
        @classmethod
        def support(cls, attn_configs, attn_inputs):
            return True

        @classmethod
        def support_parallelism_config(cls, parallelism_config):
            return True

        def __init__(self, attn_configs, attn_inputs, parallelism_config=None):
            pass

        def support_cuda_graph(self):
            return True

    _Fake.__name__ = name
    return _Fake


def _fmha_config(**overrides):
    """SimpleNamespace mirroring the fields FMHAConfig exposes."""
    cfg = dict(
        enable_fmha=True,
        enable_trt_fmha=True,
        enable_paged_trt_fmha=True,
        enable_open_source_fmha=True,
        enable_paged_open_source_fmha=True,
        enable_trtv1_fmha=True,
        disable_flash_infer=False,
        enable_xqa=True,
        use_aiter_pa=True,
        use_asm_pa=True,
        use_triton_pa=False,
    )
    cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


def _patch_impls(names):
    fakes = [_make_fake_impl(n) for n in names]
    return mock.patch(f"{_ATTN_FACTORY}.DECODE_MHA_IMPS", fakes)


_NAMES = ["PyFlashinferDecodeImpl", "XQADecodeImpl"]


# ─── _eligible respects fmha_config ───────────────────────────────────────────
def test_eligible_excludes_flashinfer_when_disabled():
    with _patch_impls(_NAMES):
        eligible = backend_selector._eligible(
            None, None, None, _fmha_config(disable_flash_infer=True)
        )
    assert "PyFlashinferDecodeImpl" not in eligible
    assert "XQADecodeImpl" in eligible


def test_eligible_excludes_xqa_when_disabled():
    with _patch_impls(_NAMES):
        eligible = backend_selector._eligible(
            None, None, None, _fmha_config(enable_xqa=False)
        )
    assert "XQADecodeImpl" not in eligible
    assert "PyFlashinferDecodeImpl" in eligible


def test_eligible_keeps_all_when_nothing_disabled():
    with _patch_impls(_NAMES):
        eligible = backend_selector._eligible(None, None, None, _fmha_config())
    assert set(eligible) == set(_NAMES)


def test_eligible_none_fmha_config_keeps_all():
    with _patch_impls(_NAMES):
        eligible = backend_selector._eligible(None, None, None, None)
    assert set(eligible) == set(_NAMES)


# ─── instantiate_decode_impl respects fmha_config ─────────────────────────────
def test_instantiate_decode_impl_returns_none_when_disabled():
    # Disabled short-circuit happens before touching model.config, so a bare
    # namespace with only fmha_config is enough.
    model = types.SimpleNamespace(fmha_config=_fmha_config(disable_flash_infer=True))
    with _patch_impls(_NAMES):
        inst = backend_selector.instantiate_decode_impl(
            model, types.SimpleNamespace(), "PyFlashinferDecodeImpl", True
        )
    assert inst is None


def test_instantiate_decode_impl_instantiates_when_enabled():
    model = types.SimpleNamespace(
        fmha_config=_fmha_config(),
        config=types.SimpleNamespace(
            getAttentionConfigs=lambda tp: None, headwise_config=None
        ),
        parallelism_config=types.SimpleNamespace(get_attn_tp_size=lambda: 1),
    )
    with _patch_impls(_NAMES):
        inst = backend_selector.instantiate_decode_impl(
            model, types.SimpleNamespace(), "PyFlashinferDecodeImpl", True
        )
    assert inst is not None


# Bind the module-level test_* functions onto a TestCase so bazel's unittest
# runner (no pytest available) discovers and runs them.
class BackendSelectorTest(unittest.TestCase):
    pass


for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        setattr(BackendSelectorTest, _name, staticmethod(_fn))
del _name, _fn  # don't leak a class/func ref that unittest would re-collect


if __name__ == "__main__":
    unittest.main()
