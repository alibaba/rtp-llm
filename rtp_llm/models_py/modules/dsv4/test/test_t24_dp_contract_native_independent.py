"""Native-independent execution checks for the T24 production boundaries.

The repository image used for this contract round has no torch installation
and cannot load the RTP native extensions.  This harness therefore loads the
 production source files directly and supplies only tiny shells for unrelated
 imports. The production functions under test are not copied or reimplemented
 here: ``select_strategy``, ``_get_decode_topk_workspace``, decode constructor
 wiring, and public decode signatures all execute from production source.
 Strategy selection uses a synthetic registry because concrete strategy
 imports require unavailable runtime dependencies; the checked-in registry
 shape is also inspected directly.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class _Tensor:
    _next = 1000

    def __init__(self, value=0):
        self.value = value
        self._ptr = _Tensor._next
        _Tensor._next += 1

    def data_ptr(self):
        return self._ptr

    def clone(self):
        return _Tensor(self.value)

    @property
    def device(self):
        return _Device("cpu")

    def to(self, *args, **kwargs):
        return self

    def copy_(self, other):
        self.value = other.value
        return self


class _Device:
    def __init__(self, kind):
        self.type = str(kind).split(":", 1)[0]

    def __hash__(self):
        return hash(self.type)

    def __eq__(self, other):
        return isinstance(other, _Device) and self.type == other.type

    def __repr__(self):
        return self.type


def _torch_shell() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.int32 = object()
    torch.int64 = object()
    torch.Tensor = _Tensor
    torch.device = _Device
    torch.zeros = lambda *args, **kwargs: _Tensor()
    torch.full = lambda *args, **kwargs: _Tensor(kwargs.get("fill_value", 0))
    torch.empty = lambda *args, **kwargs: _Tensor()
    torch.clamp = lambda value, **kwargs: value
    torch.cuda = types.SimpleNamespace(is_current_stream_capturing=lambda: False)
    nn = types.ModuleType("torch.nn")

    class Module:
        def __init__(self):
            pass

    nn.Module = Module
    torch.nn = nn
    sys.modules["torch.nn.functional"] = types.ModuleType("torch.nn.functional")
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    return torch


def _module(name: str, **attrs: object) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class _Metadata:
    start_pos: _Tensor


def _load_modules():
    _torch_shell()
    for name in (
        "rtp_llm",
        "rtp_llm.models_py",
        "rtp_llm.models_py.modules",
        "rtp_llm.models_py.modules.dsv4",
        "rtp_llm.models_py.modules.dsv4.moe",
        "rtp_llm.models_py.modules.dsv4.moe.strategies",
    ):
        if name not in sys.modules:
            _module(name)
    # Production base.py imports these lazily only for EP>1 policy checks.
    _module(
        "rtp_llm.models_py.modules.dsv4.moe.strategies.mega_buf",
        _mega_moe_disabled_or_unavailable_reason=lambda: "test shell: Mega absent",
    )
    _module(
        "rtp_llm.models_py.modules.dsv4.moe.mega_fused_buf",
        mega_moe_fused_requested=lambda: False,
    )
    _module(
        "rtp_llm.models_py.modules.dsv4.moe.mega_se_buf",
        mega_moe_se_requested=lambda: False,
    )
    _module(
        "rtp_llm.models_py.modules.dsv4.moe.mega_buf",
        _mega_moe_disabled_or_unavailable_reason=lambda: "test shell: Mega absent",
    )
    _module(
        "rtp_llm.models_py.modules.dsv4.platform_provider",
        Dsv4ProviderCapability=types.SimpleNamespace(DEEPEP_MOE=object()),
        get_dsv4_platform_provider_capabilities=lambda: frozenset(),
    )
    base = _load(
        "t24_production_base",
        ROOT / "moe" / "strategies" / "base.py",
    )

    class _Strategy(base.RoutedExpertsStrategy):
        name = "local_loop"

        @classmethod
        def can_handle(cls, cfg):
            return cfg.ep_size == 1

    class _Mega(base.RoutedExpertsStrategy):
        name = "mega"

        @classmethod
        def can_handle(cls, cfg):
            return False

    class _DeepEP(base.RoutedExpertsStrategy):
        name = "deepep"

        @classmethod
        def can_handle(cls, cfg):
            return cfg.ep_size > 1

    base._STRATEGY_PRIORITY[:] = [_Mega, _DeepEP, _Strategy]

    indexer_pkg = "t24_production_indexer"
    # indexer imports kernels and model helpers; these are deliberately
    # unrelated to the workspace allocation policy being exercised.
    for name in (
        "rtp_llm.models_py.modules.dsv4._profiler",
        "rtp_llm.models_py.modules.dsv4.chunk_env",
        "rtp_llm.models_py.modules.dsv4.cp",
        "rtp_llm.models_py.modules.dsv4.fp8._indexer_q_quant_triton",
        "rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton",
        "rtp_llm.models_py.modules.dsv4.fp8._indexer_score",
        "rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils",
        "rtp_llm.models_py.modules.dsv4.fp8.compressor",
        "rtp_llm.models_py.modules.dsv4.prefill_workspace",
        "rtp_llm.models_py.modules.dsv4.qlinear",
    ):
        _module(name, **{
            "record_function_range": lambda *args, **kwargs: None,
            "dsv4_chunk_tokens_from_env": lambda *args, **kwargs: 0,
            "_CP_ROLE_INDEXER": "indexer",
            "CPContext": object,
            "build_cp_full_prefill_positions": lambda *args, **kwargs: None,
            "indexer_q_fp8_quant_fold": lambda *args, **kwargs: None,
            "indexer_q_rope_fp8_quant_fold": lambda *args, **kwargs: None,
            "INDEXER_ENTRY_BYTES": 1,
            "INDEXER_HEAD_DIM": 1,
            "fp8_mqa_indexer_score": lambda *args, **kwargs: None,
            "fp8_paged_indexer_score": lambda *args, **kwargs: None,
            "has_fp8_mqa_logits": lambda: False,
            "has_fp8_paged_mqa_logits": lambda: False,
            "PoolBackedModule": object,
            "CompressorFP8": object,
            "CompressorMeta": object,
            "_CompressorPending": object,
            "PrefillWorkspace": object,
            "QuantizedLinear": object,
        })
    _module("rtp_llm.ops.compute_ops", rtp_llm_ops=types.SimpleNamespace())
    indexer = _load(indexer_pkg, ROOT / "fp8" / "indexer.py")

    # The decode impl's allocator/update are unavailable tensor dependencies;
    # their small shells exercise constructor/update wiring only.
    metadata = types.ModuleType("rtp_llm.models_py.modules.dsv4.decode.decode_attn_metadata")
    metadata.DSv4DecodeAttnMetadata = _Metadata
    metadata.allocate_decode_metadata = lambda **kwargs: _Metadata(_Tensor())
    metadata.update_decode_metadata_in_place = lambda meta, start_pos, **kwargs: meta.start_pos.copy_(start_pos)
    sys.modules[metadata.__name__] = metadata
    _module(
        "rtp_llm.models_py.modules.dsv4.kv_cache_utils",
        primary_attention_inputs=lambda value: value,
        as_attention_inputs_by_tag=lambda value: {},
    )
    decode_impl = _load(
        "t24_production_decode_impl",
        ROOT / "decode" / "decode_fmha_impl.py",
    )
    return base, indexer, decode_impl


class T24NativeIndependentTest(unittest.TestCase):
    _STRATEGY_ENV = (
        "DSV4_MOE_STRATEGY",
        "DSV4_USE_MEGA_MOE",
        "DSV4_USE_MEGA_MOE_SE",
        "DSV4_USE_MEGA_MOE_FUSED",
        "DSV4_USE_GROUPED_FP4",
    )

    @classmethod
    def setUpClass(cls):
        cls._saved_strategy_env = {
            key: os.environ.get(key) for key in cls._STRATEGY_ENV
        }
        for key in cls._STRATEGY_ENV:
            os.environ.pop(key, None)
        try:
            cls.base, cls.indexer, cls.decode_impl = _load_modules()
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        for key, value in getattr(cls, "_saved_strategy_env", {}).items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _cfg(self, ep_size):
        return self.base.MoeCfg(
            layer_id=0, dim=1, moe_inter_dim=1, n_routed_experts=8,
            n_activated_experts=2, swiglu_limit=1.0, ep_size=ep_size,
            ep_rank=0, n_local_experts=8 // ep_size,
            local_expert_start=0, local_expert_end=8 // ep_size,
            max_tokens_per_rank=4,
        )

    def test_production_selector_control_flow_with_synthetic_registry(self):
        source = (ROOT / "moe" / "strategies" / "__init__.py").read_text()
        for name in ("MegaMoEStrategy", "DeepEPStrategy", "GroupedFP4Strategy", "LocalLoopStrategy"):
            self.assertIn(name, source)
        self.assertIs(self.base.select_strategy(self._cfg(1)), self.base._STRATEGY_PRIORITY[-1])
        for ep in (2, 4, 8):
            with self.assertRaisesRegex(RuntimeError, "requires MegaMoEStrategy"):
                self.base.select_strategy(self._cfg(ep))
        with self.assertRaisesRegex(RuntimeError, "bypass Mega"):
            self.base.select_strategy(self._cfg(4), forced="deepep")

    def test_real_workspace_helper_fails_before_allocation_during_capture(self):
        self.indexer._decode_topk_workspace_cache.clear()
        self.indexer._decode_topk_capture_active = lambda device: True
        with self.assertRaisesRegex(RuntimeError, "warmed before graph capture"):
            self.indexer._get_decode_topk_workspace(_Device("cuda"))
        self.assertFalse(self.indexer._decode_topk_workspace_cache)

    def test_decode_impl_constructor_update_wiring_with_metadata_shell(self):
        cfg = self.decode_impl.DSv4DecodeFmhaImplConfig(
            max_batch_size=1, q_len=1, window_size=8, head_dim=32,
            max_seq_len=64, compress_ratios=[4], index_topk=4,
        )
        a_input = types.SimpleNamespace(sequence_lengths=_Tensor(1))
        b_input = types.SimpleNamespace(sequence_lengths=_Tensor(2))
        a = self.decode_impl.DSv4DecodeFmhaImpl(cfg, _Device("cpu"), a_input)
        b = self.decode_impl.DSv4DecodeFmhaImpl(cfg, _Device("cpu"), b_input)
        self.assertNotEqual(a.metadata.start_pos.data_ptr(), b.metadata.start_pos.data_ptr())
        before = b.metadata.start_pos.value
        a.prepare_cuda_graph(types.SimpleNamespace(sequence_lengths=_Tensor(7)))
        self.assertEqual(a.metadata.start_pos.value, 7)
        self.assertEqual(b.metadata.start_pos.value, before)

    def test_decode_forward_abi_is_production_callable(self):
        _module("rtp_llm.models_py.modules.dsv4._forward_tensor_debug")
        _module("rtp_llm.models_py.modules.dsv4._record_tensor")
        _module("rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils", require_pool_tokens_per_block=lambda *a, **k: 1)
        _module("rtp_llm.models_py.modules.dsv4.kv_cache_utils", **{k: k for k in ("CSA_KV", "CSA_STATE", "HCA_KV", "HCA_STATE", "INDEXER_KV", "INDEXER_STATE", "SWA_KV", "DSV4_KERNEL_ROW_TAGS")}, primary_attention_inputs=lambda x: x, group_tags=lambda x: (), build_block_tables_for_tags=lambda *a, **k: {})
        forward = _load(
            "t24_production_decode_forward",
            ROOT / "decode" / "forward.py",
        )
        for name in ("forward_layers", "forward_decode"):
            self.assertIn("numerical_status", inspect.signature(getattr(forward, name)).parameters)


if __name__ == "__main__":
    unittest.main()
