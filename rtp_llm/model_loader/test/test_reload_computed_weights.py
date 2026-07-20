"""Coverage tests for the level-2 wake reload's computed-weight source.

The checkpoint reload generators (``prepare_weights`` /
``prepare_weights_fastsafetensor``) only emit weights backed by a checkpoint
tensor. Live weights that cold load computes afterwards -- the rope cos/sin
cache and the EPLB placement buffers -- are discarded by the level-2 pause and
must be recomputed on wake. ``ModelLoader.prepare_computed_weights`` supplies
them so the reload's full-coverage assertion does not spuriously fail on models
that have them (DeepSeek rope cache, any EPLB-enabled MoE). These tests exercise
the computation directly without a real checkpoint or GPU.
"""

import os
import types
import unittest

import torch
import torch.nn.functional as F

from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.ops import TaskType
from rtp_llm.utils.model_weight import W


def _make_loader(
    load_config, weights_info, model_weights_info=None, task_type=None
) -> ModelLoader:
    """A bare ModelLoader carrying only the attributes the computed-weight
    generators read, so the pure computation can be tested in isolation.

    ``model_weights_info`` supplies the global :class:`WeightModule` list the
    lm_head / positional derivation searches; it defaults to empty so tests that
    only exercise rope / EPLB do not emit those. ``task_type`` defaults to None
    (not LANGUAGE_MODEL) so lm_head is not emitted unless a test opts in."""
    loader = object.__new__(ModelLoader)
    loader._load_config = load_config
    loader._weights_info = weights_info
    loader._model_weights_info = model_weights_info or types.SimpleNamespace(weights=[])
    loader._task_type = task_type
    return loader


class _FakeModule:
    """Minimal WeightModule stub: ``load`` returns ``{name: tensor}``."""

    def __init__(self, name: str, tensor: torch.Tensor) -> None:
        self.name = name
        self._tensor = tensor

    def load(self, tensor_source, layer_id, device, load_config):
        return {self.name: self._tensor}


def _lm_config(**overrides):
    """A load_config / weights_info pair for a dense LM with no rope/EPLB, so
    prepare_computed_weights emits only lm_head / positional."""
    load_config = types.SimpleNamespace(
        expert_num=0, phy_exp_num=0, num_layers=1, phy2log=[], database=object()
    )
    model_config = types.SimpleNamespace(
        normalize_lm_head_weight=overrides.get("normalize", False),
        logit_scale=overrides.get("logit_scale", 1.0),
        max_seq_len=overrides.get("max_seq_len", 4),
    )
    weights_info = types.SimpleNamespace(
        enable_eplb_=False,
        create_dynamic_weights=lambda: [],
        model_config=model_config,
    )
    return load_config, weights_info


class IterEplbWeightsTest(unittest.TestCase):
    def test_yields_placement_buffers_from_phy2log(self) -> None:
        # 2 logical experts, 1 redundant (phy_exp_num=3). Layer 0 maps
        # phy0->log0, phy1->log1, phy2->log0.
        load_config = types.SimpleNamespace(
            expert_num=2, phy_exp_num=3, num_layers=1, phy2log=[[0, 1, 0]]
        )
        weights_info = types.SimpleNamespace(enable_eplb_=True)
        loader = _make_loader(load_config, weights_info)

        yielded = list(loader._iter_eplb_weights("cpu"))
        by_name = {(layer_id, name): t for layer_id, name, t in yielded}

        self.assertEqual(len(yielded), 2)
        cnt = by_name[(0, W.logic_expert_cnt)]
        log2phy = by_name[(0, W.log2phy)]
        # log0 is served by phy0 and phy2 (cnt 2), log1 by phy1 (cnt 1).
        self.assertEqual(cnt.tolist(), [2, 1])
        self.assertEqual(cnt.dtype, torch.int32)
        # log2phy row per logical expert, width redundant+1=2, unused slots -1.
        self.assertEqual(log2phy.tolist(), [[0, 2], [1, -1]])
        self.assertEqual(log2phy.dtype, torch.int32)

    def test_yields_nothing_when_eplb_disabled_and_no_redundancy(self) -> None:
        load_config = types.SimpleNamespace(
            expert_num=4, phy_exp_num=4, num_layers=2, phy2log=[[0, 1, 2, 3]] * 2
        )
        weights_info = types.SimpleNamespace(enable_eplb_=False)
        loader = _make_loader(load_config, weights_info)
        self.assertEqual(list(loader._iter_eplb_weights("cpu")), [])

    def test_yields_nothing_for_dense_model(self) -> None:
        load_config = types.SimpleNamespace(
            expert_num=0, phy_exp_num=0, num_layers=1, phy2log=[]
        )
        weights_info = types.SimpleNamespace(enable_eplb_=True)
        loader = _make_loader(load_config, weights_info)
        self.assertEqual(list(loader._iter_eplb_weights("cpu")), [])


class PrepareComputedWeightsTest(unittest.TestCase):
    def test_chains_rope_dynamic_weight_before_eplb(self) -> None:
        rope_tensor = torch.zeros(4, 8, dtype=torch.float32)

        class _FakeRope:
            name = W.rope_cos_sin_cache

            def load(self, tensor_source, layer_id, device, load_config):
                return {W.rope_cos_sin_cache: rope_tensor}

        load_config = types.SimpleNamespace(
            expert_num=2,
            phy_exp_num=3,
            num_layers=1,
            phy2log=[[0, 1, 0]],
            database=object(),
        )
        weights_info = types.SimpleNamespace(
            enable_eplb_=True, create_dynamic_weights=lambda: [_FakeRope()]
        )
        loader = _make_loader(load_config, weights_info)

        yielded = list(loader.prepare_computed_weights("cpu"))
        names = [(layer_id, name) for layer_id, name, _ in yielded]

        # Rope (global, layer_id=None) comes first, then the two EPLB buffers.
        self.assertEqual(names[0], (None, W.rope_cos_sin_cache))
        self.assertIs(yielded[0][2], rope_tensor)
        self.assertIn((0, W.logic_expert_cnt), names)
        self.assertIn((0, W.log2phy), names)
        self.assertEqual(len(yielded), 3)

    def test_no_dynamic_weights_yields_only_eplb(self) -> None:
        load_config = types.SimpleNamespace(
            expert_num=2,
            phy_exp_num=3,
            num_layers=1,
            phy2log=[[0, 1, 0]],
            database=object(),
        )
        weights_info = types.SimpleNamespace(
            enable_eplb_=True, create_dynamic_weights=lambda: []
        )
        loader = _make_loader(load_config, weights_info)
        yielded = list(loader.prepare_computed_weights("cpu"))
        self.assertEqual(len(yielded), 2)
        self.assertTrue(all(layer_id == 0 for layer_id, _, _ in yielded))

    def test_dense_model_yields_nothing(self) -> None:
        load_config = types.SimpleNamespace(
            expert_num=0, phy_exp_num=0, num_layers=1, phy2log=[], database=object()
        )
        weights_info = types.SimpleNamespace(
            enable_eplb_=False, create_dynamic_weights=lambda: []
        )
        loader = _make_loader(load_config, weights_info)
        self.assertEqual(list(loader.prepare_computed_weights("cpu")), [])


class DeriveLmHeadTest(unittest.TestCase):
    def test_normalize_applied(self) -> None:
        load_config, weights_info = _lm_config(normalize=True)
        loader = _make_loader(load_config, weights_info)
        raw = torch.randn(3, 8)
        out = loader._derive_lm_head_tensor(raw)
        self.assertTrue(torch.equal(out, F.normalize(raw)))

    def test_logit_scale_applied(self) -> None:
        load_config, weights_info = _lm_config(logit_scale=2.0)
        loader = _make_loader(load_config, weights_info)
        raw = torch.randn(3, 8)
        out = loader._derive_lm_head_tensor(raw)
        self.assertTrue(torch.equal(out, 2.0 * raw))

    def test_identity_when_no_transform(self) -> None:
        load_config, weights_info = _lm_config()
        loader = _make_loader(load_config, weights_info)
        raw = torch.randn(3, 8)
        # No normalize, logit_scale == 1.0 -> raw returned unchanged (same object).
        self.assertIs(loader._derive_lm_head_tensor(raw), raw)

    def test_normalize_then_scale(self) -> None:
        load_config, weights_info = _lm_config(normalize=True, logit_scale=3.0)
        loader = _make_loader(load_config, weights_info)
        raw = torch.randn(3, 8)
        out = loader._derive_lm_head_tensor(raw)
        self.assertTrue(torch.equal(out, 3.0 * F.normalize(raw)))


class DerivePositionalTest(unittest.TestCase):
    def test_sliced_to_max_seq_len(self) -> None:
        load_config, weights_info = _lm_config(max_seq_len=4)
        loader = _make_loader(load_config, weights_info)
        raw = torch.randn(8, 5)
        out = loader._derive_positional_tensor(raw, "cpu")
        self.assertEqual(out.shape, (4, 5))
        self.assertTrue(torch.equal(out, raw[:4]))

    def test_raises_when_too_short(self) -> None:
        load_config, weights_info = _lm_config(max_seq_len=8)
        loader = _make_loader(load_config, weights_info)
        raw = torch.randn(4, 5)
        with self.assertRaises(Exception):
            loader._derive_positional_tensor(raw, "cpu")


class IterLmHeadWeightTest(unittest.TestCase):
    def test_from_dedicated_lm_head_module(self) -> None:
        load_config, weights_info = _lm_config()
        tensor = torch.randn(6, 8)
        mwi = types.SimpleNamespace(weights=[_FakeModule(W.lm_head, tensor)])
        loader = _make_loader(load_config, weights_info, mwi, TaskType.LANGUAGE_MODEL)
        yielded = list(loader._iter_lm_head_weight("cpu"))
        self.assertEqual(len(yielded), 1)
        self.assertEqual((yielded[0][0], yielded[0][1]), (None, W.lm_head))
        self.assertIs(yielded[0][2], tensor)

    def test_falls_back_to_embedding_module(self) -> None:
        load_config, weights_info = _lm_config()
        emb = torch.randn(6, 8)
        mwi = types.SimpleNamespace(weights=[_FakeModule(W.embedding, emb)])
        loader = _make_loader(load_config, weights_info, mwi, TaskType.LANGUAGE_MODEL)
        yielded = list(loader._iter_lm_head_weight("cpu"))
        # No lm_head module -> the embedding tensor is used as lm_head.
        self.assertEqual(len(yielded), 1)
        self.assertEqual((yielded[0][0], yielded[0][1]), (None, W.lm_head))
        self.assertIs(yielded[0][2], emb)

    def test_normalize_applied_to_loaded_lm_head(self) -> None:
        load_config, weights_info = _lm_config(normalize=True)
        tensor = torch.randn(6, 8)
        mwi = types.SimpleNamespace(weights=[_FakeModule(W.lm_head, tensor)])
        loader = _make_loader(load_config, weights_info, mwi, TaskType.LANGUAGE_MODEL)
        yielded = list(loader._iter_lm_head_weight("cpu"))
        self.assertTrue(torch.equal(yielded[0][2], F.normalize(tensor)))

    def test_nothing_for_non_language_model(self) -> None:
        load_config, weights_info = _lm_config()
        mwi = types.SimpleNamespace(weights=[_FakeModule(W.lm_head, torch.randn(6, 8))])
        loader = _make_loader(load_config, weights_info, mwi, task_type=None)
        self.assertEqual(list(loader._iter_lm_head_weight("cpu")), [])

    def test_nothing_when_no_module(self) -> None:
        load_config, weights_info = _lm_config()
        loader = _make_loader(
            load_config, weights_info, task_type=TaskType.LANGUAGE_MODEL
        )
        self.assertEqual(list(loader._iter_lm_head_weight("cpu")), [])


class IterPositionalWeightTest(unittest.TestCase):
    def test_sliced_positional_yielded(self) -> None:
        load_config, weights_info = _lm_config(max_seq_len=4)
        full = torch.randn(8, 5)
        mwi = types.SimpleNamespace(weights=[_FakeModule(W.positional_embedding, full)])
        loader = _make_loader(load_config, weights_info, mwi)
        yielded = list(loader._iter_positional_weight("cpu"))
        self.assertEqual(len(yielded), 1)
        self.assertEqual((yielded[0][0], yielded[0][1]), (None, W.positional_embedding))
        self.assertEqual(yielded[0][2].shape, (4, 5))
        self.assertTrue(torch.equal(yielded[0][2], full[:4]))

    def test_nothing_when_no_module(self) -> None:
        load_config, weights_info = _lm_config()
        loader = _make_loader(load_config, weights_info)
        self.assertEqual(list(loader._iter_positional_weight("cpu")), [])


class PrepareComputedWeightsDerivedTest(unittest.TestCase):
    def test_emits_lm_head_and_positional_for_lm(self) -> None:
        load_config, weights_info = _lm_config(max_seq_len=4)
        lm = torch.randn(6, 8)
        pos = torch.randn(8, 5)
        mwi = types.SimpleNamespace(
            weights=[
                _FakeModule(W.lm_head, lm),
                _FakeModule(W.positional_embedding, pos),
            ]
        )
        loader = _make_loader(load_config, weights_info, mwi, TaskType.LANGUAGE_MODEL)
        yielded = list(loader.prepare_computed_weights("cpu"))
        names = [(layer_id, name) for layer_id, name, _ in yielded]
        # Dense, no rope/EPLB -> exactly lm_head then positional, in that order.
        self.assertEqual(names, [(None, W.lm_head), (None, W.positional_embedding)])

    def test_no_derived_globals_for_non_lm_without_positional(self) -> None:
        load_config, weights_info = _lm_config()
        loader = _make_loader(load_config, weights_info, task_type=None)
        self.assertEqual(list(loader.prepare_computed_weights("cpu")), [])


class ReloadFallbackTest(unittest.TestCase):
    """Fault-tolerance control flow of ``reload_weights_from_loader``: which
    source path runs, exercised without CUDA by stubbing ``_do_reload``."""

    _FORCE_SCRATCH_ENV = "SLEEP_L2_WAKE_RELOAD_FORCE_SCRATCH"

    def _make_wm(self, can_fast: bool):
        from rtp_llm.model_loader.weight_manager import WeightManager

        wm = object.__new__(WeightManager)
        wm._device = "cpu"
        wm._weights_loader = types.SimpleNamespace(
            can_reload_from_fastsafetensor=lambda: can_fast,
            prepare_weights_fastsafetensor=lambda *a, **k: iter([]),
            prepare_weights=lambda device: iter([]),
        )
        return wm

    def setUp(self) -> None:
        os.environ.pop(self._FORCE_SCRATCH_ENV, None)

    def tearDown(self) -> None:
        os.environ.pop(self._FORCE_SCRATCH_ENV, None)

    def test_uses_fast_when_available_and_ok(self) -> None:
        wm = self._make_wm(can_fast=True)
        calls = []
        wm._do_reload = lambda factory, method, device: calls.append(method)
        wm._discard_reload_transients = lambda: calls.append("discard")
        wm.reload_weights_from_loader()
        self.assertEqual(calls, ["fastsafetensors"])

    def test_falls_back_to_scratch_when_fast_raises(self) -> None:
        wm = self._make_wm(can_fast=True)
        calls = []

        def fake_do_reload(factory, method, device):
            calls.append(method)
            factory()  # consume the source factory like the real path does
            if method == "fastsafetensors":
                raise RuntimeError("simulated fast-path fault")

        wm._do_reload = fake_do_reload
        wm._discard_reload_transients = lambda: calls.append("discard")
        wm.reload_weights_from_loader()
        self.assertEqual(calls, ["fastsafetensors", "discard", "scratch"])

    def test_scratch_when_fast_unavailable(self) -> None:
        wm = self._make_wm(can_fast=False)
        calls = []
        wm._do_reload = lambda factory, method, device: calls.append(method)
        wm._discard_reload_transients = lambda: calls.append("discard")
        wm.reload_weights_from_loader()
        self.assertEqual(calls, ["scratch"])

    def test_force_scratch_env_skips_fast(self) -> None:
        os.environ[self._FORCE_SCRATCH_ENV] = "1"
        wm = self._make_wm(can_fast=True)
        calls = []
        wm._do_reload = lambda factory, method, device: calls.append(method)
        wm._discard_reload_transients = lambda: calls.append("discard")
        wm.reload_weights_from_loader()
        self.assertEqual(calls, ["scratch"])


class FastsafetensorsKnobPassthroughTest(unittest.TestCase):
    """nogds / use_shm forward to the fastsafetensors ParallelLoader."""

    def _run(self, **iter_kwargs):
        import fastsafetensors

        from rtp_llm.utils.database import CkptDatabase

        recorded = {}

        class _FakeLoader:
            def __init__(self, **kwargs):
                recorded.update(kwargs)
                self.loader = types.SimpleNamespace(close=lambda: None)

            def iterate_weights(self):
                return iter([])

        orig = fastsafetensors.ParallelLoader
        fastsafetensors.ParallelLoader = _FakeLoader
        try:
            db = object.__new__(CkptDatabase)
            db.pretrain_file_list = []
            gen = db.fastsafetensors_weights_iterator("cpu", False, **iter_kwargs)
            list(gen)  # drive the generator so ParallelLoader is constructed
        finally:
            fastsafetensors.ParallelLoader = orig
        return recorded

    def test_defaults_preserve_cold_load_behavior(self) -> None:
        recorded = self._run()
        self.assertEqual(recorded["use_shm"], True)
        self.assertEqual(recorded["nogds"], False)

    def test_overrides_forwarded(self) -> None:
        recorded = self._run(nogds=True, use_shm=False)
        self.assertEqual(recorded["use_shm"], False)
        self.assertEqual(recorded["nogds"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
