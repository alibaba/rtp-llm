"""CPU-only unit tests for RL frozen PLE exclusion and BF16 HC paused in-place unlock."""

import gc
import unittest
import weakref
from types import SimpleNamespace

import torch

from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.model_loader.rl_weight_update import RlWeightUpdate
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.utils.model_weight import W


class _DummyWeight:
    def __init__(self, name: str, resident_device=None):
        self.name = name
        self.resident_device = resident_device


class _FakeLoader:
    def __init__(self, static_table_keys, static_meta_keys, emit_items):
        self._static_table_keys = set(static_table_keys)
        self._static_meta_keys = set(static_meta_keys)
        self._emit_items = emit_items

    def get_rl_static_target_keys(self, weights):
        return set(self._static_table_keys), set(self._static_meta_keys)

    def prepare_weights_from_tensors(
        self, named_tensors, device, *, max_buffer_bytes, borrow_singleton_sources=True
    ):
        received = dict(named_tensors)
        for layer, target_name, source_name in self._emit_items:
            yield layer, target_name, received[source_name]


class RlWeightUpdateUnitTest(unittest.TestCase):
    def test_converted_weight_lifetime_does_not_depend_on_cyclic_gc(self):
        weight = AtomicWeight("nested", [])
        weight._load_raw_tensor = lambda *_: torch.ones(1)
        weight._split = lambda tensor, _: tensor
        weight._postprocess = lambda tensor, *_: {"nested": {"leaf": tensor}}
        config = SimpleNamespace(merge_lora=False)
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            for convert in (
                lambda: weight.load(None, None, "cpu", config),
                lambda: weight.update(torch.ones(1), "cpu", config),
            ):
                outputs = convert()
                tensor_ref = weakref.ref(outputs["leaf"])
                del outputs
                self.assertIsNone(tensor_ref())
        finally:
            if was_enabled:
                gc.enable()

    def test_frozen_ple_and_int64_metadata_excluded_while_ple_dense_kept(self):
        ple_shard = _DummyWeight(W.qwen4_ple_ngram_shards, resident_device="cpu")
        hash_mult = _DummyWeight(W.qwen4_ple_multipliers)
        hash_mod = _DummyWeight(W.qwen4_ple_ngram_offsets)
        hash_tok = _DummyWeight(W.qwen4_ple_ngram_vocab_sizes)
        cpu_only = _DummyWeight("custom_cpu_static", resident_device="cpu")
        ple_dense = _DummyWeight("ple_key_proj")

        self.assertTrue(ModelLoader._is_rl_static_component(ple_shard))
        self.assertTrue(ModelLoader._is_rl_static_component(hash_mult))
        self.assertTrue(ModelLoader._is_rl_static_component(hash_mod))
        self.assertTrue(ModelLoader._is_rl_static_component(hash_tok))
        self.assertTrue(ModelLoader._is_rl_static_component(cpu_only))
        self.assertFalse(ModelLoader._is_rl_static_component(ple_dense))

    def test_bf16_hc_paused_inplace_unlock_live_refusal_and_tail_check(self):
        # Simulate qwen4_exp merged HC buffer: [16, 8] BF16,
        # active rows [0..3], zero-padded tail [4..15].
        merged = torch.zeros((16, 8), dtype=torch.bfloat16)
        mix_down = merged[0:2]
        inject = merged[2:4]
        for view in (mix_down, inject):
            view._rtp_requires_restart_for_update = True
            view._rtp_allow_paused_inplace_update = True
            view._rtp_merged_parent = merged
            view._rtp_merged_tail_start = 4

        mw = ModelWeights(num_layers=1, device="cpu", dtype=torch.bfloat16)
        mw.set_layer_weight(0, "hc.mix_down", mix_down)
        mw.set_layer_weight(0, "hc.inject", inject)
        # Static INT64 metadata + CPU PLE shard must stay untouched.
        meta_tensor = torch.tensor([11, 22, 33], dtype=torch.int64)
        ple_shard = torch.zeros((4, 8), dtype=torch.bfloat16)
        mw.set_layer_weight(0, W.qwen4_ple_multipliers, meta_tensor)
        mw.set_layer_weight(0, f"{W.qwen4_ple_ngram_shards}.0", ple_shard)

        new_mix = torch.full((2, 8), 3.0, dtype=torch.bfloat16)
        new_inject = torch.full((2, 8), 5.0, dtype=torch.bfloat16)

        # Live update (default allow_paused_inplace=False) MUST still refuse.
        with self.assertRaisesRegex(PermissionError, "requires a service restart"):
            mw.update_layer_weight(0, "hc.mix_down", new_mix)

        # Full paused RlWeightUpdate transaction succeeds in-place and preserves pointers + zero tail.
        loader = _FakeLoader(
            static_table_keys={(0, f"{W.qwen4_ple_ngram_shards}.0")},
            static_meta_keys={(0, W.qwen4_ple_multipliers)},
            emit_items=[
                (0, "hc.mix_down", "model.layers.0.hc.down"),
                (0, "hc.inject", "model.layers.0.hc.inject"),
            ],
        )
        before_mix_ptr = mix_down.data_ptr()
        before_inj_ptr = inject.data_ptr()
        tx = RlWeightUpdate(
            loader=loader,
            weights=mw,
            device="cpu",
            source_shapes={
                "model.layers.0.hc.down": (2, 8),
                "model.layers.0.hc.inject": (2, 8),
            },
            max_buffer_bytes=1024 * 1024,
            source_dtypes={
                "model.layers.0.hc.down": torch.bfloat16,
                "model.layers.0.hc.inject": torch.bfloat16,
            },
        )
        tx.push("model.layers.0.hc.down", new_mix)
        tx.push("model.layers.0.hc.inject", new_inject)
        receipt = tx.finish()
        self.assertEqual(receipt["hf_sources"], 2)
        self.assertEqual(receipt["internal_tensors"], 2)
        self.assertEqual(receipt["static_table_tensors"], 1)
        self.assertEqual(receipt["static_meta_tensors"], 1)
        self.assertEqual(mix_down.data_ptr(), before_mix_ptr)
        self.assertEqual(inject.data_ptr(), before_inj_ptr)
        self.assertTrue(torch.equal(merged[0:2], new_mix))
        self.assertTrue(torch.equal(merged[2:4], new_inject))
        self.assertTrue(torch.equal(merged[4:], torch.zeros((12, 8), dtype=torch.bfloat16)))

        # Corrupting the zero-padded tail before commit MUST fail transaction validation.
        tx2 = RlWeightUpdate(
            loader=loader,
            weights=mw,
            device="cpu",
            source_shapes={
                "model.layers.0.hc.down": (2, 8),
                "model.layers.0.hc.inject": (2, 8),
            },
            max_buffer_bytes=1024 * 1024,
        )
        tx2.push("model.layers.0.hc.down", new_mix)
        merged[4, 0] = 1.0
        tx2.push("model.layers.0.hc.inject", new_inject)
        with self.assertRaisesRegex(RuntimeError, "trailing padding rows became non-zero"):
            tx2.finish()

    def test_restart_only_target_without_paused_allow_flag_rejected_pre_write(self):
        t = torch.zeros((2, 8), dtype=torch.bfloat16)
        t._rtp_requires_restart_for_update = True
        mw = ModelWeights(num_layers=1, device="cpu", dtype=torch.bfloat16)
        mw.set_layer_weight(0, "fp8_repacked", t)
        loader = _FakeLoader(set(), set(), [(0, "fp8_repacked", "src.w")])
        with self.assertRaisesRegex(PermissionError, "requires a service restart"):
            RlWeightUpdate(
                loader=loader,
                weights=mw,
                device="cpu",
                source_shapes={"src.w": (2, 8)},
                max_buffer_bytes=1024,
            )


if __name__ == "__main__":
    unittest.main()
