import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.models_py.modules.qwen4_exp.pd_cache_store import (
    SIDE_TAGS,
    prepare_side_cache_store_writers,
    publish_layer_side_caches,
    validate_pd_cache_inputs,
)


class Qwen4PDCacheStoreTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"RTP_LLM_QWEN4_ENABLE_PD": "0"})
        env.start()
        self.addCleanup(env.stop)
        self.events = []
        self.native_writer = SimpleNamespace(write=self.write)

    def write(self, inputs, cache):
        self.events.append((inputs.tag, cache.tag, cache.payload.clone()))

    def inputs(self, tag, *, store=True, **overrides):
        result = SimpleNamespace(
            is_prefill=True,
            is_cuda_graph=False,
            is_target_verify=False,
            cache_store_inputs=SimpleNamespace(tag=tag) if store else None,
            cache_store_writer=self.native_writer,
        )
        result.__dict__.update(overrides)
        return result

    def pools(self):
        tags = sorted(SIDE_TAGS)
        caches = [SimpleNamespace(tag=tag, payload=torch.zeros(3)) for tag in tags]
        main = SimpleNamespace(tag="full", payload=torch.zeros(3))
        kv_cache = SimpleNamespace(
            get_layer_cache_groups=lambda layer: (
                [main, *caches[:2]] if layer == 0 else [main, *caches[2:]]
            ),
        )
        return kv_cache, {tag: self.inputs(tag) for tag in ["full", *tags]}, caches

    def test_default_off_and_incomplete_plan_fail_before_publication(self):
        kv, inputs, caches = self.pools()
        with self.assertRaisesRegex(RuntimeError, "RTP_LLM_QWEN4_ENABLE_PD"):
            prepare_side_cache_store_writers(inputs, kv, layer_count=2)
        with patch.dict(os.environ, {"RTP_LLM_QWEN4_ENABLE_PD": "1"}):
            inputs[sorted(SIDE_TAGS)[0]].cache_store_inputs = None
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                prepare_side_cache_store_writers(inputs, kv, layer_count=2)
        self.assertEqual(self.events, [])
        self.assertTrue(all(not cache.payload.any() for cache in caches))

    def test_missing_writer_graph_and_verify_are_rejected(self):
        with patch.dict(os.environ, {"RTP_LLM_QWEN4_ENABLE_PD": "1"}):
            for values, expected in (
                ({"cache_store_writer": None}, "writer is missing"),
                ({"is_cuda_graph": True}, "eager prefill"),
                ({"is_target_verify": True}, "target-verify"),
                ({"is_prefill": False}, "prefill inputs"),
            ):
                with self.subTest(values=values), self.assertRaisesRegex(
                    RuntimeError, expected
                ):
                    validate_pd_cache_inputs((self.inputs("full", **values),))
        self.assertEqual(self.events, [])

    def test_missing_tag_fails_before_layer_mutation(self):
        kv, inputs, _ = self.pools()
        del inputs[sorted(SIDE_TAGS)[0]]
        with patch.dict(os.environ, {"RTP_LLM_QWEN4_ENABLE_PD": "1"}):
            with self.assertRaisesRegex(RuntimeError, "inputs are missing"):
                prepare_side_cache_store_writers(inputs, kv, layer_count=2)
        self.assertEqual(self.events, [])

    def test_tag_local_writers_observe_completed_layer_state(self):
        kv, inputs, caches = self.pools()

        def factory(value, cache):
            return WriteCacheStoreOp(value.cache_store_writer, value.cache_store_inputs)

        with patch.dict(os.environ, {"RTP_LLM_QWEN4_ENABLE_PD": "1"}), patch(
            "rtp_llm.models_py.modules.factory.attention.common.create_write_cache_store_impl",
            side_effect=factory,
        ):
            writers = prepare_side_cache_store_writers(inputs, kv, layer_count=2)
            self.assertEqual(set(writers), SIDE_TAGS)
            for cache in caches[:2]:
                cache.payload.fill_(11)
            publish_layer_side_caches(kv, 0, writers)
            for cache in caches[2:]:
                cache.payload.fill_(17)
            publish_layer_side_caches(kv, 1, writers)
        self.assertEqual(len(self.events), 4)
        for planned_tag, actual_tag, snapshot in self.events:
            self.assertEqual(planned_tag, actual_tag)
            expected = 11 if actual_tag in {x.tag for x in caches[:2]} else 17
            torch.testing.assert_close(snapshot, torch.full_like(snapshot, expected))

    def test_pdfusion_and_decode_without_publication_are_noops(self):
        kv, inputs, _ = self.pools()
        for value in inputs.values():
            value.cache_store_inputs = None
            value.is_prefill = False
        self.assertEqual(
            prepare_side_cache_store_writers(inputs, kv, layer_count=2), {}
        )
        publish_layer_side_caches(kv, 0, {})
        self.assertEqual(self.events, [])

    def test_model_forward_publishes_after_decoder_completed(self):
        from rtp_llm.models_py.model_desc import qwen4_exp

        order = []
        caches = [
            SimpleNamespace(tag=tag, payload=torch.zeros(3))
            for tag in ("ple_conv_state", "ple_ngram_ctx")
        ]
        main_cache = SimpleNamespace(tag="full")
        inputs_by_tag = {
            tag: self.inputs(tag) for tag in ("full", "ple_conv_state", "ple_ngram_ctx")
        }

        class Embed(nn.Module):
            def forward(self, inputs):
                return torch.zeros(1, 4)

        class Layer(nn.Module):
            self_attn = SimpleNamespace(qsa_indexer=None)

            def forward(self, hidden, fmha, **kwargs):
                for cache in caches:
                    cache.payload.fill_(19)
                order.append("layer_completed")
                return hidden + 1

        class Mixer(nn.Module):
            def forward(self, hidden):
                return hidden, None, None

        def write(plan, cache):
            order.append("published")
            torch.testing.assert_close(
                cache.payload, torch.full_like(cache.payload, 19)
            )

        def factory(value, cache):
            return WriteCacheStoreOp(
                SimpleNamespace(write=write), value.cache_store_inputs
            )

        model = qwen4_exp.Qwen4ExpModel.__new__(qwen4_exp.Qwen4ExpModel)
        nn.Module.__init__(model)
        model.word_embedding = lambda inputs: torch.zeros(1, 4)
        model.layers = nn.ModuleList([Layer()])
        model.ple_layers = nn.ModuleDict({"0": nn.Identity()})
        model.kv_cache = SimpleNamespace(
            get_layer_cache_groups=lambda layer: [main_cache, *caches],
            get_layer_cache=lambda layer, tag: main_cache,
        )
        model.hyper_connection_mixer = Mixer()
        model._initial_hyper_states = lambda hidden: hidden
        model._build_attn_meta = lambda *args: None
        model._ple_input_ids = lambda inputs: inputs.input_ids
        model._apply_ple = lambda layer, hidden, *args: hidden
        model._attention_tag = lambda layer: "full"
        model._layer_fmha_impl = lambda *args: None
        inputs = SimpleNamespace(
            input_ids=torch.zeros(1, dtype=torch.long), attention_inputs=inputs_by_tag
        )
        with patch.dict(os.environ, {"RTP_LLM_QWEN4_ENABLE_PD": "1"}), patch(
            "rtp_llm.models_py.modules.factory.attention.common.create_write_cache_store_impl",
            side_effect=factory,
        ), patch.object(qwen4_exp, "_is_cuda_graph_forward", return_value=False):
            model.forward(inputs, fmha_impl=object())
        self.assertEqual(order, ["layer_completed", "published", "published"])


if __name__ == "__main__":
    unittest.main()
