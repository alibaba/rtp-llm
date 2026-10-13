"""Shared token routing must preserve visual/hash rows and request lifetime."""

import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rtp_llm.models_py.modules.dsv4.moe import gate as gate_module
from rtp_llm.models_py.modules.dsv4.moe import moe_layer
from rtp_llm.models_py.modules.dsv4.moe import routing_metadata as routing


class _CountComparisons(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.ge.Scalar:
            self.count += 1
        return func(*args, **(kwargs or {}))


def _gate(*, hashed, device="cpu", score_func="sqrtsoftplus", visual_bias=True):
    gate = gate_module.Gate.__new__(gate_module.Gate)
    torch.nn.Module.__init__(gate)
    gate.dim = 3
    gate.topk = 2
    gate.score_func = score_func
    gate.route_scale = 1.25
    gate.hash = hashed
    gate._dbg_prefix = None
    gate.vocab_size = 8
    gate._has_visual_tokens = True
    gate.weight = torch.tensor(
        [[0.5, 1.0, -0.5], [0.0, 0.5, 0.75], [1.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    gate.bias = None if hashed else torch.tensor([0.5, 0.0, -0.5, 1.0], device=device)
    gate.bias_vl = (
        torch.tensor([0.0, 2.0, 1.0, -1.0], device=device) if visual_bias else None
    )
    gate.tid2eid = torch.tensor([[i % 4, (i + 1) % 4] for i in range(8)], device=device)
    return gate


class GateRoutingMetadataTest(unittest.TestCase):
    def test_cp_local_order_padding_and_noncontiguous_decode_ids(self):
        # Local zigzag order includes visual sentinel rows and zero padding;
        # metadata must not infer positions or remove the padded rows.
        ids = torch.tensor([[8, 4, 0], [2, 9, 0]]).t()
        metadata = routing.prepare_gate_routing_metadata(
            ids, vocab_size=8, has_hash_routing=True
        )
        torch.testing.assert_close(metadata.input_ids, ids.reshape(-1))
        torch.testing.assert_close(
            metadata.image_mask, torch.tensor([True, False, False, True, False, False])
        )
        torch.testing.assert_close(
            metadata.safe_hash_input_ids, torch.tensor([0, 2, 4, 0, 0, 0])
        )
        self.assertEqual(metadata.image_mask_column.shape, (6, 1))

    def test_v41_no_hash_and_text_forward_do_not_build_unused_tensors(self):
        model = SimpleNamespace(
            args=SimpleNamespace(vocab_size=8, n_hash_layers=0),
            _has_visual_tokens=False,
        )
        ids = torch.tensor([1, 8, 2])
        with patch.object(routing, "prepare_gate_routing_metadata") as prepare:
            self.assertIsNone(routing.prepare_model_gate_routing_metadata(model, ids))
            prepare.assert_not_called()
        model._has_visual_tokens = True
        with patch.object(torch, "where", side_effect=AssertionError("no hash ids")):
            metadata = routing.prepare_model_gate_routing_metadata(model, ids)
        self.assertIsNone(metadata.safe_hash_input_ids)

    def test_shared_metadata_preserves_all_gate_score_and_hash_modes(self):
        ids = torch.tensor([0, 8, 3, 9, 0, 7])
        x = torch.arange(18).reshape(6, 3).float() / 11 - 0.5
        metadata = routing.prepare_gate_routing_metadata(
            ids, vocab_size=8, has_hash_routing=True
        )
        with patch.dict(os.environ, {"DSV4_GATE_FUSED": "0", "DSV4_GATE_FP32": "1"}):
            for hashed in (False, True):
                for score_func in ("sqrtsoftplus", "sigmoid", "softmax"):
                    with self.subTest(hashed=hashed, score_func=score_func):
                        gate = _gate(hashed=hashed, score_func=score_func)
                        expected = gate(x, ids)
                        with _CountComparisons() as count:
                            actual = gate(x, ids, routing_metadata=metadata)
                        self.assertEqual(count.count, 0)
                        for value, reference in zip(actual, expected):
                            torch.testing.assert_close(value, reference, rtol=0, atol=0)

    def test_layer_without_visual_bias_preserves_legacy_behavior(self):
        ids = torch.tensor([0, 3, 7])
        x = torch.arange(9).reshape(3, 3).float() / 11
        metadata = routing.prepare_gate_routing_metadata(
            ids, vocab_size=8, has_hash_routing=True
        )
        with patch.dict(os.environ, {"DSV4_GATE_FUSED": "0", "DSV4_GATE_FP32": "1"}):
            for hashed in (True, False):
                gate = _gate(hashed=hashed, visual_bias=False)
                for value, reference in zip(
                    gate(x, ids, routing_metadata=metadata), gate(x, ids)
                ):
                    torch.testing.assert_close(value, reference, rtol=0, atol=0)

    def test_chunked_moe_slices_metadata_and_reuses_views_across_layers(self):
        ids = torch.tensor([8, 3, 0, 2, 9, 7, 0])
        x = torch.arange(21).reshape(7, 3).float() / 11
        metadata = routing.prepare_gate_routing_metadata(
            ids, vocab_size=8, has_hash_routing=True
        )
        chunk_metadata = []
        with patch.dict(os.environ, {"DSV4_GATE_FUSED": "0", "DSV4_GATE_FP32": "1"}):
            for layer in (0, 1):
                moe = moe_layer.MoE.__new__(moe_layer.MoE)
                torch.nn.Module.__init__(moe)
                moe.dim, moe.layer_id, moe.max_tokens_per_rank = 3, layer, 3
                moe._gate_pack_static = False
                moe._routed_includes_shared = True
                moe._strategy = lambda x, weights, indices: (
                    x + weights.sum(dim=-1, keepdim=True) + indices[:, :1]
                )
                real_gate = _gate(hashed=True)
                seen = []

                def gate(x, input_ids, **kwargs):
                    if kwargs:
                        seen.append(kwargs["routing_metadata"])
                    return real_gate(x, input_ids, **kwargs)

                moe.gate = gate
                expected = moe._forward_chunked(x, ids, x.shape).clone()
                actual = moe._forward_chunked(x, ids, x.shape, metadata).clone()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                chunk_metadata.append(seen)
        self.assertEqual(
            [chunk.input_ids.numel() for chunk in chunk_metadata[0]], [3, 3, 1]
        )
        for first, second in zip(*chunk_metadata):
            self.assertIs(first, second)

    def test_empty_and_reused_input_buffer_get_fresh_metadata(self):
        empty = routing.prepare_gate_routing_metadata(
            torch.empty(0, dtype=torch.long), vocab_size=8, has_hash_routing=True
        )
        self.assertEqual(empty.image_mask.numel(), 0)
        ids = torch.tensor([8, 1, 3])
        first = routing.prepare_gate_routing_metadata(
            ids, vocab_size=8, has_hash_routing=True
        )
        first_chunk = first.slice(0, 2)
        ids.copy_(torch.tensor([2, 9, 0]))
        second = routing.prepare_gate_routing_metadata(
            ids, vocab_size=8, has_hash_routing=True
        )
        self.assertIsNot(second, first)
        self.assertIsNot(second.slice(0, 2), first_chunk)
        torch.testing.assert_close(
            second.image_mask, torch.tensor([False, True, False])
        )
        torch.testing.assert_close(second.safe_hash_input_ids, torch.tensor([2, 0, 0]))

    def test_prefill_and_decode_share_once_and_refresh_the_next_forward(self):
        from rtp_llm.models_py.modules.dsv4.decode import forward as decode
        from rtp_llm.models_py.modules.dsv4.prefill import forward as prefill
        from rtp_llm.models_py.modules.dsv4.test.test_prefill_fast_path import _FakeV4

        for mode in ("prefill", "prefill_fast", "decode"):
            with self.subTest(mode=mode), ExitStack() as stack:
                model = _FakeV4()
                model.args = SimpleNamespace(vocab_size=8, n_hash_layers=3)
                model._has_visual_tokens = True
                seen = []

                def layer(hidden, *args, routing_metadata=None, **kwargs):
                    seen.append(routing_metadata)
                    return hidden

                class ForwardLayer:
                    def __init__(self, layer_id):
                        self.layer_id = layer_id

                    def __call__(self, *args, **kwargs):
                        return layer(*args, **kwargs)

                    forward_decode = __call__

                model.layers = [ForwardLayer(i) for i in range(3)]
                ids = torch.tensor([8, 1, 0])
                stack.enter_context(patch.object(prefill._rt, "ENABLED", False))
                stack.enter_context(
                    patch.object(prefill._fwd_dbg, "enabled", return_value=False)
                )
                stack.enter_context(
                    patch.object(prefill, "build_and_propagate_prefill_meta_fp8")
                )
                stack.enter_context(
                    patch.object(prefill, "clear_prefill_meta_shared_fp8")
                )
                stack.enter_context(patch.dict(os.environ, {"DSV41_CED": "0"}))
                if mode.startswith("prefill"):
                    stack.enter_context(
                        patch.object(
                            prefill,
                            "_prefill_fast_path_layer_calls",
                            return_value=(layer,) * 3,
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            prefill,
                            "_prefill_fast_path_enabled",
                            return_value=mode == "prefill_fast",
                        )
                    )

                    def run():
                        return prefill.forward_layers(
                            model,
                            None,
                            ids,
                            torch.arange(3),
                            torch.tensor([0, 3]),
                            None,
                        )

                else:
                    model._hc_head_reduce = lambda hidden: hidden.squeeze(2)

                    def run():
                        return decode.forward_layers(
                            model,
                            None,
                            ids,
                            SimpleNamespace(
                                batch_size=1, q_len_per_req=3, is_cuda_graph=False
                            ),
                        )

                with _CountComparisons() as count:
                    run()
                self.assertEqual(count.count, 1)
                self.assertEqual(len(seen), 3)
                self.assertTrue(all(value is seen[0] for value in seen))
                first = seen[0]
                seen.clear()
                ids.copy_(torch.tensor([2, 9, 0]))
                run()
                self.assertIsNot(seen[0], first)
                torch.testing.assert_close(
                    seen[0].image_mask, torch.tensor([False, True, False])
                )

    def test_standalone_transformer_shares_prefill_and_decode_metadata(self):
        from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer

        seen = []

        class Layer:
            def __call__(self, hidden, *args, routing_metadata=None, **kwargs):
                seen.append(routing_metadata)
                return hidden

            forward_decode = __call__

        model = SimpleNamespace(
            args=SimpleNamespace(vocab_size=8, n_hash_layers=3, ep_size=1, dim=3),
            _has_visual_tokens=True,
            hc_mult=1,
            layers=[Layer(), Layer()],
            embed=lambda ids: torch.zeros((*ids.shape, 3)),
            _propagate_cp_ctx=lambda ctx: None,
            _hc_head_reduce=lambda h: h.squeeze(-2),
            norm=lambda h: h,
        )
        ids = torch.tensor([[8, 1, 0]])
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode), _CountComparisons() as count:
                seen.clear()
                if mode == "prefill":
                    V4Transformer.forward(model, ids, apply_lm_head=False)
                else:
                    V4Transformer.forward_decode(
                        model, ids, SimpleNamespace(batch_size=1, q_len_per_req=3)
                    )
            self.assertEqual(count.count, 1)
            self.assertIs(seen[0], seen[1])
            torch.testing.assert_close(
                seen[0].image_mask, torch.tensor([True, False, False])
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GateRoutingMetadataCudaGraphTest(unittest.TestCase):
    def test_graph_replay_rebuilds_mask_for_changed_input_ids(self):
        ids = torch.tensor([8, 1, 0, 2, 9, 3], device="cuda")
        x = torch.arange(18, device="cuda").reshape(6, 3).float() / 11
        gate = _gate(hashed=True, device="cuda")
        with patch.dict(os.environ, {"DSV4_GATE_FUSED": "0", "DSV4_GATE_FP32": "1"}):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    metadata = routing.prepare_gate_routing_metadata(
                        ids, vocab_size=8, has_hash_routing=True
                    )
                    gate(x, ids, routing_metadata=metadata)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                metadata = routing.prepare_gate_routing_metadata(
                    ids, vocab_size=8, has_hash_routing=True
                )
                output = gate(x, ids, routing_metadata=metadata)
            for values in ([0, 9, 2, 8, 1, 0], [8, 9, 8, 9, 0, 0], [0, 1, 2, 3, 4, 5]):
                ids.copy_(torch.tensor(values, device="cuda"))
                graph.replay()
                expected = gate(x, ids)
                for actual, reference in zip(output, expected):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
