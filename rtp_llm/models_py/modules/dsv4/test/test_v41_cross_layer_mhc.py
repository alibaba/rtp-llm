"""Cross-layer delayed mHC boundaries, fallback, and graph replay."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.block import Block
from rtp_llm.models_py.modules.dsv4.hc import decode_transition as transition
from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCHead, DelayedHCUnit

_ENV = {"DSV41_FUSED_CROSS_LAYER_MHC": "1", "DSV41_MEGA_MHC": "1"}


class ReferenceNorm(torch.nn.Module):
    def forward(self, x):
        value = x.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)).to(
            x.dtype
        )


class TestAttention(torch.nn.Module):
    def forward_decode(self, x, _metadata, kv_cache=None):
        return x * 0.25


class TestFFN(torch.nn.Module):
    def forward(self, x, input_ids):
        return (
            x.float() * 0.125 + input_ids.reshape(*x.shape[:-1], 1).float() * 0.01
        ).to(x.dtype)


class TestEngram(torch.nn.Module):
    def forward(self, x, _hashes, _mask):
        return x + 0.125


def make_blocks(dim=8, device="cpu", count=3, engram_layers=()):
    torch.manual_seed(413)
    blocks = []
    previous = None
    for index in range(count):
        block = Block.__new__(Block)
        torch.nn.Module.__init__(block)
        block.layer_id = index
        block.engram = TestEngram() if index in engram_layers else None
        block.engram_hashes = torch.empty(0, device=device)
        block.engram_token_mask = None
        block.attn = TestAttention()
        block.ffn = TestFFN()
        for name in ("attn", "ffn"):
            unit = DelayedHCUnit(
                torch.randn(24, 4 * dim, device=device) * 0.003,
                torch.randn(24, device=device) * 0.1,
                torch.tensor([0.2, 0.4, 0.3], device=device),
                dim=dim,
                hc_mult=4,
                hc_sinkhorn_iters=20,
                norm_eps=1e-6,
                hc_eps=1e-6,
                layer_id=index,
                name=name,
            )
            unit.set_previous(previous)
            setattr(block, name + "_hc", unit)
            previous = unit
            if device == "cpu":
                norm = ReferenceNorm()
            else:
                from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm

                norm = RMSNorm(
                    torch.ones(dim, device=device, dtype=torch.bfloat16), 1e-6
                )
            setattr(block, name + "_norm", norm)
        blocks.append(block)
    return blocks


def run_blocks(blocks, hidden, input_ids, captures=(), attn_fn=None):
    hidden = hidden.clone()
    captured = []
    for index, block in enumerate(blocks):
        hidden = transition.forward_decode_layer(
            block,
            hidden,
            None,
            input_ids,
            next_layer=blocks[index + 1] if index + 1 < len(blocks) else None,
            preserve_output=index in captures,
            attn_fn=attn_fn,
        )
        if index in captures:
            assert isinstance(hidden, torch.Tensor)
            captured.append(hidden.clone())
    assert isinstance(hidden, torch.Tensor)
    return hidden, DelayedHCHead(blocks[-1].ffn_hc).head(hidden), *captured


def reference_transition(output, residual, post, comb, previous, next_hc, norm):
    residual = previous.post(output, residual, post, comb)
    x, post, comb = next_hc.pre(residual)
    normalized = norm(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    return residual, normalized, post, comb


class CrossLayerMHCCPUTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, _ENV)
        env.start()
        self.addCleanup(env.stop)
        tracing = patch.object(
            transition._record_tensor, "should_record_layer", return_value=False
        )
        tracing.start()
        self.addCleanup(tracing.stop)
        self.hidden = torch.randn(2, 5, 4, 8, dtype=torch.bfloat16)
        self.ids = torch.arange(10).reshape(2, 5)

    def compare(self, *, captures=(), engram_layers=(), backend=None, attn_fn=None):
        baseline = make_blocks(engram_layers=engram_layers)
        candidate = make_blocks(engram_layers=engram_layers)
        with patch.dict(os.environ, {"DSV41_FUSED_CROSS_LAYER_MHC": "0"}):
            expected = run_blocks(baseline, self.hidden, self.ids, captures, attn_fn)
        with patch.object(
            transition, "try_fused_post_pre", side_effect=backend
        ) as fused:
            actual = run_blocks(candidate, self.hidden, self.ids, captures, attn_fn)
        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref, rtol=0, atol=0)
        for a, b in zip(candidate, baseline):
            torch.testing.assert_close(
                a.attn_hc.pre_mix_out, b.attn_hc.pre_mix_out, rtol=0, atol=0
            )
            torch.testing.assert_close(
                a.ffn_hc.pre_mix_out, b.ffn_hc.pre_mix_out, rtol=0, atol=0
            )
        return fused.call_count

    def test_two_transitions_and_final_delayed_head(self):
        self.assertEqual(self.compare(backend=reference_transition), 2)

    def test_attention_override_uses_prepared_input(self):
        self.assertEqual(
            self.compare(backend=reference_transition, attn_fn=lambda x: x * 0.5), 2
        )

    def test_engram_and_aux_capture_boundaries(self):
        self.assertEqual(
            self.compare(engram_layers=(1,), backend=reference_transition), 1
        )
        self.assertEqual(self.compare(captures=(1,), backend=reference_transition), 1)
        self.assertEqual(
            self.compare(
                captures=(1,), engram_layers=(1,), backend=reference_transition
            ),
            0,
        )

    def test_unsupported_backend_materializes_original_post(self):
        self.assertEqual(self.compare(backend=lambda *args: None), 2)

    def test_pending_values_do_not_escape_a_forward(self):
        blocks = make_blocks()
        with patch.object(
            transition, "try_fused_post_pre", side_effect=reference_transition
        ):
            first = run_blocks(blocks, self.hidden, self.ids)
            second = run_blocks(blocks, self.hidden * -0.75, self.ids + 1)
        with patch.dict(os.environ, {"DSV41_FUSED_CROSS_LAYER_MHC": "0"}):
            expected = run_blocks(make_blocks(), self.hidden * -0.75, self.ids + 1)
        self.assertFalse(torch.equal(first[0], second[0]))
        for got, ref in zip(second, expected):
            torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_enable_gate_predecessor_and_debug_guards(self):
        left, right, _ = make_blocks()
        self.assertTrue(transition.can_defer_decode_post(left, right, False))
        self.assertFalse(transition.can_defer_decode_post(left, right, True))
        self.assertFalse(transition.can_defer_decode_post(left, None, False))
        for variable in _ENV:
            with patch.dict(os.environ, {variable: "0"}):
                self.assertFalse(transition.can_defer_decode_post(left, right, False))
        for recorded in (left.layer_id, right.layer_id):
            with patch.object(
                transition._record_tensor,
                "should_record_layer",
                side_effect=lambda index: index == recorded,
            ):
                self.assertFalse(transition.can_defer_decode_post(left, right, False))
        right.attn_hc.set_previous(left.attn_hc)
        self.assertFalse(transition.can_defer_decode_post(left, right, False))

    def test_backend_error_propagates_without_materializing_post(self):
        blocks = make_blocks()
        pending = transition.forward_decode_layer(
            blocks[0], self.hidden, None, self.ids, next_layer=blocks[1]
        )
        self.assertIsInstance(pending, transition.PendingDecodePost)
        with patch.object(
            transition,
            "try_fused_post_pre",
            side_effect=RuntimeError("backend failure"),
        ), patch.object(pending.previous, "post", wraps=pending.previous.post) as post:
            with self.assertRaisesRegex(RuntimeError, "backend failure"):
                transition.forward_decode_layer(blocks[1], pending, None, self.ids)
            post.assert_not_called()

    def test_custom_block_keeps_original_call_signature(self):
        class CustomBlock:
            def forward_decode(self, hidden, metadata, input_ids, kv_cache=None):
                return hidden + 1

        result = transition.forward_decode_layer(
            CustomBlock(), self.hidden, None, self.ids
        )
        torch.testing.assert_close(result, self.hidden + 1, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CrossLayerMHCCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("mega_mhc requires Blackwell")

    def setUp(self):
        env = patch.dict(os.environ, {**_ENV, "MOEDBG": "0"})
        env.start()
        self.addCleanup(env.stop)

    def assert_close(self, actual, expected):
        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref, rtol=0.04, atol=0.0625)
            rms = (got.float() - ref.float()).square().mean().sqrt()
            bound = 1e-5 + 0.01 * ref.float().square().mean().sqrt()
            self.assertLessEqual(rms.item(), bound.item())

    @torch.no_grad()
    def test_proposal_and_verify_layouts(self):
        for batch, width in ((1, 5), (4, 5), (4, 6), (8, 6)):
            with self.subTest(batch=batch, width=width):
                baseline = make_blocks(5120, "cuda")
                candidate = make_blocks(5120, "cuda")
                hidden = torch.randn(
                    batch, width, 4, 5120, device="cuda", dtype=torch.bfloat16
                )
                ids = torch.arange(batch * width, device="cuda").view(batch, width)
                with patch.dict(os.environ, {"DSV41_FUSED_CROSS_LAYER_MHC": "0"}):
                    expected = run_blocks(baseline, hidden, ids)
                with patch.object(
                    transition,
                    "try_fused_post_pre",
                    wraps=transition.try_fused_post_pre,
                ) as fused:
                    actual = run_blocks(candidate, hidden, ids)
                self.assertEqual(fused.call_count, 2)
                self.assert_close(actual, expected)

    @torch.no_grad()
    def test_graph_replays_updated_inputs_with_boundary_and_head(self):
        baseline = make_blocks(5120, "cuda", count=5, engram_layers=(1,))
        candidate = make_blocks(5120, "cuda", count=5, engram_layers=(1,))
        hidden = torch.randn(4, 6, 4, 5120, device="cuda", dtype=torch.bfloat16)
        ids = torch.arange(24, device="cuda").view(4, 6)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run_blocks(candidate, hidden, ids, captures=(3,))
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = run_blocks(candidate, hidden, ids, captures=(3,))
        initial = None
        for scale in (1.0, -0.75, 2.0):
            hidden.copy_(torch.randn_like(hidden) * scale)
            ids.add_(1)
            with patch.dict(os.environ, {"DSV41_FUSED_CROSS_LAYER_MHC": "0"}):
                expected = run_blocks(baseline, hidden, ids, captures=(3,))
            graph.replay()
            self.assert_close(actual, expected)
            if initial is None:
                initial = actual[1].clone()
            else:
                self.assertFalse(torch.equal(initial, actual[1]))


if __name__ == "__main__":
    unittest.main()
