"""Executable V4.1 block ordering, forward ownership and final readout."""

import dataclasses
import sys
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models.test.dsv41_attention_projection_test import queries, weights_fixture
from rtp_llm.models.test.dsv41_attention_test import attention
from rtp_llm.models.test.dsv41_cache_codec_test import load_codec_source, make_cache
from rtp_llm.models.test.dsv41_delayed_hc_test import hc
from rtp_llm.models.test.dsv41_expert_test import expert, shared_weights
from rtp_llm.models.test.dsv41_feed_forward_test import ffn
from rtp_llm.models.test.dsv41_indexer_test import indexer
from rtp_llm.models.test.dsv41_moe_test import moe, routed, routed_weights
from rtp_llm.models.test.dsv41_projection_test import norm, projection

with patch.dict(
    sys.modules,
    {
        "rtp_llm.models_py.modules.dsv4.hc.delayed": hc,
        "rtp_llm.models_py.modules.dsv4.v41_attention_projection": queries,
        "rtp_llm.models_py.modules.dsv4.v41_feed_forward": ffn,
        "rtp_llm.models_py.modules.dsv4.v41_moe": moe,
        "rtp_llm.models_py.modules.dsv4.v41_projection": projection,
    },
):
    block = load_codec_source("_dsv41_block_test", "v41_block.py")


def fixture(layer=20):
    sources, weights, frequencies = weights_fixture(layer)
    for kind in ("attn", "ffn"):
        weights[f"v4.hc.{kind}_fn"] = torch.randn(24, 512) * 0.01
        weights[f"v4.hc.{kind}_base"] = torch.randn(24) * 0.1
        weights[f"v4.hc.{kind}_scale"] = torch.tensor([0.5, 0.6, 0.7])
        weights[f"v4.{kind}_norm.weight"] = torch.randn(128).bfloat16()
    projections, kv = queries.build_v41_projections(
        sources, weights, frequencies, backend="eager"
    )
    feed_forward = ffn.V41FeedForward(
        layer,
        ffn.V41Router(
            torch.randn(3, 128).bfloat16() / 128**0.5, torch.randn(3), topk=2
        ),
        expert.V41SharedExpert(shared_weights(dim=128, inter=128), backend="eager"),
        moe.V41LocalMoE(
            routed.V41RoutedExperts(routed_weights(dim=128, inter=128), backend="eager")
        ),
    )
    return block.build_v41_block(weights, projections, kv, feed_forward), weights


def inputs(layer=20):
    valid = torch.tensor([True, True, False])
    residual = torch.randn(3, 4, 128).bfloat16()
    residual[-1] = float("nan")
    previous = torch.rand(3, 4)
    previous[-1] = float("nan")
    return (
        block.V41ResidualState(residual, previous, layer),
        torch.tensor([0, 1, -999]),
        valid,
    )


def reference_post(unit, output, residual, mixes):
    mixed = torch.stack(
        [
            sum(
                mixes.residual_mix[:, source, dest, None] * residual[:, source].float()
                for source in range(4)
            )
            + mixes.post[:, dest, None] * output.float()
            for dest in range(4)
        ],
        dim=1,
    )
    return mixed.bfloat16()


def reference_finish(op, state, attended, frame, valid):
    residual = torch.where(valid[:, None, None], state.residual, 0)
    previous = torch.where(valid[:, None], state.previous_pre_mix, 0)
    _, mixes = op.attn_hc.pre(residual, previous)
    residual = reference_post(
        op.attn_hc, op.attention.output(attended, frame.query), residual, mixes
    )
    collapsed = (residual.float() * mixes.next_pre_mix[..., None]).sum(1).bfloat16()
    _, next_mixes = op.ffn_hc.pre(residual, mixes.next_pre_mix)
    result = reference_post(
        op.ffn_hc,
        op.feed_forward(norm(collapsed, op.ffn_norm), valid),
        residual,
        next_mixes,
    )
    return torch.where(valid[:, None, None], result, 0), torch.where(
        valid[:, None], next_mixes.next_pre_mix, 0
    )


def attend(op, frame, positions, valid, *, memory=None, selected=None):
    swa, _ = make_cache("swa_kv")
    writes = torch.tensor([0, 1, -999])
    if op.attention.sources.owns_kv:
        global_cache, _ = make_cache("hca_kv")
        index_cache, _ = make_cache("indexer_kv")
        frame.kv.publish(
            swa,
            writes,
            global_cache=global_cache,
            global_slots=writes,
            index_cache=index_cache,
            index_slots=writes,
        )
        select = indexer.V41Indexer(
            op.attention.sources, max_positions=256, backend="eager"
        )
        selected = select(
            frame.query.index_query,
            frame.query.index_weights,
            index_cache,
            torch.zeros(3, 1, dtype=torch.int32),
            positions,
            valid_rows=valid,
        )
        memory = global_cache, index_cache
    else:
        frame.kv.publish(swa, writes)
    reads = torch.tensor([[0, -1], [0, 1], [-1, -1]])
    kwargs = (
        {}
        if memory is None
        else {"global_cache": memory[0], "global_slots": selected.topk.long()}
    )
    attended = attention.V41SparseAttention(backend="eager")(
        frame.query.query, op.attention.sinks, swa, reads, valid_rows=valid, **kwargs
    )
    return attended, memory, selected


class V41BlockTest(unittest.TestCase):
    def test_prepare_uses_predecessor_then_norm_and_same_input_for_query_and_kv(self):
        op, _ = fixture()
        state, positions, valid = inputs()
        frame = op.prepare(state, positions, valid)
        clean = torch.where(valid[:, None, None], state.residual, 0)
        prior = torch.where(valid[:, None], state.previous_pre_mix, 0)
        expected_hidden = norm(
            (clean.float() * prior[..., None]).sum(1).bfloat16(), op.attn_norm
        )
        expected_query = op.attention.prepare(expected_hidden, positions, valid)
        expected_kv = op.kv.prepare(expected_hidden, positions, valid)
        for actual, expected in (
            (frame.query.query, expected_query.query),
            (frame.kv.swa, expected_kv.swa),
            (frame.kv.global_kv, expected_kv.global_kv),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        wrong = norm(hc.collapse_delayed(clean, frame.mixes.next_pre_mix), op.attn_norm)
        self.assertFalse(torch.equal(expected_hidden, wrong))

    def test_full_projection_publication_selection_attention_and_ffn_order(self):
        op, _ = fixture()
        state, positions, valid = inputs()
        frame = op.prepare(state, positions, valid)
        attended, _, selected = attend(op, frame, positions, valid)
        actual = op.finish(frame, attended)
        expected, pre_mix = reference_finish(op, state, attended, frame, valid)
        torch.testing.assert_close(actual.residual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual.previous_pre_mix, pre_mix, rtol=0, atol=0)
        self.assertEqual(actual.next_layer_id, 21)
        self.assertEqual(selected.candidate_counts.tolist(), [1, 1, 0])
        self.assertTrue(bool((actual.residual[~valid] == 0).all()))

    def test_l20_l21_handoff_keeps_delayed_coefficients_and_source_selection(self):
        first, _ = fixture(20)
        second, _ = fixture(21)
        state, positions, valid = inputs()
        frame = first.prepare(state, positions, valid)
        attended, memory, selected = attend(first, frame, positions, valid)
        boundary = first.finish(frame, attended)
        copied = block.V41ResidualState(
            boundary.residual.clone(), boundary.previous_pre_mix.clone(), 21
        )
        results = []
        for value in (boundary, copied):
            frame = second.prepare(value, positions, valid)
            self.assertIsNone(frame.kv.global_kv)
            self.assertIsNone(frame.query.index_query)
            attended, _, _ = attend(
                second, frame, positions, valid, memory=memory, selected=selected
            )
            results.append(second.finish(frame, attended))
        torch.testing.assert_close(
            results[0].residual, results[1].residual, rtol=0, atol=0
        )
        torch.testing.assert_close(
            results[0].previous_pre_mix, results[1].previous_pre_mix, rtol=0, atol=0
        )

    def test_interleaved_frames_snapshot_residual_and_mask_without_module_state(self):
        op, _ = fixture(0)
        state, positions, valid = inputs(0)
        frame_a = op.prepare(state, positions, valid)
        output = torch.randn(3, 64, 512).bfloat16()
        expected = op.finish(frame_a, output)
        state.residual[:2].add_(3)
        state.previous_pre_mix.zero_()
        frame_b = op.prepare(state, positions, valid)
        valid.zero_()
        positions.fill_(-999)
        op.finish(frame_b, output)
        actual = op.finish(frame_a, output)
        torch.testing.assert_close(actual.residual, expected.residual, rtol=0, atol=0)

    def test_embedding_expansion_empty_block_and_final_checkpoint_readout(self):
        valid = torch.tensor([True, False])
        embeddings = torch.randn(2, 128).bfloat16()
        embeddings[1] = float("nan")
        start = block.start_v41_residual(embeddings, valid)
        torch.testing.assert_close(
            start.residual[0], embeddings[0].repeat(4, 1), rtol=0, atol=0
        )
        self.assertEqual(
            start.previous_pre_mix.tolist(),
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        )
        state, _, valid = inputs()
        final = dataclasses.replace(state, next_layer_id=40)
        weight = torch.randn(128).bfloat16()
        actual = block.readout_v41_residual(final, weight, valid)
        expected = norm(
            hc.collapse_delayed(state.residual[:2], state.previous_pre_mix[:2]), weight
        )
        torch.testing.assert_close(actual[:2], expected, rtol=0, atol=0)
        self.assertTrue(bool((actual[-1] == 0).all()))
        with self.assertRaisesRegex(ValueError, "global layer"):
            block.readout_v41_residual(state, weight, valid)
        op, _ = fixture(0)
        empty = block.start_v41_residual(
            torch.empty(0, 128, dtype=torch.bfloat16), torch.empty(0, dtype=torch.bool)
        )
        frame = op.prepare(
            empty, torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.bool)
        )
        result = op.finish(frame, torch.empty(0, 64, 512, dtype=torch.bfloat16))
        self.assertEqual(result.residual.shape, (0, 4, 128))

    def test_wrong_layer_frame_and_checkpoint_parameters_fail_before_ffn(self):
        op, weights = fixture(0)
        state, positions, valid = inputs(0)
        frame = op.prepare(state, positions, valid)
        with patch.object(op.feed_forward, "forward") as forward:
            with self.assertRaisesRegex(ValueError, "different block"):
                op.finish(
                    dataclasses.replace(frame, owner=object()), torch.zeros(3, 64, 512)
                )
            with self.assertRaisesRegex(ValueError, "global layer"):
                op.prepare(
                    dataclasses.replace(state, next_layer_id=1), positions, valid
                )
            forward.assert_not_called()
        original = dict(weights)
        block.build_v41_block(weights, op.attention, op.kv, op.feed_forward)
        self.assertTrue(all(weights[k] is v for k, v in original.items()))
        op.attn_hc.norm_eps = 1e-6
        with self.assertRaisesRegex(ValueError, "epsilons"):
            block.V41Block(
                op.attention,
                op.kv,
                op.attn_hc,
                op.ffn_hc,
                op.attn_norm,
                op.ffn_norm,
                op.feed_forward,
            )


if __name__ == "__main__":
    unittest.main()
