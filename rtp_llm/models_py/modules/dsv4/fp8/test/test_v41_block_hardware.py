"""SM120 full-width block: delayed HC, cache production, attention and FFN."""

import unittest

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_v41_attention_projection_hardware as attention_fixture,
)
from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_v41_expert_hardware as shared_fixture,
)
from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_v41_routed_hardware as routed_fixture,
)
from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCUnit
from rtp_llm.models_py.modules.dsv4.v41_attention import V41SparseAttention
from rtp_llm.models_py.modules.dsv4.v41_block import V41Block, V41ResidualState
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache
from rtp_llm.models_py.modules.dsv4.v41_compressor import V41PairMetadata, V41PairState
from rtp_llm.models_py.modules.dsv4.v41_expert import V41SharedExpert
from rtp_llm.models_py.modules.dsv4.v41_feed_forward import V41FeedForward, V41Router
from rtp_llm.models_py.modules.dsv4.v41_indexer import V41Indexer
from rtp_llm.models_py.modules.dsv4.v41_moe import V41LocalMoE
from rtp_llm.models_py.modules.dsv4.v41_routed import V41RoutedExperts


def fixture(ratio):
    query, kv, _ = attention_fixture.fixture(ratio, full=True)
    layer_id = query.sources.layer_id

    def hc():
        return DelayedHCUnit(
            torch.randn(24, 4 * 5120, device="cuda") * 0.001,
            torch.randn(24, device="cuda") * 0.1,
            torch.tensor([0.5, 0.6, 0.7], device="cuda"),
            dim=5120,
            norm_eps=1e-20,
        )

    ffn = V41FeedForward(
        layer_id,
        V41Router(
            (torch.randn(8, 5120, device="cuda") / 5120**0.5).bfloat16(),
            torch.randn(8, device="cuda"),
            max_rows=8,
        ),
        V41SharedExpert(shared_fixture.V41ExpertHardwareTest.weights()),
        V41LocalMoE(
            V41RoutedExperts(routed_fixture.V41RoutedHardwareTest.weights(8)),
            max_tokens=8,
            masked_token_limit=8,
        ),
    )
    return V41Block(
        query,
        kv,
        hc(),
        hc(),
        torch.randn(5120, device="cuda").bfloat16(),
        torch.randn(5120, device="cuda").bfloat16(),
        ffn,
    )


class V41BlockHardwareTest(unittest.TestCase):
    def setUp(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        previous = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)
        torch.backends.cuda.matmul.allow_tf32 = False

    def test_full_width_both_ratios_and_dynamic_block_graph(self):
        for ratio in (1, 2):
            op = fixture(ratio)
            state = V41ResidualState(
                torch.randn(5, 4, 5120, device="cuda").bfloat16(),
                torch.rand(5, 4, device="cuda"),
                op.layer_id,
            )
            meta = V41PairMetadata(
                torch.tensor([0, 0, 0, 0, -999], device="cuda"),
                torch.tensor([0, 1, 2, 3, -999], device="cuda"),
                torch.tensor([True, True, True, True, False], device="cuda"),
                torch.tensor([-1, 0, -1, 2, 99999], device="cuda"),
                torch.full((5,), -1, dtype=torch.int64, device="cuda"),
                torch.tensor([0, 1, 2, 3, 99999], device="cuda"),
            )
            state.residual[-1] = float("nan")
            state.previous_pre_mix[-1] = float("nan")
            pair = V41PairState(torch.zeros(2, 4096, device="cuda"))

            def cache(tag):
                return V41PagedCache(
                    torch.zeros(
                        2,
                        dsv41_page_layouts(1)[tag].stride_bytes,
                        dtype=torch.uint8,
                        device="cuda",
                    ),
                    tag,
                )

            swa, global_kv, index_k = (
                cache("swa_kv"),
                cache("hca_kv" if ratio == 1 else "csa_kv"),
                cache("indexer_kv"),
            )
            writes = torch.tensor([0, 1, 2, 3, -999], device="cuda")
            global_writes = torch.tensor(
                [0, 1, 2, 3, -999] if ratio == 1 else [0, 0, 1, 1, -999], device="cuda"
            )
            index_writes = global_writes.clone()
            reads = torch.full((5, 128), -1, dtype=torch.int64, device="cuda")
            for row in range(4):
                reads[row, : row + 1] = torch.arange(row + 1, device="cuda")
            pages = torch.zeros(5, 1, dtype=torch.int32, device="cuda")
            indexer = V41Indexer(op.attention.sources, max_positions=256 // ratio)
            attention = V41SparseAttention()

            def run():
                kwargs = (
                    {"pair_state": pair, "pair_metadata": meta} if ratio == 2 else {}
                )
                frame = op.prepare(state, meta.positions, meta.valid_rows, **kwargs)
                frame.kv.publish(
                    swa,
                    writes,
                    global_cache=global_kv,
                    global_slots=global_writes,
                    index_cache=index_k,
                    index_slots=index_writes,
                )
                selected = indexer(
                    frame.query.index_query,
                    frame.query.index_weights,
                    index_k,
                    pages,
                    meta.positions,
                    valid_rows=meta.valid_rows,
                )
                slots = torch.where(
                    selected.topk >= 0,
                    selected.topk.long()
                    + pages[:, :1].long() * global_kv.layout.entries,
                    -1,
                )
                attended = attention(
                    frame.query.query,
                    op.attention.sinks,
                    swa,
                    reads,
                    valid_rows=meta.valid_rows,
                    global_cache=global_kv,
                    global_slots=slots,
                )
                return op.finish(frame, attended)

            for _ in range(3):
                run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = run()
            for live, page in ((True, 1), (False, 0), (True, 0)):
                state.residual[:4].normal_()
                state.previous_pre_mix[:4].uniform_()
                meta.valid_rows[:4] = live
                if not live:
                    state.residual.fill_(float("nan"))
                    state.previous_pre_mix.fill_(float("nan"))
                pages.fill_(page)
                logical = torch.arange(4, device="cuda") // ratio
                global_writes[:4] = logical + page * global_kv.layout.entries
                index_writes[:4] = logical + page * index_k.layout.entries
                graph.replay()
                expected = run()
                self.assertEqual(actual.next_layer_id, op.layer_id + 1)
                torch.testing.assert_close(
                    actual.residual, expected.residual, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    actual.previous_pre_mix, expected.previous_pre_mix, rtol=0, atol=0
                )
                self.assertTrue(bool(torch.isfinite(actual.residual).all()))
                self.assertTrue(bool((actual.residual[~meta.valid_rows] == 0).all()))
                self.assertTrue(
                    bool((actual.previous_pre_mix[~meta.valid_rows] == 0).all())
                )
            del graph, op


if __name__ == "__main__":
    unittest.main()
