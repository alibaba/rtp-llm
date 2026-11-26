import os
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.chunking import (
    MegaMoePrefillChunker,
)
from rtp_llm.models_py.modules.qwen4_exp.moe.shared_overlap import Qwen4MoeLayer
from rtp_llm.ops import MoeConfig, ParallelismConfig, RoleType, SpeculativeType


def _uneven_prefill_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        local = 9 if rank else 0
        inputs = torch.arange(local, dtype=torch.float32).view(-1, 1)
        group = dist.group.WORLD
        plan = MegaMoePrefillChunker(4, group)
        all_reduce = dist.all_reduce
        with patch.object(dist, "all_reduce", wraps=all_reduce) as exchange:
            with plan.plan(local, inputs.device):
                for _ in range(2):
                    layer = MegaMoePrefillChunker(4, group)
                    scratch = torch.empty((4, 1))

                    def run_chunk(start, end):
                        result = scratch[: end - start]
                        result.copy_(inputs[start:end] * 3)
                        return result

                    forward_once = Mock(side_effect=AssertionError("expected chunks"))
                    run = Mock(side_effect=run_chunk)
                    actual = layer.forward(inputs, forward_once, lambda _: run)
                    expected = [(0, 4), (4, 8), (8, 9)] if rank else [(0, 0)] * 3
                    assert [call.args for call in run.call_args_list] == expected
                    torch.testing.assert_close(actual, inputs * 3)
            assert exchange.call_count == 1
    finally:
        dist.destroy_process_group()


class MegaMoePrefillChunkingTest(unittest.TestCase):
    @unittest.skipUnless(dist.is_gloo_available(), "Gloo is required")
    def test_one_exchange_and_empty_rank_with_real_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _uneven_prefill_worker, args=(f"file://{directory}/store",), nprocs=2
            )

    def test_qwen4_routes_pack_and_shared_gates_across_chunks(self):
        for use_fused_pack in (False, True):
            with self.subTest(fused=use_fused_pack):
                self._check_qwen4_routing(use_fused_pack)

    def _check_qwen4_routing(self, use_fused_pack):
        layer = Qwen4MoeLayer.__new__(Qwen4MoeLayer)
        torch.nn.Module.__init__(layer)
        group = Mock(spec=dist.ProcessGroup)
        group.size.return_value = 72
        layer.prefill_chunker = MegaMoePrefillChunker(4, group)
        layer._uses_shared_expert_gates = use_fused_pack
        layer._uses_fused_gate_pack = Mock(return_value=use_fused_pack)
        layer.shared_expert = None
        layer.shared_expert_gate = None
        layer.gate = Mock(side_effect=lambda x: x + 10)
        layer._routed_shared_gates = Mock(
            side_effect=lambda x, *, use_static_buffer: x[:, 0] + 20
        )
        layer._select_routing = Mock(side_effect=lambda scores: (scores, scores.long()))
        scratch = torch.empty((4, 1))
        seen = []

        def routed(x, weights, ids, extra):
            self.assertIsNone(extra)
            torch.testing.assert_close(weights, x + 10)
            torch.testing.assert_close(ids, (x + 10).long())
            return run(x)

        def packed(x, scores, extra):
            torch.testing.assert_close(scores, x + 10)
            torch.testing.assert_close(extra["shared_expert_gates"], x[:, 0] + 20)
            return run(x)

        def run(x):
            seen.append(x.size(0))
            scratch[: x.size(0)].copy_(x * 2)
            return scratch[: x.size(0)]

        layer._forward_with_routing = routed
        layer._forward_gate_pack = packed
        inputs = torch.arange(5, dtype=torch.float32).view(-1, 1)

        def exchange(count, *, op, group):
            self.assertEqual(group.size(), 72)
            self.assertEqual(op, dist.ReduceOp.MAX)
            count.fill_(9)

        with patch.object(dist, "all_reduce", side_effect=exchange) as collective:
            with layer.prefill_chunker.plan(inputs.size(0), inputs.device):
                actual = layer(inputs)
        torch.testing.assert_close(actual, inputs * 2)
        self.assertEqual(seen, [4, 1, 0])
        self.assertEqual(collective.call_count, 1)
        layer.gate.assert_called_once()
        self.assertEqual(layer._select_routing.call_count, int(not use_fused_pack))
        self.assertEqual(layer._routed_shared_gates.call_count, int(use_fused_pack))
        if use_fused_pack:
            layer._routed_shared_gates.assert_called_once_with(
                inputs, use_static_buffer=False
            )

    def test_model_skips_decode_verify_commit_and_capture(self):
        chunker = Mock()
        model = SimpleNamespace(
            config=SimpleNamespace(is_mtp=True),
            _mega_moe_prefill_chunker=chunker,
        )
        token_ids = torch.ones(13, dtype=torch.long)
        inputs = SimpleNamespace(input_ids=token_ids, is_mtp_decode_commit=False)
        primary = SimpleNamespace(is_prefill=True, is_target_verify=False)
        module = "rtp_llm.models_py.model_desc.qwen4_exp"
        with (
            patch(f"{module}.get_primary_attention_inputs", return_value=primary),
            patch(f"{module}._is_cuda_graph_forward", return_value=False) as graph,
        ):
            for is_prefill, verify, commit, capture in (
                (False, False, False, False),
                (True, True, False, False),
                (True, False, True, False),
                (True, False, False, True),
            ):
                primary.is_prefill = is_prefill
                primary.is_target_verify = verify
                inputs.is_mtp_decode_commit = commit
                graph.return_value = capture
                with Qwen4ExpModel._mega_moe_prefill_scope(model, inputs, None):
                    pass
            chunker.plan.assert_not_called()
            graph.return_value = False
            Qwen4ExpModel._mega_moe_prefill_scope(model, inputs, None)
            chunker.plan.assert_called_once_with(13, token_ids.device)

    def test_capacity_preserves_unchunked_mtp_and_decode(self):
        for strategy in ("mega_moe", "mega_moe_se"):
            with self.subTest(strategy=strategy):
                model = ModelConfig()
                model.model_type = "qwen4_exp"
                model._qwen4_mxfp4_source = True
                model.expert_num = 864
                model.moe_prefill_max_tokens_per_rank = 32768
                parallel = ParallelismConfig()
                parallel.dp_size = parallel.ep_size = parallel.world_size = 72
                moe = MoeConfig()
                moe.moe_strategy = strategy
                engine = SimpleNamespace(
                    moe_config=moe,
                    parallelism_config=parallel,
                    pd_sep_config=SimpleNamespace(role_type=RoleType.PREFILL),
                    runtime_config=SimpleNamespace(
                        max_generate_batch_size=8,
                        fifo_scheduler_config=SimpleNamespace(
                            pdfusion_coord_mode="off"
                        ),
                    ),
                    hw_kernel_config=SimpleNamespace(
                        enable_cuda_graph=True,
                        generation_prefill_capture_token_buckets=[],
                        decode_capture_batch_sizes=[16],
                    ),
                    sp_config=SimpleNamespace(
                        type=SpeculativeType.MTP, gen_num_per_cycle=3
                    ),
                )
                with patch.dict(os.environ, {"RTP_MEGAMOE_CHUNK_TOKENS": "32"}):
                    ModelFactory._configure_qwen4_mega_moe_prefill(engine, model)
                adapter = MoEConfigAdapter(model, parallel, moe)
                self.assertEqual(adapter.mega_moe_prefill_chunk_tokens, 32)
                self.assertEqual(adapter.max_tokens_per_rank, 64)
                self.assertEqual(adapter.prefill_max_tokens_per_rank, 32768)


if __name__ == "__main__":
    unittest.main()
