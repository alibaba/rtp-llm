"""Scheduling contracts; actual NCCL execution is covered by the CUDA test."""

import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.modules.factory.fused_moe.utils.config import TpMoeChunkConfig


def make_layer(chunks=4, mode="overlap"):
    layer = GenericMoeLayer.__new__(GenericMoeLayer)
    torch.nn.Module.__init__(layer)
    layer.tp_chunk_config = TpMoeChunkConfig(chunks, mode, 1)
    layer.use_unified_tp_allreduce = True
    layer.use_ep_shared_allreduce = False
    layer.ffn_tp_size = 2
    layer.parallelism_config = SimpleNamespace(
        dp_size=1, prefill_cp_config=SimpleNamespace(is_enabled=lambda: False)
    )
    layer.top_k = 2
    layer.correction_bias = None
    layer.fake_balance_expert = None
    layer.gate = Mock(side_effect=lambda x: x[:, :4])

    def topk(scores, ids, weights):
        values, indices = scores.topk(2, dim=-1)
        ids.copy_(indices)
        weights.copy_(values.softmax(dim=-1))

    layer.select_topk = Mock(side_effect=topk)

    def routed(*, hidden_states, topk_weights, topk_ids, **kwargs):
        assert kwargs["skip_tp_allreduce"]
        scale = (topk_weights * (topk_ids + 1)).sum(-1, keepdim=True)
        return hidden_states * scale

    layer.fused_moe = Mock(side_effect=routed)
    layer.fused_moe.topk_ids_dtype = torch.int32
    layer.shared_expert = Mock(side_effect=lambda x, **_: x.square() * 0.1)
    layer.shared_expert_gate = Mock(side_effect=lambda x: x[:, :1])
    layer.sigmoid_gate_scale_add = Mock(
        side_effect=lambda gate, shared, out: out.add_(gate.sigmoid() * shared)
    )
    return layer


class Pending:
    def __init__(self, tensor, trace):
        self.tensor = tensor
        self.trace = trace
        self.wait_count = 0

    def wait(self):
        self.trace.append("wait")
        self.wait_count += 1
        if self.wait_count == 1:
            self.tensor.mul_(2)
        return self.tensor


class GenericMoeTpChunkingTest(unittest.TestCase):
    def test_modes_match_unified_reference_with_gated_shared_and_tail(self):
        torch.manual_seed(42)
        for chunks in (2, 4):
            for mode in ("serial", "overlap"):
                for tokens in (4, 5, 17, 31):
                    with self.subTest(chunks=chunks, mode=mode, tokens=tokens):
                        layer = make_layer(chunks, mode)
                        x = torch.randn(tokens, 8)
                        original = x.clone()
                        trace, handles = [], []
                        compute = layer._forward_impl

                        def partial(*args, **kwargs):
                            trace.append("compute")
                            return compute(*args, **kwargs)

                        def launch(tensor, group):
                            trace.append("launch")
                            handle = Pending(tensor, trace)
                            handles.append(handle)
                            return handle

                        with patch(
                            "rtp_llm.models_py.model_desc.generic_moe.all_reduce",
                            side_effect=lambda tensor, **_: tensor * 2,
                        ):
                            expected = layer._forward_impl(x)
                        with patch.object(
                            layer, "_forward_impl", side_effect=partial
                        ), patch(
                            "rtp_llm.models_py.model_desc.generic_moe.all_reduce_async",
                            side_effect=launch,
                        ):
                            actual = layer._forward_tp_chunks(x)
                        torch.testing.assert_close(actual, expected)
                        torch.testing.assert_close(x, original)
                        self.assertTrue(all(h.wait_count == 1 for h in handles))
                        self.assertEqual(
                            len({h.tensor.data_ptr() for h in handles}), len(handles)
                        )
                        if mode == "serial":
                            self.assertEqual(
                                trace, ["compute", "launch", "wait"] * len(handles)
                            )
                        else:
                            self.assertEqual(
                                trace,
                                ["compute", "launch"] * len(handles)
                                + ["wait"] * len(handles),
                            )

    def test_compute_exception_joins_already_launched_buffers(self):
        layer = make_layer()
        handles = []

        def launch(tensor, group):
            handle = Pending(tensor, [])
            handles.append(handle)
            return handle

        with patch.object(
            layer,
            "_forward_impl",
            side_effect=[torch.ones(2, 8), RuntimeError("compute")],
        ), patch(
            "rtp_llm.models_py.model_desc.generic_moe.all_reduce_async",
            side_effect=launch,
        ):
            with self.assertRaisesRegex(RuntimeError, "compute"):
                layer._forward_tp_chunks(torch.ones(8, 8))
        self.assertEqual(len(handles), 1)
        self.assertEqual(handles[0].wait_count, 1)

    def test_chunks_reuse_full_batch_routing_and_shared_gate(self):
        layer = make_layer()
        x = torch.randn(13, 8)
        with patch(
            "rtp_llm.models_py.model_desc.generic_moe.all_reduce_async",
            side_effect=lambda tensor, _: Pending(tensor, []),
        ):
            layer._forward_tp_chunks(x)
        layer.gate.assert_called_once_with(x)
        layer.select_topk.assert_called_once()
        layer.shared_expert_gate.assert_called_once_with(x)
        self.assertEqual(layer.fused_moe.call_count, 4)
        for name, position in (("topk_ids", 1), ("topk_weights", 2)):
            self.assertTrue(
                torch.equal(
                    torch.cat(
                        [call.kwargs[name] for call in layer.fused_moe.call_args_list]
                    ),
                    layer.select_topk.call_args.args[position],
                )
            )
        self.assertTrue(
            torch.equal(
                torch.cat(
                    [
                        call.args[0]
                        for call in layer.sigmoid_gate_scale_add.call_args_list
                    ]
                ),
                x[:, :1],
            )
        )

    def test_default_forward_keeps_original_path(self):
        layer = make_layer()
        x = torch.ones(8, 8)
        with patch.object(
            layer, "_forward_impl", return_value=x
        ) as original, patch.object(layer, "_forward_tp_chunks") as chunked:
            self.assertIs(layer(x), x)
        original.assert_called_once_with(x)
        chunked.assert_not_called()

    def test_wait_failure_attempts_all_cleanup_and_preserves_original_error(self):
        layer = make_layer()
        handles = [Mock() for _ in range(4)]
        handles[0].wait.side_effect = [
            RuntimeError("original wait failed"),
            RuntimeError("cleanup wait failed"),
        ]
        with patch.object(
            layer, "_forward_impl", side_effect=lambda x, **_: x.clone()
        ), patch(
            "rtp_llm.models_py.model_desc.generic_moe.all_reduce_async",
            side_effect=handles,
        ), patch(
            "rtp_llm.models_py.model_desc.generic_moe.logger.exception"
        ) as log:
            with self.assertRaisesRegex(RuntimeError, "original wait failed"):
                layer._forward_tp_chunks(torch.ones(8, 8))
        self.assertEqual([handle.wait.call_count for handle in handles], [2, 1, 1, 1])
        log.assert_called_once()

    def test_eligibility_rejects_unsupported_execution_modes(self):
        x = SimpleNamespace(shape=(8192, 2048), is_cuda=True)
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ), patch.object(torch.version, "hip", None), patch.dict(
            os.environ, {"RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "0"}
        ):
            layer = make_layer()
            self.assertTrue(layer._can_chunk_tp_prefill(x, True))
            self.assertFalse(layer._can_chunk_tp_prefill(x, False))
            for attr, value in (
                ("use_unified_tp_allreduce", False),
                ("ffn_tp_size", 4),
            ):
                with patch.object(layer, attr, value):
                    self.assertFalse(layer._can_chunk_tp_prefill(x, True))
            with patch.object(layer.parallelism_config, "dp_size", 2):
                self.assertFalse(layer._can_chunk_tp_prefill(x, True))
            with patch.object(
                layer.parallelism_config.prefill_cp_config,
                "is_enabled",
                return_value=True,
            ):
                self.assertFalse(layer._can_chunk_tp_prefill(x, True))
            with patch.dict(os.environ, {"RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "1"}):
                self.assertFalse(layer._can_chunk_tp_prefill(x, True))
            with patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ):
                self.assertFalse(layer._can_chunk_tp_prefill(x, True))
            with patch.object(torch.version, "hip", "7.0"):
                self.assertFalse(layer._can_chunk_tp_prefill(x, True))
            layer.tp_chunk_config = replace(layer.tp_chunk_config, min_tokens=16384)
            self.assertFalse(layer._can_chunk_tp_prefill(x, True))
            layer.tp_chunk_config = TpMoeChunkConfig()
            self.assertFalse(layer._can_chunk_tp_prefill(x, True))

    def test_invalid_experiment_configuration_fails_at_initialization(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(TpMoeChunkConfig.from_env(), TpMoeChunkConfig())
        for key, value in (
            ("MOE_TP_CHUNKS", "3"),
            ("MOE_TP_CHUNKS", "invalid"),
            ("MOE_TP_CHUNK_MODE", "unknown"),
            ("MOE_TP_CHUNK_MIN_TOKENS", "0"),
        ):
            with self.subTest(key=key), patch.dict(
                os.environ, {key: value}, clear=True
            ):
                with self.assertRaises(ValueError):
                    TpMoeChunkConfig.from_env()


if __name__ == "__main__":
    unittest.main()
