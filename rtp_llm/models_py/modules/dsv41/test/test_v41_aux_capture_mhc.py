"""Aux capture observes layer entry without forcing a materialized mHC boundary."""

import os
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.decode import forward as decode
from rtp_llm.models_py.modules.dsv41.hc import decode_transition as transition
from rtp_llm.models_py.modules.dsv41.hc.delayed import DelayedHCHead
from rtp_llm.models_py.modules.dsv41.test.test_v41_cross_layer_mhc import (
    make_blocks,
    reference_transition,
)
from rtp_llm.models_py.modules.dsv41.transformer import V4Transformer


def capture_model(blocks, hidden, captures):
    rows = hidden.shape[0] * hidden.shape[1]
    dim = hidden.shape[-1]
    model = SimpleNamespace(
        layers=blocks,
        hc_mult=4,
        capture_aux_hidden_layer_ids=captures,
        args=SimpleNamespace(vocab_size=1024, n_hash_layers=0),
        _has_visual_tokens=False,
        _mtp_hidden_buffer=torch.empty(
            rows, len(captures) * dim, device=hidden.device, dtype=hidden.dtype
        ),
        norm=blocks[-1].attn_norm,
        _hc_head_reduce=DelayedHCHead(blocks[-1].ffn_hc).head,
    )
    model.capture_aux_hidden = partial(V4Transformer.capture_aux_hidden, model)
    model._note_aux_hidden_rows = partial(V4Transformer._note_aux_hidden_rows, model)
    return model


def run_model(model, hidden, ids, is_graph=False):
    return decode.forward_layers(
        model,
        None,
        ids,
        SimpleNamespace(
            batch_size=hidden.shape[0],
            q_len_per_req=hidden.shape[1],
            is_cuda_graph=is_graph,
        ),
        prepare_hidden_fn=lambda **kwargs: hidden.clone(),
    )


class AuxCaptureMHCCPUTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(
            os.environ, {"DSV41_FUSED_CROSS_LAYER_MHC": "1", "DSV41_MEGA_MHC": "1"}
        )
        env.start()
        self.addCleanup(env.stop)
        trace = patch.object(decode._rt, "ENABLED", False)
        trace.start()
        self.addCleanup(trace.stop)

    def check_capture(self, engram=(), backend=reference_transition):
        hidden = torch.randn(2, 5, 4, 8, dtype=torch.bfloat16)
        ids = torch.arange(10)
        captures = (3, 1, 0, 2)
        blocks = make_blocks(count=4, engram_layers=engram)
        candidate = capture_model(blocks, hidden, captures)
        baseline_blocks = make_blocks(count=4, engram_layers=engram)
        value = hidden.clone()
        expected = {}
        for index, block in enumerate(baseline_blocks):
            expected[index] = value.mean(-2).reshape(10, 8).clone()
            value = block.forward_decode(value, None, ids)
        with patch.object(
            transition, "try_fused_post_pre", side_effect=backend
        ) as fused:
            actual = run_model(candidate, hidden, ids)
        for segment, index in enumerate(captures):
            torch.testing.assert_close(
                candidate._mtp_hidden_buffer[:, segment * 8 : (segment + 1) * 8],
                expected[index],
                rtol=0,
                atol=0,
            )
        ref = DelayedHCHead(baseline_blocks[-1].ffn_hc).head(value)
        ref = baseline_blocks[-1].attn_norm(ref.reshape(-1, 8)).reshape(2, 5, 8)
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
        self.assertEqual(candidate._mtp_hidden_valid_tokens, 10)
        self.assertEqual(fused.call_count, 3 - len(engram))

    def test_first_last_adjacent_and_reordered_capture_keep_fusion(self):
        self.check_capture()

    def test_engram_capture_is_before_injection(self):
        self.check_capture(engram=(1,))

    def test_backend_fallback_still_captures_materialized_entry(self):
        self.check_capture(backend=lambda *args, **kwargs: None)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class AuxCaptureMHCCudaTest(unittest.TestCase):
    @torch.inference_mode()
    def test_graph_updates_captured_entry_after_real_mega_mhc(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("Blackwell required")
        hidden = torch.randn(64, 6, 4, 5120, device="cuda", dtype=torch.bfloat16)
        ids = torch.arange(384, device="cuda")
        model = capture_model(
            make_blocks(5120, "cuda", count=4, engram_layers=(1,)),
            hidden,
            (3, 1, 0, 2),
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run_model(model, hidden, ids)
        torch.cuda.current_stream().wait_stream(stream)
        captured = []
        native = transition.try_fused_post_pre

        def observe(*args, **kwargs):
            prepared = native(*args, **kwargs)
            captured.append(prepared is not None)
            return prepared

        graph = torch.cuda.CUDAGraph()
        with patch.object(transition, "try_fused_post_pre", side_effect=observe):
            with torch.cuda.graph(graph, stream=stream):
                output = run_model(model, hidden, ids, is_graph=True)
        self.assertEqual(captured, [True, True])
        for scale in (1.0, -0.75, 2.0):
            hidden.copy_(torch.randn_like(hidden) * scale)
            ids.add_(1)
            expected = run_model(model, hidden, ids).clone()
            expected_aux = model._mtp_hidden_buffer.clone()
            model._mtp_hidden_buffer.zero_()
            graph.replay()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                model._mtp_hidden_buffer, expected_aux, rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
