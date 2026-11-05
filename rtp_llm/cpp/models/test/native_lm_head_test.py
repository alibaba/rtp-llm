import datetime
import gc
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.cpp.models.test.pywrapped_model_cache_store_integration_test import (
    ForwardModel,
    _extension,
)


def indexes(rows):
    return torch.arange(rows, dtype=torch.int64).pin_memory()


def reference(hidden, weight, scale=None):
    if weight.dtype != torch.float8_e4m3fn:
        rows = hidden.to(weight.dtype)
        if weight.dtype == torch.bfloat16:
            return rows.float().mm(weight.float().t())
        return rows.mm(weight.t()).float()
    rows = hidden.to(torch.bfloat16).contiguous()
    if not rows.shape[0]:
        return torch.empty((0, weight.shape[0]), device=rows.device)
    if weight.shape[0] % 16 or weight.shape[1] % 16:
        return rows.mm((weight.float() * scale).bfloat16().t()).float()
    # The native dynamic quantizer uses only selected, live BF16 rows.
    activation_scale = (rows.float().abs().amax() / 448).reshape(1)
    quantized = torch.zeros(
        ((rows.shape[0] + 15) // 16 * 16, rows.shape[1]),
        device=rows.device,
        dtype=torch.float8_e4m3fn,
    )
    quantized[: rows.shape[0]].copy_(
        (rows.float() * activation_scale.clamp_min(1e-9).reciprocal())
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    return torch._scaled_mm(
        quantized,
        weight.t(),
        scale_a=activation_scale,
        scale_b=scale,
        out_dtype=torch.bfloat16,
    )[: rows.shape[0]].float()


def make_head(weight, scale=None, **kwargs):
    model = ForwardModel()

    def forbidden(_):
        raise AssertionError("native LM head called a Python output-head hook")

    model.compute_logits = forbidden
    return _extension.make_native_lm_head(model, weight, scale, **kwargs)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class NativeLmHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1543)

    def test_converter_preserves_head_format_under_mixed_moe_quantization(self):
        for dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
            torch.float8_e4m3fn,
        ):
            weight = torch.randn(32, 32, device="cuda").to(dtype)
            scale = (
                torch.tensor(0.019713, device="cuda")
                if dtype == torch.float8_e4m3fn
                else None
            )
            for mixed in (False, True):
                with self.subTest(dtype=dtype, mixed=mixed):
                    actual, actual_scale = _extension.convert_native_lm_head(
                        weight, scale, mixed
                    )
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(actual.data_ptr(), weight.data_ptr())
                    if scale is None:
                        self.assertIsNone(actual_scale)
                    else:
                        self.assertEqual(actual_scale.shape, (1,))
                        self.assertEqual(actual_scale.data_ptr(), scale.data_ptr())

    def test_invalid_scale_or_weight_fails_during_conversion(self):
        weight = torch.ones(32, 32, device="cuda").to(torch.float8_e4m3fn)
        for scale in (
            None,
            torch.tensor([0.0], device="cuda"),
            torch.tensor([-1.0], device="cuda"),
            torch.tensor([float("nan")], device="cuda"),
            torch.tensor([float("inf")], device="cuda"),
            torch.ones(2, device="cuda"),
            torch.ones(1, device="cuda", dtype=torch.float64),
            torch.ones(1),
        ):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(RuntimeError, "FP8 LM head requires"):
                    _extension.convert_native_lm_head(weight, scale)
        for invalid in (weight.t(), weight[:0], weight.reshape(-1), weight.bfloat16()):
            with self.assertRaisesRegex(RuntimeError, "FP8 LM head requires"):
                _extension.convert_native_lm_head(invalid, torch.ones(1, device="cuda"))

    def test_fp8_projection_handles_live_rows_padding_and_unaligned_weights(self):
        for vocab, width in ((32, 32), (128, 2560), (33, 32), (32, 30)):
            weight = (torch.randn(vocab, width, device="cuda") * 0.1).to(
                torch.float8_e4m3fn
            )
            scale = torch.tensor([0.019713], device="cuda")
            runner = make_head(weight, scale)
            for rows in (0, 1, 3, 4, 7, 16, 17, 31, 32, 64, 257):
                for dtype in (torch.bfloat16, torch.float32):
                    with self.subTest(vocab=vocab, width=width, rows=rows, dtype=dtype):
                        hidden = torch.randn(rows, width, device="cuda", dtype=dtype)
                        actual = runner(hidden, indexes(rows), compact=True)["logits"]
                        self.assertEqual(actual.dtype, torch.float32)
                        self.assertTrue(actual.is_contiguous())
                        torch.testing.assert_close(
                            actual, reference(hidden, weight, scale), rtol=0, atol=0
                        )

    def test_row_selection_precedes_fp8_quantization(self):
        weight = torch.randn(32, 32, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.tensor([0.03], device="cuda")
        runner = make_head(weight, scale)
        hidden = torch.randn(8, 32, device="cuda", dtype=torch.bfloat16)
        hidden[-1].mul_(10000)
        selected = torch.tensor([3, 0, 1], dtype=torch.int64).pin_memory()
        live = hidden[selected.cuda()]
        expected = reference(live, weight, scale)
        full = runner(hidden, selected)["logits"]
        compact = runner(live, indexes(3), compact=True)["logits"]
        torch.testing.assert_close(full, expected, rtol=0, atol=0)
        torch.testing.assert_close(compact, expected, rtol=0, atol=0)
        all_outputs = runner(hidden, selected, need_all_logits=True)
        all_reference = reference(hidden, weight, scale)
        torch.testing.assert_close(
            all_outputs["all_logits"], all_reference, rtol=0, atol=0
        )
        torch.testing.assert_close(
            all_outputs["logits"], all_reference[selected.cuda()], rtol=0, atol=0
        )
        self.assertFalse(torch.equal(expected, all_reference[selected.cuda()]))

    def test_zero_activation_and_noncontiguous_hidden(self):
        weight = torch.randn(32, 32, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.tensor([0.02], device="cuda")
        runner = make_head(weight, scale)
        hidden = torch.randn(32, 7, device="cuda", dtype=torch.bfloat16).t()
        self.assertFalse(hidden.is_contiguous())
        for values in (hidden, hidden * 0):
            actual = runner(values, indexes(7), compact=True)["logits"]
            torch.testing.assert_close(
                actual, reference(values, weight, scale), rtol=0, atol=0
            )

    def test_original_native_weight_dtypes_and_vocabulary_trimming(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            weight = torch.randn(32, 32, device="cuda").to(dtype)
            hidden = torch.randn(5, 32, device="cuda", dtype=torch.bfloat16)
            runner = make_head(weight, vocab_size=29)
            selected = torch.tensor([4, 1], dtype=torch.int64).pin_memory()
            actual = runner(hidden, selected, need_all_logits=True)
            expected = reference(hidden, weight)[:, :29]
            torch.testing.assert_close(
                actual["all_logits"], expected, rtol=1e-5, atol=1e-5
            )
            torch.testing.assert_close(
                actual["logits"], expected[selected.cuda()], rtol=1e-5, atol=1e-5
            )

    def test_fp8_graph_replay_recomputes_scale_from_current_values(self):
        weight = torch.randn(128, 2560, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.tensor([0.02], device="cuda")
        runner = make_head(weight, scale)
        for rows in (1, 3, 17):
            hidden = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
            selected = indexes(rows)
            for _ in range(3):
                runner(hidden, selected, compact=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = runner(hidden, selected, compact=True)["logits"]
            pointer = actual.data_ptr()
            for magnitude in (0.0, 0.01, 1.0, 100.0):
                hidden.normal_().mul_(magnitude)
                graph.replay()
                torch.testing.assert_close(
                    actual, reference(hidden, weight, scale), rtol=0, atol=0
                )
                self.assertEqual(actual.data_ptr(), pointer)
            graph.reset()
        del runner
        gc.collect()


def _tp_worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=90),
    )

    def unexpected(*_):
        raise AssertionError("LM head requested an unexpected collective")

    def gather(recv, mode, send, inplace):
        assert mode == 0 and len(recv) == 1 and not send
        dest = recv[0].view(-1)
        size = dest.numel() // 2
        local = dest[rank * size : (rank + 1) * size].clone()
        dist.all_gather_into_tensor(dest, local)

    _extension.register_comm_ops(unexpected, unexpected, gather)
    try:
        for dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
            torch.float8_e4m3fn,
        ):
            torch.manual_seed(1543)
            hidden = torch.randn(7, 32, device="cuda", dtype=torch.bfloat16)
            shards = [torch.randn(32, 32, device="cuda").to(dtype) for _ in range(2)]
            scales = [torch.tensor([value], device="cuda") for value in (0.02, 0.07)]
            scale = scales[rank] if dtype == torch.float8_e4m3fn else None
            runner = make_head(
                shards[rank], scale, tp_size=2, tp_rank=rank, vocab_size=61
            )
            selected = torch.tensor([5, 0, 2], dtype=torch.int64).pin_memory()
            for need_all in (False, True):
                live = hidden if need_all else hidden[selected.cuda()]
                expected = torch.cat(
                    [
                        reference(
                            live,
                            shards[i],
                            scales[i] if dtype == torch.float8_e4m3fn else None,
                        )
                        for i in range(2)
                    ],
                    dim=1,
                )[:, :61]
                outputs = runner(hidden, selected, need_all_logits=need_all)
                torch.testing.assert_close(
                    outputs["logits"],
                    expected[selected.cuda()] if need_all else expected,
                    rtol=1e-5,
                    atol=1e-5,
                )
                if need_all:
                    torch.testing.assert_close(
                        outputs["all_logits"], expected, rtol=1e-5, atol=1e-5
                    )
            del runner
        dist.barrier()
    finally:
        _extension.clear_comm_ops()
        dist.destroy_process_group()


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class NativeLmHeadTpTest(unittest.TestCase):
    def test_real_tp_collective_preserves_rank_order_and_per_rank_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _tp_worker,
                args=(str(Path(directory) / "rendezvous"),),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
