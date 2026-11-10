"""Cache bytes must match the installed SGL kernel, including scale rounding."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.ppu.mla_kv_quant import concat_and_cache_mla_fp8
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.ops import KvCacheDataType


class MlaKvQuantTest(TestCase):
    @classmethod
    def setUpClass(cls):
        data = json.loads(
            Path(__file__).with_name("test_mla_kv_sgl_golden.json").read_text()
        )
        cls.ckv = torch.tensor(
            data["input_ckv_u16"], dtype=torch.uint16, device="cuda"
        ).view(torch.bfloat16)
        cls.kpe = torch.tensor(
            data["input_kpe_u16"], dtype=torch.uint16, device="cuda"
        ).view(torch.bfloat16)
        cls.golden = torch.tensor(data["expected_u8"], dtype=torch.uint8, device="cuda")

    def test_sgl_bytes_scattered_slots_and_noncontiguous_rows(self):
        ckv = torch.empty(6, 1024, device="cuda", dtype=torch.bfloat16)[:, :512]
        kpe = torch.empty(6, 128, device="cuda", dtype=torch.bfloat16)[:, :64]
        ckv.copy_(self.ckv)
        kpe.copy_(self.kpe)
        slots = torch.tensor(
            [65, 130, -1, 199, 64, 191], device="cuda", dtype=torch.int64
        )
        cache = torch.full((4, 64, 656), 137, device="cuda", dtype=torch.uint8)
        expected = cache.clone()
        for row, slot in enumerate(slots.cpu().tolist()):
            if slot >= 0:
                expected.view(-1, 656)[slot].copy_(self.golden[row])
        writer = MlaKVCacheWriteOp(KvCacheDataType.FP8)
        self.assertTrue(writer._ppu_fp8)
        writer.forward(
            ckv,
            kpe,
            SimpleNamespace(kv_cache_base=cache),
            SimpleNamespace(slot_mapping=slots),
        )
        self.assertTrue(torch.equal(cache, expected))

    def test_graph_reads_new_values_and_new_slot_mapping(self):
        ckv, kpe = self.ckv.clone(), self.kpe.clone()
        slots = torch.arange(64, 70, dtype=torch.int64, device="cuda")
        cache = torch.zeros(4, 64, 656, device="cuda", dtype=torch.uint8)
        writer = MlaKVCacheWriteOp(KvCacheDataType.FP8)
        kv = SimpleNamespace(kv_cache_base=cache)
        fmha = SimpleNamespace(slot_mapping=slots)
        for _ in range(3):
            writer.forward(ckv, kpe, kv, fmha)
        torch.cuda.synchronize()
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=stream):
            writer.forward(ckv, kpe, kv, fmha)
        for multiplier, start in ((2, 128), (4, 192)):
            ckv.copy_(self.ckv * multiplier)
            kpe.copy_(self.kpe * multiplier)
            slots.copy_(
                torch.arange(start, start + 6, dtype=torch.int64, device="cuda")
            )
            cache.zero_()
            expected = self.golden.clone()
            scales = expected[:, 512:528].contiguous().view(torch.float32) * multiplier
            expected[:, 512:528] = scales.view(torch.uint8)
            expected[:, 528:] = kpe.view(torch.uint8)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(cache.view(-1, 656)[start : start + 6], expected)
            )
            self.assertEqual(int(torch.count_nonzero(cache.view(-1, 656)[:start])), 0)

    def test_zero_tiles_and_empty_input_remain_finite(self):
        cache = torch.zeros(2, 64, 656, device="cuda", dtype=torch.uint8)
        ckv = torch.zeros(1, 512, device="cuda", dtype=torch.bfloat16)
        kpe = torch.ones(1, 64, device="cuda", dtype=torch.bfloat16)
        slots = torch.tensor([64], device="cuda", dtype=torch.int64)
        concat_and_cache_mla_fp8(ckv, kpe, cache, slots)
        packed = cache[1, 0]
        self.assertEqual(int(torch.count_nonzero(packed[:512])), 0)
        scales = packed[512:528].view(torch.float32)
        self.assertTrue(bool(torch.isfinite(scales).all() and (scales > 0).all()))
        before = cache.clone()
        concat_and_cache_mla_fp8(ckv[:0], kpe[:0], cache, slots[:0])
        self.assertTrue(torch.equal(before, cache))

    def test_padded_inputs_only_write_rows_present_in_slot_mapping(self):
        slots = torch.tensor([65, 129, 193], device="cuda", dtype=torch.int64)
        cache = torch.full((4, 64, 656), 137, device="cuda", dtype=torch.uint8)
        expected = cache.clone()
        for row, slot in enumerate((65, 129, 193)):
            expected.view(-1, 656)[slot].copy_(self.golden[row])
        concat_and_cache_mla_fp8(self.ckv, self.kpe, cache, slots)
        self.assertTrue(torch.equal(cache, expected))
        before = cache.clone()
        concat_and_cache_mla_fp8(self.ckv, self.kpe, cache, slots[:0])
        self.assertTrue(torch.equal(before, cache))

    def test_non_ppu_writer_retains_native_dispatch(self):
        module = "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op."
        with patch(module + "is_ppu", return_value=False):
            writer = MlaKVCacheWriteOp(KvCacheDataType.FP8)
        slots = torch.arange(6, dtype=torch.int64, device="cuda")
        cache = torch.zeros(1, 64, 656, device="cuda", dtype=torch.uint8)
        with patch(module + "compute_ops.concat_and_cache_mla") as native:
            writer.forward(
                self.ckv,
                self.kpe,
                SimpleNamespace(kv_cache_base=cache),
                SimpleNamespace(slot_mapping=slots),
            )
        native.assert_called_once()
        self.assertEqual(native.call_args.args[4], "fp8_ds_mla")


if __name__ == "__main__":
    main()
