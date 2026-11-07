"""Real SGL K-cache bytes, graph updates and IndexerOp dispatch scope."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.ppu.indexer_k_quant import quantize_and_cache_indexer_k
from rtp_llm.models_py.modules.base.cuda import indexer_op


class IndexerKeyQuantTest(TestCase):
    @classmethod
    def setUpClass(cls):
        data = json.loads(
            Path(__file__).with_name("test_indexer_k_sgl_golden.json").read_text()
        )
        cls.keys = torch.tensor(
            data["input_key_u16"], dtype=torch.uint16, device="cuda"
        ).view(torch.bfloat16)
        cls.q = torch.tensor(data["expected_q_u8"], dtype=torch.uint8, device="cuda")
        cls.scales = torch.tensor(
            data["expected_scale_u32"], dtype=torch.uint32, device="cuda"
        ).view(torch.float32)
        cls.real_rows = len(data["real_rows"])

    def expected(self, slots, multiplier=1):
        result = torch.full((6, 64, 132), 137, dtype=torch.uint8, device="cuda")
        flat = result.view(6, -1)
        scales = self.scales * multiplier
        for row, slot in enumerate(slots.cpu().tolist()):
            if slot >= 0:
                page, offset = divmod(slot, 64)
                flat[page, offset * 128 : (offset + 1) * 128] = self.q[row]
                flat[page, 64 * 128 + offset * 4 : 64 * 128 + (offset + 1) * 4] = (
                    scales[row : row + 1].view(torch.uint8)
                )
        return result

    def test_scattered_slots_noncontiguous_rows_and_native_sgl_bytes(self):
        keys = torch.empty(len(self.keys), 256, dtype=torch.bfloat16, device="cuda")[
            :, :128
        ]
        keys.copy_(self.keys)
        slots = torch.arange(len(keys), dtype=torch.int64, device="cuda") * 37 % 384
        slots[::9] = -1
        cache = torch.full((6, 64, 132), 137, dtype=torch.uint8, device="cuda")
        quantize_and_cache_indexer_k(keys, cache, slots)
        self.assertTrue(torch.equal(cache, self.expected(slots)))

    def test_graph_updates_values_slots_and_padded_input(self):
        rows = self.real_rows
        keys = torch.zeros(rows + 5, 128, dtype=torch.bfloat16, device="cuda")
        keys[:rows].copy_(self.keys[:rows])
        slots = torch.arange(rows, dtype=torch.int64, device="cuda")
        cache = torch.full((6, 64, 132), 137, dtype=torch.uint8, device="cuda")
        for _ in range(3):
            quantize_and_cache_indexer_k(keys, cache, slots)
        torch.cuda.synchronize()
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=stream):
            quantize_and_cache_indexer_k(keys, cache, slots)
        for multiplier, start in ((2, 128), (4, 256)):
            keys[:rows].copy_(self.keys[:rows] * multiplier)
            slots.copy_(
                torch.arange(start, start + rows, dtype=torch.int64, device="cuda")
            )
            slots[::7] = -1
            cache.fill_(137)
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(cache, self.expected(slots, multiplier)))

    def test_empty_negative_slots_and_special_values(self):
        cache = torch.full((6, 64, 132), 137, dtype=torch.uint8, device="cuda")
        before = cache.clone()
        slots = torch.full((len(self.keys),), -1, dtype=torch.int64, device="cuda")
        quantize_and_cache_indexer_k(self.keys, cache, slots)
        quantize_and_cache_indexer_k(self.keys[:0], cache, slots[:0])
        self.assertTrue(torch.equal(cache, before))
        slots = torch.arange(len(self.keys), dtype=torch.int64, device="cuda")
        quantize_and_cache_indexer_k(self.keys, cache, slots)
        self.assertTrue(torch.equal(cache, self.expected(slots)))

    def test_indexer_entrypoints_and_native_fallback_scope(self):
        rows = len(self.keys)
        slots = torch.arange(rows, dtype=torch.int64, device="cuda")
        restore = torch.arange(rows, dtype=torch.int64, device="cuda")
        query = torch.zeros(rows, 1, 128, dtype=torch.bfloat16, device="cuda")
        operator = indexer_op.IndexerOp(
            1, 128, 16, 64, scale_fmt="none", query_scale_fmt="ue8m0"
        )
        for method in ("quant_k_only", "quant_k_cp_only", "quant_q_k", "quant_q_k_cp"):
            cache = torch.full((6, 64 * 132), 137, dtype=torch.uint8, device="cuda")
            kv = SimpleNamespace(kv_cache_base=cache, seq_size_per_block=64)
            with patch.object(
                indexer_op, "all_gather", return_value=self.keys
            ), patch.object(
                indexer_op.rtp_llm_ops, "indexer_k_quant_and_cache"
            ) as native:
                args = [self.keys, kv, slots]
                if method.startswith("quant_q"):
                    args.insert(0, query)
                if method.endswith("_cp") or method == "quant_k_cp_only":
                    args.append(restore)
                getattr(operator, method)(*args)
                native.assert_not_called()
            self.assertTrue(
                torch.equal(cache.view(6, 64, 132), self.expected(slots)), method
            )
        for ppu, key_fmt, query_fmt in (
            (False, "none", "ue8m0"),
            (True, "none", None),
            (True, "ue8m0", "ue8m0"),
        ):
            other = indexer_op.IndexerOp(
                1, 128, 16, 64, scale_fmt=key_fmt, query_scale_fmt=query_fmt
            )
            with patch.object(indexer_op, "is_ppu", return_value=ppu), patch.object(
                indexer_op.rtp_llm_ops, "indexer_k_quant_and_cache"
            ) as native:
                other.quant_k_only(self.keys, kv, slots)
                native.assert_called_once()


if __name__ == "__main__":
    main()
