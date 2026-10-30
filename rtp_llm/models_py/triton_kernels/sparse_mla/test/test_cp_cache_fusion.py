import unittest

import torch
from rtp_llm.models_py.triton_kernels.sparse_mla.cp_cache_fusion import (
    try_indexed_cache_write,
    try_read_new_kv,
)
from rtp_llm.ops import compute_ops


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CPCacheFusionTest(unittest.TestCase):
    def test_fp8_cache_storage_view(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
            _as_uint8,
        )

        src = torch.randn(32, 576, device="cuda", dtype=torch.bfloat16)
        ids = torch.arange(17, device="cuda", dtype=torch.int64) - 17
        slots = torch.randperm(64, device="cuda")[:17]
        slots[0] = -1
        ref = torch.full((1, 64, 656), 91, device="cuda", dtype=torch.uint8)
        got = ref.clone().view(torch.float8_e4m3fn)
        byte_view = _as_uint8(got)
        self.assertEqual(byte_view.data_ptr(), got.data_ptr())
        compute_ops.concat_and_cache_mla(
            src[:, :512][ids],
            src[:, 512:][ids],
            ref,
            slots,
            "fp8_ds_mla",
            torch.ones((), device="cuda"),
        )
        self.assertTrue(
            try_indexed_cache_write(src[:, :512], src[:, 512:], ids, slots, byte_view)
        )
        self.assertTrue(torch.equal(ref, byte_view))

    def test_mla_bytes_and_read_padding(self):
        torch.manual_seed(831)
        scale = torch.ones((), device="cuda")
        for n in (0, 1, 17, 2024):
            src = torch.randn(max(n + 7, 32), 576, device="cuda", dtype=torch.bfloat16)
            src[-1].zero_()
            src[-2].fill_(1.0e-40)
            src[-3].fill_(2000.0)
            src[-4, 0] = float("nan")
            src[-5, 1] = float("inf")
            ids = torch.arange(n, device="cuda", dtype=torch.int64) - n
            slots = torch.randperm(max(n + 7, 32), device="cuda")[:n].long()
            if n > 1:
                slots[0] = -1
            count = max(n + 7, 32)
            ref = torch.full(
                ((count + 63) // 64, 64, 656), 91, device="cuda", dtype=torch.uint8
            )
            got = ref.clone()
            if n:
                compute_ops.concat_and_cache_mla(
                    src[:, :512][ids],
                    src[:, 512:][ids],
                    ref,
                    slots,
                    "fp8_ds_mla",
                    scale,
                )
            self.assertTrue(
                try_indexed_cache_write(src[:, :512], src[:, 512:], ids, slots, got)
            )
            self.assertTrue(torch.equal(ref, got), f"cache bytes N={n}")
            valid_slots = slots[slots >= 0].int().reshape(1, -1)
            lengths = torch.tensor(
                [valid_slots.numel()], device="cuda", dtype=torch.int32
            )
            starts = torch.zeros(1, device="cuda", dtype=torch.int32)
            expected = torch.zeros((n + 5, 576), device="cuda", dtype=torch.bfloat16)
            flat = ref.reshape(-1, 1, 656)
            if valid_slots.numel():
                compute_ops.cp_gather_and_upconvert_fp8_kv_cache_v2(
                    flat, expected, valid_slots, lengths, starts, 1, valid_slots.numel()
                )
            out = try_read_new_kv(flat, valid_slots, n + 5)
            self.assertIsNotNone(out)
            self.assertTrue(
                torch.equal(out.view(torch.int16), expected.view(torch.int16))
            )

    def test_unsupported(self):
        self.assertIsNone(
            try_read_new_kv(
                torch.zeros(1, 1, 656, dtype=torch.uint8),
                torch.zeros(1, dtype=torch.int32),
                1,
            )
        )
        k = torch.zeros(2, 512, device="cuda", dtype=torch.float32)
        self.assertFalse(try_indexed_cache_write(k, None, None, None, None))


if __name__ == "__main__":
    unittest.main()
