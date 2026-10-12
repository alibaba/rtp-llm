"""Real CUDA contracts for absorbed-query concatenation and paged KV writes."""

import unittest

import torch

from rtp_llm.ops import compute_ops


class FusedMlaDecodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("This CUDA kernel test requires a GPU")
        torch.cuda.set_device(0)

    def make_inputs(self, tokens, heads, latent, pe, page, dtype):
        # Match bmm's [head, token, latent] result and a slice of packed Q.
        q_latent = torch.randn(
            heads, tokens, latent, device="cuda", dtype=dtype
        ).transpose(0, 1)
        q_pe = torch.randn(tokens, heads, 128 + pe, device="cuda", dtype=dtype)[
            ..., 128:
        ]
        kv = torch.randn(tokens, latent + 7, device="cuda", dtype=dtype)[:, :latent]
        k_pe = torch.randn(tokens, pe + 5, device="cuda", dtype=dtype)[:, :pe]
        width = latent + pe
        blocks = max(3, (tokens + page - 1) // page + 1)
        backing = torch.full(
            (blocks, page + 3, width + 7), -7, device="cuda", dtype=dtype
        )
        cache = backing[:, :page, :width]
        out = torch.empty(tokens, heads, width, device="cuda", dtype=dtype)
        # Unique positive slots, including page transitions; any negative skips KV.
        slots = [page - 1 + i if i % 4 else -1 - i for i in range(tokens)]
        slot_storage = torch.full((tokens * 2,), -99, device="cuda", dtype=torch.int64)
        slot_tensor = slot_storage[::2]
        slot_tensor.copy_(torch.tensor(slots, device="cuda", dtype=torch.int64))
        return [q_latent, q_pe, kv, k_pe, out, cache, slot_tensor], backing, slots

    def assert_bits_equal(self, actual, expected):
        storage_dtype = torch.int32 if actual.dtype == torch.float32 else torch.int16
        self.assertTrue(
            torch.equal(
                actual.contiguous().view(storage_dtype),
                expected.contiguous().view(storage_dtype),
            )
        )

    def assert_result(self, args, backing, slots, previous):
        q_latent, q_pe, kv, k_pe, out, cache, _ = args
        self.assert_bits_equal(out, torch.cat((q_latent, q_pe), dim=-1))
        expected = previous.clone()
        page = cache.shape[1]
        latent = kv.shape[1]
        for token, slot in enumerate(slots):
            if slot >= 0:
                expected[slot // page, slot % page, :latent] = kv[token]
                expected[slot // page, slot % page, latent : latent + k_pe.shape[1]] = (
                    k_pe[token]
                )
        # Compare full backing storage: unwritten slots and padding must survive.
        self.assert_bits_equal(backing, expected)

    def test_strides_pages_negative_slots_and_storage_types(self):
        torch.manual_seed(73)
        cases = [
            (t, 16, 512, 64, p, torch.bfloat16)
            for t in (1, 3, 17, 65)
            for p in (64, 128)
        ]
        cases += [
            (19, 3, 13, 3, 17, dtype)
            for dtype in (torch.bfloat16, torch.float16, torch.float32)
        ]
        for shape in cases:
            with self.subTest(shape=shape):
                args, backing, slots = self.make_inputs(*shape)
                before = backing.clone()
                compute_ops.fused_mla_decode_q_concat_kv_cache_insert(*args)
                self.assert_result(args, backing, slots, before)

    def test_special_values_are_copied_without_arithmetic(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                args, backing, slots = self.make_inputs(3, 2, 4, 4, 8, dtype)
                values = torch.tensor(
                    [-0.0, float("inf"), -float("inf"), float("nan")],
                    dtype=dtype,
                    device="cuda",
                )
                for tensor in args[:4]:
                    tensor.copy_(values)
                before = backing.clone()
                compute_ops.fused_mla_decode_q_concat_kv_cache_insert(*args)
                self.assert_result(args, backing, slots, before)

    def test_empty_batch_does_not_write_cache(self):
        args, backing, slots = self.make_inputs(0, 3, 13, 3, 17, torch.bfloat16)
        before = backing.clone()
        compute_ops.fused_mla_decode_q_concat_kv_cache_insert(*args)
        self.assert_result(args, backing, slots, before)

    def test_graph_replay_reads_updated_inputs_and_slots_on_current_stream(self):
        args, backing, slots = self.make_inputs(17, 4, 512, 64, 128, torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                compute_ops.fused_mla_decode_q_concat_kv_cache_insert(*args)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            compute_ops.fused_mla_decode_q_concat_kv_cache_insert(*args)
        for iteration in (1, 2):
            # CUDA Graph keeps addresses, but each replay must consume fresh data.
            slots = [-1 if (i + iteration) % 3 == 0 else 128 + i for i in range(17)]
            before = backing.clone()
            with torch.cuda.stream(stream):
                for tensor in args[:4]:
                    tensor.fill_(iteration)
                args[-1].copy_(torch.tensor(slots, device="cuda", dtype=torch.int64))
                graph.replay()
            stream.synchronize()
            self.assert_result(args, backing, slots, before)

    def test_invalid_metadata_is_rejected_before_launch(self):
        args, _, _ = self.make_inputs(3, 2, 13, 3, 17, torch.bfloat16)
        invalid = [
            (1, args[1][:, :1], "dimensions"),
            (2, args[2].float(), "dtype"),
            (6, args[6].int(), "int64"),
            (6, args[6].cpu(), "CUDA int64"),
            (
                0,
                torch.empty(3, 2, 26, dtype=torch.bfloat16, device="cuda")[..., ::2],
                "strides",
            ),
            (
                4,
                torch.empty(3, 2, 32, dtype=torch.bfloat16, device="cuda")[..., ::2],
                "contiguous",
            ),
        ]
        for index, value, message in invalid:
            with self.subTest(index=index, message=message):
                operands = args.copy()
                operands[index] = value
                with self.assertRaisesRegex(RuntimeError, message):
                    compute_ops.fused_mla_decode_q_concat_kv_cache_insert(*operands)


if __name__ == "__main__":
    unittest.main()
