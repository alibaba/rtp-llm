"""Exact operator tests for the BF16 MLA decode Q/cache epilogue.

The inputs already contain the final positional channels and local cache slots.
The TP, MTP and PageRR cases below are tensor-layout fixtures, not distributed
attention, service, or scheduler integration tests.
"""

import unittest

import torch


ARGUMENTS = (
    "q_latent",
    "q_pe",
    "kv_c",
    "k_pe",
    "q_out",
    "kv_cache",
    "slot_mapping",
)


def reference(q_latent, q_pe, kv_c, k_pe, kv_cache, slot_mapping):
    """Independent PyTorch reference; every Q row is written, even padding."""
    expected_q = torch.cat((q_latent, q_pe), dim=-1)
    expected_cache = kv_cache.clone()
    cache_rows = torch.cat((kv_c, k_pe), dim=-1)
    page_size = expected_cache.shape[1]
    for token, slot in enumerate(slot_mapping.cpu().tolist()):
        if slot >= 0:
            expected_cache[slot // page_size, slot % page_size].copy_(
                cache_rows[token]
            )
    return expected_q, expected_cache


def values(shape, device, offset=0):
    # Small integers are exactly representable in BF16. Distinct channel/token
    # patterns catch swapped segments, wrong head strides and missing writes.
    count = 1
    for size in shape:
        count *= size
    return (
        (torch.arange(count, device=device) + offset) % 127 - 63
    ).to(torch.bfloat16).reshape(shape)


def make_inputs(tokens, heads, page_size, device, slots=None, strided=True):
    if strided:
        q_latent = values((heads, tokens, 512), device, 3).transpose(0, 1)
        q_pe = values((tokens, heads, 192), device, 11)[..., 128:]
        kv_c = values((tokens, 1024), device, 23)[:, 16:528]
        k_pe = values((tokens, 128), device, 41)[:, 32:96]
        q_out = torch.full(
            (tokens, 2 * heads + 1, 576), -97, dtype=torch.bfloat16, device=device
        )[:, 1::2]
    else:
        q_latent = values((tokens, heads, 512), device, 3)
        q_pe = values((tokens, heads, 64), device, 11)
        kv_c = values((tokens, 512), device, 23)
        k_pe = values((tokens, 64), device, 41)
        q_out = torch.full(
            (tokens, heads, 576), -97, dtype=torch.bfloat16, device=device
        )
    return dict(
        q_latent=q_latent,
        q_pe=q_pe,
        kv_c=kv_c,
        k_pe=k_pe,
        q_out=q_out,
        kv_cache=values((4, page_size, 576), device, 59),
        slot_mapping=torch.tensor(
            list(range(tokens)) if slots is None else slots,
            dtype=torch.int64,
            device=device,
        ),
    )


class TestMLADecodeReference(unittest.TestCase):
    def test_reference_writes_slot_zero_skips_all_negative_slots_and_keeps_q(self):
        # Hand-derived small fixture also checks the oracle without CUDA/ops.
        expected_q, expected_cache = reference(
            torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]]]),
            torch.tensor([[[7]], [[8]], [[9]]]),
            torch.tensor([[10, 11], [12, 13], [14, 15]]),
            torch.tensor([[16], [17], [18]]),
            torch.tensor([[[20, 21, 22], [23, 24, 25]]]),
            torch.tensor([0, -1, -7]),
        )
        torch.testing.assert_close(
            expected_q,
            torch.tensor([[[1, 2, 7]], [[3, 4, 8]], [[5, 6, 9]]]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            expected_cache,
            torch.tensor([[[10, 11, 16], [23, 24, 25]]]),
            rtol=0,
            atol=0,
        )

    def test_reference_empty_tokens_preserve_cache(self):
        args = make_inputs(0, 2, 64, "cpu")
        expected_q, expected_cache = reference(
            args["q_latent"], args["q_pe"], args["kv_c"], args["k_pe"],
            args["kv_cache"], args["slot_mapping"],
        )
        self.assertEqual(tuple(expected_q.shape), (0, 2, 576))
        torch.testing.assert_close(
            expected_cache, args["kv_cache"], rtol=0, atol=0
        )


class TestMLADecodeQConcatCacheInsert(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        # Keep the reference tests usable without loading the RTP CUDA library.
        from rtp_llm.ops import compute_ops

        cls.op = staticmethod(compute_ops.mla_decode_q_concat_cache_insert)
        cls.device = torch.device("cuda", torch.cuda.current_device())

    def call(self, args):
        self.op(*(args[name] for name in ARGUMENTS))

    def run_case(self, args):
        expected_q, expected_cache = reference(
            args["q_latent"], args["q_pe"], args["kv_c"], args["k_pe"],
            args["kv_cache"], args["slot_mapping"],
        )
        read_only = {
            name: args[name].clone()
            for name in ("q_latent", "q_pe", "kv_c", "k_pe", "slot_mapping")
        }
        self.call(args)
        torch.testing.assert_close(args["q_out"], expected_q, rtol=0, atol=0)
        # Comparing the whole cache also checks every untouched cache entry.
        torch.testing.assert_close(args["kv_cache"], expected_cache, rtol=0, atol=0)
        for name, before in read_only.items():
            torch.testing.assert_close(args[name], before, rtol=0, atol=0)

    def test_contiguous_and_production_strides_are_copied_exactly(self):
        for page_size in (64, 128):
            for strided in (False, True):
                with self.subTest(page_size=page_size, strided=strided):
                    args = make_inputs(
                        7, 16, page_size, self.device,
                        slots=[0, page_size - 1, page_size, -1,
                               2 * page_size - 1, -7, 2 * page_size],
                        strided=strided,
                    )
                    if strided:
                        self.assertFalse(args["q_latent"].is_contiguous())
                        self.assertEqual(args["q_latent"].stride(-1), 1)
                        self.assertEqual(args["q_pe"].stride(1), 192)
                        self.assertEqual(args["kv_c"].stride(0), 1024)
                        self.assertEqual(args["k_pe"].stride(0), 128)
                        self.assertEqual(args["q_out"].stride(-1), 1)
                    self.run_case(args)

    def test_strided_output_does_not_overwrite_padding_heads(self):
        args = make_inputs(3, 2, 64, self.device)
        storage = torch.full(
            (3, 5, 576), -97, dtype=torch.bfloat16, device=self.device
        )
        args["q_out"] = storage[:, 1::2]
        expected_storage = storage.clone()
        expected_storage[:, 1::2] = torch.cat(
            (args["q_latent"], args["q_pe"]), dim=-1
        )
        self.run_case(args)
        torch.testing.assert_close(storage, expected_storage, rtol=0, atol=0)

    def test_aligned_strided_cache_preserves_other_storage(self):
        for page_size in (64, 128):
            with self.subTest(page_size=page_size):
                args = make_inputs(
                    3, 2, page_size, self.device,
                    slots=[0, page_size - 1, page_size],
                )
                storage = values((8, 2 * page_size + 8, 576), self.device, 67)
                args["kv_cache"] = storage[::2, 8::2]
                expected_storage = storage.clone()
                _, expected_cache = reference(
                    args["q_latent"], args["q_pe"], args["kv_c"], args["k_pe"],
                    args["kv_cache"], args["slot_mapping"],
                )
                expected_storage[::2, 8::2] = expected_cache
                self.run_case(args)
                torch.testing.assert_close(
                    storage, expected_storage, rtol=0, atol=0
                )

    def test_tp4_tp8_query_layouts_and_replicated_heads_with_mtp_q1_q3_q4(self):
        # 128 global heads yield 32/16 local heads at TP4/8; the 64/128-head
        # fixtures represent repetition used by an attention backend. The op
        # must copy the supplied head layout without interpreting TP or Q.
        for tp_size, local_heads in ((4, 32), (8, 16)):
            for replication in (1, 64 // local_heads, 128 // local_heads):
                for query_length in (1, 3, 4):
                    for page_size in (64, 128):
                        with self.subTest(
                            tp=tp_size, replication=replication,
                            q=query_length, page_size=page_size,
                        ):
                            tokens = 3 * query_length
                            slots = [
                                0, page_size - 1, page_size, -1,
                                page_size + 1, 2 * page_size - 1,
                                2 * page_size, -7, 2 * page_size + 1,
                                3 * page_size - 1, 3 * page_size, 4 * page_size - 1,
                            ][:tokens]
                            args = make_inputs(
                                tokens, local_heads * replication,
                                page_size, self.device, slots=slots,
                            )
                            args["q_latent"] = values(
                                (local_heads, tokens, 512), self.device, 3
                            ).repeat_interleave(replication, dim=0).transpose(0, 1)
                            args["q_pe"] = values(
                                (tokens, local_heads, 192), self.device, 11
                            ).repeat_interleave(replication, dim=1)[..., 128:]
                            self.run_case(args)

    def test_token_counts_64_65_and_128_copy_all_queries_and_cache_rows(self):
        # Exercise both sides of the small-batch launch strategy boundary.
        # Positive slots are unique and deliberately scattered across pages;
        # the final token stays valid to catch a truncated final work group.
        for tokens in (64, 65, 128):
            for heads in (16, 32):
                for page_size in (64, 128):
                    with self.subTest(
                        tokens=tokens, heads=heads, page_size=page_size
                    ):
                        slots = [
                            (5 * token) % (4 * page_size)
                            for token in range(tokens)
                        ]
                        slots[7] = -1
                        slots[31] = -7
                        self.run_case(make_inputs(
                            tokens, heads, page_size, self.device, slots=slots
                        ))

    def test_preserves_raw_bf16_bits_in_queries_and_entire_cache(self):
        # Signed zero, subnormals, infinities, and distinct quiet/signaling NaN
        # payloads must survive raw copies. Integer views avoid NaN equality
        # rules and expose conversion, canonicalization, or flush-to-zero.
        patterns = torch.tensor(
            [0, -32768, 1, -32767, 127, -32641, 128, -32640,
             32639, -129, 32640, -128, 32705, -63, 32641, -127],
            dtype=torch.int16,
            device=self.device,
        )
        for page_size in (64, 128):
            with self.subTest(page_size=page_size):
                args = make_inputs(
                    4, 16, page_size, self.device,
                    slots=[0, page_size - 1, -7, page_size],
                )
                for index, name in enumerate(
                    ("q_latent", "q_pe", "kv_c", "k_pe", "kv_cache")
                ):
                    count = args[name].numel()
                    bits = patterns.roll(index).repeat(
                        (count + patterns.numel() - 1) // patterns.numel()
                    )[:count]
                    args[name].view(torch.int16).copy_(
                        bits.reshape(args[name].shape)
                    )
                expected_q, expected_cache = reference(
                    args["q_latent"], args["q_pe"], args["kv_c"], args["k_pe"],
                    args["kv_cache"], args["slot_mapping"],
                )
                self.call(args)
                torch.testing.assert_close(
                    args["q_out"].view(torch.int16),
                    expected_q.view(torch.int16), rtol=0, atol=0,
                )
                torch.testing.assert_close(
                    args["kv_cache"].view(torch.int16),
                    expected_cache.view(torch.int16), rtol=0, atol=0,
                )

    def test_page_rr_already_local_slots_cross_page_boundaries(self):
        # Only the owner's already-local slots reach this op. Negative entries
        # model tokens owned elsewhere; no global-to-local mapping runs here.
        for page_size in (64, 128):
            with self.subTest(page_size=page_size):
                self.run_case(make_inputs(
                    7, 32, page_size, self.device,
                    slots=[page_size - 1, -1, page_size, 2 * page_size - 1,
                           -11, 0, 2 * page_size],
                ))

    def test_all_negative_slots_still_write_every_query(self):
        self.run_case(make_inputs(
            4, 1, 64, self.device, slots=[-1, -2, -7, -(2**63)]
        ))

    def test_empty_tokens_preserve_cache(self):
        for page_size in (64, 128):
            with self.subTest(page_size=page_size):
                self.run_case(make_inputs(0, 16, page_size, self.device))

    def test_cuda_graph_replay_uses_new_slots_including_zero(self):
        for page_size in (64, 128):
            with self.subTest(page_size=page_size):
                args = make_inputs(
                    4, 16, page_size, self.device, slots=[-1, -2, -3, -4]
                )
                expected_cache = args["kv_cache"].clone()
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.call(args)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    self.call(args)
                for slots in (
                    [0, page_size - 1, page_size, -7],
                    [-1, -3, -7, -4],
                    [2 * page_size - 1, 1, -1, page_size + 1],
                ):
                    with torch.cuda.stream(stream):
                        args["slot_mapping"].copy_(torch.tensor(
                            slots, dtype=torch.int64, device=self.device
                        ))
                        args["q_out"].fill_(-97)
                        graph.replay()
                    stream.synchronize()
                    expected_q, expected_cache = reference(
                        args["q_latent"], args["q_pe"], args["kv_c"], args["k_pe"],
                        expected_cache, args["slot_mapping"],
                    )
                    torch.testing.assert_close(
                        args["q_out"], expected_q, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        args["kv_cache"], expected_cache, rtol=0, atol=0
                    )

    def test_uses_current_cuda_stream_for_reads_and_writes(self):
        args = make_inputs(4, 16, 64, self.device, slots=[0, 63, -1, 64])
        cache_before = args["kv_cache"].clone()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            # Make reads on an incorrectly hard-coded default stream race with
            # these producers. Snapshot consumers stay on the tested stream.
            torch.cuda._sleep(10_000_000)
            for name, value in (("q_latent", 7), ("q_pe", 13),
                                ("kv_c", 17), ("k_pe", 19)):
                args[name].fill_(value)
            self.call(args)
            observed_q = args["q_out"].clone()
            observed_cache = args["kv_cache"].clone()
        stream.synchronize()
        expected_q, expected_cache = reference(
            args["q_latent"], args["q_pe"], args["kv_c"], args["k_pe"],
            cache_before, args["slot_mapping"],
        )
        torch.testing.assert_close(observed_q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(observed_cache, expected_cache, rtol=0, atol=0)

    def test_rejects_non_bf16_data_and_non_int64_slots(self):
        for name in ARGUMENTS:
            with self.subTest(argument=name):
                args = make_inputs(3, 2, 64, self.device)
                args[name] = args[name].to(
                    torch.int32 if name == "slot_mapping" else torch.float32
                )
                with self.assertRaises(RuntimeError):
                    self.call(args)

    def test_rejects_cpu_tensors(self):
        for name in ARGUMENTS:
            with self.subTest(argument=name):
                args = make_inputs(3, 2, 64, self.device)
                args[name] = args[name].cpu()
                with self.assertRaises(RuntimeError):
                    self.call(args)

    def test_rejects_wrong_ranks(self):
        for name in ARGUMENTS:
            with self.subTest(argument=name):
                args = make_inputs(3, 2, 64, self.device)
                args[name] = args[name].unsqueeze(0)
                with self.assertRaises(RuntimeError):
                    self.call(args)

    def test_rejects_wrong_token_head_and_channel_dimensions(self):
        cases = [(name, "tokens") for name in ARGUMENTS if name != "kv_cache"]
        cases += [(name, "heads") for name in ("q_pe", "q_out")]
        cases += [(name, "channels") for name in ARGUMENTS[:-1]]
        for name, dimension in cases:
            with self.subTest(argument=name, dimension=dimension):
                args = make_inputs(3, 2, 64, self.device)
                if dimension == "tokens":
                    args[name] = args[name][:2]
                elif dimension == "heads":
                    args[name] = args[name][:, :1]
                else:
                    args[name] = args[name][..., :-1]
                with self.assertRaises(RuntimeError):
                    self.call(args)

    def test_rejects_noncontiguous_last_dimensions(self):
        for name in ARGUMENTS[:-1]:
            with self.subTest(argument=name):
                args = make_inputs(3, 2, 64, self.device)
                shape = list(args[name].shape)
                shape[-1] *= 2
                args[name] = values(tuple(shape), self.device)[..., ::2]
                with self.assertRaises(RuntimeError):
                    self.call(args)

    def test_rejects_misaligned_pointers_and_outer_strides(self):
        for name in ARGUMENTS[:-1]:
            for misalignment in ("pointer", "stride"):
                with self.subTest(argument=name, misalignment=misalignment):
                    args = make_inputs(3, 2, 64, self.device)
                    shape = tuple(args[name].shape)
                    if misalignment == "pointer":
                        count = args[name].numel()
                        args[name] = values((count + 8,), self.device)[
                            1:count + 1
                        ].view(shape)
                    else:
                        padded = shape[:-1] + (shape[-1] + 1,)
                        args[name] = values(padded, self.device)[..., :-1]
                    with self.assertRaises(RuntimeError):
                        self.call(args)

    def test_rejects_noncontiguous_slots(self):
        args = make_inputs(3, 2, 64, self.device)
        args["slot_mapping"] = torch.arange(
            6, dtype=torch.int64, device=self.device
        )[::2]
        with self.assertRaises(RuntimeError):
            self.call(args)

    def test_rejects_zero_heads(self):
        args = make_inputs(3, 2, 64, self.device)
        for name in ("q_latent", "q_pe", "q_out"):
            args[name] = args[name][:, :0]
        with self.assertRaises(RuntimeError):
            self.call(args)

    def test_rejects_unsupported_cache_page_size(self):
        args = make_inputs(3, 2, 64, self.device)
        args["kv_cache"] = values((4, 32, 576), self.device)
        with self.assertRaises(RuntimeError):
            self.call(args)

    def test_rejects_tensors_on_different_cuda_devices(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("requires two CUDA devices")
        args = make_inputs(3, 2, 64, self.device)
        other_device = (self.device.index + 1) % torch.cuda.device_count()
        args["k_pe"] = args["k_pe"].to(torch.device("cuda", other_device))
        with self.assertRaises(RuntimeError):
            self.call(args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
