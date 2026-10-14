import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
    build_cos_sin_cache,
    run_fused_compress_kv_write,
    run_save_partial_states,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class Fp4SpeculativeStateTest(unittest.TestCase):
    def test_rejected_suffix_preserves_accepted_cache(self):
        torch.manual_seed(741)
        count, ring, block = 4120, 12, 256
        blocks = (count + block - 1) // block
        device = torch.device("cuda")
        kv = torch.randn((count, 256), device=device)
        scores = torch.randn_like(kv)
        ape = torch.randn((4, 256), device=device)
        norm = torch.randn(128, device=device)
        angles = torch.randn((count, 32), device=device)
        cos_sin, _ = build_cos_sin_cache(torch.polar(torch.ones_like(angles), angles))
        table = torch.arange(1, blocks + 1, dtype=torch.int32, device=device)[None]

        def compress(state, cache, begin, end, values, logits):
            positions = torch.arange(begin, end, dtype=torch.int64, device=device)
            slots = (positions // block + 1) * ring + positions % ring
            limit = torch.minimum(
                (positions // block + 1) * block, torch.full_like(positions, end)
            )
            write_slots = torch.where(positions + ring >= limit, slots, -1)
            run_save_partial_states(
                values, logits, ape, positions, state, write_slots, 4, raw_scores=True
            )
            run_fused_compress_kv_write(
                state,
                torch.zeros_like(positions, dtype=torch.int32),
                positions,
                slots,
                table,
                norm,
                1e-6,
                cos_sin,
                cache,
                torch.where((positions + 1) % 4 == 0, 64 + positions // 4, -1),
                values,
                logits,
                ape,
                begin,
                False,
                head_dim=128,
                rope_head_dim=64,
                compress_ratio=4,
                overlap=True,
                state_tokens_per_block=block,
                fp4_indexer=True,
            )

        def empty():
            return (
                torch.zeros((blocks + 1, ring, 512), device=device),
                torch.full(
                    (blocks + 1, 64, 68), 0xAB, dtype=torch.uint8, device=device
                ),
            )

        expected_state, expected_cache = empty()
        compress(expected_state, expected_cache, 0, count, kv, scores)

        def check(cache, end):
            for page in range((end // 4 + 63) // 64):
                rows = min(64, end // 4 - page * 64)
                actual = cache[page + 1].flatten()
                reference = expected_cache[page + 1].flatten()
                torch.testing.assert_close(
                    actual[: rows * 64], reference[: rows * 64], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    actual[4096 : 4096 + rows * 4],
                    reference[4096 : 4096 + rows * 4],
                    rtol=0,
                    atol=0,
                )
            self.assertTrue(torch.all(cache[0] == 0xAB).item())

        for prefix in (
            1,
            3,
            4,
            7,
            8,
            11,
            127,
            128,
            129,
            253,
            255,
            256,
            257,
            4093,
            4095,
            4096,
            4097,
        ):
            for accepted in (1, 2, 3, 4):
                with self.subTest(prefix=prefix, accepted=accepted):
                    state, cache = empty()
                    compress(state, cache, 0, prefix, kv[:prefix], scores[:prefix])
                    proposed_k = kv[prefix : prefix + 4].clone()
                    proposed_s = scores[prefix : prefix + 4].clone()
                    # Rejected rows must not affect accepted bytes or the next cycle.
                    proposed_k[accepted:] = torch.randn_like(proposed_k[accepted:]) * 3
                    proposed_s[accepted:] = torch.randn_like(proposed_s[accepted:]) * 3
                    compress(state, cache, prefix, prefix + 4, proposed_k, proposed_s)
                    check(cache, prefix + accepted)
                    for pos in range(prefix + accepted, prefix + 12):
                        compress(
                            state,
                            cache,
                            pos,
                            pos + 1,
                            kv[pos : pos + 1],
                            scores[pos : pos + 1],
                        )
                    check(cache, prefix + 12)


if __name__ == "__main__":
    unittest.main()
