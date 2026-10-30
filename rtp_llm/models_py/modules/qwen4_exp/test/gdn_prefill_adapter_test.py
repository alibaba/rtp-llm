import unittest

import torch
import triton

from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_adapter import (
    _store_page_checkpoints,
)
from rtp_llm.models_py.modules.qwen4_exp.test.gdn_prefill_reference import (
    sm120_prefill_with_checkpoints,
)
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd
from rtp_llm.models_py.triton_kernels.fla.utils import assert_close


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class GdnSm120PrefillTest(unittest.TestCase):
    def test_output_final_and_all_page_states_with_ragged_prefix(self):
        # fail-loud rather than skip: this target is pinned to RTX_5000_PRO_CU13.
        self.assertEqual(
            torch.cuda.get_device_capability(),
            (12, 0),
            "target pool is RTX_5000_PRO_CU13 (SM120)",
        )
        for lengths in ((257,), (129, 257)):
            for prefix in (False, True):
                torch.manual_seed(7330 + sum(lengths) + int(prefix))
                total = sum(lengths)
                q = torch.randn(1, total, 6, 128, device="cuda", dtype=torch.bfloat16)
                k = torch.randn_like(q)
                v = torch.randn(1, total, 12, 128, device="cuda", dtype=torch.bfloat16)
                g = -torch.rand(1, total, 12, device="cuda") * 0.1
                beta = torch.rand_like(g).to(torch.bfloat16)
                initial = (
                    torch.randn(len(lengths), 12, 128, 128, device="cuda") * 0.05
                    if prefix
                    else None
                )
                starts = [0]
                for n in lengths:
                    starts.append(starts[-1] + n)
                cu = torch.tensor(starts, device="cuda", dtype=torch.int32)
                expected, _, final = chunk_gated_delta_rule(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state=initial,
                    output_final_state=True,
                    cu_seqlens=cu,
                    use_qk_l2norm_in_kernel=True,
                )
                actual, checkpoints, actual_final, checkpoint_starts = (
                    sm120_prefill_with_checkpoints(
                        q, k, v, g, beta, initial, cu, lengths
                    )
                )
                # FLA and CP each approximate the recurrence. Apply the existing
                # .005 math-reference threshold independently, rather than
                # assuming FLA is an exact oracle for a different backend.
                ref, ref_final, ref_pages = recurrence_reference(
                    q, k, v, g, beta, initial, lengths
                )
                assert_close("legacy output versus FP64", ref, expected, 0.005)
                assert_close("SM120 output versus FP64", ref, actual, 0.005)
                assert_close("legacy final versus FP64", ref_final, final, 0.005)
                assert_close("SM120 final versus FP64", ref_final, actual_final, 0.005)
                assert_close("SM120 pages versus FP64", ref_pages, checkpoints, 0.005)
                self.assertEqual(checkpoint_starts.dtype, torch.int64)

    def test_fused_post_conv_preparation_preserves_all_boundaries(self):
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_prepare import (
            prepare_sm120_prefill,
        )
        from rtp_llm.models_py.triton_kernels.common.scatter_qkv import scatter_qkv
        from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating

        for rows, kh, vh in ((129, 3, 6), (4096, 6, 12), (12288, 6, 12)):
            torch.manual_seed(7680 + rows)
            mixed = torch.randn(
                rows, (2 * kh + vh) * 128, device="cuda", dtype=torch.bfloat16
            )
            ab = torch.randn(rows, 2 * vh, device="cuda", dtype=torch.bfloat16)
            a, b = ab[:, :vh], ab[:, vh:]
            alog = torch.randn(vh, device="cuda") - 2
            bias = torch.randn(vh, device="cuda") * 0.1
            q, k, v = scatter_qkv(mixed, kh, vh, 128, 128)
            g, beta = fused_gdn_gating(alog, a, b, bias)
            expected = (
                l2norm_fwd(q).squeeze(0),
                l2norm_fwd(k).squeeze(0),
                v.squeeze(0),
                g.squeeze(0).exp(),
                beta.squeeze(0).float(),
            )
            actual = prepare_sm120_prefill(mixed, a, b, alog, bias, kh, vh)
            for name, candidate, reference in zip(
                ("q", "k", "v", "alpha", "beta"), actual, expected
            ):
                with self.subTest(rows=rows, field=name):
                    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)

    def test_fp32_ba_padded_rows_and_flat_weight_views(self):
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_prepare import (
            prepare_sm120_prefill,
        )
        from rtp_llm.models_py.triton_kernels.common.scatter_qkv import scatter_qkv
        from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating

        for rows, kh, vh in ((3850, 8, 24), (4096, 6, 12)):
            torch.manual_seed(79001 + rows)
            width = (2 * kh + vh) * 128
            mixed = torch.randn(rows, width + 128, device="cuda", dtype=torch.bfloat16)[
                :, :width
            ]
            for dtype in (torch.bfloat16, torch.float32):
                ab = torch.randn(rows, 2 * vh, device="cuda", dtype=dtype)
                a, b = ab[:, :vh], ab[:, vh:]
                alog = torch.randn(1, vh, device="cuda") - 2
                bias = torch.randn(vh, 1, device="cuda") * 0.1
                # Original scatter requires a dense row owner; the fused
                # producer directly consumes the equivalent padded-row view.
                q, k, v = scatter_qkv(mixed.contiguous(), kh, vh, 128, 128)
                g, beta = fused_gdn_gating(alog, a, b, bias)
                expected = (
                    l2norm_fwd(q).squeeze(0),
                    l2norm_fwd(k).squeeze(0),
                    v.squeeze(0),
                    g.squeeze(0).exp(),
                    beta.squeeze(0).float(),
                )
                actual = prepare_sm120_prefill(mixed, a, b, alog, bias, kh, vh)
                for got, ref in zip(actual, expected):
                    torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_real_qwen4_three_value_heads_per_key_head(self):
        for lengths in ((257,), (129, 257)):
            torch.manual_seed(79301 + sum(lengths))
            total = sum(lengths)
            q = torch.randn(1, total, 8, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.randn_like(q)
            v = torch.randn(1, total, 24, 128, device="cuda", dtype=torch.bfloat16)
            g = -torch.rand(1, total, 24, device="cuda") * 0.1
            beta = torch.rand_like(g).bfloat16()
            cu = torch.tensor(
                [0] + list(__import__("itertools").accumulate(lengths)),
                device="cuda",
                dtype=torch.int32,
            )
            initial = torch.randn(len(lengths), 24, 128, 128, device="cuda") * 0.05
            actual, pages, final, starts = sm120_prefill_with_checkpoints(
                q, k, v, g, beta, initial, cu, lengths
            )
            ref, ref_final, ref_pages = recurrence_reference(
                q, k, v, g, beta, initial, lengths
            )
            assert_close("GQA3 CP output versus FP64", ref, actual, 0.005)
            assert_close("GQA3 final versus FP64", ref_final, final, 0.005)
            assert_close("GQA3 every page versus FP64", ref_pages, pages, 0.005)

    def test_checkpoint_store_all_pages_prefix_sentinel_and_dtype(self):
        lengths, prefixes = (257, 128, 7), (128, 256, 0)
        cu = torch.tensor([0, 257, 385, 392], device="cuda", dtype=torch.int32)
        prefix = torch.tensor(prefixes, device="cuda", dtype=torch.int32)
        starts = torch.tensor([0, 2, 3, 3], device="cuda", dtype=torch.int64)
        table = torch.tensor(
            [[7, 4, 8, 2, 0], [9, 6, 3, 0, 0], [5, 0, 0, 0, 0]],
            device="cuda",
            dtype=torch.int32,
        )
        heads = 2
        pages = torch.randn(3, heads, 128, 128, device="cuda")
        final = torch.randn(3, heads, 128, 128, device="cuda")
        for dtype in (torch.float32, torch.bfloat16):
            cache = torch.randn(11, heads, 128, 128, device="cuda", dtype=dtype)
            expected = cache.clone()
            for batch, length in enumerate(lengths):
                for page in range(triton.cdiv(length, 128)):
                    block = int(table[batch, prefixes[batch] // 128 + page])
                    if block:
                        source = (
                            final[batch]
                            if (page + 1) * 128 >= length
                            else pages[int(starts[batch]) + page]
                        )
                        expected[block].copy_(source)
            _store_page_checkpoints[(3, 3, triton.cdiv(heads * 128 * 128, 1024))](
                pages,
                final,
                starts,
                cu,
                prefix,
                table,
                cache,
                TABLE_STRIDE=table.stride(0),
                CACHE_STRIDE=cache.stride(0),
                HEADS=heads,
                PAGE=128,
                BLOCK=1024,
            )
            torch.testing.assert_close(cache, expected, rtol=0, atol=0)
            # Allocator page zero is reserved even if a page table references it.
            table[0, 2] = 0
            before_zero = cache[0].clone()
            _store_page_checkpoints[(3, 3, triton.cdiv(heads * 128 * 128, 1024))](
                pages,
                final,
                starts,
                cu,
                prefix,
                table,
                cache,
                TABLE_STRIDE=table.stride(0),
                CACHE_STRIDE=cache.stride(0),
                HEADS=heads,
                PAGE=128,
                BLOCK=1024,
            )
            torch.testing.assert_close(cache[0], before_zero, rtol=0, atol=0)


def recurrence_reference(q, k, v, g, beta, initial, lengths):
    q = (
        l2norm_fwd(q)
        .squeeze(0)
        .repeat_interleave(v.shape[2] // q.shape[2], dim=1)
        .double()
    )
    k = (
        l2norm_fwd(k)
        .squeeze(0)
        .repeat_interleave(v.shape[2] // k.shape[2], dim=1)
        .double()
    )
    v, forget, beta = (
        v.squeeze(0).double(),
        g.squeeze(0).double().exp(),
        beta.squeeze(0).double(),
    )
    outputs, states, pages, offset = [], [], [], 0
    for batch, length in enumerate(lengths):
        state = (
            initial[batch].double().clone()
            if initial is not None
            else torch.zeros(v.shape[1], 128, 128, device=v.device, dtype=torch.float64)
        )
        for token in range(length):
            idx = offset + token
            state = state * forget[idx, :, None, None]
            delta = (v[idx] - (state * k[idx, :, None, :]).sum(-1)) * beta[idx, :, None]
            state = state + delta[:, :, None] * k[idx, :, None, :]
            outputs.append((state * q[idx, :, None, :]).sum(-1) * (128**-0.5))
            if (token + 1) % 128 == 0:
                pages.append(state.clone())
        states.append(state)
        offset += length
    return torch.stack(outputs).unsqueeze(0), torch.stack(states), torch.stack(pages)


if __name__ == "__main__":
    unittest.main()
