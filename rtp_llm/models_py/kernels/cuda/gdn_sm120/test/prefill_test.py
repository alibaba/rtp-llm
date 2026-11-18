import unittest

import torch
import triton

from rtp_llm.models_py.kernels.cuda.gdn_sm120.test.reference import (
    sm120_prefill_with_checkpoints,
)
from rtp_llm.models_py.triton_kernels.fla.block import store_page_checkpoints_kernel
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd
from rtp_llm.models_py.triton_kernels.fla.utils import assert_close


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class GdnSm120PrefillTest(unittest.TestCase):
    def test_segmented_conv_and_gdn_restore_long_prefix_and_all_cache_bytes(self):
        from types import SimpleNamespace

        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextMetadata
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.segmented_prefill import (
            QwenPrefillContext,
        )
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_adapter import (
            Qwen4Sm120GdnPrefill,
        )
        from rtp_llm.models_py.triton_kernels.causal_conv1d.causal_conv1d import (
            prepare_causal_conv1d_metadata,
        )
        from rtp_llm.ops import DataType
        from rtp_llm.utils.model_weight import W

        torch.manual_seed(8631)
        device = torch.device("cuda", torch.cuda.current_device())
        width, capacity, heads = 96, 8192, 4
        module = Qwen4Sm120GdnPrefill(
            SimpleNamespace(
                linear_key_head_dim=128,
                linear_value_head_dim=128,
                linear_num_key_heads=2,
                linear_num_value_heads=heads,
                linear_conv_kernel_dim=4,
                ssm_state_dtype=DataType.TYPE_FP32,
                conv_state_dtype=DataType.TYPE_BF16,
            ),
            SimpleNamespace(get_attn_tp_size=lambda: 1),
            {
                W.linear_attn_conv1d_w: torch.randn(
                    1024, 1, 4, device=device, dtype=torch.bfloat16
                )
                * 0.1,
                W.linear_attn_dt_b: torch.randn(heads, device=device),
                W.linear_attn_alog: torch.randn(heads, device=device),
            },
        )
        storage = torch.zeros(
            4 * width + 1,
            module.linear_cache_converter.block_size_bytes,
            device=device,
            dtype=torch.int8,
        )
        module._get_ssm_states(storage).normal_(std=0.01)
        module._get_conv_states(storage).normal_(std=0.01)
        baseline = storage.clone()
        cache = SimpleNamespace(kv_cache_base=storage, seq_size_per_block=128)

        def inputs(lengths, prefixes):
            cu = torch.tensor(
                [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
            )
            table = torch.arange(
                1, len(lengths) * width + 1, dtype=torch.int32
            ).reshape(len(lengths), width)
            return (
                SimpleNamespace(
                    is_prefill=True,
                    is_cuda_graph=False,
                    input_lengths=torch.tensor(lengths, dtype=torch.int32),
                    prefix_lengths=torch.tensor(prefixes, dtype=torch.int32),
                    prefix_lengths_device=torch.tensor(
                        prefixes, dtype=torch.int32, device=device
                    ),
                    cu_seqlens_device=cu.to(device),
                    kv_cache_kernel_block_id_device=table.to(device),
                    cache_store_inputs=None,
                    cache_store_writer=None,
                ),
                cu,
            )

        dummy, _ = inputs([capacity], [0])
        owner = SimpleNamespace(
            device=device,
            max_requests=4,
            linear_layers=[(0, None)],
            recording=True,
            model=SimpleNamespace(
                config=SimpleNamespace(max_seq_len=12288),
                _attention_tag=lambda _: "linear",
                kv_cache=SimpleNamespace(get_kernel_seq_size_per_block=lambda _: 128),
            ),
            pool=torch.cuda.graph_pool_handle(),
            capture_stream=torch.cuda.Stream(),
        )
        context = QwenPrefillContext(
            owner, capacity, SimpleNamespace(attention_inputs={"linear": dummy})
        )
        context.gdn(
            0,
            module,
            (
                torch.randn(capacity, 1024, device=device, dtype=torch.bfloat16),
                torch.randn(capacity, heads, device=device, dtype=torch.bfloat16),
                torch.randn(capacity, heads, device=device, dtype=torch.bfloat16),
            ),
            cache,
        )
        owner.recording = False
        for lengths, prefixes in (
            ((2048, 2048), (6144, 7168)),
            ((3072, 3072), (7168, 0)),
            ((2048,), (0,)),
            ((2048,), (7168,)),
        ):
            live, cu = inputs(lengths, prefixes)
            table = live.kv_cache_kernel_block_id_device
            table.copy_(
                torch.randperm(4 * width, device=device, dtype=torch.int32)[
                    : table.numel()
                ].reshape_as(table)
                + 1
            )
            for row, prefix in enumerate(prefixes):
                if prefix:
                    table[row, : (prefix - 1) // 128] = -1
            arguments = (
                torch.randn(sum(lengths), 1024, device=device, dtype=torch.bfloat16),
                torch.randn(sum(lengths), heads, device=device, dtype=torch.bfloat16),
                torch.randn(sum(lengths), heads, device=device, dtype=torch.bfloat16),
            )
            storage.copy_(baseline)
            reference_cache = SimpleNamespace(
                kv_cache_base=baseline.clone(), seq_size_per_block=128
            )
            context.refresh(SimpleNamespace(attention_inputs={"linear": live}), lengths)
            actual = context.gdn(0, module, arguments, cache)
            expected = module(
                *arguments,
                live,
                reference_cache,
                Qwen3NextMetadata(
                    prefill_conv1d_meta=prepare_causal_conv1d_metadata(cu, device)
                ),
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                storage, reference_cache.kv_cache_base, rtol=0, atol=0
            )
        for region in context.regions.values():
            region.graph.reset()

    def test_prepared_graphs_keep_separate_capacity_workspaces(self):
        from rtp_llm.models_py.kernels.cuda.gdn_sm120.prefill import (
            sm120_prefill_prepared,
        )
        from rtp_llm.models_py.triton_kernels.fla.block import (
            load_initial_state_from_block_map,
        )
        from rtp_llm.models_py.triton_kernels.fla.prepared import GdnPrefillPrepared

        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        fixtures = []
        for capacity, key_heads, heads, batch, live_output in (
            (2048, 2, 4, 2, False),
            (4096, 2, 4, 2, False),
            (8192, 8, 24, 4, True),
            (8192, 4, 12, 4, True),
            (8192, 2, 6, 4, True),
        ):
            torch.manual_seed(837 + capacity)
            prepared = GdnPrefillPrepared(
                capacity, batch, "cuda", output_live_only=live_output
            )
            prepared.refresh((257, 129), (0, 128))
            prepared.require_prefix_alignment(128)
            q = torch.nn.functional.normalize(
                torch.randn(capacity, key_heads, 128, device="cuda"), dim=-1
            ).bfloat16()
            k = torch.nn.functional.normalize(
                torch.randn_like(q).float(), dim=-1
            ).bfloat16()
            v = torch.randn(capacity, heads, 128, device="cuda", dtype=torch.bfloat16)
            original = (q.clone(), k.clone(), v.clone())
            alpha = torch.exp(-torch.rand(capacity, heads, device="cuda") * 0.1)
            beta = torch.rand_like(alpha).bfloat16().float()
            width = (capacity + 7168) // 128 + 1
            table = torch.arange(
                1, batch * width + 1, device="cuda", dtype=torch.int32
            ).view(batch, width)
            pool = torch.randn(table.numel() + 1, heads, 128, 128, device="cuda") * 0.01
            base_pool = pool.clone()
            initial = torch.empty(batch, heads, 128, 128, device="cuda")

            def run():
                load_initial_state_from_block_map(
                    prepared.prefix_lengths,
                    table,
                    pool,
                    initial,
                    128,
                    prepared=prepared,
                )
                output, checkpoints, final, starts = sm120_prefill_prepared(
                    q,
                    k,
                    v,
                    alpha,
                    beta,
                    initial,
                    prepared.cu_seqlens,
                    prepared=prepared,
                )
                store_page_checkpoints_kernel[(batch, capacity // 128, heads * 16)](
                    checkpoints,
                    final,
                    starts,
                    prepared.cu_seqlens,
                    prepared.prefix_lengths,
                    table,
                    pool,
                    TABLE_STRIDE=table.stride(0),
                    CACHE_STRIDE=pool.stride(0),
                    HEADS=heads,
                    PAGE=128,
                    BLOCK=1024,
                    request_live=prepared.request_live,
                )
                return output, checkpoints, final

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output, checkpoints, final = run()
            fixtures.append(
                (
                    prepared,
                    (q, k, v),
                    original,
                    alpha,
                    beta,
                    table,
                    pool,
                    base_pool,
                    initial,
                    graph,
                    output,
                    checkpoints,
                    final,
                )
            )
        pointers = [
            {tensor.data_ptr() for tensor in item[0].workspace.values()}
            | {item[0].sm120_cp_chunk_len.data_ptr()}
            for item in fixtures
        ]
        for index, pointer_set in enumerate(pointers):
            for other in pointers[index + 1 :]:
                self.assertFalse(pointer_set & other)
        for lengths, explicit_prefixes in (
            ((257, 129), None),
            ((65, 0), None),
            ((0, 0), None),
            ((129, 257), None),
            ((8192,), None),
            ((4093, 2051, 2048), None),
            ((6141, 1025, 1026), None),
            ((4096, 4096), None),
            ((2048, 2048), (6144, 7168)),
            ((3072, 3072), (7168, 6144)),
            ((2048,), (0,)),
        ):
            for (
                prepared,
                inputs,
                original,
                alpha,
                beta,
                table,
                pool,
                base_pool,
                initial,
                graph,
                output,
                checkpoints,
                final,
            ) in fixtures:
                if (
                    sum(lengths) > prepared.token_capacity
                    or len(lengths) > prepared.batch_capacity
                ):
                    continue
                prefixes = explicit_prefixes or tuple(
                    128 if row == 1 and sum(lengths) < 2048 else 0
                    for row in range(len(lengths))
                )
                prepared.refresh(lengths, prefixes)
                table.copy_(
                    torch.arange(1, table.numel() + 1, device="cuda", dtype=torch.int32)
                    .flip(0)
                    .view_as(table)
                )
                for row, length in enumerate(lengths):
                    if length and not prefixes[row]:
                        table[row, : (length - 1) // 128] = -1
                for current, source in zip(inputs, original):
                    source.neg_()
                    current.copy_(source)
                    current[sum(lengths) :] = float("nan")
                alpha.mul_(0.999)
                beta.mul_(0.99)
                base_pool.neg_()
                pool.copy_(base_pool)
                initial.fill_(float("nan"))
                checkpoints.fill_(float("nan"))
                final.fill_(float("nan"))
                # Every replay must overwrite live output, including after a
                # larger request left this shared bank populated.
                output.fill_(float("nan"))
                graph.replay()
                for current, source in zip(inputs, original):
                    torch.testing.assert_close(
                        current[: sum(lengths)],
                        source[: sum(lengths)],
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        current[sum(lengths) :],
                        torch.zeros_like(current[sum(lengths) :]),
                        rtol=0,
                        atol=0,
                    )
                expected_output = torch.zeros_like(output)
                expected_pool = base_pool.clone()
                touched = set()
                cu = [0]
                for length in lengths:
                    cu.append(cu[-1] + length)
                if cu[-1]:
                    start_state = torch.zeros_like(initial[: len(lengths)])
                    for row, prefix in enumerate(prefixes):
                        if prefix and lengths[row]:
                            start_state[row] = base_pool[
                                int(table[row, (prefix - 1) // 128])
                            ]
                    # Compare the entire live batch, retaining eager's original
                    # CP policy. Per-row calls can select a different CP length.
                    q, k, v = (tensor[: cu[-1]].contiguous() for tensor in inputs)
                    reference, pages, end, _ = sm120_prefill_prepared(
                        q,
                        k,
                        v,
                        alpha[: cu[-1]].contiguous(),
                        beta[: cu[-1]].contiguous(),
                        start_state,
                        torch.tensor(cu, device="cuda", dtype=torch.int32),
                        lengths,
                    )
                    expected_output[:, : cu[-1]] = reference
                    torch.testing.assert_close(
                        checkpoints[: sum(n // 128 for n in lengths)],
                        pages,
                        rtol=0,
                        atol=0,
                    )
                checkpoint = 0
                for row, length in enumerate(lengths):
                    if not length:
                        continue
                    torch.testing.assert_close(final[row], end[row], rtol=0, atol=0)
                    for page in range((length + 127) // 128):
                        slot = int(table[row, prefixes[row] // 128 + page])
                        if slot > 0:
                            expected_pool[slot] = (
                                end[row]
                                if (page + 1) * 128 >= length
                                else pages[checkpoint + page]
                            )
                            touched.add(slot)
                    checkpoint += length // 128
                if prepared.output_live_only:
                    torch.testing.assert_close(
                        output[:, : cu[-1]],
                        expected_output[:, : cu[-1]],
                        rtol=0,
                        atol=0,
                    )
                else:
                    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
                torch.testing.assert_close(pool, expected_pool, rtol=0, atol=0)
                untouched = [
                    slot for slot in range(pool.shape[0]) if slot not in touched
                ]
                torch.testing.assert_close(
                    pool[untouched], base_pool[untouched], rtol=0, atol=0
                )
        for fixture in fixtures:
            fixture[9].reset()

    def test_overwritten_cuda13_wheel_uses_native_prefill(self):
        from unittest.mock import patch

        from rtp_llm.models_py.kernels.cuda.gdn_sm120.prefill import sm120_dsl_available

        sm120_dsl_available.cache_clear()
        self.assertTrue(
            sm120_dsl_available(), "SM120 pool must provide the intact CUDA 13 compiler"
        )
        sm120_dsl_available.cache_clear()
        try:
            with patch(
                "rtp_llm.models_py.utils.cutlass.cuda13_cutlass_is_intact",
                return_value=False,
            ):
                self.assertFalse(sm120_dsl_available())
        finally:
            sm120_dsl_available.cache_clear()

    def test_output_final_and_all_page_states_with_ragged_prefix(self):
        # fail-loud rather than skip: this target is pinned to RTX_5000_PRO_CU13.
        self.assertEqual(
            torch.cuda.get_device_capability(),
            (12, 0),
            "target pool is RTX_5000_PRO_CU13 (SM120)",
        )
        for lengths in ((257,), (129, 257), (1, 7, 63)):
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
                self.assertEqual(checkpoints.shape, ref_pages.shape)
                if ref_pages.numel():
                    assert_close(
                        "SM120 pages versus FP64", ref_pages, checkpoints, 0.005
                    )
                self.assertEqual(checkpoint_starts.dtype, torch.int64)

    def test_fused_post_conv_preparation_preserves_all_boundaries(self):
        from rtp_llm.models_py.triton_kernels.common.scatter_qkv import scatter_qkv
        from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
        from rtp_llm.models_py.triton_kernels.fla.gdn_prefill_prepare import (
            prepare_sm120_prefill,
        )

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
        from rtp_llm.models_py.triton_kernels.common.scatter_qkv import scatter_qkv
        from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
        from rtp_llm.models_py.triton_kernels.fla.gdn_prefill_prepare import (
            prepare_sm120_prefill,
        )

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
        for key_heads, heads, lengths in (
            (8, 24, (257,)),
            (8, 24, (129, 257)),
            (8, 24, (1, 7, 63)),
            (4, 12, (257,)),
            (2, 6, (257,)),
            (2, 6, (4223,)),
        ):
            torch.manual_seed(79301 + sum(lengths))
            total = sum(lengths)
            q = torch.randn(
                1, total, key_heads, 128, device="cuda", dtype=torch.bfloat16
            )
            k = torch.randn_like(q)
            v = torch.randn(1, total, heads, 128, device="cuda", dtype=torch.bfloat16)
            g = -torch.rand(1, total, heads, device="cuda") * 0.1
            beta = torch.rand_like(g).bfloat16()
            cu = torch.tensor(
                [0] + list(__import__("itertools").accumulate(lengths)),
                device="cuda",
                dtype=torch.int32,
            )
            initial = torch.randn(len(lengths), heads, 128, 128, device="cuda") * 0.05
            actual, pages, final, starts = sm120_prefill_with_checkpoints(
                q, k, v, g, beta, initial, cu, lengths
            )
            ref, ref_final, ref_pages = recurrence_reference(
                q, k, v, g, beta, initial, lengths
            )
            assert_close("GQA3 CP output versus FP64", ref, actual, 0.005)
            assert_close("GQA3 final versus FP64", ref_final, final, 0.005)
            self.assertEqual(pages.shape, ref_pages.shape)
            if ref_pages.numel():
                assert_close("GQA3 every page versus FP64", ref_pages, pages, 0.005)
            if lengths == (4223,):
                # TP8 resumes its last 127 tokens from the 4096-token page.
                tail = [tensor[:, 4096:].contiguous() for tensor in (q, k, v, g, beta)]
                tail_cu = torch.tensor([0, 127], device="cuda", dtype=torch.int32)
                cached = pages[31:32].clone().contiguous()
                resumed, _, resumed_final, _ = sm120_prefill_with_checkpoints(
                    *tail, cached, tail_cu, (127,)
                )
                legacy, _, legacy_final = chunk_gated_delta_rule(
                    *tail,
                    initial_state=cached,
                    output_final_state=True,
                    cu_seqlens=tail_cu,
                    use_qk_l2norm_in_kernel=True,
                )
                for name, output, state in (
                    ("CP", resumed, resumed_final),
                    ("FLA", legacy, legacy_final),
                ):
                    assert_close(name + " resumed output", ref[:, 4096:], output, 0.005)
                    assert_close(name + " resumed final", ref_final, state, 0.005)

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
            store_page_checkpoints_kernel[(3, 3, triton.cdiv(heads * 128 * 128, 1024))](
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
            store_page_checkpoints_kernel[(3, 3, triton.cdiv(heads * 128 * 128, 1024))](
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
    page_states = (
        torch.stack(pages) if pages else states[0].new_empty((0, *states[0].shape))
    )
    return torch.stack(outputs).unsqueeze(0), torch.stack(states), page_states


if __name__ == "__main__":
    unittest.main()
