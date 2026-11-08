"""Paged-state parity with cuLA; output precision versus an FP32 recurrence."""
import unittest
from unittest.mock import patch

import torch
from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path

setup_cutlass_import_path()
from cula.kda import chunk_kda
from rtp_llm.models_py.modules.factory.attention.cuda_kda_impl import (
    cula_kda_paged_prefill,
    prepare_cula_kda_prefill_metadata,
)


def recurrent_reference(q, k, v, gate, beta, state, alog, bias):
    # cuLA's L2 normalization publishes BF16 projections before the FP32
    # recurrence. Retaining unrounded FP32 Q/K tests a different operator.
    def normalize(projection):
        values = projection.float()
        normalized = values * torch.rsqrt(
            values.square().sum(-1, keepdim=True) + 1e-6
        )
        return normalized.to(projection.dtype).float()

    q = normalize(q) * 128**-0.5
    k = normalize(k)
    decay = (-5.0 * torch.sigmoid(alog.exp()[None, :, None] *
                                (gate.float() + bias.reshape(-1, 128)))).exp()
    beta = beta.float().sigmoid()
    state = state.clone()
    outputs = []
    for t in range(q.shape[0]):
        state = state * decay[t, :, :, None]
        correction = (state * k[t, :, :, None]).sum(dim=-2)
        state = state + k[t, :, :, None] * ((v[t].float() - correction) * beta[t, :, None])[:, None, :]
        outputs.append((q[t, :, :, None] * state).sum(dim=-2))
    return torch.stack(outputs)


def native_checkpoint_reference(q, k, v, gate, beta, initial, alog, bias, prefix, expected):
    # A float32 checkpoint is storage precision, not an FP32-only recurrence:
    # cuLA's chunk algorithm also rounds intermediate projections to BF16.
    # Compare publication against the separate, unpaged final-state API, and
    # retain the independent FP32 output-accuracy gate for every endpoint.
    length = q.shape[0]
    ends = [*range(64 - prefix % 64, length, 64), length]
    states = {}
    for end in ends:
        with torch.inference_mode():
            output, state = chunk_kda(
                *(x[:end].unsqueeze(0).contiguous() for x in (q, k, v, gate)),
                beta[:end].unsqueeze(0).contiguous(),
                scale=128**-0.5,
                initial_state=initial.unsqueeze(0).clone(),
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=True, lower_bound=-5.0,
                use_intracard_cp=False, A_log=alog, dt_bias=bias,
                disable_recompute=False,
            )
        torch.testing.assert_close(output[0].float(), expected[:end], rtol=0.03, atol=0.003)
        states[end - 1] = state[0].clone()
    return states


class NativeKdaCheckpointTest(unittest.TestCase):

    def test_short_context_budget_split_matches_single_round(self):
        # A 2560-token context fits the configured 4096-token chunk. Another
        # request can leave only 1024 tokens, so continuation must use the
        # published LINEAR state even when that boundary is not retained.
        length, split, page, interval, heads = 2560, 1024, 128, 1024, 12
        torch.manual_seed(3108)
        q, k, v = [
            torch.randn(length, heads, 128, device="cuda", dtype=torch.bfloat16) * 0.1
            for _ in range(3)
        ]
        v = v * 10
        gate = torch.full_like(q, -8.0)
        beta = torch.randn(length, heads, device="cuda", dtype=torch.bfloat16) * 0.1
        alog = torch.zeros(heads, device="cuda", dtype=torch.float32)
        bias = torch.zeros(heads * 128, device="cuda", dtype=torch.float32)
        for last_n in (1, 5):
            with self.subTest(last_n=last_n):
                points = list(range(interval, length + 1, interval))[-last_n:]
                retained = {point: index + 1 for index, point in enumerate(points)}
                borrowed = len(retained) + 1
                chunk_pool = torch.zeros(
                    borrowed + 1, heads, 128, 128, device="cuda", dtype=torch.float32
                )
                full_pool = torch.zeros_like(chunk_pool)

                def run(pool, start, end):
                    row = [-1] * ((length + page - 1) // page)
                    if start:
                        row[start // page - 1] = retained.get(start, borrowed)
                    for point, block in retained.items():
                        if start < point <= end:
                            row[point // page - 1] = block
                    if end not in retained:
                        row[(end - 1) // page] = borrowed
                    return cula_kda_paged_prefill(
                        q[start:end],
                        k[start:end],
                        v[start:end],
                        gate[start:end],
                        beta[start:end],
                        alog,
                        bias,
                        -5.0,
                        pool,
                        [0, end - start],
                        [start],
                        [row],
                        page,
                    )

                with torch.inference_mode():
                    expected = run(full_pool, 0, length)
                    first = run(chunk_pool, 0, split)
                    state_block = retained.get(split, borrowed)
                    self.assertGreater(state_block, 0)
                    prefix_state = chunk_pool[state_block].clone()
                    self.assertTrue(torch.isfinite(prefix_state).all().item())
                    self.assertGreater(prefix_state.abs().max().item(), 0)
                    missing_prefix_pool = chunk_pool.clone()
                    missing_prefix_pool[state_block].zero_()
                    second = run(chunk_pool, split, length)
                    without_prefix = run(missing_prefix_pool, split, length)
                torch.testing.assert_close(
                    torch.cat((first, second)), expected, rtol=0.03, atol=0.003
                )
                # The same tolerance must reject losing the first round's state.
                with self.assertRaises(AssertionError):
                    torch.testing.assert_close(
                        without_prefix, expected[split:], rtol=0.03, atol=0.003
                    )
                for block in [*retained.values(), borrowed]:
                    torch.testing.assert_close(
                        chunk_pool[block], full_pool[block], rtol=0.03, atol=0.003
                    )

    def test_last_n_borrowed_chunks_match_disjoint_recurrent_storage(self):
        # N also exceeds the two CP checkpoints covered by one chunk.
        for n in range(1, 6):
            for tail in (0, 128):
                with self.subTest(n=n, tail=tail):
                    self._check_borrowed_chunks(n, 12*1024+tail, 2048)

    def test_last5_one_million_with_64k_chunks_and_optional_tail(self):
        for tail in (0, 128):
            with self.subTest(tail=tail):
                self._check_borrowed_chunks(5, 1_048_576+tail, 65_536)

    def _check_borrowed_chunks(self, n, length, chunk):
        page, interval, heads = 128, 1024, 12
        torch.manual_seed(3105+n)
        tensors = [torch.randn(length, heads, 128, device="cuda", dtype=torch.bfloat16)*0.1
                   for _ in range(4)]
        beta = torch.randn(length, heads, device="cuda", dtype=torch.bfloat16)*0.1
        alog = torch.zeros(heads, device="cuda", dtype=torch.float32)
        bias = torch.zeros(heads*128, device="cuda", dtype=torch.float32)
        complete = length//interval
        retained = {(complete-n+index+1)*interval: index+1 for index in range(n)}
        borrowed = n+1 if length % interval else n
        candidate = torch.zeros(borrowed+1, heads, 128, 128, device="cuda", dtype=torch.float32)
        # Every reference write gets its own state, including all earlier
        # chunk boundaries. It never aliases a source with a destination.
        points = sorted(set(range(chunk, length, chunk)) | set(retained) | {length})
        reference_blocks = {point: index+1 for index, point in enumerate(points)}
        reference = torch.zeros(len(points)+1, heads, 128, 128, device="cuda", dtype=torch.float32)
        previous, frozen = 0, {}
        for prefix in range(0, length, chunk):
            end = min(length, prefix+chunk)
            row = [-1]*((length+page-1)//page)
            control = row.copy()
            if prefix:
                row[prefix//page-1] = previous
                control[prefix//page-1] = reference_blocks[prefix]
            for point, block in retained.items():
                if prefix < point <= end:
                    row[point//page-1] = block
                    control[point//page-1] = reference_blocks[point]
            output = retained.get(end, borrowed)
            if end not in retained:
                row[(end-1)//page] = output
                control[(end-1)//page] = reference_blocks[end]
            inputs = [value[prefix:end] for value in tensors]
            args = [*inputs, beta[prefix:end], alog, bias, -5.0]
            actual = cula_kda_paged_prefill(*args, candidate,
                [0, end-prefix], [prefix], [row], page)
            expected = cula_kda_paged_prefill(*args, reference,
                [0, end-prefix], [prefix], [control], page)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(candidate[output], reference[reference_blocks[end]], atol=0, rtol=0)
            for block, state in frozen.items():
                torch.testing.assert_close(candidate[block], state, atol=0, rtol=0)
            for point, block in retained.items():
                if prefix < point <= end:
                    torch.testing.assert_close(candidate[block], reference[reference_blocks[point]], atol=0, rtol=0)
                    frozen[block] = candidate[block].clone()
            previous = output

    def test_packed_batch_strided_cache_and_layer_workspace_reuse(self):
        # Empty requests must not add a cuLA sequence; virtual padding keeps its
        # token range, publishes no state, and has exactly zero output.
        cu = [0, 129, 129, 194, 201]
        prefixes = [64, 0, 128, 0]
        tables = [[3, 5, -1, 7], [], [8, 9, 10, -1], [0]]
        heads, blocks = 12, 16
        row_size = heads * 128 * 128
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):

                def new_pool():
                    storage = torch.full(
                        (blocks, row_size + 19),
                        -77.0,
                        device="cuda",
                        dtype=dtype,
                    )
                    pool = storage.as_strided(
                        (blocks, heads, 128, 128),
                        (row_size + 19, 128 * 128, 128, 1),
                    )
                    return storage, pool

                storage, pool = new_pool()
                metadata = prepare_cula_kda_prefill_metadata(
                    cu,
                    prefixes,
                    tables,
                    64,
                    pool,
                )
                self.assertIsNotNone(metadata.packed)
                packed = metadata.packed
                pointers = (
                    packed.initial_states.data_ptr(),
                    packed.checkpoints.data_ptr(),
                )
                self.assertEqual(packed.cu_seqlens_cpu.tolist(), [0, 129, 194, 201])
                self.assertEqual(packed.checkpoint_offsets.tolist(), [0, 3, 5, 6])
                self.assertEqual(
                    packed.checkpoint_blocks.tolist(), [5, -1, 7, 10, -1, -1]
                )

                # Each layer has a different pool and different prefix state.
                # Reusing the workspace must not reuse the previous layer's H0.
                for layer in range(2):
                    with self.subTest(layer=layer):
                        if layer:
                            storage, pool = new_pool()
                        torch.manual_seed(8320 + layer)
                        for block in (3, 9):
                            pool[block].copy_(torch.randn_like(pool[block]) * 0.02)
                        before = storage.clone()
                        shape = (cu[-1], heads, 128)
                        q, k, v, gate = [
                            torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                            for _ in range(4)
                        ]
                        beta = torch.randn(
                            cu[-1],
                            heads,
                            device="cuda",
                            dtype=torch.bfloat16,
                        )
                        alog = torch.randn(heads, device="cuda") * 0.1
                        bias = torch.randn(heads * 128, device="cuda") * 0.01
                        expected = torch.zeros_like(v, dtype=torch.float32)
                        states = []
                        for request in (0, 2):
                            start, end = cu[request : request + 2]
                            initial = pool[
                                tables[request][prefixes[request] // 64 - 1]
                            ].float()
                            reference = recurrent_reference(
                                q[start:end],
                                k[start:end],
                                v[start:end],
                                gate[start:end],
                                beta[start:end],
                                initial,
                                alog,
                                bias,
                            )
                            expected[start:end].copy_(reference)
                            states.append(
                                (
                                    request,
                                    native_checkpoint_reference(
                                        q[start:end],
                                        k[start:end],
                                        v[start:end],
                                        gate[start:end],
                                        beta[start:end],
                                        initial,
                                        alog,
                                        bias,
                                        prefixes[request],
                                        reference,
                                    ),
                                )
                            )

                        packed.initial_states.fill_(float("nan"))
                        packed.checkpoints.fill_(float("nan"))
                        with patch("cula.kda.chunk_kda", wraps=chunk_kda) as call:
                            actual = cula_kda_paged_prefill(
                                q,
                                k,
                                v,
                                gate,
                                beta,
                                alog,
                                bias,
                                -5.0,
                                pool,
                                cu,
                                prefixes,
                                tables,
                                64,
                                metadata=metadata,
                            )
                        call.assert_called_once()
                        self.assertIs(
                            call.call_args.kwargs["cu_seqlens_cpu"],
                            packed.cu_seqlens_cpu,
                        )
                        self.assertIs(
                            call.call_args.kwargs["checkpoint_offsets"],
                            packed.checkpoint_offsets,
                        )
                        self.assertEqual(call.call_args.args[0].shape[1], cu[-1])
                        torch.testing.assert_close(
                            actual.float(),
                            expected,
                            rtol=0.03,
                            atol=0.003,
                        )
                        self.assertEqual(torch.count_nonzero(actual[194:]).item(), 0)
                        written = set()
                        for request, checkpoints in states:
                            for endpoint, state in checkpoints.items():
                                block = tables[request][
                                    (prefixes[request] + endpoint) // 64
                                ]
                                if block > 0:
                                    torch.testing.assert_close(
                                        pool[block].float(),
                                        state,
                                        rtol=0.03,
                                        atol=0.003,
                                    )
                                    written.add(block)
                        untouched = [
                            block for block in range(blocks) if block not in written
                        ]
                        torch.testing.assert_close(
                            storage[untouched], before[untouched], rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            storage[:, row_size:], before[:, row_size:], rtol=0, atol=0
                        )
                        self.assertEqual(
                            pointers,
                            (
                                packed.initial_states.data_ptr(),
                                packed.checkpoints.data_ptr(),
                            ),
                        )

    def test_bounded_gate_and_every_checkpoint(self):
        original_empty = torch.empty
        poisoned = []

        def empty_with_poison(*args, **kwargs):
            result = original_empty(*args, **kwargs)
            if result.ndim == 5 and result.shape[-2:] == (128, 128) and result.dtype == torch.float32:
                result.fill_(float("nan"))
                poisoned.append(result.shape)
            return result

        def checked_chunk(*args, **kwargs):
            initial = kwargs["initial_state"].clone()
            result = chunk_kda(*args, **kwargs)
            torch.testing.assert_close(kwargs["initial_state"], initial, rtol=0, atol=0)
            return result

        for heads, length, prefix in ((12, 1, 0), (12, 63, 0), (12, 64, 0),
                                      (24, 65, 0), (12, 131, 0), (24, 129, 31),
                                      (12, 401, 0)):
            with self.subTest(heads=heads, length=length, prefix=prefix):
                torch.manual_seed(5700 + heads + length)
                shape = (length, heads, 128)
                q, k, v, gate = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
                beta = torch.randn(length, heads, device="cuda", dtype=torch.bfloat16)
                alog = torch.randn(heads, device="cuda") * 0.1
                bias = torch.randn(heads * 128, device="cuda") * 0.01
                initial = torch.randn(heads, 128, 128, device="cuda") * 0.02
                blocks = (prefix + length + 63) // 64
                table = list(range(1, blocks + 1))
                if length == 401:
                    # Retain prefix history and make continuation reads observable.
                    gate = (gate.float() * 0.1 - 4.0).to(gate.dtype)
                    for boundary in (128, 256, 384):
                        q[boundary].copy_(k[boundary - 1])
                    table.reverse()
                pool = torch.full((blocks + 3, heads, 128, 128), -77.0, device="cuda")
                if prefix:
                    pool[table[(prefix - 1) // 64]].copy_(initial)
                else:
                    initial.zero_()
                initial_pool = pool.clone()
                expected = recurrent_reference(q, k, v, gate, beta, initial, alog, bias)
                states = native_checkpoint_reference(
                    q, k, v, gate, beta, initial, alog, bias, prefix, expected,
                )
                chunk_sizes = (length, 128) if length == 401 else (length,)
                for chunk_size in chunk_sizes:
                    pool.copy_(initial_pool)
                    for start in range(0, length, chunk_size):
                        end = min(start + chunk_size, length)
                        with self.subTest(chunk_size=chunk_size, start=start):
                            before = pool.clone()
                            if start:
                                reset = recurrent_reference(
                                    q[start:start + 1], k[start:start + 1], v[start:start + 1],
                                    gate[start:start + 1], beta[start:start + 1], initial, alog, bias,
                                )
                                self.assertGreater((expected[start] - reset[0]).abs().max().item(), 0.01)
                            with patch("torch.empty", side_effect=empty_with_poison), patch("cula.kda.chunk_kda", side_effect=checked_chunk):
                                actual = cula_kda_paged_prefill(
                                    q[start:end], k[start:end], v[start:end], gate[start:end],
                                    beta[start:end], alog, bias, -5.0, pool,
                                    [0, end - start], [prefix + start], [table], 64,
                                )
                            # Preserve the existing bounded-KDA operator tolerances.
                            torch.testing.assert_close(actual.float(), expected[start:end], rtol=0.03, atol=0.003)
                            written = set()
                            for endpoint, state in states.items():
                                if not start <= endpoint < end:
                                    continue
                                block = table[(prefix + endpoint) // 64]
                                torch.testing.assert_close(pool[block], state, rtol=0.03, atol=0.003)
                                self.assertTrue(torch.isfinite(pool[block]).all().item())
                                written.add(block)
                            untouched = [i for i in range(pool.shape[0]) if i not in written]
                            torch.testing.assert_close(pool[untouched], before[untouched], rtol=0, atol=0)
        self.assertTrue(poisoned)


if __name__ == "__main__":
    unittest.main()
