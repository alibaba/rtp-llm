"""CPU-only producer wiring: compact rows, immutable halo, carry and fallback."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_fp4_triton as codec
from rtp_llm.models_py.modules.dsv4.fp8 import attention_v41 as attention


class PrefillProducerIntegrationCPU(unittest.TestCase):
    def test_slots_forward_physical_owner_and_state_end_without_rewriting(self):
        positions = torch.tensor([0, 1, 7, 15, 16, 31])
        requests = torch.tensor([0, 0, 0, 1, 1, 1])
        ends = torch.tensor([8, 32])
        table = torch.arange(1, 33).reshape(2, 16).int()
        sentinel = torch.full_like(positions, 73)
        for ratio in (1, 2):
            for cp_size in (1, 4):
                for state_region in (False, True):
                    region = (
                        attention.CSA_STATE if state_region else attention.INDEXER_KV
                    )
                    cp = SimpleNamespace(
                        cp_size=cp_size, cp_rank=0, kv_cache_sharded=True
                    )
                    source = SimpleNamespace(
                        compress_ratio=ratio,
                        _cp_ctx=cp,
                        _kv_cache=SimpleNamespace(seq_size_per_block=16),
                        _source_pool=lambda region: torch.empty(32, 4),
                        _source_entries=lambda region, pool: 2,
                        _block_tables_by_type={region: table},
                    )
                    with patch.object(
                        attention, "require_pool_tokens_per_block", return_value=4
                    ), patch.object(
                        attention.prefill_metadata,
                        "try_slot_mapping",
                        return_value=sentinel,
                    ) as fused:
                        actual = attention.AttentionV41FP8._slots(
                            source,
                            region,
                            positions,
                            requests,
                            state_end=ends if state_region else None,
                        )
                    self.assertIs(actual, sentinel)
                    args, kwargs = fused.call_args
                    self.assertIs(args[0], positions)
                    self.assertIs(args[1], requests)
                    self.assertIs(args[2], table)
                    self.assertEqual(args[3:], (2, 4, ratio, cp_size, 0))
                    self.assertEqual(
                        kwargs["owner_tokens_per_block"], 16 if cp_size > 1 else 4
                    )
                    self.assertEqual(kwargs["state"], state_region)
                    self.assertIs(kwargs["seq_ends"], ends if state_region else None)

    def test_state_store_fast_and_fallback_keep_unowned_sentinel_unchanged(self):
        slots = torch.tensor([-1, 2, 3])
        values, scores = torch.arange(12).view(3, 4).float(), torch.ones(3, 4)
        expected = torch.full((4, 8), -31.0)
        expected[2:] = torch.cat((values[1:], scores[1:]), -1)
        for fast in (True, False):
            pool = torch.full((4, 8), -31.0)
            source = SimpleNamespace(
                _slots=lambda *args, **kwargs: slots, _source_pool=lambda region: pool
            )

            def store(v, s, mapped, target):
                self.assertIs(target, pool)
                self.assertIs(mapped, slots)
                if fast:
                    target[mapped[1:]] = torch.cat((v[1:], s[1:]), -1)
                return fast

            with patch.object(
                attention.prefill_global, "store_states", side_effect=store
            ) as kernel:
                attention.AttentionV41FP8._write_states(
                    source,
                    values,
                    scores,
                    torch.arange(3),
                    torch.zeros(3).long(),
                    torch.tensor([3]),
                )
            kernel.assert_called_once()
            torch.testing.assert_close(pool, expected, rtol=0, atol=0)

    def _run(
        self,
        ratio,
        tile_rows,
        main_fused,
        index_fused,
        pool_bound,
        empty=False,
        index_only=False,
    ):
        starts_host, lengths_host = ([0], [0]) if empty else ([1, 0, 3], [4, 1, 2])
        starts = torch.tensor(starts_host)
        lengths = torch.tensor(lengths_host)
        positions = torch.tensor(
            [
                p
                for start, length in zip(starts_host, lengths_host)
                for p in range(start, start + length)
            ],
            dtype=torch.long,
        )
        req_ids = torch.repeat_interleave(torch.arange(len(starts)), lengths)
        rows, head = len(positions), 4
        generator = torch.Generator().manual_seed(219)
        raw = torch.randn(rows, 2 * head, generator=generator)
        original_state = torch.randn(len(starts), 2 * head, generator=generator)
        state = original_state.clone()
        main_pool, index_pool = (
            torch.zeros(400, head).bfloat16(),
            torch.zeros(400, head).bfloat16(),
        )
        norm = torch.tensor([0.5, 1, 1.5, 2.0]).bfloat16()
        projection = torch.eye(head).bfloat16()
        freqs = torch.polar(torch.ones(16, 1), torch.arange(16).float()[:, None] * 0.15)
        calls, writes, latents = [], [], []
        owner = SimpleNamespace(
            global_norm=norm, index_wk=projection, index_k_norm=norm
        )

        def slots(region, pos, req, **kwargs):
            # CP4 owner filtering; inactive rows must still produce compact latent.
            slot = req * 100 + pos // ratio + 1
            return torch.where((pos // 2) % 4 == 1, slot, -1)

        def store(data, pool, mapping):
            valid = mapping >= 0
            pool[mapping[valid]] = data[valid]

        def read(pool, mapping):
            return torch.where((mapping >= 0)[:, None], pool[mapping.clamp_min(0)], 0)

        def write_states(values, scores, pos, req, ends):
            calls.append("state")
            writes.append(
                (values.clone(), scores.clone(), pos.clone(), req.clone(), ends.clone())
            )
            # A real ring may overwrite the predecessor. The snapshot must not alias it.
            state.fill_(999)

        def compress(
            values,
            scores,
            norm,
            eps,
            pos,
            req,
            begin,
            previous,
            boundaries,
            frequencies,
            pool,
            mapping,
            actual_ratio,
            carry,
        ):
            calls.append("main")
            self.assertEqual(actual_ratio, ratio)
            if ratio == 2:
                torch.testing.assert_close(previous, original_state)
            if not main_fused:
                return None
            pooled = []
            for token in boundaries.tolist():
                if ratio == 1:
                    pooled.append(values[token])
                    continue
                if int(pos[token]) == int(begin[req[token]]):
                    before_v, before_s = previous[req[token]].chunk(2)
                elif token == 0:
                    before_v, before_s = (
                        (carry[0][0], carry[1][0])
                        if carry is not None
                        else (values[token], scores[token])
                    )
                else:
                    before_v, before_s = values[token - 1], scores[token - 1]
                w = torch.stack((before_s, scores[token])).softmax(0)
                pooled.append((torch.stack((before_v, values[token])) * w).sum(0))
            pooled = torch.stack(pooled) if pooled else torch.empty(0, head)
            latent = attention.rms_norm(pooled, norm, eps).bfloat16()
            latents.append(latent.clone())
            angles = frequencies[pos[boundaries] // ratio * ratio]
            store(attention.rope_only(latent.clone(), angles, 2), pool, mapping)
            return latent

        def store_index(
            projected, weight, eps, pos, frequencies, pool, mapping, actual_ratio
        ):
            calls.append("index")
            self.assertEqual(actual_ratio, ratio)
            if not index_fused:
                return False
            keys = attention.rope_only(
                attention.rms_norm(projected, weight, eps),
                frequencies[pos // ratio * ratio],
                2,
            )
            store(keys, pool, mapping)
            return True

        # Reuse one tile buffer: carry must be cloned before generator resumes.
        def tiles():
            buffer = torch.empty(tile_rows, 2 * head)
            for t0 in range(0, max(rows, 1), tile_rows):
                count = min(tile_rows, rows - t0)
                buffer.fill_(-991)
                buffer[:count].copy_(raw[t0 : t0 + count])
                yield t0, buffer[:count]

        shared = {"prefill_chunk_meta": object()}
        attn = SimpleNamespace(
            compress_ratio=ratio,
            head_dim=head,
            rope_head_dim=2,
            eps=1e-6,
            layer_id=2,
            index_n_heads=32,
            freqs_cis=freqs,
            _cp_ctx=SimpleNamespace(
                input_lengths_global_host=lengths_host, prefix_lengths_host=starts_host
            ),
            _owner=lambda: owner,
            _global_region=lambda: attention.CSA_KV if ratio == 2 else attention.HCA_KV,
            _source_pool=lambda region: (
                (index_pool if region == attention.INDEXER_KV else main_pool)
                if pool_bound or (index_only and region == attention.INDEXER_KV)
                else None
            ),
            _slots=slots,
            _read_state=lambda *args: state.clone(),
            _write_states=write_states,
            _gather_shards=lambda value: value,
            _shared_attention=shared,
        )
        with ExitStack() as stack:
            main_mock = stack.enter_context(
                patch.object(
                    attention.prefill_global, "compress_main", side_effect=compress
                )
            )
            index_mock = stack.enter_context(
                patch.object(
                    attention.prefill_global, "store_index", side_effect=store_index
                )
            )
            stack.enter_context(
                patch.object(
                    attention.prefill_indexer, "is_supported", return_value=False
                )
            )
            stack.enter_context(
                patch.object(attention, "fp8_roundtrip", side_effect=lambda x: x)
            )
            stack.enter_context(
                patch.object(
                    codec, "quantize_and_insert_k_cache_fp4", side_effect=store
                )
            )
            stack.enter_context(
                patch.object(
                    codec,
                    "quantize_indexer_k_fp4",
                    side_effect=lambda value, mapping, pool: store(
                        value, pool, mapping
                    ),
                )
            )
            stack.enter_context(
                patch.object(codec, "dequantize_indexer_k_fp4", side_effect=read)
            )
            stack.enter_context(
                patch.object(codec, "gather_k_cache_bytes_fp4", side_effect=read)
            )
            stack.enter_context(
                patch.object(
                    codec, "dequantize_k_cache_bytes_fp4", side_effect=lambda x: x
                )
            )
            attention.AttentionV41FP8._produce_global(
                attn,
                torch.empty(rows, 1),
                positions,
                req_ids,
                starts,
                lengths,
                prefill=True,
                projected_tiles=tiles(),
            )
        self.assertNotIn("prefill_chunk_meta", shared)
        self.assertEqual(
            main_mock.call_count,
            (max(rows, 1) + tile_rows - 1) // tile_rows if pool_bound else 0,
        )
        self.assertEqual(index_mock.call_count, main_mock.call_count)
        if ratio == 2:
            self.assertEqual(len(writes), (max(rows, 1) + tile_rows - 1) // tile_rows)
            for _, _, _, _, ends in writes:
                torch.testing.assert_close(ends, starts + lengths)
            if pool_bound:
                self.assertEqual(
                    calls,
                    [stage for _ in writes for stage in ("main", "state", "index")],
                )
        else:
            self.assertEqual(writes, [])
        return main_pool, index_pool, shared["global"][2], writes, latents

    def test_ratio1_and_2_all_fast_slow_combinations_match_across_tiles(self):
        for ratio in (1, 2):
            for tile_rows in (1, 2, 3, 7):
                reference = self._run(ratio, tile_rows, False, False, True)
                for main_fused, index_fused in (
                    (True, True),
                    (False, True),
                    (True, False),
                ):
                    with self.subTest(
                        ratio=ratio, tile=tile_rows, main=main_fused, index=index_fused
                    ):
                        actual = self._run(
                            ratio, tile_rows, main_fused, index_fused, True
                        )
                        for got, expected in zip(actual[:3], reference[:3]):
                            torch.testing.assert_close(got, expected, rtol=0, atol=0)
                        if ratio == 2 and tile_rows == 1 and main_fused:
                            self.assertTrue(
                                any(len(latent) == 0 for latent in actual[4])
                            )

    def test_pool_free_warmup_uses_new_boundary_keys_across_request_prefixes(self):
        for ratio in (1, 2):
            reference = self._run(ratio, 7, False, False, False)
            for tile_rows in (1, 2, 3):
                actual = self._run(ratio, tile_rows, True, True, False)
                torch.testing.assert_close(actual[2], reference[2], rtol=0, atol=0)
                expected = [
                    length if ratio == 1 else (start + length) // 2 - start // 2
                    for start, length in zip((1, 0, 3), (4, 1, 2))
                ]
                self.assertEqual([len(g) for g, k in actual[2]], expected)

    def test_main_pool_unbound_keeps_warmup_keys_with_index_pool_bound(self):
        for ratio in (1, 2):
            expected = self._run(ratio, 2, False, False, False)
            actual = self._run(ratio, 2, True, True, False, index_only=True)
            torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)

    def test_zero_rows_and_empty_compact_outputs_keep_valid_shapes(self):
        for ratio in (1, 2):
            for pool_bound in (False, True):
                for main_fused, index_fused in ((False, False), (True, True)):
                    with self.subTest(ratio=ratio, pool=pool_bound, fused=main_fused):
                        actual = self._run(
                            ratio, 1, main_fused, index_fused, pool_bound, empty=True
                        )
                        self.assertEqual(actual[2][0][0].shape, (0, 4))
                        self.assertEqual(actual[2][0][1].shape, (0, 4))


if __name__ == "__main__":
    unittest.main()
