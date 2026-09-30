"""Reference pooling, chunk carry and explicit ratio-2 state commit tests."""

import dataclasses
import unittest

import torch

from rtp_llm.models.test.dsv41_cache_codec_test import load_codec_source

compressor = load_codec_source("_dsv41_compressor_test", "v41_compressor.py")


def state(pages=2, gen=1):
    entries = (2 + gen + 1) & ~1
    raw = torch.full((pages, entries * 1024 + 64), 173.0)
    return compressor.V41PairState(raw[:, : entries * 1024], gen), raw


def metadata(records, entries, device="cpu"):
    lookup = {item: row for row, item in enumerate(records) if item is not None}
    final = {}
    for item in lookup:
        request, position = item
        final[request] = max(final.get(request, -1), position)
    data = [[] for _ in range(6)]
    for item in records:
        if item is None:
            fields = (-999, -999, False, 999999, 999999, 999999)
        else:
            request, position = item
            prev = lookup.get((request, position - 1), -1) if position % 2 else -1
            slot = (
                request * entries + (position - 1) % entries
                if position % 2 and prev < 0
                else -1
            )
            write = (
                request * entries + position % entries
                if position + entries > final[request]
                else -1
            )
            fields = (request, position, True, prev, slot, write)
        for values, value in zip(data, fields):
            values.append(value)
    return compressor.V41PairMetadata(
        *[
            torch.tensor(
                values, device=device, dtype=torch.bool if i == 2 else torch.int64
            )
            for i, values in enumerate(data)
        ]
    )


def reference(v0, v1, s0, s1, norm):
    weights = torch.stack((s0, s1)).softmax(0)
    pooled = (torch.stack((v0, v1)) * weights).sum(0).bfloat16().float()
    return (pooled * torch.rsqrt(pooled.square().mean() + 1e-20) * norm).bfloat16()


class V41CompressorTest(unittest.TestCase):
    def test_full_and_chunked_pairs_with_state_relocation(self):
        torch.manual_seed(417)
        values, scores = torch.randn(12, 512), torch.randn(12, 512)
        norm = torch.randn(512).bfloat16()
        expected = torch.stack(
            [
                reference(values[p - 1], values[p], scores[p - 1], scores[p], norm)
                for p in range(1, 12, 2)
            ]
        )
        for chunks in ((12,), (1, 3, 2, 1, 5), (3, 1, 1, 7)):
            pool, backing = state()
            start = 0
            state_offset = 0
            produced = []
            for count in chunks:
                records = [(0, position) for position in range(start, start + count)]
                meta = metadata(records, pool.entries)
                meta = dataclasses.replace(
                    meta,
                    previous_slots=torch.where(
                        meta.previous_slots >= 0, meta.previous_slots + state_offset, -1
                    ),
                    write_slots=torch.where(
                        meta.write_slots >= 0, meta.write_slots + state_offset, -1
                    ),
                )
                before = pool.raw.clone()
                frame = compressor.prepare_ratio2(
                    values[start : start + count],
                    scores[start : start + count],
                    norm,
                    pool,
                    meta,
                    backend="eager",
                )
                torch.testing.assert_close(pool.raw, before, atol=0, rtol=0)
                produced.append(frame.latent[frame.complete])
                frame.commit()
                self.assertTrue(bool((backing[:, -64:] == 173).all()))
                # A PD destination can resume from exactly the transferred FP32 ring.
                saved = pool.raw[state_offset // pool.entries].clone()
                pool, backing = state()
                pool.raw[1].copy_(saved)
                state_offset = pool.entries
                start += count
            torch.testing.assert_close(torch.cat(produced), expected, atol=0, rtol=0)

    def test_cp_permutation_keeps_same_request_predecessors(self):
        torch.manual_seed(418)
        records = [(0, 3), (1, 7), (0, 0), (1, 4), (0, 1), (1, 5), (0, 2), (1, 6), None]
        values, scores = torch.randn(9, 512), torch.randn(9, 512)
        values[-1] = scores[-1] = float("nan")
        pool, _ = state()
        norm = torch.ones(512).bfloat16()
        meta = metadata(records, pool.entries)
        frame = compressor.prepare_ratio2(
            values, scores, norm, pool, meta, backend="eager"
        )
        for row, item in enumerate(records):
            if item is not None and item[1] % 2:
                prev = records.index((item[0], item[1] - 1))
                expected = reference(
                    values[prev], values[row], scores[prev], scores[row], norm
                )
                torch.testing.assert_close(frame.latent[row], expected, atol=0, rtol=0)
            else:
                self.assertTrue(bool((frame.latent[row] == 0).all()))
        frame.commit()

    def test_pool_rounds_to_bf16_before_normalization(self):
        pool, _ = state()
        values = torch.arange(1024).reshape(2, 512).float() / 997 + 0.0013
        scores = torch.zeros_like(values)
        norm = torch.ones(512).bfloat16()
        frame = compressor.prepare_ratio2(
            values,
            scores,
            norm,
            pool,
            metadata([(0, 0), (0, 1)], pool.entries),
            backend="eager",
        )
        expected = reference(values[0], values[1], scores[0], scores[1], norm)
        torch.testing.assert_close(frame.latent[1], expected, atol=0, rtol=0)
        unrounded = values.mean(0)
        wrong = (unrounded * torch.rsqrt(unrounded.square().mean() + 1e-20)).bfloat16()
        self.assertFalse(torch.equal(frame.latent[1], wrong))

    def test_deferred_commit_snapshots_values_and_slots_before_ring_wrap(self):
        pool, _ = state(gen=0)
        old = torch.cat((torch.full((1, 512), 3.0), torch.zeros(1, 512)), -1)
        pool.write(torch.tensor([0]), old, backend="eager")
        values = torch.stack((torch.full((512,), 5.0), torch.full((512,), 99.0)))
        scores = torch.zeros_like(values)
        meta = metadata([(0, 3), (0, 4)], pool.entries)
        before = pool.raw.clone()
        frame = compressor.prepare_ratio2(
            values, scores, torch.ones(512).bfloat16(), pool, meta, backend="eager"
        )
        torch.testing.assert_close(pool.raw, before, atol=0, rtol=0)
        torch.testing.assert_close(
            frame.latent[0], torch.ones(512).bfloat16(), atol=0, rtol=0
        )
        values.fill_(-123)
        meta.write_slots.fill_(-1)
        frame.commit()
        expected = torch.cat((torch.full((1, 512), 99.0), torch.zeros(1, 512)), -1)
        torch.testing.assert_close(
            pool.read(torch.tensor([0]), backend="eager"), expected, atol=0, rtol=0
        )

    def test_rejects_cross_request_predecessors_and_duplicate_writes_before_mutation(
        self,
    ):
        pool, _ = state()
        values, scores, norm = (
            torch.ones(2, 512),
            torch.zeros(2, 512),
            torch.ones(512).bfloat16(),
        )
        meta = metadata([(0, 0), (0, 1)], pool.entries)
        before = pool.raw.clone()
        bad = [
            dataclasses.replace(meta, request_ids=torch.tensor([0, 1])),
            dataclasses.replace(meta, previous_slots=torch.tensor([-1, 0])),
            dataclasses.replace(meta, write_slots=torch.tensor([0, 0])),
            dataclasses.replace(meta, previous_rows=torch.tensor([-1, 999])),
        ]
        for item in bad:
            with self.assertRaises(ValueError):
                compressor.prepare_ratio2(
                    values, scores, norm, pool, item, backend="eager"
                )
            torch.testing.assert_close(pool.raw, before, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "alias"):
            pool.write(
                pool.raw[0].view(torch.int64)[:1], torch.zeros(1, 1024), backend="eager"
            )

    def test_empty_batch_and_nonfinite_previous_state(self):
        pool, _ = state()
        norm = torch.ones(512).bfloat16()
        frame = compressor.prepare_ratio2(
            torch.empty(0, 512),
            torch.empty(0, 512),
            norm,
            pool,
            metadata([], pool.entries),
            backend="eager",
        )
        frame.commit()
        self.assertEqual(frame.latent.shape, (0, 512))
        pool.raw.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "state must be finite"):
            compressor.prepare_ratio2(
                torch.ones(1, 512),
                torch.zeros(1, 512),
                norm,
                pool,
                metadata([(0, 1)], pool.entries),
                backend="eager",
            )


if __name__ == "__main__":
    unittest.main()
