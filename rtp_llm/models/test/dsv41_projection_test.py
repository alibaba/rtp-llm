"""Reference projection order, distinct cache planes and delayed publication."""

import dataclasses
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.config.dsv41_config import V41LayerSources
from rtp_llm.models.test.dsv41_cache_codec_test import (
    codec,
    load_codec_source,
    make_cache,
    reference_rows,
)
from rtp_llm.models.test.dsv41_compressor_test import compressor, metadata, state
from rtp_llm.models.test.dsv41_mxfp8_test import (
    grouped_reference,
    linear,
    quantization_reference,
)

rope = load_codec_source("_dsv41_projection_rope", "rope.py")
with patch.dict(
    sys.modules,
    {
        "rtp_llm.models_py.modules.dsv4.rope": rope,
        "rtp_llm.models_py.modules.dsv4.v41_cache": codec,
        "rtp_llm.models_py.modules.dsv4.v41_compressor": compressor,
        "rtp_llm.models_py.modules.dsv4.v41_mxfp8": linear,
    },
):
    projection = load_codec_source("_dsv41_projection_test", "v41_projection.py")


def fixture(ratio, device="cpu", hidden=128):
    torch.manual_seed(420)
    backend = "eager" if device == "cpu" else "flashinfer"
    wkv = linear.V41MXFP8Linear(
        (torch.randn(512, hidden, device=device) * 8).to(torch.float8_e4m3fn),
        torch.full((16, hidden // 32), 121, dtype=torch.uint8, device=device),
        backend=backend,
    )
    sources = (
        V41LayerSources(0, 0, None, None, None)
        if ratio == 0
        else (
            V41LayerSources(2, 2, 2, 2, None)
            if ratio == 2
            else V41LayerSources(20, 1, 20, 20, 20)
        )
    )
    freq = rope.precompute_freqs_cis(
        64,
        1024,
        65536 if ratio else 0,
        160000 if ratio else 10000,
        16,
        32,
        1,
        device=torch.device(device),
    )
    kwargs = {}
    if ratio:
        kwargs = {
            "compressor_weight": (
                torch.randn(512, hidden, device=device) / hidden**0.5
            ).bfloat16(),
            "compressor_norm": torch.randn(512, device=device).bfloat16(),
            "index_weight": (
                torch.randn(128, 512, device=device) / 512**0.5
            ).bfloat16(),
            "index_norm": torch.randn(128, device=device).bfloat16(),
        }
        if ratio == 2:
            kwargs["gate_weight"] = (
                torch.randn(512, hidden, device=device) / hidden**0.5
            ).bfloat16()
    op = projection.V41KVProjection(
        sources, wkv, torch.randn(512, device=device).bfloat16(), freq, **kwargs
    )
    return op


def norm(x, weight):
    value = x.float()
    return (
        value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-20) * weight
    ).to(x.dtype)


def rotate(x, frequencies):
    tail = x[..., -64:].float().reshape(*x.shape[:-1], 32, 2)
    rotated = torch.view_as_real(torch.view_as_complex(tail) * frequencies).flatten(-2)
    return torch.cat((x[..., :-64], rotated.to(x.dtype)), -1)


def swa_reference(op, x, positions):
    raw, scales = quantization_reference(x)
    projected = grouped_reference(
        raw.view(torch.float8_e4m3fn), scales, op.wkv.weight, op.wkv.scales
    )
    return rotate(norm(projected, op.kv_norm), op.frequencies[positions])


class V41ProjectionTest(unittest.TestCase):
    def test_ratio1_projects_index_from_unrotated_latent_and_publishes_exact_bytes(
        self,
    ):
        op = fixture(1)
        x = torch.randn(4, 128).bfloat16()
        positions = torch.tensor([255, 0, 256, -999])
        valid = torch.tensor([True, True, True, False])
        x[-1] = float("nan")
        frame = op.prepare(x, positions, valid)
        latent = norm(F.linear(x[:3], op.compressor_weight), op.compressor_norm)
        index = rotate(
            norm(F.linear(latent, op.index_weight), op.index_norm),
            op.frequencies[positions[:3]],
        )
        global_kv = rotate(latent, op.frequencies[positions[:3]])
        swa = swa_reference(op, x[:3], positions[:3])
        for actual, expected in (
            (frame.swa[:3], swa),
            (frame.global_kv[:3], global_kv),
            (frame.index_k[:3], index),
        ):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        # Applying GLOBAL RoPE before the index projection changes the key.
        wrong = rotate(
            norm(F.linear(global_kv, op.index_weight), op.index_norm),
            op.frequencies[positions[:3]],
        )
        self.assertFalse(torch.equal(index, wrong))
        caches = [make_cache(tag, gap=64) for tag in ("swa_kv", "hca_kv", "indexer_kv")]
        slots = torch.tensor([257, 1, 256, 999999])
        swa_slots = torch.tensor([3, -1, 2, -999])
        frame.publish(
            caches[0][0],
            swa_slots,
            global_cache=caches[1][0],
            global_slots=slots,
            index_cache=caches[2][0],
            index_slots=slots,
        )
        for (cache, backing), rows, selected in zip(
            caches, (swa, global_kv, index), (swa_slots[:3], slots[:3], slots[:3])
        ):
            live = selected >= 0
            torch.testing.assert_close(
                cache.gather(selected[live], backend="eager"),
                reference_rows(rows[live], cache.tag),
                atol=0,
                rtol=0,
            )
            self.assertTrue(bool((backing[:, -64:] == 173).all()))
        for value in (frame.swa, frame.index_k, frame.global_kv):
            self.assertTrue(bool((value[-1] == 0).all()))

    def test_ratio2_cp_permutation_and_distinct_index_page_stride(self):
        op = fixture(2)
        x = torch.randn(5, 128).bfloat16()
        records = [(0, 257), (0, 254), (0, 255), (0, 256), None]
        pair_state, _ = state()
        meta = metadata(records, pair_state.entries)
        x[-1] = float("nan")
        before = pair_state.raw.clone()
        frame = op.prepare(
            x,
            meta.positions,
            meta.valid_rows,
            pair_state=pair_state,
            pair_metadata=meta,
        )
        torch.testing.assert_close(pair_state.raw, before, atol=0, rtol=0)
        values = F.linear(x[:4].float(), op.compressor_weight)
        scores = F.linear(x[:4].float(), op.gate_weight)
        for row, prev in ((0, 3), (2, 1)):
            pooled = (
                (
                    torch.stack((values[prev], values[row]))
                    * torch.stack((scores[prev], scores[row])).softmax(0)
                )
                .sum(0)
                .bfloat16()
            )
            latent = norm(pooled, op.compressor_norm)
            freqs = op.frequencies[meta.positions[row] - 1]
            torch.testing.assert_close(
                frame.global_kv[row], rotate(latent, freqs), atol=0, rtol=0
            )
            torch.testing.assert_close(
                frame.index_k[row],
                rotate(norm(F.linear(latent, op.index_weight), op.index_norm), freqs),
                atol=0,
                rtol=0,
            )
        swa, _ = make_cache("swa_kv")
        global_cache, _ = make_cache("csa_kv")
        index_cache, _ = make_cache("indexer_kv")
        global_slots = torch.tensor([128, -1, 127, -1, -999])
        index_slots = torch.tensor([256, -1, 127, -1, -999])
        frame.publish(
            swa,
            torch.tensor([0, 1, 2, 3, -999]),
            global_cache=global_cache,
            global_slots=global_slots,
            index_cache=index_cache,
            index_slots=index_slots,
        )
        for cache, slots, rows in (
            (global_cache, global_slots, frame.global_kv),
            (index_cache, index_slots, frame.index_k),
        ):
            torch.testing.assert_close(
                cache.gather(slots[frame.complete], backend="eager"),
                reference_rows(rows[frame.complete], cache.tag),
                atol=0,
                rtol=0,
            )
        saved = pair_state.read(torch.tensor([1, 2, 3, 0]), backend="eager")
        torch.testing.assert_close(
            saved, torch.cat((values, scores), -1), atol=0, rtol=0
        )
        self.assertTrue(bool((index_cache.raw[0, 128 * 64 : 256 * 64] == 173).all()))

    def test_chunk_carry_and_relocated_state_match_full_projection(self):
        op = fixture(2)
        x = torch.randn(8, 128).bfloat16()
        pool, _ = state()
        all_meta = metadata([(0, p) for p in range(8)], pool.entries)
        expected = op.prepare(
            x,
            all_meta.positions,
            all_meta.valid_rows,
            pair_state=pool,
            pair_metadata=all_meta,
        )
        produced = []
        offset, start = 0, 0
        for count in (1, 2, 3, 2):
            meta = metadata([(0, p) for p in range(start, start + count)], pool.entries)
            meta = dataclasses.replace(
                meta,
                previous_slots=torch.where(
                    meta.previous_slots >= 0, meta.previous_slots + offset, -1
                ),
                write_slots=torch.where(
                    meta.write_slots >= 0, meta.write_slots + offset, -1
                ),
            )
            frame = op.prepare(
                x[start : start + count],
                meta.positions,
                meta.valid_rows,
                pair_state=pool,
                pair_metadata=meta,
            )
            produced.append(frame.global_kv)
            frame.pair.commit()
            old = pool.raw[offset // pool.entries].clone()
            pool, _ = state()
            pool.raw[1].copy_(old)
            offset = pool.entries
            start += count
        torch.testing.assert_close(
            torch.cat(produced), expected.global_kv, atol=0, rtol=0
        )

    def test_all_destinations_checked_before_any_cache_or_state_mutation(self):
        op = fixture(2)
        pool, _ = state()
        meta = metadata([(0, 0), (0, 1)], pool.entries)
        frame = op.prepare(
            torch.ones(2, 128).bfloat16(),
            meta.positions,
            meta.valid_rows,
            pair_state=pool,
            pair_metadata=meta,
        )
        swa, _ = make_cache("swa_kv")
        global_cache, _ = make_cache("csa_kv")
        index, _ = make_cache("indexer_kv")
        before = [
            item.clone() for item in (swa.raw, global_cache.raw, index.raw, pool.raw)
        ]
        for bad_slots in (
            torch.tensor([-1, 999999]),
            torch.tensor([-1, -1]),
            torch.tensor([-1.0, 0.0]),
        ):
            with self.assertRaises(ValueError):
                frame.publish(
                    swa,
                    torch.tensor([0, 1]),
                    global_cache=global_cache,
                    global_slots=torch.tensor([-1, 0]),
                    index_cache=index,
                    index_slots=bad_slots,
                )
            for actual, expected in zip(
                (swa.raw, global_cache.raw, index.raw, pool.raw), before
            ):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_nonowners_publish_only_swa_and_empty_rows_are_noops(self):
        for sources in (
            V41LayerSources(0, 0, None, None, None),
            V41LayerSources(21, 1, 20, 20, 20),
        ):
            base = fixture(0)
            op = projection.V41KVProjection(
                sources, base.wkv, base.kv_norm, base.frequencies
            )
            for count in (0, 3):
                valid = torch.ones(count, dtype=torch.bool)
                x = torch.randn(count, 128).bfloat16()
                pos = torch.arange(count)
                frame = op.prepare(x, pos, valid)
                self.assertIsNone(frame.global_kv)
                self.assertIsNone(frame.pair)
                swa, _ = make_cache("swa_kv")
                frame.publish(swa, pos)
                if count:
                    torch.testing.assert_close(
                        swa.gather(pos, backend="eager"),
                        reference_rows(swa_reference(op, x, pos), "swa_kv"),
                        atol=0,
                        rtol=0,
                    )

    def test_metadata_admission_and_owner_weight_contracts(self):
        op = fixture(2)
        pool, _ = state()
        meta = metadata([(0, 0), (0, 1)], pool.entries)
        x = torch.zeros(2, 128).bfloat16()
        with self.assertRaisesRegex(ValueError, "projection metadata"):
            op.prepare(
                x,
                meta.positions.clone(),
                meta.valid_rows,
                pair_state=pool,
                pair_metadata=meta,
            )
        for positions, values in (
            (torch.tensor([-1, 0]), x),
            (torch.tensor([0, 1024]), x),
            (meta.positions, torch.full_like(x, float("nan"))),
        ):
            with self.assertRaisesRegex(ValueError, "live rows"):
                op.prepare(values, positions, meta.valid_rows)
        with self.assertRaisesRegex(ValueError, "compressor weight"):
            projection.V41KVProjection(op.sources, op.wkv, op.kv_norm, op.frequencies)
        with self.assertRaisesRegex(ValueError, "non-owner"):
            projection.V41KVProjection(
                V41LayerSources(21, 1, 20, 20, 20),
                op.wkv,
                op.kv_norm,
                op.frequencies,
                index_norm=op.index_norm,
            )


if __name__ == "__main__":
    unittest.main()
