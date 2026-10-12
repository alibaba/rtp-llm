"""Independent joint-softmax oracle and sparse attention contract checks."""

import sys
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models.test.dsv41_cache_codec_test import (
    codec,
    load_codec_source,
    make_cache,
)

with patch.dict(sys.modules, {"rtp_llm.models_py.modules.dsv4.v41_cache": codec}):
    attention = load_codec_source("_dsv41_attention_test", "v41_attention.py")


def oracle(query, sinks, pools, valid):
    result = torch.zeros_like(query)
    for row in range(query.shape[0]):
        if not valid[row]:
            continue
        keys = torch.cat(
            [
                cache.read(slots[row][slots[row] >= 0].contiguous(), backend="eager")
                for cache, slots in pools
            ]
        ).double()
        logits = query[row].double() @ keys.T * (512**-0.5)
        logits = torch.cat((logits, sinks.double()[:, None]), -1)
        probabilities = logits.softmax(-1)[:, :-1]
        result[row] = (probabilities @ keys).to(result.dtype)
    return result


def fixture(rows=4, heads=3, global_tag="hca_kv", swa_width=35, global_width=67):
    torch.manual_seed(413)
    swa, _ = make_cache("swa_kv", gap=64)
    global_cache, _ = make_cache(global_tag, gap=128)
    for cache in (swa, global_cache):
        count = cache.raw.shape[0] * cache.layout.entries
        cache.write(
            torch.randn(count, 512).bfloat16(), torch.arange(count), backend="eager"
        )
    query = torch.randn(rows, heads, 512).bfloat16()
    sinks = torch.linspace(-1, 2, heads)
    swa_slots = torch.randint(
        0, swa.raw.shape[0] * swa.layout.entries, (rows, swa_width)
    )
    global_slots = torch.randint(
        0, global_cache.raw.shape[0] * global_cache.layout.entries, (rows, global_width)
    )
    swa_slots[:, 1::4] = -1
    global_slots[:, 2::5] = -1
    valid = torch.ones(rows, dtype=torch.bool)
    if rows > 1:
        valid[-1] = False
        query[-1].fill_(float("nan"))
        swa_slots[-1].fill_(2**30)
        global_slots[-1].fill_(-99)
    return query, sinks, swa, swa_slots, valid, global_cache, global_slots


class V41AttentionTest(unittest.TestCase):
    def test_joint_softmax_matches_independent_double_oracle(self):
        for tag in ("csa_kv", "hca_kv"):
            q, sink, swa, slots, valid, glob, selected = fixture(global_tag=tag)
            before = (swa.raw.clone(), glob.raw.clone())
            output = attention.V41SparseAttention(backend="eager")(
                q,
                sink,
                swa,
                slots,
                valid_rows=valid,
                global_cache=glob,
                global_slots=selected,
            )
            expected = oracle(q, sink, [(swa, slots), (glob, selected)], valid)
            torch.testing.assert_close(output, expected, atol=0.004, rtol=0.01)
            self.assertTrue(bool((output[-1] == 0).all()))
            torch.testing.assert_close(swa.raw, before[0], atol=0, rtol=0)
            torch.testing.assert_close(glob.raw, before[1], atol=0, rtol=0)

    def test_sink_is_shared_once_across_pools(self):
        q, sink, swa, slots, valid, glob, selected = fixture(
            rows=1, heads=1, swa_width=1, global_width=1
        )
        q.zero_()
        sink.zero_()
        slots.fill_(0)
        selected.fill_(0)
        swa.write(torch.full((1, 512), 3.0), torch.tensor([0]), backend="eager")
        glob.write(torch.full((1, 512), 3.0), torch.tensor([0]), backend="eager")
        out = attention.V41SparseAttention(backend="eager")(
            q,
            sink,
            swa,
            slots,
            valid_rows=valid,
            global_cache=glob,
            global_slots=selected,
        )
        # (3 + 3 + sink_value=0) / (1 + 1 + sink_weight=1)
        torch.testing.assert_close(out, torch.full_like(out, 2), atol=0, rtol=0)

    def test_empty_pools_and_extreme_sinks_remain_finite(self):
        q, sink, swa, slots, valid, glob, selected = fixture(rows=1, heads=2)
        op = attention.V41SparseAttention(backend="eager")
        for width in (0, 35):
            holes = torch.full((1, width), -1, dtype=torch.int32)
            sink[:] = torch.tensor([-1e38, 1e38])
            out = op(q, sink, swa, holes, valid_rows=valid)
            torch.testing.assert_close(out, torch.zeros_like(q), atol=0, rtol=0)
        out = op(q, sink, swa, slots, valid_rows=valid)
        expected = oracle(q, sink, [(swa, slots)], valid)
        torch.testing.assert_close(out, expected, atol=0.004, rtol=0.01)

    def test_rejects_invalid_metadata_before_output_write(self):
        q, sink, swa, slots, valid, glob, selected = fixture()
        op = attention.V41SparseAttention(backend="eager")
        out = torch.full_like(q, 123)
        cases = [
            {"global_cache": glob},
            {"global_cache": swa, "global_slots": selected},
            {"global_cache": glob, "global_slots": selected.float()},
            {
                "global_cache": glob,
                "global_slots": torch.zeros(4, 513, dtype=torch.int32),
            },
        ]
        for args in cases:
            with self.assertRaises(ValueError):
                op(q, sink, swa, slots, valid_rows=valid, out=out, **args)
            self.assertTrue(bool((out == 123).all()))
        with self.assertRaisesRegex(ValueError, "alias"):
            op(q, sink, swa, slots, valid_rows=valid, out=q)
        with self.assertRaisesRegex(ValueError, "bounded"):
            op(q, sink, swa, torch.zeros(4, 129, dtype=torch.int32), valid_rows=valid)
        with self.assertRaisesRegex(ValueError, "sinks"):
            op(q, sink.double(), swa, slots, valid_rows=valid)

    def test_live_invalid_values_rejected_dead_rows_ignored(self):
        q, sink, swa, slots, valid, _, _ = fixture()
        op = attention.V41SparseAttention(backend="eager")
        op(q, sink, swa, slots, valid_rows=valid)
        slots[0, 0] = -2
        with self.assertRaisesRegex(ValueError, "outside"):
            op(q, sink, swa, slots, valid_rows=valid)
        slots[0, 0] = swa.raw.shape[0] * swa.layout.entries
        with self.assertRaisesRegex(ValueError, "outside"):
            op(q, sink, swa, slots, valid_rows=valid)
        slots[0, 0] = 0
        q[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            op(q, sink, swa, slots, valid_rows=valid)

    def test_empty_batch_and_explicit_output(self):
        q, sink, swa, slots, valid, _, _ = fixture(rows=0)
        out = torch.empty_like(q)
        actual = attention.V41SparseAttention(backend="eager")(
            q, sink, swa, slots, valid_rows=valid, out=out
        )
        self.assertIs(actual, out)
        with self.assertRaisesRegex(RuntimeError, "SM120"):
            attention.V41SparseAttention()(q, sink, swa, slots, valid_rows=valid)


if __name__ == "__main__":
    unittest.main()
