"""Public dispatch keeps fallback and every cache/undo byte contract."""

import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.qwen4_exp.qsa.draft_prepare_triton import try_prepare
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_compressor import (
    restore_indexer_cache,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_decode_triton import (
    write_draft_window_with_undo_,
)
from rtp_llm.ops import RopeStyle


def equal(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def undo_equal(a, b):
    for field in ("state_slots", "original_state", "kv_slots", "original_kv"):
        assert equal(getattr(a, field), getattr(b, field)), field


def baseline(*args, **kwargs):
    with patch(
        "rtp_llm.models_py.modules.qwen4_exp.qsa.draft_writer_triton.try_write_draft_window",
        return_value=None,
    ):
        return write_draft_window_with_undo_(*args, **kwargs)


def candidate(*args, **kwargs):
    from rtp_llm.models_py.modules.qwen4_exp.qsa.draft_writer_triton import (
        try_write_draft_window,
    )

    def checked(*values, **options):
        undo = try_write_draft_window(*values, **options)
        assert undo is not None, "Expected fused draft writer"
        return undo

    with patch(
        "rtp_llm.models_py.modules.qwen4_exp.qsa.draft_writer_triton.try_write_draft_window",
        side_effect=checked,
    ):
        return write_draft_window_with_undo_(*args, **kwargs)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DraftQsaFusedTest(unittest.TestCase):

    def setUp(self):
        if torch.version.hip is not None or torch.cuda.get_device_capability() not in (
            (9, 0),
            (12, 0),
        ):
            self.skipTest("SM90 or SM120 required")

    def test_unsupported_rope_and_shapes_fall_back(self):
        q = torch.zeros(4, 4, 128, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(4, device="cuda", dtype=torch.int64)
        visible = torch.ones((1, 4), device="cuda", dtype=torch.int32)
        frequency = torch.ones(32, device="cuda")
        base = dict(
            style=RopeStyle.Base, index_factor=3, dim=64, scale=1.0, base=10000.0
        )
        for field, value in [
            ("dim", 32),
            ("index_factor", 2),
            ("indexer_is_neox_style", False),
            ("scale", 0.0),
        ]:
            rope = SimpleNamespace(**{**base, field: value})
            self.assertIsNone(
                try_prepare(
                    q, positions, positions, visible, frequency, rope_config=rope
                )
            )
        self.assertIsNone(
            try_prepare(
                q,
                positions,
                positions,
                visible,
                None,
                rope_config=SimpleNamespace(**base),
            )
        )
        self.assertIsNone(
            try_prepare(
                q,
                positions,
                positions,
                visible.t(),
                frequency,
                rope_config=SimpleNamespace(**base),
            )
        )

    def test_public_writer_ragged_pages_undo_and_dynamic_graph(self):
        result = dict(cases=[], dynamic_generations=0)
        for batch, page in product((1, 2, 3, 8, 16), (64, 128)):
            torch.manual_seed(20261020 + batch)
            raw = torch.randn(batch * 4, 640, device="cuda", dtype=torch.bfloat16)[
                :, 512:
            ]
            cos = torch.randn(batch * 4, 64, device="cuda", dtype=torch.bfloat16)
            sin = torch.randn_like(cos)
            gamma = torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.1
            kt = torch.arange(
                1, batch * 2 + 1, device="cuda", dtype=torch.int32
            ).reshape(batch, 2)
            st = kt.clone()
            kv = torch.randn(
                batch * 2 + 1, page // 4, 128, device="cuda", dtype=torch.bfloat16
            )
            state = torch.randn(
                batch * 2 + 1, 8, 128, device="cuda", dtype=torch.float32
            )
            initial_kv = kv.clone()
            initial_state = state.clone()
            prefixes = torch.empty(batch, device="cuda", dtype=torch.int32)
            lengths = torch.empty_like(prefixes)
            cu = torch.empty(batch + 1, device="cuda", dtype=torch.int32)

            def call(fn):
                return fn(
                    raw,
                    cu,
                    prefixes,
                    lengths,
                    cos,
                    sin,
                    gamma,
                    kv,
                    kt,
                    state,
                    st,
                    norm_eps=1e-6,
                )

            for prefix in (
                0,
                1,
                3,
                4,
                5,
                7,
                8,
                page - 4,
                page - 3,
                page - 2,
                page - 1,
                page,
                page + 1,
            ):
                for length in range(5):
                    lens = [(length + i) % 5 for i in range(batch)]
                    lengths.copy_(torch.tensor(lens, device="cuda", dtype=torch.int32))
                    cu.copy_(
                        torch.tensor(
                            [0] + list(torch.tensor(lens).cumsum(0).tolist()),
                            device="cuda",
                            dtype=torch.int32,
                        )
                    )
                    prefixes.copy_(
                        torch.tensor([prefix] * batch, device="cuda", dtype=torch.int32)
                    )
                    kv.copy_(initial_kv)
                    state.copy_(initial_state)
                    expected = call(baseline)
                    expected_kv = kv.clone()
                    expected_state = state.clone()
                    kv.copy_(initial_kv)
                    state.copy_(initial_state)
                    actual = call(candidate)
                    assert equal(kv, expected_kv) and equal(state, expected_state), (
                        batch,
                        prefix,
                        length,
                    )
                    undo_equal(actual, expected)
                    restore_indexer_cache(actual)
                    assert equal(kv, initial_kv) and equal(state, initial_state)
                    result["cases"].append(
                        dict(
                            batch=batch,
                            prefix=prefix,
                            lengths=lens,
                            pools_and_undo_byte_equal=True,
                        )
                    )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured_undo = call(candidate)
            for generation in range(32):
                lens = [(generation + i) % 5 for i in range(batch)]
                lengths.copy_(torch.tensor(lens, device="cuda", dtype=torch.int32))
                cu.copy_(
                    torch.tensor(
                        [0] + list(torch.tensor(lens).cumsum(0).tolist()),
                        device="cuda",
                        dtype=torch.int32,
                    )
                )
                prefixes.copy_(
                    torch.tensor(
                        [page - 3 + (generation + i) % 5 for i in range(batch)],
                        device="cuda",
                        dtype=torch.int32,
                    )
                )
                raw.copy_(torch.randn_like(raw))
                cos.copy_(torch.randn_like(cos))
                sin.copy_(torch.randn_like(sin))
                kt.copy_(kt.flip(1))
                st.copy_(st.flip(1))
                kv.copy_(initial_kv)
                state.copy_(initial_state)
                expected = call(baseline)
                expected_kv = kv.clone()
                expected_state = state.clone()
                kv.copy_(initial_kv)
                state.copy_(initial_state)
                graph.replay()
                torch.cuda.synchronize()
                assert equal(kv, expected_kv) and equal(state, expected_state), (
                    batch,
                    generation,
                )
                undo_equal(captured_undo, expected)
                restore_indexer_cache(captured_undo)
                assert equal(kv, initial_kv) and equal(state, initial_state)
                result["dynamic_generations"] += 1
            graph.reset()
        self.assertEqual(len(result["cases"]), 650)
        self.assertEqual(result["dynamic_generations"], 320)

    def test_shared_sentinel_pages_are_unchanged_and_undo_is_exact(self):
        for page, invalid in product((64, 128), (False, True)):
            raw = torch.randn(12, 640, device="cuda", dtype=torch.bfloat16)[:, 512:]
            cos = torch.randn(12, 64, device="cuda", dtype=torch.bfloat16)
            sin = torch.randn_like(cos)
            gamma = torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.1
            prefixes = torch.tensor(
                [page - 1, page, page + 1], device="cuda", dtype=torch.int32
            )
            lengths = torch.tensor([1, 0, 3], device="cuda", dtype=torch.int32)
            cu = torch.tensor([0, 1, 1, 4], device="cuda", dtype=torch.int32)
            table = torch.tensor(
                [[1, 2], [3, 4], [5, 6]], device="cuda", dtype=torch.int32
            )
            if invalid:
                table[0, 0] = table[2, 1] = 0
            kv = torch.randn(7, page // 4, 128, device="cuda", dtype=torch.bfloat16)
            state = torch.randn(7, 8, 128, device="cuda", dtype=torch.float32)
            initial_kv, initial_state = kv.clone(), state.clone()
            args = (
                raw,
                cu,
                prefixes,
                lengths,
                cos,
                sin,
                gamma,
                kv,
                table,
                state,
                table,
            )
            expected = baseline(*args, norm_eps=1e-6)
            expected_kv, expected_state = kv.clone(), state.clone()
            kv.copy_(initial_kv)
            state.copy_(initial_state)
            actual = candidate(*args, norm_eps=1e-6)
            self.assertTrue(equal(kv, expected_kv) and equal(state, expected_state))
            self.assertTrue(
                equal(kv[0], initial_kv[0]) and equal(state[0], initial_state[0])
            )
            undo_equal(actual, expected)
            restore_indexer_cache(actual)
            self.assertTrue(equal(kv, initial_kv) and equal(state, initial_state))


if __name__ == "__main__":
    unittest.main()
