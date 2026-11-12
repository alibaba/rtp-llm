import json
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
    exact_head_rms_norm,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.cub_topk import _TORCH_GIT
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_prefill_score import (
    try_prefill_score,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.prefill_filtered_topk import (
    BLOCK_SELECTION_WIDTH,
    _load_module,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.prefill_packed_indexer import (
    _packed_norm_rope,
    try_packed_prefill,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_topk_triton import (
    try_select_qsa_tokens,
)


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (9, 0)
    and torch.version.git_version == _TORCH_GIT,
    "requires SM90 and the pinned Torch version",
)
class PackedIndexerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_visible_special_values(self):
        for columns in (64, 576, 4096, 8192):
            cases = [
                (length, min(nans, length), negative)
                for length in sorted(
                    {
                        1,
                        min(511, columns),
                        min(512, columns),
                        min(513, columns),
                        columns,
                    }
                )
                for nans in (1, 31, 32, 33, 64, 65, 512, 513)
                for negative in (False, True)
            ]
            scores = torch.arange(columns, device="cuda", dtype=torch.float32).repeat(
                len(cases), 1
            )
            visible = torch.tensor(
                [c[0] for c in cases], device="cuda", dtype=torch.int32
            )
            for row, (length, nans, negative) in enumerate(cases):
                scores[row, :nans] = float("nan")
                if negative:
                    scores[row, :nans].view(torch.int32).fill_(-4194304)
                if length - nans >= 4:
                    scores[row, nans : nans + 4] = torch.tensor(
                        [-torch.inf, torch.inf, -0.0, 0.0], device="cuda"
                    )
                scores[row, length:] = -torch.inf
            tail_counts = torch.arange(len(cases), device="cuda", dtype=torch.int32) % 4
            positions = visible * 4 + tail_counts - 1
            out = torch.empty((len(cases), 2051), device="cuda", dtype=torch.int32)
            _load_module().select(scores, visible, positions, out, 0)
            values, indices = scores.topk(min(512, columns), dim=-1)
            expected = torch.full_like(out, -1)
            tokens = indices[:, :, None] * 4 + torch.arange(4, device="cuda")
            tokens.masked_fill_(~(values > -torch.inf)[:, :, None], -1)
            expected[:, : tokens.shape[1] * 4] = tokens.flatten(1).int()
            tail_offsets = torch.arange(3, device="cuda", dtype=torch.int32)
            expected[:, 2048:] = torch.where(
                tail_offsets < tail_counts[:, None],
                visible[:, None] * 4 + tail_offsets,
                -1,
            )
            torch.testing.assert_close(
                out, expected, rtol=0, atol=0, msg=f"width={columns}"
            )
            packed = torch.empty(
                (len(cases), BLOCK_SELECTION_WIDTH), device="cuda", dtype=torch.int32
            )
            _load_module().select(scores, visible, positions, packed, 0)
            valid = expected >= 0
            slot = torch.arange(2051, device="cuda")
            stable_order = torch.where(valid, slot, slot + 2051).argsort(dim=-1)
            torch.testing.assert_close(packed[:, -1], valid.sum(dim=-1).int())
            expanded = packed[:, slot // 4] * 4 + (slot % 4).int()
            expanded.masked_fill_(slot >= packed[:, -1:], -1)
            torch.testing.assert_close(expanded, expected.gather(1, stable_order))

    def test_build_failure_falls_back_once(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa import (
            prefill_filtered_topk as topk,
        )
        from rtp_llm.models_py.modules.qwen4_exp.qsa import (
            prefill_packed_indexer as packed,
        )

        q = torch.zeros((256, 4, 128), device="cuda", dtype=torch.bfloat16)
        keys = torch.zeros((1, 64, 128), device="cuda", dtype=q.dtype)
        rope = torch.zeros((1, 256, 64), device="cuda", dtype=q.dtype)
        cu = torch.tensor([0, 256], device="cuda", dtype=torch.int32)
        gamma = torch.zeros(128, device="cuda", dtype=q.dtype)
        args = (q, keys, rope, rope, cu, [256], gamma, 1e-6)
        with (
            patch.object(topk, "_MODULE", None),
            patch.object(topk, "_BUILD_FAILED", False),
            patch(
                "torch.utils.cpp_extension.load_inline",
                side_effect=RuntimeError("build failed"),
            ) as build,
            patch.object(packed, "_packed_norm_rope") as norm,
            self.assertLogs(topk._LOGGER, level="WARNING") as logged,
        ):
            self.assertIsNone(try_packed_prefill(*args))
            self.assertIsNone(try_packed_prefill(*args))
            build.assert_called_once()
            norm.assert_not_called()
            self.assertEqual(len(logged.records), 1)
        with patch.object(topk, "_MODULE") as selector:
            selector.select.side_effect = RuntimeError("execution failed")
            with self.assertRaisesRegex(RuntimeError, "execution failed"):
                try_packed_prefill(*args)

    def test_topk_boundary_and_ties(self):
        for columns in [64, 256, 512, 576, 1024, 1088, 2048, 4096, 8192]:
            lens = torch.tensor(
                sorted(
                    set(
                        [
                            0,
                            1,
                            min(511, columns),
                            min(512, columns),
                            min(513, columns),
                            columns - 1,
                            columns,
                        ]
                    )
                ),
                device="cuda",
                dtype=torch.int32,
            )
            for mode in ["random", "ties", "constant", "narrow"]:
                scores = torch.randn((lens.numel(), columns), device="cuda")
                if mode == "ties":
                    scores = scores.round()
                if mode == "constant":
                    scores.fill_(1)
                if mode == "narrow":
                    scores = 1 + scores * 1e-5
                offset = 3
                vis = torch.cat(
                    [torch.zeros(offset, device="cuda", dtype=torch.int32), lens]
                )
                pos = vis * 4 + 1
                out = torch.full(
                    (offset + lens.numel() + 2, 2051),
                    -7,
                    device="cuda",
                    dtype=torch.int32,
                )
                # Poison invisible memory: the selector may read only its visible prefix.
                for row, length in enumerate(lens.tolist()):
                    scores[row, length:] = float("nan")
                _load_module().select(scores, vis, pos, out, offset)
                torch.cuda.synchronize()
                self.assertTrue(
                    bool((out[:offset] == -7).all() and (out[-2:] == -7).all())
                )
                for row, length in enumerate(lens.tolist()):
                    result = out[offset + row]
                    count = min(length, 512)
                    blocks = result[: count * 4 : 4] // 4
                    self.assertEqual(blocks.unique().numel(), count)
                    masked = scores[row].clone()
                    masked[length:] = -torch.inf
                    native = masked.topk(min(512, columns))
                    expected_blocks = native.indices[native.values > -torch.inf]
                    torch.testing.assert_close(
                        blocks.long(),
                        expected_blocks,
                        rtol=0,
                        atol=0,
                        msg=f"columns={columns}, mode={mode}, visible={length}",
                    )

                    self.assertTrue(bool(((blocks >= 0) & (blocks < length)).all()))
                    torch.testing.assert_close(
                        scores[row, blocks.long()].sort().values,
                        scores[row, :length].topk(count).values.sort().values,
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        result[: count * 4].reshape(-1, 4),
                        blocks[:, None] * 4
                        + torch.arange(4, device="cuda", dtype=torch.int32),
                        rtol=0,
                        atol=0,
                    )
                    self.assertTrue(bool((result[count * 4 : 2048] == -1).all()))
                    torch.testing.assert_close(
                        result[2048:],
                        torch.tensor(
                            [length * 4, length * 4 + 1, -1],
                            device="cuda",
                            dtype=torch.int32,
                        ),
                        rtol=0,
                        atol=0,
                    )
        print("TOPK_BOUNDARIES_AND_TIES_OK", flush=True)

    def test_prepare_score_and_full_selection(self):
        for lengths in [
            [3, 5, 129],
            [256],
            [257],
            [1029, 2053, 3077, 4101],
            [8192],
            [16384],
            [32768],
            [32768, 256],
        ]:
            n = sum(lengths)
            mx = max(lengths)
            dtype = torch.bfloat16
            projected = torch.randn((n, 640), device="cuda", dtype=dtype)
            raw = projected[:, :512].view(n, 4, 128)
            gamma = torch.randn(128, device="cuda", dtype=dtype) * 0.2
            angle = torch.randn((len(lengths), mx, 32), device="cuda")
            cos = torch.cat([angle.cos()] * 2, -1).to(dtype)
            sin = torch.cat([angle.sin()] * 2, -1).to(dtype)
            keys = torch.randn((len(lengths), mx // 4, 128), device="cuda", dtype=dtype)
            cu = torch.tensor(
                [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                device="cuda",
                dtype=torch.int32,
            )
            qout = torch.empty((n, 4, 128), device="cuda", dtype=dtype)
            visible = torch.empty(n, device="cuda", dtype=torch.int32)
            positions = torch.empty_like(visible)
            _packed_norm_rope[(len(lengths), mx, 4)](
                raw,
                gamma,
                cos,
                sin,
                cu,
                qout,
                visible,
                positions,
                raw.stride(0),
                mx,
                EPS=1e-6,
                num_warps=1,
                enable_fp_fusion=False,
            )
            norm = exact_head_rms_norm(raw, gamma, 1e-6)
            offset = 0
            for i, l in enumerate(lengths):
                # Independent partial-NeoX RoPE reference. Keep BF16 rounding
                # after each product, as in the original indexer.
                ref = norm[offset : offset + l].clone()
                partial = ref[..., :64]
                rotated = torch.cat((-partial[..., 32:], partial[..., :32]), -1)
                ref[..., :64] = (
                    partial * cos[i, :l, None, :] + rotated * sin[i, :l, None, :]
                )
                torch.testing.assert_close(
                    qout[offset : offset + l], ref, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    visible[offset : offset + l],
                    (torch.arange(l, device="cuda", dtype=torch.int32) + 1) // 4,
                    rtol=0,
                    atol=0,
                )
                offset += l
            out = try_packed_prefill(raw, keys, cos, sin, cu, lengths, gamma, 1e-6)
            if min(lengths) < 256:
                self.assertIsNone(out)  # Preserve the native short-row tie order.
                continue
            self.assertIsNotNone(out)
            # Independent sampled dense FP32 oracle, including causal/tail boundaries.
            offset = 0
            for i, l in enumerate(lengths):
                old_scores = try_prefill_score(
                    qout[offset : offset + l].unsqueeze(0),
                    keys[i : i + 1, : l // 4],
                )
                old_selected = try_select_qsa_tokens(
                    old_scores.reshape(l, l // 4),
                    torch.arange(1, l + 1, device="cuda", dtype=torch.int32),
                    compress_ratio=4,
                    token_budget=2048,
                    finite_only=False,
                )
                torch.testing.assert_close(
                    out[offset : offset + l],
                    old_selected,
                    rtol=0,
                    atol=0,
                    msg=f"original selection order: lengths={lengths}, request={i}",
                )
                sample = sorted(
                    set(
                        [
                            0,
                            min(3, l - 1),
                            min(2046, l - 1),
                            min(2047, l - 1),
                            min(2050, l - 1),
                            min(2051, l - 1),
                            l // 2,
                            l - 1,
                        ]
                    )
                )
                for p in sample:
                    visible_n = (p + 1) // 4
                    count = min(visible_n, 512)
                    sel = out[offset + p]
                    blocks = sel[: count * 4 : 4].long() // 4
                    self.assertEqual(blocks.unique().numel(), count)
                    self.assertTrue(bool(((blocks >= 0) & (blocks < visible_n)).all()))
                    if visible_n:
                        score = (
                            qout[offset + p].float() @ keys[i, :visible_n].float().T
                        ).relu().sum(0) * (128**-0.5)
                        selected = score[blocks].sort().values
                        expected = score.topk(count).values.sort().values
                        torch.testing.assert_close(
                            selected, expected, rtol=3e-5, atol=3e-5
                        )
                    self.assertTrue(bool((sel[count * 4 : 2048] == -1).all()))
                    tail = (
                        torch.arange(3, device="cuda", dtype=torch.int32)
                        + visible_n * 4
                    )
                    tail = torch.where(tail <= p, tail, -1)
                    torch.testing.assert_close(sel[2048:], tail, rtol=0, atol=0)
                offset += l
            print("PACKED_SELECTION_OK " + json.dumps(lengths), flush=True)

    def test_unsupported_fallback(self):
        q = torch.zeros((4, 4, 128), dtype=torch.bfloat16)
        keys = torch.zeros((1, 1, 128), dtype=q.dtype)
        rope = torch.zeros((1, 4, 64), dtype=q.dtype)
        cu = torch.tensor([0, 4], dtype=torch.int32)
        g = torch.zeros(128, dtype=q.dtype)
        self.assertIsNone(try_packed_prefill(q, keys, rope, rope, cu, [4], g, 1e-6))
        self.assertIsNone(try_packed_prefill(q, keys, rope, rope, cu, [], g, 1e-6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
