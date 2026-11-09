"""Vectorized QSA draft-tail MRoPE verdict: batched == per-request loop.

Locks the fusion item that replaces the per-request
``mrope_decode_tail_verdict`` loop in the draft/preplan step build with one
batched packed verdict:

* the packed verdict equals the AND of every per-request verdict across
  randomized and boundary cases (equal axes, unequal-axis tails, valid and
  invalid tails, B=1..16, lengths hitting and missing the compress ratio);
* the plan builders keep the exact legacy RuntimeErrors on invalid input and
  only re-run the per-request slow path after a failed batched verdict;
* the batched success path launches no per-request nonzero/assert kernel
  sequence (its launch count does not grow with the batch size).
"""

import os
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime import qsa_step_plan
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context import (
    Qwen4ExpQSARuntimeContext,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_step_plan import (
    _STEP_PLAN_CACHE,
    build_cold_step_plan,
    build_incremental_step_plan,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    mrope_decode_tail_verdict,
    mrope_decode_tail_verdict_packed,
)
from rtp_llm.ops import RopeStyle

RATIO = 4
D = 8
PAGE = 8
F = 3


def _mrope_config():
    return SimpleNamespace(
        style=RopeStyle.Mrope,
        index_factor=F,
        dim=4,
        mrope_dim1=2,
        mrope_dim2=0,
        mrope_dim3=0,
        mrope_interleaved=True,
        base=10_000,
        scale=1.0,
        indexer_is_neox_style=True,
    )


def _loop_verdict(rows, lengths, ratio):
    """The per-request verdict the legacy plan loop computes."""
    verdict = torch.ones((), dtype=torch.bool, device=rows.device)
    offset = 0
    for length in lengths:
        verdict = verdict & mrope_decode_tail_verdict(
            rows[offset : offset + length], ratio
        )
        offset += length
    return verdict


def _packed_case(rng, *, ratio, index_factor, batch):
    lengths = [rng.randint(0, 2 * ratio + 2) for _ in range(batch)]
    rows = []
    for length in lengths:
        prefix = rng.randint(0, 64)
        style = rng.choice(("text", "offset_axes", "garbage", "image_tail"))
        for i in range(length):
            if style == "text":
                rows.append([prefix + i] * index_factor)
            elif style == "offset_axes":
                base = prefix + i
                rows.append(
                    [base + rng.randint(-3, 3) for _ in range(index_factor)]
                )
            elif style == "garbage":
                rows.append([rng.randint(0, 40) for _ in range(index_factor)])
            else:
                if i < length - 1 or rng.random() < 0.5:
                    rows.append(
                        [prefix + i]
                        + [
                            prefix + i + rng.randint(1, 9)
                            for _ in range(index_factor - 1)
                        ]
                    )
                else:
                    rows.append([prefix + i] * index_factor)
    if not rows:
        rows = []
    return lengths, torch.tensor(rows, dtype=torch.int32).reshape(-1, index_factor)


class QsaDraftTailVectorTest(unittest.TestCase):
    def test_packed_verdict_matches_per_request_loop_randomized(self):
        rng = random.Random(20261109)
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for device in devices:
            for step in range(60):
                ratio = rng.choice((1, 2, 3, 4, 8))
                index_factor = rng.choice((1, 3))
                batch = rng.randint(1, 16)
                lengths, host_rows = _packed_case(
                    rng, ratio=ratio, index_factor=index_factor, batch=batch
                )
                for dtype in (torch.int32, torch.int64):
                    with self.subTest(
                        device=device, step=step, ratio=ratio, dtype=dtype
                    ):
                        rows = host_rows.to(device=device, dtype=dtype)
                        cu = torch.tensor(
                            [0]
                            + list(torch.tensor(lengths).cumsum(0).tolist()),
                            dtype=torch.int32,
                            device=device,
                        )
                        self.assertEqual(
                            bool(mrope_decode_tail_verdict_packed(rows, cu, ratio)),
                            bool(_loop_verdict(rows, lengths, ratio)),
                        )

    def test_packed_verdict_matches_per_request_loop_boundaries(self):
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for device in devices:
            cases = (
                # lengths, ratio, index_factor, per-request row builders
                ([4], 4, 3),  # single aligned request: no tail to validate
                ([1], 4, 3),  # single token: tail is the whole request
                ([3, 4], 4, 3),  # miss + hit the boundary
                ([1, 2, 3, 4, 5, 6, 7, 8], 4, 3),
                ([2] * 16, 4, 3),  # B=16
                ([0, 4, 0], 4, 3),  # zero-length requests stay unconstrained
                ([7, 1], 8, 3),  # ratio 8
                ([5, 0, 3], 2, 1),  # ratio 2, index_factor 1
            )
            for lengths, ratio, index_factor in cases:
                for variant in ("valid", "invalid_tail", "unequal_axis"):
                    with self.subTest(
                        device=device,
                        lengths=lengths,
                        ratio=ratio,
                        variant=variant,
                    ):
                        rows = []
                        for length in lengths:
                            prefix = 32
                            for i in range(length):
                                if variant == "valid" or i < length - 1:
                                    rows.append([prefix + i] * index_factor)
                                elif variant == "invalid_tail":
                                    rows.append([prefix + i + 5] * index_factor)
                                else:
                                    row = [prefix + i] * index_factor
                                    row[-1] += 3
                                    rows.append(row)
                        if rows:
                            matrix = torch.tensor(rows, dtype=torch.int64).to(device)
                        else:
                            matrix = torch.zeros(
                                (0, index_factor), dtype=torch.int64, device=device
                            )
                        cu = torch.tensor(
                            [0]
                            + list(torch.tensor(lengths).cumsum(0).tolist()),
                            dtype=torch.int32,
                            device=device,
                        )
                        self.assertEqual(
                            bool(mrope_decode_tail_verdict_packed(matrix, cu, ratio)),
                            bool(_loop_verdict(matrix, lengths, ratio)),
                        )

    def test_packed_verdict_rejects_malformed_metadata(self):
        rows = torch.zeros(2, 3, dtype=torch.int32)
        cu = torch.tensor([0, 2], dtype=torch.int32)
        for kwargs, message in (
            (
                dict(position_rows=rows[0], cu_seqlens=cu, compress_ratio=4),
                "position_rows must be",
            ),
            (
                dict(position_rows=rows, cu_seqlens=cu.reshape(1, 2), compress_ratio=4),
                "cu_seqlens must be",
            ),
            (
                dict(
                    position_rows=rows,
                    cu_seqlens=cu.to(torch.float32),
                    compress_ratio=4,
                ),
                "cu_seqlens must be int32 or int64",
            ),
            (
                dict(position_rows=rows, cu_seqlens=cu, compress_ratio=0),
                "compress_ratio must be positive",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    mrope_decode_tail_verdict_packed(**kwargs)

    def test_packed_verdict_follows_rows_across_devices(self):
        # Production transports combo_position_ids as a pinned CPU tensor while
        # the expected cu partition rides a pinned device upload, so the packed
        # verdict must follow the rows' device (the [B + 1] copy is a no-op on
        # the matching-device hot path) exactly like the legacy per-request
        # verdict computed on each slice's own device.
        if not torch.cuda.is_available():
            self.skipTest("requires cuda")
        lengths = [5, 4, 1]
        ratio = 4
        rows = []
        for length in lengths:
            for i in range(length):
                rows.append([32 + i] * 3)
        cpu_rows = torch.tensor(rows, dtype=torch.int64)
        cu_values = [0] + list(torch.tensor(lengths).cumsum(0).tolist())
        device_cu = torch.tensor(cu_values, dtype=torch.int32, device="cuda")
        expected = bool(
            mrope_decode_tail_verdict_packed(
                cpu_rows.to("cuda"), device_cu, ratio
            )
        )
        self.assertEqual(
            bool(mrope_decode_tail_verdict_packed(cpu_rows, device_cu, ratio)),
            expected,
        )
        self.assertEqual(
            bool(
                mrope_decode_tail_verdict_packed(
                    cpu_rows.to("cuda"), device_cu.to("cpu"), ratio
                )
            ),
            expected,
        )

    # ------------------------------------------------------------------
    # Plan-level fixtures (mirror the runtime contract fixtures).
    # ------------------------------------------------------------------

    def _incremental_context(self, *, lengths, prefixes, rows, device):
        token_count = sum(lengths)
        input_lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
        prefix_lengths = torch.tensor(prefixes, dtype=torch.int32, device=device)
        sequence_lengths = torch.empty(0, dtype=torch.int32, device=device)
        cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
        cu[1:] = input_lengths.cumsum(0)
        cu_kv = torch.zeros_like(cu)
        cu_kv[1:] = (prefix_lengths + input_lengths).cumsum(0)
        position_ids = (
            torch.tensor(rows, dtype=torch.int32, device=device).reshape(-1)
            if rows
            else torch.zeros(0, dtype=torch.int32, device=device)
        )
        blocks = max(2, max(prefixes) // PAGE + 2)
        kv_table = torch.arange(
            1, len(lengths) * blocks + 1, dtype=torch.int32, device=device
        ).view(len(lengths), blocks)
        state_table = kv_table + len(lengths) * blocks
        pool_blocks = 4 * len(lengths) * blocks

        def make_inputs():
            return SimpleNamespace(
                is_prefill=True,
                is_target_verify=False,
                is_cuda_graph=False,
                is_s_padded=False,
                context_parallel_info=None,
                cache_store_inputs=None,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                sequence_lengths=sequence_lengths,
                cu_seqlens_device=cu,
                cu_kv_seqlens_device=cu_kv,
                combo_position_ids=position_ids,
                kv_cache_kernel_block_id_device=kv_table,
                kv_cache_block_id_device=state_table,
            )

        return Qwen4ExpQSARuntimeContext(
            main_cache=SimpleNamespace(tag="full"),
            main_inputs=make_inputs(),
            indexer_kv_cache=SimpleNamespace(
                tag=INDEXER_KV_TAG,
                kv_cache_base=torch.zeros(pool_blocks, PAGE // RATIO * D * 2, dtype=torch.uint8, device=device),
                seq_size_per_block=PAGE,
            ),
            indexer_kv_inputs=make_inputs(),
            indexer_state_cache=SimpleNamespace(
                tag=INDEXER_STATE_TAG,
                kv_cache_base=torch.zeros(pool_blocks, 2 * RATIO * D, dtype=torch.float32, device=device),
                seq_size_per_block=PAGE,
            ),
            indexer_state_inputs=make_inputs(),
        )

    def _incremental_indexer(self, device):
        return SimpleNamespace(
            head_dim=D,
            n_heads=2,
            compress_ratio=RATIO,
            token_budget=64,
            k_norm_gamma=torch.rand(D, device=device),
            norm_eps=1e-6,
        )

    def _tail_rows(self, lengths, prefixes, *, break_request=None):
        rows = []
        for request, (prefix, length) in enumerate(zip(prefixes, lengths)):
            for i in range(length):
                value = prefix + i
                if request == break_request and i == length - 1:
                    value += 7
                rows.extend([value] * F)
        return rows

    def _build_incremental(self, context, token_count, device):
        return build_incremental_step_plan(
            context,
            indexer=self._incremental_indexer(device),
            token_count=token_count,
            device=torch.device(device),
            rope_config=_mrope_config(),
            draft=False,
            draft_prefix_reuse=False,
        )

    def test_incremental_plan_uses_packed_verdict_on_success(self):
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        lengths = [3, 4, 2]
        prefixes = [16, 24, 32]
        for device in devices:
            with self.subTest(device=device):
                for strict in ("", "1"):
                    _STEP_PLAN_CACHE.clear()
                    context = self._incremental_context(
                        lengths=lengths,
                        prefixes=prefixes,
                        rows=self._tail_rows(lengths, prefixes),
                        device=device,
                    )
                    with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": strict}), patch.object(
                        qsa_step_plan,
                        "mrope_decode_tail_verdict_packed",
                        wraps=qsa_step_plan.mrope_decode_tail_verdict_packed,
                    ) as packed, patch.object(
                        qsa_step_plan,
                        "validate_mrope_decode_tail",
                        wraps=qsa_step_plan.validate_mrope_decode_tail,
                    ) as slow:
                        plan = self._build_incremental(context, sum(lengths), device)
                    self.assertEqual(plan.token_count, sum(lengths))
                    packed.assert_called_once()
                    slow.assert_not_called()

    def test_incremental_plan_preserves_error_behavior(self):
        lengths = [3, 4, 2]
        prefixes = [16, 24, 32]
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                _STEP_PLAN_CACHE.clear()
                context = self._incremental_context(
                    lengths=lengths,
                    prefixes=prefixes,
                    rows=self._tail_rows(lengths, prefixes, break_request=0),
                    device=device,
                )
                with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "1"}):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        r"qwen4_exp QSA draft incremental prefill qwen4 QSA "
                        r"unfinished MRoPE compression group requires a "
                        r"consecutive equal-axis text tail ending at the "
                        r"maximum position; historical image positions are not "
                        r"stored in indexer_state",
                    ):
                        self._build_incremental(context, sum(lengths), device)
        # Non-strict CPU keeps the enqueue-only device-assert message as a
        # host RuntimeError; CUDA intentionally traps the device instead.
        _STEP_PLAN_CACHE.clear()
        context = self._incremental_context(
            lengths=lengths,
            prefixes=prefixes,
            rows=self._tail_rows(lengths, prefixes, break_request=0),
            device="cpu",
        )
        with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": ""}):
            with self.assertRaisesRegex(
                RuntimeError,
                r"qwen4_exp QSA unfinished MRoPE compression group requires "
                r"a consecutive equal-axis text tail ending at the maximum "
                r"position",
            ):
                self._build_incremental(context, sum(lengths), "cpu")

    def test_incremental_plan_falls_back_per_request_only_on_failure(self):
        lengths = [3, 4, 2]
        prefixes = [16, 24, 32]
        _STEP_PLAN_CACHE.clear()
        context = self._incremental_context(
            lengths=lengths,
            prefixes=prefixes,
            rows=self._tail_rows(lengths, prefixes, break_request=2),
            device="cpu",
        )
        with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "1"}), patch.object(
            qsa_step_plan,
            "mrope_decode_tail_verdict_packed",
            wraps=qsa_step_plan.mrope_decode_tail_verdict_packed,
        ) as packed, patch.object(
            qsa_step_plan,
            "validate_mrope_decode_tail",
            wraps=qsa_step_plan.validate_mrope_decode_tail,
        ) as slow:
            with self.assertRaisesRegex(RuntimeError, "unfinished MRoPE"):
                self._build_incremental(context, sum(lengths), "cpu")
        packed.assert_called_once()
        slow.assert_called()
        # Requests 0 and 1 validate before request 2 raises; request 1 is
        # ratio-aligned and early-returns inside the legacy validator.
        self.assertEqual(slow.call_count, 3)

    def test_cold_plan_preserves_per_request_error_attribution(self):
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        lengths = [3, 5]
        for device in devices:
            for strict in ("", "1"):
                with self.subTest(device=device, strict=strict):
                    _STEP_PLAN_CACHE.clear()
                    token_count = sum(lengths)
                    rows = []
                    for request, length in enumerate(lengths):
                        for i in range(length):
                            value = 40 + i
                            if request == 1 and i == 0:
                                value += 9
                            rows.extend([value] * F)
                    position_ids = torch.tensor(rows, dtype=torch.int32).reshape(-1)
                    input_lengths = torch.tensor(lengths, dtype=torch.int32)
                    prefix_lengths = torch.zeros(2, dtype=torch.int32)
                    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
                    device_cu = cu.to(device)

                    def make_inputs():
                        return SimpleNamespace(
                            is_prefill=True,
                            is_target_verify=False,
                            is_cuda_graph=False,
                            is_s_padded=False,
                            context_parallel_info=None,
                            cache_store_inputs=None,
                            input_lengths=input_lengths,
                            prefix_lengths=prefix_lengths,
                            sequence_lengths=torch.empty(0, dtype=torch.int32),
                            cu_seqlens_device=device_cu,
                            cu_kv_seqlens_device=device_cu,
                            combo_position_ids=position_ids.to(device),
                            kv_cache_kernel_block_id_device=torch.tensor(
                                [[1, 2], [3, 4]], dtype=torch.int32, device=device
                            ),
                            kv_cache_block_id_device=torch.tensor(
                                [[5, 6], [7, 8]], dtype=torch.int32, device=device
                            ),
                        )

                    context = Qwen4ExpQSARuntimeContext(
                        main_cache=SimpleNamespace(tag="full"),
                        main_inputs=make_inputs(),
                        indexer_kv_cache=SimpleNamespace(
                            tag=INDEXER_KV_TAG,
                            kv_cache_base=torch.zeros(16, PAGE // RATIO * D * 2, dtype=torch.uint8, device=device),
                            seq_size_per_block=PAGE,
                        ),
                        indexer_kv_inputs=make_inputs(),
                        indexer_state_cache=SimpleNamespace(
                            tag=INDEXER_STATE_TAG,
                            kv_cache_base=torch.zeros(16, 2 * RATIO * D, dtype=torch.float32, device=device),
                            seq_size_per_block=PAGE,
                        ),
                        indexer_state_inputs=make_inputs(),
                    )
                    indexer = SimpleNamespace(
                        head_dim=D,
                        n_heads=2,
                        compress_ratio=RATIO,
                        token_budget=64,
                        k_norm_gamma=torch.rand(D),
                        norm_eps=1e-6,
                    )
                    with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": strict}):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            r"qwen4_exp QSA prefill qwen4 QSA unfinished MRoPE "
                            r"compression group requires a consecutive "
                            r"equal-axis text tail ending at the maximum "
                            r"position; historical image positions are not "
                            r"stored in indexer_state "
                            r"\[request=1 seq_len=5 lengths\[:6\]=\[3, 5\] "
                            r"prefix_lengths\[:6\]=\[0, 0\] "
                            r"is_mtp_draft=False\]",
                        ):
                            build_cold_step_plan(
                                context,
                                indexer=indexer,
                                token_count=token_count,
                                device=torch.device(device),
                                rope_config=_mrope_config(),
                                rope_dtype=torch.bfloat16,
                            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_packed_verdict_launch_count_is_batch_independent(self):
        from torch.profiler import ProfilerActivity, profile

        def measure(batch):
            lengths = [3] * batch
            rows = []
            for _ in range(batch):
                for i in range(3):
                    rows.extend([64 + i] * F)
            matrix = torch.tensor(rows, dtype=torch.int32, device="cuda").reshape(-1, F)
            cu = torch.tensor(
                [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                dtype=torch.int32,
                device="cuda",
            )
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                mrope_decode_tail_verdict_packed(matrix, cu, RATIO)
                torch.cuda.synchronize()
            kernels = [
                event
                for event in prof.key_averages()
                if event.device_type == torch.autograd.DeviceType.CUDA
                and "memcpy" not in event.key.lower()
            ]
            launches = sum(event.count for event in kernels)
            names = [event.key for event in kernels]
            return launches, names

        small, small_names = measure(4)
        large, large_names = measure(16)
        self.assertLessEqual(large - small, 4, (small, large))
        for name in large_names:
            lowered = name.lower()
            self.assertNotIn("nonzero", lowered)
            self.assertNotIn("assert_async", lowered)

        # The legacy per-request loop for the same batch grows linearly.
        lengths = [3] * 16
        rows = []
        for _ in lengths:
            for i in range(3):
                rows.extend([64 + i] * F)
        matrix = torch.tensor(rows, dtype=torch.int32, device="cuda").reshape(-1, F)
        _loop_verdict(matrix, lengths, RATIO)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            _loop_verdict(matrix, lengths, RATIO)
            torch.cuda.synchronize()
        loop_launches = sum(
            event.count
            for event in prof.key_averages()
            if event.device_type == torch.autograd.DeviceType.CUDA
            and "memcpy" not in event.key.lower()
        )
        self.assertLess(large * 2, loop_launches, (large, loop_launches))


if __name__ == "__main__":
    unittest.main()
