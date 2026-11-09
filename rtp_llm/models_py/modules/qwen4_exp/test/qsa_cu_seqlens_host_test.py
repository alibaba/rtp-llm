"""QSA step-plan cu_seqlens validation must build its expected vectors on the host.

Serving traces (sm90) show every step-plan window paying two validation
chains: ``expected_cu = torch.zeros(B + 1, device); expected_cu[1:] =
torch.tensor(lengths, device=cuda).cumsum(0)`` followed by a device
comparison, once for ``cu_seqlens_device`` and once for
``cu_kv_seqlens_device`` over ``prefixes + lengths``.  Each chain launched a
device fill, a pageable H2D of the lengths, a cub scan and comparison kernels
(~15 launches per step), and the strict ``torch.equal`` variant read the
verdict back to the host.

This test locks the host-side rebuild:

* wrong cu_seqlens / cu_kv_seqlens values are still rejected with the exact
  legacy RuntimeErrors -- through the device comparison (strict mode, and
  async mode on CPU tensors) and through the engine's pinned host mirror of
  cu_seqlens -- while correct values still build a plan;
* the happy path builds and compares the expected vectors without any device
  work for them: no cub DeviceScan kernel, no device ``zeros`` of the
  expected vectors, no ``cumsum`` op and no pageable HtoD for the lengths
  upload (every staging upload is pinned), both for the host-mirror engine
  path and the device-only engine path;
* the plan keeps a device-resident int32 [B + 1] cu_seqlens tensor whose
  values are the expected cumsum.
"""

import itertools
import json
import os
import tempfile
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
from rtp_llm.ops import RopeStyle


def _trace_events(prof) -> list:
    path = os.path.join(tempfile.mkdtemp(prefix="qsa_cu_host_"), "t.json")
    prof.export_chrome_trace(path)
    with open(path) as handle:
        return json.load(handle)["traceEvents"]


def _scan_kernel_count(trace_events) -> int:
    """Device cumsum kernels (cub DeviceScan) recorded by the profiler."""
    return sum(
        1
        for e in trace_events
        if e.get("cat") == "kernel"
        and (
            "DeviceScan" in (e.get("name") or "")
            or "scan_innermost" in (e.get("name") or "")
        )
    )


def _pageable_htod_count(trace_events) -> int:
    return sum(
        1
        for e in trace_events
        if e.get("cat") == "gpu_memcpy" and "HtoD (Pageable" in (e.get("name") or "")
    )


def _pinned_htod_count(trace_events) -> int:
    return sum(
        1
        for e in trace_events
        if e.get("cat") == "gpu_memcpy" and "HtoD (Pinned" in (e.get("name") or "")
    )


def _dtoh_count(trace_events) -> int:
    return sum(
        1
        for e in trace_events
        if e.get("cat") == "gpu_memcpy" and "DtoH" in (e.get("name") or "")
    )


def _aten_op_count(trace_events, op_name: str) -> int:
    return sum(
        1
        for e in trace_events
        if e.get("cat") == "cpu_op" and e.get("name") == op_name
    )


def _device_zeros_of_size(trace_events, size: int) -> int:
    """``aten::zeros`` allocations of exactly ``size`` elements ([B + 1])."""
    count = 0
    for e in trace_events:
        if e.get("cat") != "cpu_op" or e.get("name") != "aten::zeros":
            continue
        concrete = (e.get("args") or {}).get("Concrete Inputs") or []
        if concrete and concrete[0] == f"[{size}]":
            count += 1
    return count


class Qwen4ExpQsaCuSeqlensHostTest(unittest.TestCase):
    D = 4
    RATIO = 4
    PAGE = 8
    HEADS = 2
    PREFIXES = [8, 12]
    LENGTHS = [4, 3]

    def setUp(self):
        qsa_step_plan.clear_step_plan_cache()
        # Hermetic strict flag: the async verdicts (and the plan cache) only
        # exist on the production default; strict-mode cases override this to
        # "1" themselves.
        strict_off = patch.dict(
            "os.environ", {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}
        )
        strict_off.start()
        self.addCleanup(strict_off.stop)
        self.rope = SimpleNamespace(
            style=RopeStyle.Base,
            index_factor=3,
            dim=self.D,
            indexer_is_neox_style=True,
            base=10_000.0,
            scale=1.0,
        )
        self.indexer = SimpleNamespace(
            head_dim=self.D,
            n_heads=self.HEADS,
            compress_ratio=self.RATIO,
            token_budget=8,
            k_norm_gamma=torch.rand(self.D),
            norm_eps=1e-6,
        )

    # ------------------------------------------------------------------
    # fixtures
    # ------------------------------------------------------------------

    def _expected_cu(self, lengths=None):
        return list(
            itertools.accumulate(lengths or self.LENGTHS, initial=0)
        )

    def _expected_cu_kv(self, prefixes=None, lengths=None):
        prefixes = self.PREFIXES if prefixes is None else prefixes
        lengths = self.LENGTHS if lengths is None else lengths
        return list(
            itertools.accumulate(
                (p + l for p, l in zip(prefixes, lengths)), initial=0
            )
        )

    def _device_of(self, device: str) -> torch.device:
        """A concrete device (with index) like the engine's pool devices."""
        spec = torch.device(device)
        if spec.type == "cuda":
            return torch.device("cuda", torch.cuda.current_device())
        return spec

    def _context(
        self,
        device,
        *,
        prefixes=None,
        lengths=None,
        cu_values=None,
        cu_kv_values=None,
        host_cu_values=None,
        with_host_mirror=True,
        with_host_lengths=True,
        is_mtp_draft=True,
    ):
        """One draft-incremental step; ``device`` places every tensor.

        ``with_host_lengths`` emulates the host-input engine (pinned CPU
        length/prefix sources); clearing it emulates ``RTP_LLM_DEVICE_INPUT``
        engines that keep them on the device.  ``with_host_mirror`` adds the
        engine's pinned host ``cu_seqlens`` mirror that PyWrappedModel
        uploads to ``cu_seqlens_device``.
        """
        prefixes = self.PREFIXES if prefixes is None else prefixes
        lengths = self.LENGTHS if lengths is None else lengths
        batch = len(lengths)
        device = self._device_of(device)
        cu_values = self._expected_cu(lengths) if cu_values is None else cu_values
        cu_kv_values = (
            self._expected_cu_kv(prefixes, lengths)
            if cu_kv_values is None
            else cu_kv_values
        )
        host_cu_values = (
            self._expected_cu(lengths) if host_cu_values is None else host_cu_values
        )
        if with_host_lengths:
            input_lengths = torch.tensor(lengths, dtype=torch.int32).pin_memory()
            prefix_lengths = torch.tensor(prefixes, dtype=torch.int32).pin_memory()
        else:
            input_lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
            prefix_lengths = torch.tensor(
                prefixes, dtype=torch.int32, device=device
            )
        positions = []
        for prefix, length in zip(prefixes, lengths):
            positions.extend(
                [pos, pos, pos] for pos in range(prefix, prefix + length)
            )
        position_ids = torch.tensor(positions, dtype=torch.int64).reshape(-1)
        kv_table = torch.arange(
            1, batch * 3 + 1, dtype=torch.int32, device=device
        ).view(batch, 3)
        state_table = kv_table.clone()

        # The three cache regions share one inputs tensor per attribute
        # (the production engine materialises them once), so the metadata
        # agreement sees identical objects instead of paying device
        # comparisons.
        shared = dict(
            is_prefill=True,
            is_target_verify=False,
            is_cuda_graph=False,
            is_s_padded=False,
            context_parallel_info=None,
            cache_store_inputs=None,
            input_lengths=input_lengths,
            prefix_lengths=prefix_lengths,
            sequence_lengths=torch.empty(0, dtype=torch.int32),
            cu_seqlens_device=torch.tensor(
                cu_values, dtype=torch.int32, device=device
            ),
            cu_kv_seqlens_device=torch.tensor(
                cu_kv_values, dtype=torch.int32, device=device
            ),
            combo_position_ids=position_ids.to(device),
            kv_cache_kernel_block_id_device=kv_table,
            kv_cache_block_id_device=state_table,
        )
        if with_host_mirror:
            shared["cu_seqlens"] = torch.tensor(
                host_cu_values, dtype=torch.int32
            ).pin_memory()

        def _inputs():
            return SimpleNamespace(**shared)

        return Qwen4ExpQSARuntimeContext(
            main_cache=SimpleNamespace(tag="full"),
            main_inputs=_inputs(),
            indexer_kv_cache=SimpleNamespace(
                tag=INDEXER_KV_TAG,
                kv_cache_base=torch.zeros(
                    8, (self.PAGE // self.RATIO) * self.D * 2, dtype=torch.uint8,
                    device=device,
                ),
                seq_size_per_block=self.PAGE,
            ),
            indexer_kv_inputs=_inputs(),
            indexer_state_cache=SimpleNamespace(
                tag=INDEXER_STATE_TAG,
                kv_cache_base=torch.zeros(
                    8, 2 * self.RATIO * self.D, dtype=torch.float32, device=device
                ),
                seq_size_per_block=self.PAGE,
            ),
            indexer_state_inputs=_inputs(),
            is_mtp_draft=is_mtp_draft,
        )

    def _build(self, context, device="cuda", *, draft=True, draft_prefix_reuse=False):
        return qsa_step_plan.get_prefill_step_plan(
            context,
            indexer=self.indexer,
            token_count=sum(context.main_inputs.input_lengths.tolist()),
            device=self._device_of(device),
            rope_config=self.rope,
            draft=draft,
            draft_prefix_reuse=draft_prefix_reuse,
        )

    # ------------------------------------------------------------------
    # acceptance and the frozen plan tensors
    # ------------------------------------------------------------------

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_plan_freezes_expected_cu_boundaries_on_device(self):
        plan = self._build(self._context("cuda"))
        expected = self._expected_cu()
        # The device-resident expected tensor keeps the legacy dtype/shape and
        # the plan freezes the same host boundaries.
        self.assertIsNotNone(plan.cu_seqlens)
        self.assertEqual(plan.cu_seqlens.dtype, torch.int32)
        self.assertEqual(tuple(plan.cu_seqlens.shape), (len(self.LENGTHS) + 1,))
        self.assertEqual(plan.cu_seqlens.device.type, "cuda")
        self.assertEqual(plan.cu_seqlens.tolist(), expected)
        self.assertEqual(plan.cu_host, expected)
        self.assertEqual(plan.lengths, self.LENGTHS)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_plan_builds_with_device_only_engine_sources(self):
        # The RTP_LLM_DEVICE_INPUT emulation has no host cu_seqlens mirror.
        plan = self._build(
            self._context("cuda", with_host_mirror=False, with_host_lengths=False)
        )
        self.assertEqual(plan.cu_seqlens.tolist(), self._expected_cu())

    # ------------------------------------------------------------------
    # rejection semantics (legacy messages, both comparison sites)
    # ------------------------------------------------------------------

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_rejects_mismatched_device_cu_seqlens_strict(self):
        with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "1"}):
            context = self._context(
                "cuda", cu_values=[0, 4, 9], with_host_mirror=False
            )
            with self.assertRaisesRegex(
                RuntimeError, "cu_seqlens do not match input_lengths"
            ):
                self._build(context)

    def test_rejects_mismatched_device_cu_seqlens_async_cpu(self):
        # CPU tensors: the async verdict raises fail-fast on the host.
        context = self._context("cpu", cu_values=[0, 4, 9], with_host_mirror=False)
        with self.assertRaisesRegex(
            RuntimeError, "cu_seqlens do not match input_lengths"
        ):
            self._build(context, device="cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_rejects_mismatched_host_mirror_cu_seqlens_fail_fast(self):
        # The engine uploads cu_seqlens_device from its pinned host mirror, so
        # the host comparison must reject a mirror that disagrees with the
        # length partition even when the device copy happens to hold the
        # expected values.
        context = self._context("cuda", host_cu_values=[0, 4, 9])
        with self.assertRaisesRegex(
            RuntimeError, "cu_seqlens do not match input_lengths"
        ):
            self._build(context)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_rejects_mismatched_cu_kv_seqlens_strict(self):
        with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "1"}):
            context = self._context(
                "cuda", cu_kv_values=[0, 12, 20], with_host_mirror=False
            )
            with self.assertRaisesRegex(
                RuntimeError, "cu_kv_seqlens do not match prefix\\+input lengths"
            ):
                self._build(context)

    def test_rejects_mismatched_cu_kv_seqlens_async_cpu(self):
        context = self._context(
            "cpu", cu_kv_values=[0, 12, 20], with_host_mirror=False
        )
        with self.assertRaisesRegex(
            RuntimeError, "cu_kv_seqlens do not match prefix\\+input lengths"
        ):
            self._build(context, device="cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_rejects_non_monotonic_prefix_reuse_cu_kv_strict(self):
        with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "1"}):
            context = self._context(
                "cuda",
                prefixes=[8, 16],
                cu_kv_values=[0, 31, 12],
                with_host_mirror=False,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "MTP draft prefix-reuse has invalid local cu_kv_seqlens metadata",
            ):
                self._build(
                    context, draft=False, draft_prefix_reuse=True
                )

    # ------------------------------------------------------------------
    # the happy path pays no device kernels for the expected vectors
    # ------------------------------------------------------------------

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_happy_path_builds_expected_vectors_without_device_kernels(self):
        from torch.profiler import ProfilerActivity, profile

        batch = len(self.LENGTHS)
        # Warm up the rope-table kernels (triton JIT) outside the profile.
        self._build(self._context("cuda"))
        self._build(
            self._context("cuda", with_host_mirror=False, with_host_lengths=False)
        )
        torch.cuda.synchronize()

        for host_engine in (True, False):
            with self.subTest(host_engine=host_engine):
                context = self._context(
                    "cuda",
                    with_host_mirror=host_engine,
                    with_host_lengths=host_engine,
                )
                torch.cuda.synchronize()
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                ) as prof:
                    plan = self._build(context)
                    torch.cuda.synchronize()
                evs = _trace_events(prof)
                self.assertEqual(
                    _scan_kernel_count(evs),
                    0,
                    "the expected cumsum vectors must be built on the host "
                    "(no cub DeviceScan kernel)",
                )
                self.assertEqual(
                    _aten_op_count(evs, "aten::cumsum"),
                    0,
                    "no torch cumsum may run in the step-plan build",
                )
                self.assertEqual(
                    _device_zeros_of_size(evs, batch + 1),
                    0,
                    "the expected vectors must not be device zeros allocations",
                )
                self.assertEqual(
                    _pageable_htod_count(evs),
                    0,
                    "every host-built upload (expected vectors, lengths, "
                    "state columns) must stage through pinned memory",
                )
                self.assertGreaterEqual(
                    _pinned_htod_count(evs),
                    2,
                    "the two expected vectors must still be staged to the "
                    "device from pinned memory",
                )
                self.assertEqual(
                    plan.cu_seqlens.tolist(), self._expected_cu()
                )
                if host_engine:
                    # The only device readbacks left in this build are the two
                    # RoPE position validations (axis equality and the
                    # logical-position agreement); the host-mirror cu_seqlens
                    # comparison must not add a device readback.
                    self.assertLessEqual(
                        _dtoh_count(evs),
                        2,
                        "the host-mirror cu_seqlens comparison must not add "
                        "a device readback",
                    )


if __name__ == "__main__":
    unittest.main()
