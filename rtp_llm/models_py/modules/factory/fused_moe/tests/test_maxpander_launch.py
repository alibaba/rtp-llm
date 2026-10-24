"""Launcher lifecycle and stream contracts against independent full-WF kernels."""

import copy
import gc
import unittest
import weakref
from contextlib import ExitStack, contextmanager
from unittest import mock

import torch

from rtp_llm.eplb.maxpander_plan import build_circulant_plan
from rtp_llm.models_py.modules import maxpander_kernels
from rtp_llm.models_py.modules.factory.fused_moe.tests.maxpander_reference import (
    LegacyFullWF,
)
from rtp_llm.models_py.modules.maxpander import MaxPanderRouter

WF_KERNEL_NAMES = (
    "_wf_partial_count_kernel",
    "_wf_reduce_order_kernel",
    "_wf_quota_kernel",
    "_wf_scatter_kernel",
)
ALL_WF_KERNEL_NAMES = (
    *WF_KERNEL_NAMES,
    "_wf_count_order_kernel",
)


class MaxPanderLaunchTest(unittest.TestCase):
    tokens = 257
    kernel_names = WF_KERNEL_NAMES

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("MaxPander launcher tests require an allocated CUDA GPU")

    def setUp(self):
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.num_logical = 16
        self.num_physical = 20
        placement, candidates, replicas = build_circulant_plan(
            self.num_logical, self.num_physical, 4, seed=0
        )
        self.physical_to_logical = torch.tensor(placement, device=self.device)
        self.replicated_expert = next(
            expert for expert, count in enumerate(replicas) if count == 2
        )
        candidate_table = torch.tensor(
            candidates, dtype=torch.int32, device=self.device
        )
        replica_counts = torch.tensor(replicas, dtype=torch.int32, device=self.device)
        arguments = dict(
            num_logical=self.num_logical,
            num_physical=self.num_physical,
            ep_size=4,
            wf_passes=2,
        )
        self.router = MaxPanderRouter(candidate_table, replica_counts, **arguments)
        self.reference = LegacyFullWF(candidate_table, replica_counts, **arguments)

    def _logical_ids(self, pattern, dtype=torch.int64):
        offsets = torch.arange(self.tokens * 4, dtype=dtype, device=self.device).view(
            self.tokens, 4
        )
        if pattern == "mixed":
            return offsets % (self.num_logical + 3) - 1
        if pattern == "hot":
            return torch.full_like(offsets, self.replicated_expert)
        if pattern == "invalid":
            return torch.full_like(offsets, -1)
        raise ValueError(pattern)

    @contextmanager
    def _expect_jit_dispatches(self, expected_calls):
        with ExitStack() as stack:
            dispatches = mock.Mock()
            for name in ALL_WF_KERNEL_NAMES:
                kernel = getattr(maxpander_kernels, name)
                options = (
                    {"wraps": kernel.run}
                    if expected_calls and name in self.kernel_names
                    else {"side_effect": AssertionError("unexpected JIT dispatch")}
                )
                dispatches.attach_mock(
                    stack.enter_context(mock.patch.object(kernel, "run", **options)),
                    name,
                )
            yield
            self.assertEqual(
                [dispatch[0] for dispatch in dispatches.mock_calls],
                list(self.kernel_names) * expected_calls,
            )

    def _snapshot_buffers(self, router):
        return {name: buffer.clone() for name, buffer in router.named_buffers()}

    def _assert_buffers_unchanged(self, buffers, snapshots):
        for name, buffer in buffers.items():
            torch.testing.assert_close(buffer, snapshots[name], rtol=0, atol=0)

    def _assert_matches_reference(self, router, logical, physical, buffers=None):
        expected = logical.clone()
        self.reference.remap_inplace(expected)
        valid = (logical >= 0) & (logical < self.num_logical)
        self.assertTrue(torch.all(physical[~valid] == -1))
        physical_ids = physical[valid].long()
        self.assertTrue(
            torch.all((physical_ids >= 0) & (physical_ids < self.num_physical))
        )
        torch.testing.assert_close(
            self.physical_to_logical[physical_ids],
            logical[valid].long(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            torch.bincount(physical_ids, minlength=self.num_physical),
            torch.bincount(expected[valid].long(), minlength=self.num_physical),
            rtol=0,
            atol=0,
        )
        if buffers is None:
            buffers = dict(router.named_buffers())
        for name in ("counts", "hot_order", "quota", "alloc"):
            torch.testing.assert_close(
                buffers[name], getattr(self.reference, name), rtol=0, atol=0
            )
        torch.testing.assert_close(
            buffers["quota"].sum(1),
            torch.bincount(logical[valid].long(), minlength=self.num_logical),
            rtol=0,
            atol=0,
        )

    def _run_and_check(self, router, logical, packed):
        packed.copy_(logical)
        self.assertIs(router.remap_inplace(packed), packed)
        self._assert_matches_reference(router, logical, packed)

    def test_dispatch_reuses_workspace_without_clears_or_allocations(self):
        logical = self._logical_ids("mixed")
        packed = logical.clone()
        self._run_and_check(self.router, logical, packed)
        addresses = {
            name: buffer.data_ptr() for name, buffer in self.router.named_buffers()
        }
        for warm_cache, pattern in ((False, "hot"), (True, "invalid"), (True, "mixed")):
            with self.subTest(warm_cache=warm_cache, pattern=pattern):
                logical = self._logical_ids(pattern)
                packed.copy_(logical)
                self.router.counts.fill_(float("nan"))
                self.router.partial_counts.fill_(torch.iinfo(torch.int32).max)
                if not warm_cache:
                    self.router._launch_cache.clear()
                torch.cuda.synchronize(self.device)
                allocations_before = torch.cuda.memory_stats(self.device)[
                    "allocation.all.allocated"
                ]
                with ExitStack() as stack:
                    stack.enter_context(
                        self._expect_jit_dispatches(int(not warm_cache))
                    )
                    for name in ("zero_", "fill_"):
                        stack.enter_context(
                            mock.patch.object(
                                torch.Tensor,
                                name,
                                side_effect=AssertionError(
                                    "unexpected workspace clear"
                                ),
                            )
                        )
                    for name in (
                        "empty",
                        "empty_like",
                        "zeros",
                        "zeros_like",
                        "full",
                        "full_like",
                        "tensor",
                    ):
                        stack.enter_context(
                            mock.patch.object(
                                torch,
                                name,
                                side_effect=AssertionError("unexpected allocation"),
                            )
                        )
                    self.assertIs(self.router.remap_inplace(packed), packed)
                torch.cuda.synchronize(self.device)
                self.assertEqual(
                    torch.cuda.memory_stats(self.device)["allocation.all.allocated"],
                    allocations_before,
                )
                self.assertEqual(
                    {
                        name: buffer.data_ptr()
                        for name, buffer in self.router.named_buffers()
                    },
                    addresses,
                )
                self._assert_matches_reference(self.router, logical, packed)

    def test_apply_clones_workspace_and_rebuilds_launchers(self):
        logical = self._logical_ids("mixed")
        packed = logical.clone()
        self._run_and_check(self.router, logical, packed)
        self.assertTrue(self.router._launch_cache)
        original_buffers = dict(self.router.named_buffers())
        snapshots = self._snapshot_buffers(self.router)

        self.assertIs(self.router._apply(lambda buffer: buffer.clone()), self.router)
        self.assertEqual(len(self.router._launch_cache), 0)
        for name, original in original_buffers.items():
            self.assertNotEqual(
                getattr(self.router, name).data_ptr(), original.data_ptr()
            )

        with self._expect_jit_dispatches(1):
            self._run_and_check(self.router, self._logical_ids("hot"), packed)
        self._assert_buffers_unchanged(original_buffers, snapshots)

    def test_warm_deepcopy_does_not_mutate_source_workspace_or_cache(self):
        logical = self._logical_ids("mixed")
        packed = logical.clone()
        self._run_and_check(self.router, logical, packed)
        original_cache = dict(self.router._launch_cache)
        self.assertTrue(original_cache)
        snapshots = self._snapshot_buffers(self.router)

        duplicate = copy.deepcopy(self.router)
        self.assertEqual(len(duplicate._launch_cache), 0)
        self.assertEqual(self.router._launch_cache, original_cache)
        for name, original in self.router.named_buffers():
            self.assertNotEqual(
                getattr(duplicate, name).data_ptr(), original.data_ptr()
            )

        with self._expect_jit_dispatches(1):
            self._run_and_check(duplicate, self._logical_ids("hot"), packed)
        self._assert_buffers_unchanged(dict(self.router.named_buffers()), snapshots)
        self.assertEqual(self.router._launch_cache, original_cache)
        with self._expect_jit_dispatches(0):
            self._run_and_check(self.router, logical, packed)

    def test_equal_buffer_rebinding_rebuilds_launchers(self):
        packed = self._logical_ids("mixed")
        retired_storage = []
        for name in ("partial_counts", "counts", "cand_phys", "cand_gpu", "rep_count"):
            for replacement_method in ("attribute", "register_buffer"):
                with self.subTest(buffer=name, replacement=replacement_method):
                    self._run_and_check(self.router, self._logical_ids("mixed"), packed)
                    retired = getattr(self.router, name)
                    retired_storage.append(retired)
                    snapshot = retired.clone()
                    replacement = retired.clone()
                    self.assertNotEqual(replacement.data_ptr(), retired.data_ptr())
                    if replacement_method == "attribute":
                        setattr(self.router, name, replacement)
                    else:
                        self.router.register_buffer(name, replacement, persistent=False)
                    with self._expect_jit_dispatches(1):
                        self._run_and_check(
                            self.router, self._logical_ids("hot"), packed
                        )
                    torch.testing.assert_close(retired, snapshot, rtol=0, atol=0)
                    with self._expect_jit_dispatches(0):
                        self._run_and_check(
                            self.router, self._logical_ids("invalid"), packed
                        )

    def test_set_storage_rebuilds_without_writing_retired_storage(self):
        packed = self._logical_ids("mixed")
        retired_storage = []
        for name in ("partial_counts", "counts", "cand_phys", "cand_gpu", "rep_count"):
            with self.subTest(buffer=name):
                self._run_and_check(self.router, self._logical_ids("mixed"), packed)
                buffer = getattr(self.router, name)
                retired = buffer.detach()
                retired_storage.append(retired)
                snapshot = retired.clone()
                replacement = buffer.clone()
                buffer.set_(replacement)
                self.assertIs(getattr(self.router, name), buffer)
                self.assertNotEqual(buffer.data_ptr(), retired.data_ptr())
                with self._expect_jit_dispatches(1):
                    self._run_and_check(self.router, self._logical_ids("hot"), packed)
                torch.testing.assert_close(retired, snapshot, rtol=0, atol=0)
                with self._expect_jit_dispatches(0):
                    self._run_and_check(
                        self.router, self._logical_ids("invalid"), packed
                    )

    def test_launcher_cache_does_not_retain_input_tensors(self):
        references = []
        for pattern in ("mixed", "hot", "invalid"):
            logical = self._logical_ids(pattern)
            packed = logical.clone()
            self._run_and_check(self.router, logical, packed)
            references.append(weakref.ref(packed))
            torch.cuda.synchronize(self.device)
            del packed
            gc.collect()
            self.assertTrue(all(reference() is None for reference in references))
            self.assertTrue(self.router._launch_cache)
            self.assertLessEqual(len(self.router._launch_cache), 16)

    def test_mismatched_ambient_device_rejects_before_any_launch(self):
        for warm_cache in (False, True):
            with self.subTest(warm_cache=warm_cache):
                logical = self._logical_ids("mixed")
                packed = logical.clone()
                self._run_and_check(self.router, logical, packed)
                packed.copy_(logical)
                if not warm_cache:
                    self.router._launch_cache.clear()
                snapshots = self._snapshot_buffers(self.router)
                forbidden_launchers = {
                    cache_key: tuple(
                        mock.Mock(
                            side_effect=AssertionError("unexpected cached launch")
                        )
                        for launch_index in range(len(launchers))
                    )
                    for cache_key, launchers in self.router._launch_cache.items()
                }
                with ExitStack() as stack:
                    stack.enter_context(
                        mock.patch.dict(
                            self.router._launch_cache, forbidden_launchers, clear=True
                        )
                    )
                    stack.enter_context(self._expect_jit_dispatches(0))
                    reset = stack.enter_context(
                        mock.patch.object(
                            torch.Tensor,
                            "zero_",
                            side_effect=AssertionError("unexpected workspace reset"),
                        )
                    )
                    stack.enter_context(
                        mock.patch(
                            "torch.cuda.current_device",
                            return_value=self.device.index + 1,
                        )
                    )
                    with self.assertRaisesRegex(
                        ValueError, "current.*device|device.*current"
                    ):
                        self.router.remap_inplace(packed)
                    reset.assert_not_called()
                    for launchers in forbidden_launchers.values():
                        for launch in launchers:
                            launch.assert_not_called()
                torch.testing.assert_close(packed, logical, rtol=0, atol=0)
                self._assert_buffers_unchanged(
                    dict(self.router.named_buffers()), snapshots
                )

    def test_warm_hits_reload_inputs_on_current_stream(self):
        streams = (
            torch.cuda.Stream(device=self.device),
            torch.cuda.Stream(device=self.device),
        )
        for dtype in (torch.int32, torch.int64):
            storage = torch.full(
                (self.tokens, 9), -777, dtype=dtype, device=self.device
            )
            packed = storage[:, 2:6]
            self._run_and_check(self.router, self._logical_ids("mixed", dtype), packed)
            cache_keys = set(self.router._launch_cache)
            for index, pattern in enumerate(("hot", "invalid", "mixed", "hot")):
                with self.subTest(dtype=dtype, pattern=pattern, stream=index % 2):
                    logical = self._logical_ids(pattern, dtype)
                    stream = streams[index % 2]
                    stream.wait_stream(torch.cuda.current_stream(self.device))
                    with self._expect_jit_dispatches(0):
                        with torch.cuda.stream(stream):
                            packed.copy_(logical)
                            self.assertIs(self.router.remap_inplace(packed), packed)
                            observed = packed.clone()
                            snapshots = self._snapshot_buffers(self.router)
                            completed = torch.cuda.Event()
                            completed.record()
                        completed.synchronize()
                    self._assert_matches_reference(
                        self.router, logical, observed, buffers=snapshots
                    )
                    self.assertTrue(torch.all(storage[:, :2] == -777))
                    self.assertTrue(torch.all(storage[:, 6:] == -777))
                    self.assertEqual(set(self.router._launch_cache), cache_keys)

    def test_warm_hit_graph_replay_recomputes_reference_routes(self):
        source = self._logical_ids("mixed")
        storage = torch.full(
            (self.tokens, 9), -777, dtype=source.dtype, device=self.device
        )
        packed = storage[:, 2:6]
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            packed.copy_(source)
            self.router.remap_inplace(packed)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self._assert_matches_reference(self.router, source, packed)
        graph = torch.cuda.CUDAGraph()
        with self._expect_jit_dispatches(0):
            with torch.cuda.graph(graph, stream=stream):
                packed.copy_(source)
                self.router.remap_inplace(packed)
            for pattern in ("hot", "invalid", "mixed"):
                source.copy_(self._logical_ids(pattern))
                graph.replay()
                torch.cuda.synchronize(self.device)
                self._assert_matches_reference(self.router, source, packed)
                self.assertTrue(torch.all(storage[:, :2] == -777))
                self.assertTrue(torch.all(storage[:, 6:] == -777))


class MaxPanderSmallLaunchTest(MaxPanderLaunchTest):
    tokens = 32
    kernel_names = (
        "_wf_count_order_kernel",
        "_wf_quota_kernel",
        "_wf_scatter_kernel",
    )

    def test_same_storage_switches_partial_and_fused_warm_hits(self):
        cases = (
            (65537, "mixed"),
            (1024, "hot"),
            (65537, "invalid"),
            (1025, "mixed"),
            (1024, "invalid"),
            (65537, "hot"),
            (1025, "hot"),
        )
        for dtype_index, dtype in enumerate((torch.int32, torch.int64)):
            storage = torch.full((65537, 3), -777, dtype=dtype, device=self.device)
            address = storage[:, 1:2].data_ptr()
            seen_tokens = set()
            with ExitStack() as stack:
                dispatches = {
                    name: stack.enter_context(
                        mock.patch.object(
                            getattr(maxpander_kernels, name),
                            "run",
                            wraps=getattr(maxpander_kernels, name).run,
                        )
                    )
                    for name in ALL_WF_KERNEL_NAMES
                }
                for step, (tokens, pattern) in enumerate(cases):
                    logical = (
                        torch.arange(tokens, dtype=dtype, device=self.device)
                        * 7
                        % (self.num_logical + 2)
                        - 1
                    ).view(tokens, 1)
                    if pattern == "hot":
                        logical.fill_(self.replicated_expert)
                    elif pattern == "invalid":
                        logical.fill_(-1)
                    packed = storage[:tokens, 1:2]
                    self.assertEqual(packed.data_ptr(), address)
                    self.router.partial_counts.fill_(torch.iinfo(torch.int32).max)
                    before = {
                        name: dispatch.call_count
                        for name, dispatch in dispatches.items()
                    }
                    with self.subTest(dtype=dtype, step=step, tokens=tokens):
                        self._run_and_check(self.router, logical, packed)
                        expected = set()
                        if tokens not in seen_tokens:
                            expected = set(
                                self.kernel_names if tokens <= 1024 else WF_KERNEL_NAMES
                            )
                        seen_tokens.add(tokens)
                        for name, dispatch in dispatches.items():
                            self.assertEqual(
                                dispatch.call_count - before[name],
                                int(name in expected),
                                name,
                            )
                        self.assertTrue(torch.all(storage[:, 0] == -777))
                        self.assertTrue(torch.all(storage[:, 2] == -777))
                self.assertEqual(len(self.router._launch_cache), (dtype_index + 1) * 3)


if __name__ == "__main__":
    unittest.main()
