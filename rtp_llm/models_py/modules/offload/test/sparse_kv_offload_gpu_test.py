import unittest
from math import prod
from types import MethodType, SimpleNamespace

import torch

from rtp_llm.cpp.cache.test.libcache_config_creator_py_test import (
    make_offload_test_cache,
)
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeModel
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
    SparseMlaFp8Op,
    SparseMlaImpl,
    SparseMlaOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.models_py.modules.hybrid.mla_attention import MlaAttention
from rtp_llm.models_py.modules.offload.layout import PagedRowLayout
from rtp_llm.models_py.modules.offload.plan import LayerOffloadPlan, build_offload_plans
from rtp_llm.models_py.modules.offload.runtime import KVWrite, SparseOffloadRuntime
from rtp_llm.ops import KvCacheDataType, compute_ops
from rtp_llm.ops.compute_ops import LayerKVCache


def offload_descs(groups, tag="default"):
    layer_count = max(layer for owners in groups.values() for layer in owners) + 1
    descs = [[] for _ in range(layer_count)]
    for group, owners in groups.items():
        for layer in owners:
            descs[layer].append(
                SimpleNamespace(
                    tag=tag,
                    offload=SimpleNamespace(enabled=True),
                    offload_runtime_group=group,
                )
            )
    return descs


class _CudaStreamTestCase(unittest.TestCase):
    seed = 412

    def setUp(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA worker required")
        torch.manual_seed(self.seed)
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        self.context = torch.cuda.stream(self.stream)
        self.context.__enter__()

    def tearDown(self):
        self.stream.synchronize()
        self.context.__exit__(None, None, None)


class SparseOffloadRuntimeTest(_CudaStreamTestCase):
    def make_runtime(
        self,
        capacity=64,
        layers=2,
        planes=((torch.uint8, 288, 32),),
        layout=None,
        total=None,
        hbm=2,
        block=16,
        page=8,
        head_major=False,
    ):
        maximum = capacity
        capacity = (maximum + page - 1) // page * page
        total = total or max(128, capacity // block * 4)
        generations = torch.ones(total, dtype=torch.int64, device="cuda")
        storages, references, allocations = [], [], []
        for layer in range(layers):
            hosts, residents, refs = [], [], []
            for dtype, width, padding in planes:
                shape = width if isinstance(width, tuple) else (width,)
                width = prod(shape)
                stride = page * width + padding

                def allocate(pages, host):
                    backing = torch.empty(
                        max(pages * stride, 8),
                        dtype=dtype,
                        device="cpu" if host else "cuda",
                        pin_memory=host,
                    )
                    backing.view(torch.uint8).fill_(0xA5)
                    payload_strides = tuple(
                        prod(shape[i + 1 :]) for i in range(len(shape))
                    )
                    row_stride = width
                    if head_major:
                        row_stride = prod(shape[1:])
                        payload_strides = (page * row_stride, *payload_strides[1:])
                    tensor = backing.as_strided(
                        (pages, page, *shape), (stride, row_stride, *payload_strides)
                    )
                    allocations.append((backing, pages, stride, page * width))
                    return tensor

                host = allocate((total - hbm) * block // page, True)
                resident = allocate((hbm * block + capacity) // page, False)
                byte_width = width * resident.element_size()
                reference = torch.randint(
                    0,
                    256,
                    (total * block, byte_width),
                    dtype=torch.uint8,
                    device="cuda",
                )
                reference = reference.view(dtype).reshape(total * block, *shape)
                host.copy_(reference[hbm * block :].reshape_as(host))
                resident[: hbm * block // page].copy_(
                    reference[: hbm * block].reshape(-1, page, *shape)
                )
                hosts.append(host)
                residents.append(resident)
                refs.append(reference)
            storages.append(
                SimpleNamespace(
                    tag="default",
                    owner_layer=layer,
                    total_blocks=total,
                    hbm_blocks=hbm,
                    rows_per_block=block,
                    rows_per_page=page,
                    max_selected_rows=maximum,
                    host_planes=tuple(hosts),
                    resident_planes=tuple(residents),
                    generations=generations,
                )
            )
            references.append(refs)
        return (
            SparseOffloadRuntime(tuple(storages), layout or PagedRowLayout()),
            references,
            allocations,
        )

    def assert_rows(self, runtime, references, ids):
        valid = ids >= 0
        for layer, reference_planes in enumerate(references):
            view = runtime.acquire(layer)
            self.assertEqual(view.indices.shape, ids.shape)
            self.assertTrue(torch.equal(view.indices < 0, ~valid))
            hbm = valid & (ids < runtime.hbm_rows)
            self.assertTrue(torch.equal(view.indices[hbm], ids[hbm]))
            slots = view.indices[valid].long()
            page = runtime.storages[layer].rows_per_page
            for resident, reference in zip(view.planes, reference_planes):
                actual = (
                    resident[slots // page, slots % page].contiguous().view(torch.uint8)
                )
                expected = reference[ids[valid].long()].contiguous().view(torch.uint8)
                self.assertTrue(
                    torch.equal(actual, expected), f"wrong row bytes in storage {layer}"
                )
        runtime.drain()

    def assert_backing(self, runtime, references):
        self.stream.synchronize()
        for storage, refs in zip(runtime.storages, references):
            for host, resident, reference in zip(
                storage.host_planes, storage.resident_planes, refs
            ):
                expected_host = reference[runtime.hbm_rows :].cpu().reshape(host.shape)
                self.assertTrue(
                    torch.equal(
                        host.contiguous().view(torch.uint8),
                        expected_host.contiguous().view(torch.uint8),
                    )
                )
                expected_hbm = reference[: runtime.hbm_rows].view(torch.uint8)
                actual_hbm = (
                    resident[: runtime.hbm_rows // storage.rows_per_page]
                    .contiguous()
                    .view(torch.uint8)
                    .flatten(0, 1)
                )
                self.assertTrue(torch.equal(actual_hbm, expected_hbm))

    def assert_padding(self, allocations):
        self.stream.synchronize()
        for allocation, pages, stride, payload in allocations:
            padding = allocation[: pages * stride].reshape(pages, stride)[:, payload:]
            self.assertTrue(
                padding.contiguous().view(torch.uint8).eq(0xA5).all().item()
            )

    def test_mixed_duplicates_padding_and_empty(self):
        runtime, refs, _ = self.make_runtime()
        for values in ([7, 55, 89, 55, -1], [55, 55, 7, 89, -1], [-1] * 5, []):
            ids = torch.tensor(values, dtype=torch.int32, device="cuda").reshape(
                1, 1, -1
            )
            runtime.begin(ids)
            self.assert_rows(runtime, refs, ids)
            if 55 in values:
                self.assertEqual(
                    runtime.acquire(0).indices[ids == 55].unique().numel(), 1
                )

    def test_native_dtypes_multiple_planes_and_page_padding(self):
        for planes in (
            ((torch.bfloat16, 576, 16),),
            ((torch.float8_e4m3fn, 656, 32),),
            ((torch.uint8, 288, 160), (torch.float32, 4, 8)),
            ((torch.uint8, 7, 3),),
        ):
            with self.subTest(planes=planes):
                runtime, refs, allocations = self.make_runtime(planes=planes)
                ids = torch.tensor(
                    [3, 31, 32, 33, 200, 201, -1], dtype=torch.int32, device="cuda"
                )
                selection = ids
                runtime.begin(selection)
                self.assert_rows(runtime, refs, selection)
                write_ids = torch.tensor(
                    [3, 32, 1800, -1], dtype=torch.int32, device="cuda"
                )
                for layer, storage in enumerate(runtime.storages):
                    values = tuple(
                        torch.full(
                            (4, ref.shape[1] * ref.element_size()),
                            17 + layer,
                            dtype=torch.uint8,
                            device="cuda",
                        ).view(ref.dtype)
                        for ref in refs[layer]
                    )
                    runtime.write(layer, KVWrite(write_ids, values))
                    for reference, value in zip(refs[layer], values):
                        reference.view(torch.uint8)[write_ids[:3].long()] = value.view(
                            torch.uint8
                        )[:3]
                self.assert_rows(runtime, refs, selection)
                self.assert_backing(runtime, refs)
                self.assert_padding(allocations)

    def test_write_before_begin_and_after_prefetch_preserve_metadata(self):
        class SlowLastLayout(PagedRowLayout):
            def fetch(self, storage, misses):
                if storage.owner_layer == 1:
                    torch.cuda._sleep(1_000_000)
                super().fetch(storage, misses)

        runtime, refs, _ = self.make_runtime(layout=SlowLastLayout())
        # W can exceed S; chunks must use fixed scratch without truncation.
        write_ids = torch.arange(100, 200, dtype=torch.int32, device="cuda")
        for layer in range(2):
            values = torch.full(
                (100, 288), 21 + layer, dtype=torch.uint8, device="cuda"
            )
            runtime.write(layer, KVWrite(write_ids, (values,)))
            refs[layer][0][write_ids.long()] = values
        selection = write_ids[:64]
        runtime.begin(selection)
        for layer in range(2):
            values = torch.full(
                (100, 288), 47 + layer, dtype=torch.uint8, device="cuda"
            )
            runtime.write(layer, KVWrite(write_ids, (values,)))
            refs[layer][0][write_ids.long()] = values
        self.assert_rows(runtime, refs, selection)
        before = tuple(
            t.clone() for t in (runtime.mapping, runtime.versions, runtime.tags)
        )
        runtime.write(0, KVWrite(write_ids, (refs[0][0][write_ids.long()],)))
        for expected, actual in zip(
            before, (runtime.mapping, runtime.versions, runtime.tags)
        ):
            self.assertTrue(torch.equal(expected, actual))
        self.assert_backing(runtime, refs)

    def test_begin_joins_previous_fetch_without_acquire(self):
        class DelayedLayout(PagedRowLayout):
            def fetch(self, storage, misses):
                if storage.owner_layer == 1:
                    torch.cuda._sleep(1_000_000)
                super().fetch(storage, misses)

        runtime, refs, _ = self.make_runtime(layout=DelayedLayout())
        for step in range(12):
            ids = torch.arange(64, dtype=torch.int32, device="cuda") + 32 + step * 64
            selection = ids
            runtime.begin(selection)
        self.assert_rows(runtime, refs, selection)

    def test_sparse_eviction_with_only_one_victim_and_counter_wrap(self):
        for capacity in (64, 512, 8192):
            with self.subTest(capacity=capacity):
                runtime, refs, _ = self.make_runtime(
                    capacity=capacity, planes=((torch.uint8, 16, 0),)
                )
                ids = (
                    torch.arange(capacity, dtype=torch.int32, device="cuda")
                    + runtime.hbm_rows
                )
                selection = ids
                runtime.begin(selection)
                self.assert_rows(runtime, refs, selection)
                runtime.epoch.fill_((1 << 31) - 2)
                for step in range(12):
                    old = runtime.tags.clone()
                    ids[:-1] = old[:-1]
                    ids[-1] = runtime.hbm_rows + capacity + step
                    runtime.clock.fill_((1 << 32) * (step % 2))
                    runtime.begin(selection)
                    self.assert_rows(runtime, refs, selection)
                    self.assertEqual(
                        runtime.acquire(0).indices.unique().numel(), capacity
                    )
                    self.assertEqual(runtime.mapping[old[-1].long()].item(), -1)

    def test_two_runtimes_share_cuda_generations_and_refresh_before_write(self):
        group, refs, _ = self.make_runtime(layers=2)
        runtime = SparseOffloadRuntime((group.storages[0],), PagedRowLayout())
        other = SparseOffloadRuntime((group.storages[1],), PagedRowLayout())
        selection = torch.arange(48, 64, dtype=torch.int32, device="cuda")
        runtime.begin(selection)
        other.begin(selection)
        self.assert_rows(runtime, refs[:1], selection)
        self.assert_rows(other, refs[1:], selection)
        self.stream.synchronize()
        for layer, storage in enumerate(group.storages):
            storage.host_planes[0][2:4].fill_(81 + layer)
            refs[layer][0][48:64].fill_(81 + layer)
        runtime.generations[3] += 1
        # A stale hot slot must not be declared fresh by a one-row write.
        runtime.write(0, KVWrite(selection[:1], (refs[0][0][48:49],)))
        for cache, reference in ((runtime, refs[:1]), (other, refs[1:])):
            cache.begin(selection)
            self.assert_rows(cache, reference, selection)

    def test_concurrent_admissions_with_sparse_victims_and_duplicates(self):
        for capacity in (512, 8192):
            runtime, refs, _ = self.make_runtime(
                capacity=capacity, planes=((torch.uint8, 16, 0),)
            )
            ids = (
                torch.arange(capacity, dtype=torch.int32, device="cuda")
                + runtime.hbm_rows
            )
            runtime.begin(ids)
            self.assert_rows(runtime, refs, ids)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                runtime.begin(ids)
                runtime.drain()
            victims = capacity // 256
            positions = torch.arange(victims, device="cuda") * 256
            runtime.epoch.fill_((1 << 31) - 4)
            for duplicate in (False, True):
                keep = torch.ones(capacity, dtype=torch.bool, device="cuda")
                keep[positions] = False
                if duplicate:
                    keep[positions + 1] = False
                removed = victims * (2 if duplicate else 1)
                for step in range(12):
                    old = runtime.tags.clone()
                    ids[keep] = old[:-removed]
                    new_ids = (
                        torch.arange(victims, dtype=ids.dtype, device="cuda")
                        + runtime.hbm_rows
                        + capacity
                        + (step + 12 * duplicate) * victims
                    )
                    ids[positions] = new_ids
                    if duplicate:
                        # Each new ID also appears in another admission CTA.
                        ids[positions + 1] = new_ids.roll(1)
                    runtime.clock.fill_((1 << 32) * (step % 2))
                    graph.replay()
                    self.assert_rows(runtime, refs, ids)
                    slots = runtime.acquire(0).indices
                    self.assertEqual(
                        slots.unique().numel(), capacity - removed + victims
                    )
                    self.assertEqual(
                        runtime.mapping[old[-removed:].long()].eq(-1).sum().item(),
                        victims,
                    )
                    if duplicate:
                        self.assertTrue(
                            torch.equal(slots[positions + 1], slots[positions].roll(1))
                        )

    def test_graph_buckets_reset_version_publication_and_writes(self):
        runtime, refs, _ = self.make_runtime()
        captures = []
        for n in (16, 64):
            ids = torch.arange(n, dtype=torch.int32, device="cuda") + 32
            write_ids = torch.tensor([33], dtype=torch.int32, device="cuda")
            values = torch.full((1, 288), 19, dtype=torch.uint8, device="cuda")
            selection = ids

            def run():
                runtime.begin(selection)
                runtime.write(0, KVWrite(write_ids, (values,)))
                views = [runtime.acquire(i) for i in range(2)]
                snapshots = [v.planes[0].clone() for v in views]
                indices = views[0].indices.clone()
                runtime.drain()
                return snapshots, indices

            run()
            runtime.reset()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                snapshots, indices = run()
            runtime.reset()
            captures.append((graph, selection, write_ids, values, snapshots, indices))
        self.stream.synchronize()
        refs[0][0][33].fill_(19)
        for step in range(12):
            graph, selection, write_ids, values, snapshots, indices = captures[step % 2]
            selection.copy_(
                torch.arange(selection.numel(), device="cuda") + 32 + step * 64
            )
            write_ids.copy_(selection[:1])
            values.fill_(step + 31)
            if step == 6:
                self.stream.synchronize()
                for layer, storage in enumerate(runtime.storages):
                    storage.host_planes[0][0:2].fill_(91 + layer)
                    refs[layer][0][32:48].fill_(91 + layer)
                runtime.generations[2] += 1
                selection[:16] = torch.arange(32, 48, dtype=torch.int32, device="cuda")
                write_ids.copy_(selection[:1])
            graph.replay()
            refs[0][0][write_ids.long()] = values
            slots = indices.long()
            for layer in range(2):
                actual = snapshots[layer][slots // 8, slots % 8]
                self.assertTrue(torch.equal(actual, refs[layer][0][selection.long()]))
        self.assert_backing(runtime, refs)

    def test_drain_waits_on_callers_stream_in_eager_and_graph(self):
        payload = torch.zeros((), dtype=torch.uint8, device="cuda")

        class DelayedLayout(PagedRowLayout):
            def fetch(self, storage, misses):
                super().fetch(storage, misses)
                if storage.owner_layer == 1:
                    torch.cuda._sleep(1_000_000)
                    storage.resident_planes[0].copy_(payload)

        runtime, _, _ = self.make_runtime(layout=DelayedLayout())
        consumer = torch.cuda.Stream()
        observed = torch.empty_like(runtime.storages[1].resident_planes[0])
        selection = torch.arange(32, 64, dtype=torch.int32, device="cuda")

        def run():
            runtime.begin(selection)
            consumer.wait_stream(self.stream)
            with torch.cuda.stream(consumer):
                runtime.drain()
                observed.copy_(runtime.storages[1].resident_planes[0])
            self.stream.wait_stream(consumer)

        run()
        self.assertTrue(observed.eq(payload).all().item())
        runtime.reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            run()
        runtime.reset()
        for value in (37, 53, 79):
            payload.fill_(value)
            graph.replay()
            self.assertTrue(observed.eq(payload).all().item())

    def test_reset_preserves_backing_and_generations_and_guards_shapes(self):
        runtime, refs, _ = self.make_runtime(capacity=61)
        self.assertEqual(runtime.capacity, 64)
        self.assertEqual(runtime.maximum, 61)
        selection = torch.arange(32, 64, dtype=torch.int32, device="cuda")
        runtime.begin(selection)
        self.assert_rows(runtime, refs, selection)
        generations = runtime.generations.clone()
        runtime.reset()
        self.assertTrue(torch.equal(runtime.generations, generations))
        self.assertTrue(runtime.mapping.eq(-1).all().item())
        self.assert_backing(runtime, refs)
        with self.assertRaisesRegex(RuntimeError, "begin"):
            runtime.acquire(0)
        with self.assertRaisesRegex(ValueError, "max_selected_rows"):
            runtime.begin(torch.zeros(62, dtype=torch.int32, device="cuda"))
        with self.assertRaisesRegex(ValueError, "int32"):
            runtime.begin(selection.long())
        with self.assertRaisesRegex(ValueError, "plane count"):
            runtime.write(0, KVWrite(selection[:1], ()))
        with self.assertRaises(IndexError):
            runtime.write(2, KVWrite(selection[:0], (refs[0][0][:0],)))
        bad = SimpleNamespace(**vars(runtime.storages[1]))
        bad.generations = bad.generations.clone()
        with self.assertRaisesRegex(ValueError, "share"):
            SparseOffloadRuntime((runtime.storages[0], bad), PagedRowLayout())

    def test_sparse_mha_native_heads_and_scales_eager_and_graph(self):
        runtime, refs, allocations = self.make_runtime(
            layers=1,
            head_major=True,
            planes=(
                (torch.int8, (2, 8), 16),
                (torch.int8, (2, 8), 16),
                (torch.float32, (2,), 4),
                (torch.float32, (2,), 4),
            ),
        )
        storage = runtime.storages[0]
        storage.tag = "sparse_heads"
        for reference, host, resident in zip(
            refs[0], storage.host_planes, storage.resident_planes
        ):
            if reference.dtype == torch.int8:
                reference.random_(-8, 8)
            else:
                reference.uniform_(0.1, 0.5)
            host.copy_(reference[runtime.hbm_rows :].reshape_as(host))
            resident[: runtime.hbm_rows // storage.rows_per_page].copy_(
                reference[: runtime.hbm_rows].reshape(-1, *resident.shape[1:])
            )
        ids = torch.tensor(
            [[1, 35, 45, -1], [35, 42, 49, 50]], dtype=torch.int32, device="cuda"
        )
        write_ids = torch.tensor([1, 35], dtype=torch.int32, device="cuda")
        writes = tuple(
            reference[write_ids.long()].clone().add_(1) for reference in refs[0]
        )
        for reference, values in zip(refs[0], writes):
            reference[write_ids.long()] = values
        q = torch.randn(2, 2, 8, device="cuda")

        def attention(planes):
            k, v, ks, vs = (plane.float() for plane in planes)
            k, v = k * ks.unsqueeze(-1), v * vs.unsqueeze(-1)
            scores = torch.einsum("bhd,bshd->bhs", q, k) / 8**0.5
            scores.masked_fill_(ids[:, None, :] < 0, float("-inf"))
            return torch.einsum("bhs,bshd->bhd", scores.softmax(-1), v)

        expected = attention(
            tuple(reference[ids.clamp_min(0).long()] for reference in refs[0])
        )

        def forward():
            runtime.begin(ids)
            runtime.write(0, KVWrite(write_ids, writes))
            view = runtime.acquire(0)
            slots = view.indices.clamp_min(0).long()
            output = attention(
                tuple(plane[slots // 8, slots % 8] for plane in view.planes)
            )
            runtime.drain()
            return output

        torch.testing.assert_close(forward(), expected)
        runtime.reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            captured = forward()
        for _ in range(3):
            graph.replay()
            torch.testing.assert_close(captured, expected)
        self.assert_rows(runtime, refs, ids)
        self.assert_backing(runtime, refs)
        self.assert_padding(allocations)

    def test_csa_compressed_entries_preserve_page_padding(self):
        runtime, refs, allocations = self.make_runtime(
            layers=1,
            block=64,
            page=32,
            planes=((torch.uint8, 584, 320),),
        )
        runtime.storages[0].tag = "csa_kv"
        ids = torch.tensor(
            [[65, 129, 191, -1], [129, 192, 257, 258]], dtype=torch.int32, device="cuda"
        )
        runtime.begin(ids)
        writes = torch.randint(0, 256, (2, 584), dtype=torch.uint8, device="cuda")
        write_ids = torch.tensor([129, 192], dtype=torch.int32, device="cuda")
        runtime.write(0, KVWrite(write_ids, (writes,)))
        refs[0][0][write_ids.long()] = writes
        self.assert_rows(runtime, refs, ids)
        self.assert_backing(runtime, refs)
        self.assert_padding(allocations)

    def test_plans_use_descriptor_tags_and_distinguish_private_groups(self):
        first, _, _ = self.make_runtime(layers=3, planes=((torch.uint8, 16, 0),))
        second, _, _ = self.make_runtime(layers=3, planes=((torch.uint8, 32, 0),))
        storages = {}
        for tag, runtime in (("sparse_heads", first), ("csa_kv", second)):
            for layer, storage in enumerate(runtime.storages):
                storage.tag = tag
                storages[layer, tag] = storage
        descs = offload_descs({0: (1, 2)}, "sparse_heads")
        descs[0] = offload_descs({0: (0,)}, "sparse_heads")[0]
        descs[0][0].offload_runtime_group = None
        for layer, extra in zip(descs, offload_descs({0: (0, 1, 2)}, "csa_kv")):
            layer.extend(extra)
        cache = SimpleNamespace(
            get_offload_storage=lambda layer, tag: storages.get((layer, tag))
        )
        plans = build_offload_plans(cache, descs)
        self.assertEqual(len(plans), 6)
        self.assertIsNot(
            plans[0, "sparse_heads"].runtime, plans[1, "sparse_heads"].runtime
        )
        self.assertIs(
            plans[1, "sparse_heads"].runtime, plans[2, "sparse_heads"].runtime
        )
        self.assertIsNot(plans[1, "sparse_heads"].runtime, plans[1, "csa_kv"].runtime)
        self.assertTrue(plans[0, "csa_kv"].begin)
        self.assertFalse(plans[0, "csa_kv"].drain)
        self.assertTrue(plans[2, "csa_kv"].drain)
        descs[0][0].offload.enabled = False
        with self.assertRaisesRegex(ValueError, "declaration/storage mismatch"):
            build_offload_plans(cache, descs)

    def test_no_host_blocks(self):
        runtime, refs, _ = self.make_runtime(total=2)
        selection = torch.tensor([7, 17, 31, -1], dtype=torch.int32, device="cuda")
        runtime.begin(selection)
        self.assert_rows(runtime, refs, selection)


class SparseMlaOffloadTest(_CudaStreamTestCase):
    seed = 93

    def test_plan_uses_physical_owner_resources(self):
        cache = make_offload_test_cache(False)
        plans = build_offload_plans(cache, offload_descs({0: (0,), 1: (1,)}))
        self.assertIsNot(plans[0, "default"].runtime, plans[1, "default"].runtime)
        self.assertEqual(
            plans[0, "default"].runtime.generations.data_ptr(),
            plans[1, "default"].runtime.generations.data_ptr(),
        )
        for plan in plans.values():
            self.assertEqual(
                (plan.storage_index, plan.begin, plan.drain),
                (0, True, True),
            )
        shared = build_offload_plans(cache, offload_descs({0: (0, 1)}))
        self.assertIs(shared[0, "default"].runtime, shared[1, "default"].runtime)
        self.assertEqual(
            (
                shared[0, "default"].begin,
                shared[0, "default"].drain,
                shared[1, "default"].begin,
                shared[1, "default"].drain,
            ),
            (True, False, False, True),
        )
        # The binding is read-only and its views retain the C++ allocation.
        with self.assertRaises(AttributeError):
            shared[0, "default"].storage.hbm_blocks = 3
        with self.assertRaisesRegex(RuntimeError, "get_offload_storage"):
            cache.get_layer_cache(0, "default")
        del cache
        shared[0, "default"].storage.host_planes[0].fill_(11)
        self.assertTrue(shared[0, "default"].storage.host_planes[0].eq(11).all().item())

    def test_forward_failure_joins_pending_fetch(self):
        class DelayedLayout(PagedRowLayout):
            def fetch(self, storage, misses):
                torch.cuda._sleep(20_000_000)
                super().fetch(storage, misses)

        cache = make_offload_test_cache(False)
        plan = build_offload_plans(cache, offload_descs({0: (0,)}))[0, "default"]
        plan.runtime.layout = DelayedLayout()
        storage = plan.runtime.storages[0]
        ids = torch.tensor(
            [storage.hbm_blocks * storage.rows_per_block],
            dtype=torch.int32,
            device="cuda",
        )
        model = GenericMoeModel.__new__(GenericMoeModel)
        torch.nn.Module.__init__(model)
        model.offload_plans = {(0, "default"): plan}

        def fail(inputs, fmha_impl):
            plan.runtime.begin(ids)
            raise RuntimeError("injected after begin")

        model._forward = fail
        with self.assertRaisesRegex(RuntimeError, "injected after begin"):
            model.forward(None)
        self.assertTrue(plan.runtime.transfer_stream.query())
        self.assertFalse(plan.runtime._started)

    def _compare_attention(self, fp8, graph):
        cache = make_offload_test_cache(fp8)
        plan = build_offload_plans(cache, offload_descs({0: (0,)}))[0, "default"]
        storage = plan.runtime.storages[0]
        dtype = KvCacheDataType.FP8 if fp8 else KvCacheDataType.BASE
        writer = MlaKVCacheWriteOp(dtype)
        writer.configure_offload(storage, 2)
        width = 656 if fp8 else 576
        total = storage.total_blocks * storage.rows_per_block
        full = torch.empty(
            (total // 128, 128, width),
            dtype=storage.resident_planes[0].dtype,
            device="cuda",
        )
        scale = torch.ones((), dtype=torch.float32, device="cuda")
        codec = "fp8_ds_mla" if fp8 else "auto"
        compute_ops.concat_and_cache_mla(
            torch.randn(total, 512, dtype=torch.bfloat16, device="cuda"),
            torch.randn(total, 64, dtype=torch.bfloat16, device="cuda"),
            full,
            torch.arange(total, dtype=torch.int64, device="cuda"),
            codec,
            scale,
        )
        storage.resident_planes[0][:4].copy_(full[:4])
        storage.host_planes[0].copy_(full[4:])
        storage.generations.fill_(1)
        op_class = SparseMlaFp8Op if fp8 else SparseMlaOp
        op = op_class(128, 512, 64, 128, 128, 1.0, 128)
        reference = op_class(128, 512, 64, 128, 128, 1.0, 128)
        params = SimpleNamespace(
            batch_indice_d=torch.arange(2, dtype=torch.int32, device="cuda"),
            batch_indice_h=torch.arange(2, dtype=torch.int32),
            slot_mapping=torch.tensor([300, 700], dtype=torch.int64, device="cuda"),
        )
        table = torch.tensor(
            [list(range(2, 12)), list(range(4, 12)) + [2, 3]],
            dtype=torch.int32,
            device="cuda",
        )
        op.plan(params, table)
        reference.plan(params, table)
        topk = torch.randint(0, 1280, (2, 1, 128), dtype=torch.int32, device="cuda")
        topk[0, 0, 0] = 44
        topk[1, 0, 0] = 188
        topk[..., 1] = topk[..., 0]
        topk[..., -1] = -1
        q = torch.randn(2, 128, 576, dtype=torch.bfloat16, device="cuda")
        ckv = torch.randn(2, 512, dtype=torch.bfloat16, device="cuda")
        rope = torch.randn(2, 64, dtype=torch.bfloat16, device="cuda")

        def run():
            plan.runtime.begin(op.translate_selection(topk))
            plan.runtime.write(0, writer.pack(ckv, rope, params))
            output = op.forward_resident(q, plan.runtime.acquire(0))
            plan.runtime.drain()
            return output

        actual = run()
        captured = None
        if graph:
            plan.runtime.reset()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=self.stream):
                actual = run()
            plan.runtime.reset()
        for step in range(3):
            ckv.add_(0.125)
            rope.add_(0.25)
            compute_ops.concat_and_cache_mla(
                ckv, rope, full, params.slot_mapping, codec, scale
            )
            # SparseMlaImpl passes this flat native view to the original path.
            expected = reference.forward(q, full.view(-1, 1, width), topk)
            if captured is None:
                actual = run()
            else:
                captured.replay()
            self.assertTrue(torch.isfinite(expected).all().item())
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            packed = writer._pack_staging.view(torch.uint8).flatten(0, 1)[:2]
            expected_bytes = full.view(torch.uint8).flatten(0, 1)[params.slot_mapping]
            self.assertTrue(torch.equal(packed, expected_bytes))
        self.stream.synchronize()
        self.assertTrue(
            torch.equal(
                storage.host_planes[0].view(torch.uint8),
                full[4:].view(torch.uint8).cpu(),
            )
        )

    def test_bf16_attention_matches_original_eager_and_graph(self):
        for graph in (False, True):
            with self.subTest(graph=graph):
                self._compare_attention(False, graph)

    def test_fp8_attention_matches_original_eager_and_graph(self):
        for graph in (False, True):
            with self.subTest(graph=graph):
                self._compare_attention(True, graph)

    def test_dummy_writes_are_private_staging_only(self):
        for fp8 in (False, True):
            cache = make_offload_test_cache(fp8)
            plan = build_offload_plans(cache, offload_descs({0: (0,)}))[0, "default"]
            storage = plan.runtime.storages[0]
            writer = MlaKVCacheWriteOp(
                KvCacheDataType.FP8 if fp8 else KvCacheDataType.BASE
            )
            writer.configure_offload(storage, 4)
            params = SimpleNamespace(
                slot_mapping=torch.tensor(
                    [0, 1, 255, -1], dtype=torch.int64, device="cuda"
                )
            )
            ckv = torch.ones((4, 512), dtype=torch.bfloat16, device="cuda")
            rope = torch.ones((4, 64), dtype=torch.bfloat16, device="cuda")
            storage.host_planes[0].view(torch.uint8).fill_(13)
            storage.resident_planes[0].view(torch.uint8).fill_(19)
            for _ in range(2):
                plan.runtime.write(0, writer.pack(ckv, rope, params))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                plan.runtime.write(0, writer.pack(ckv, rope, params))
            graph.replay()
            self.stream.synchronize()
            self.assertTrue(writer._pack_backing_ids.eq(-1).all().item())
            self.assertTrue(
                storage.host_planes[0].view(torch.uint8).eq(13).all().item()
            )
            self.assertTrue(
                storage.resident_planes[0].view(torch.uint8).eq(19).all().item()
            )

    def test_early_indexer_keeps_projection_output_unchanged(self):
        attention = MlaAttention.__new__(MlaAttention)
        torch.nn.Module.__init__(attention)
        attention.q_lora_rank = attention.kv_lora_rank = attention.q_head_dim = 4
        attention.num_heads = attention.qk_rope_head_dim = attention.v_head_dim = 2
        attention.layer_idx = 0
        attention.offload_plan = None
        attention.offload_writer = None
        attention.reuse_topk_indices = False
        attention.parallelism_config = SimpleNamespace(get_attn_tp_size=lambda: 1)
        attention.fused_qkv_a_proj = torch.nn.Linear(16, 10, bias=False).cuda()
        attention.q_a_layernorm = torch.nn.LayerNorm(4).cuda()
        attention.q_b_proj = torch.nn.Linear(4, 8, bias=False).cuda()
        attention.kv_a_layernorm = torch.nn.LayerNorm(4).cuda()
        attention.o_proj = torch.nn.Identity()
        events = []
        attention.q_b_proj.register_forward_hook(lambda *args: events.append("q_b"))
        attention.kv_a_layernorm.register_forward_hook(
            lambda *args: events.append("kv_norm")
        )
        topk = torch.zeros((3, 1, 1), dtype=torch.int32, device="cuda")

        def indexer(*args, **kwargs):
            self.assertIs(args[2], caches["indexer_kv"])
            self.assertIs(args[3], indexer_fmha.fmha_params)
            events.append("indexer")
            return topk

        def forward(q, kv, rope, cache, layer, selected, **kwargs):
            events.append("attention")
            return q[:, :, :2] + kv.reshape(-1, 2, 2)

        attention.indexer = indexer
        routes = {}
        for tag, blocks in (("default", [2, 3, 4]), ("indexer_kv", [9, 10, 11])):
            impl = SparseMlaImpl.__new__(SparseMlaImpl)
            table = torch.tensor(blocks, dtype=torch.int32, device="cuda").view(3, 1)
            impl.fmha_params = SimpleNamespace(
                batch_indice_d=torch.arange(3, dtype=torch.int32, device="cuda"),
                slot_mapping=table[:, 0].to(torch.int64) * 128 + 7,
            )
            impl.rope_params = impl.fmha_params
            impl.attn_inputs = SimpleNamespace(is_prefill=False)
            impl.cp_params = None
            impl.fmha_impl = SparseMlaOp(2, 4, 2, 2, 128, 1.0, 1)
            impl.fmha_impl.plan(impl.fmha_params, table)
            routes[tag] = impl
        fmha, indexer_fmha = routes["default"], routes["indexer_kv"]
        fmha.forward = forward
        caches = {"default": LayerKVCache(), "indexer_kv": LayerKVCache()}
        hidden = torch.randn((3, 16), device="cuda")
        expected = attention(hidden, routes, caches)
        self.assertEqual(events, ["q_b", "kv_norm", "indexer", "attention"])
        events.clear()
        writes = []

        def begin(selection):
            expected_ids = torch.tensor(
                [256, 384, 512], dtype=torch.int32, device="cuda"
            )
            torch.testing.assert_close(selection.flatten(), expected_ids)
            events.append("begin")

        def pack(kv, rope, params):
            self.assertIs(params, fmha.rope_params)
            self.assertIsNot(params.slot_mapping, indexer_fmha.rope_params.slot_mapping)
            torch.testing.assert_close(
                params.slot_mapping, torch.tensor([263, 391, 519], device="cuda")
            )
            return SimpleNamespace(backing_ids=params.slot_mapping, values=(kv, rope))

        # Keep the real SparseMlaImpl.forward so pack's parameter route is exercised.
        fmha.forward = MethodType(SparseMlaImpl.forward, fmha)
        fmha.nope_head_dim = 2
        fmha.rope_impl = SimpleNamespace(forward=lambda *args: None)
        fmha._apply_input_bmm = lambda q, layer: q
        fmha._apply_output_bmm = lambda output, layer: output
        fmha.fmha_impl.forward_resident = lambda q, view, layer_id: forward(
            q, view.values[0], None, None, layer_id, topk
        )
        attention.offload_writer = SimpleNamespace(pack=pack)
        runtime = SimpleNamespace(
            begin=begin,
            write=lambda index, data: writes.append(data),
            acquire=lambda index: writes[-1],
            drain=lambda: None,
        )
        attention.offload_plan = LayerOffloadPlan(runtime, 0, True, True)
        caches["default"] = None
        actual = attention(hidden, routes, caches)
        self.assertEqual(events, ["indexer", "begin", "q_b", "kv_norm", "attention"])
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(len(writes), 1)

        events.clear()
        attention.layer_idx = 1
        attention.indexer = None
        attention.reuse_topk_indices = True
        attention.offload_plan = LayerOffloadPlan(runtime, 1, False, True)
        actual, reused = attention(hidden, routes, caches, topk, return_topk=True)
        self.assertIs(reused, topk)
        self.assertEqual(events, ["q_b", "kv_norm", "attention"])
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(len(writes), 2)
        with self.assertRaisesRegex(RuntimeError, "needs previous top-k"):
            attention(hidden, routes, caches)

        attention.offload_plan = None
        indexer_fmha.is_sparse = lambda: False
        fmha.forward = forward
        caches["default"] = LayerKVCache()
        actual, reused = attention(hidden, routes, caches, return_topk=True)
        self.assertIsNone(reused)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
