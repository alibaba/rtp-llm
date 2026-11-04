# SPDX-License-Identifier: Apache-2.0
"""Placement, publication rollback, and mapped storage lifecycle with native doubles."""

import gc
import hashlib
import sys
import tempfile
import threading
import types
import unittest
import weakref
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock
from unittest.mock import patch
from uuid import uuid4

from rtp_llm.models_py.modules.dsv41.engram_distributed import checkpoint as publish
from rtp_llm.models_py.modules.dsv41.engram_distributed import manifest as m
from rtp_llm.models_py.modules.dsv41.engram_distributed import placement
from rtp_llm.models_py.modules.dsv41.engram_distributed import runtime as runtime_module
from rtp_llm.models_py.modules.dsv41.engram_distributed import storage as s
from rtp_llm.models_py.modules.dsv41.engram_distributed.config import ShardConfig
from rtp_llm.models_py.modules.dsv41.engram_distributed.placement import (
    check_gpu_capacity,
    plan_shards,
)
from rtp_llm.models_py.modules.dsv41.engram_distributed.test.test_layout import (
    GEN,
    HASH,
    descriptors,
    prepare,
    uid,
)


class PlacementTest(unittest.TestCase):
    @patch.object(placement, "resolve_numas", return_value=(0, 1))
    def test_host_granularity_is_queried_once_per_numa_device_pair(self, resolve):
        native = NS(host_numa_granularity=mock.Mock(return_value=4096))
        layers = (1, 2, 3, 4)
        options = dict(
            memory_kind="host",
            device=0,
            numa=(0, 1),
            gpu_devices=(),
        )
        backing = placement.FabricBacking.resolve(native, layers, **options)
        self.assertEqual(backing.granularity, 4096)
        self.assertEqual(
            [backing.nodes for layer in layers],
            [(0, 1)] * len(layers),
        )
        calls = [mock.call(0, 0), mock.call(1, 0)]
        self.assertEqual(native.host_numa_granularity.call_args_list, calls)

        native.host_numa_granularity.reset_mock()
        native.host_numa_granularity.side_effect = (4096, 8192)
        with self.assertRaisesRegex(ValueError, "same allocation granularity"):
            placement.FabricBacking.resolve(native, layers, **options)
        self.assertEqual(native.host_numa_granularity.call_args_list, calls)

    def test_publisher_loads_each_row_once_and_rolls_back_partial_segments(self):
        import sys
        from contextlib import ExitStack
        from types import SimpleNamespace as NS
        from unittest.mock import patch

        from rtp_llm.models_py.modules.dsv41.engram_distributed import (
            checkpoint as publish,
        )
        from rtp_llm.models_py.modules.dsv41.engram_distributed import (
            placement as backing,
        )

        layout = NS(layer_ids=[1], num_embeddings=[32768], head_dim=64)
        tensors = [
            NS(
                name=f"layers.1.engram.embed.{role}",
                shape=(32768, columns),
                relative_path="weights",
            )
            for role, columns in (("weight", 64), ("scale", 2))
        ]
        checkpoint = NS(
            inspect_checkpoint=lambda *args: NS(
                layers=[NS(layer_id=1, tensors=tensors)],
                files=[NS(relative_path="weights")],
                index=None,
            ),
            _cache_page_size=lambda _: None,
            _discard_consumed_pages=None,
            _open_tensor_source=None,
            _same_snapshot=lambda *args, **kwargs: None,
        )
        for fail in (False, True):
            owners, loaded = [], []

            class Owner:
                def __init__(self, size, numa, device):
                    self.size, self.numa, self.closed = size, numa, False
                    owners.append(self)

                def export_handle(self):
                    return uuid4().bytes

                def close(self):
                    self.closed = True

            native = NS(
                FabricAllocation=Owner,
                host_numa_granularity=lambda *args: 4096,
                max_allocation_bytes=1 << 30,
                max_load_chunk_bytes=1 << 20,
            )

            def load(owner, root, tensor, snapshot, begin, end, columns, **kwargs):
                loaded.append((tensor.name, begin, end, owner.numa))
                if fail and len(loaded) == 3:
                    raise OSError("synthetic segment load failure")

            with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
                Path(root, "config.json").write_text("{}")
                stack.enter_context(
                    patch.dict(
                        sys.modules,
                        {
                            "rtp_llm.models_py.modules.dsv41.engram": NS(
                                EngramLayout=lambda _: layout,
                                NgramHashState=lambda *args: None,
                            ),
                            "rtp_llm.models_py.modules.dsv41.engram_distributed.checkpoint": checkpoint,
                        },
                    )
                )
                stack.enter_context(
                    patch.object(placement, "resolve_numas", return_value=(0, 1))
                )
                stack.enter_context(
                    patch.multiple(
                        publish,
                        **vars(checkpoint),
                        _load_native=lambda: native,
                        hash_state_digest=lambda _: "sha256:" + "b" * 64,
                    )
                )
                stack.enter_context(patch.object(publish, "_reclaim_legacy_tables"))
                stack.enter_context(
                    patch.object(publish, "load_rows", side_effect=load)
                )
                generation = str(uuid4())
                if fail:
                    with self.assertRaises(OSError):
                        publish.shard_publisher.publish(
                            root, "revision", "host-0", generation, 0, 2, numa=(0, 1)
                        )
                    self.assertTrue(all(owner.closed for owner in owners))
                    self.assertNotIn(generation, publish.shard_publisher._owners)
                else:
                    doc = publish.shard_publisher.publish(
                        root, "revision", "host-0", generation, 0, 2, numa=(0, 1)
                    )
                    self.assertEqual(doc["schema_version"], 4)
                    self.assertEqual(doc["granularity"], 4096)
                    self.assertEqual(
                        [
                            (s["row_begin"], s["row_end"], s["location"])
                            for s in doc["layers"][0]["segments"]
                        ],
                        [(0, 8192, 0), (8192, 16384, 1)],
                    )
                    self.assertTrue(
                        all(
                            {"weight", "scale"} <= s.keys()
                            for s in doc["layers"][0]["segments"]
                        )
                    )
                    self.assertEqual(sum(owner.size for owner in owners), 16384 * 66)
                    for name in (t.name for t in tensors):
                        self.assertEqual(
                            [
                                (begin, end, numa)
                                for tensor, begin, end, numa in loaded
                                if tensor == name
                            ],
                            [(0, 8192, 0), (8192, 16384, 1)],
                        )
                    for owner in publish.shard_publisher._owners.pop(generation):
                        owner.close()

    def test_partition_has_no_replicas_or_extra_padding(self):
        for p in (2, 3, 18):
            for dim in (32, 64, 256):
                plan = plan_shards(p * 65536 - 7, dim, p, granularity=4096)
                pieces = [
                    part
                    for i in range(p)
                    for part in placement.segment_rows(plan, i, (0, 1))
                ]
                self.assertEqual(sum(part[3] for part in pieces), plan.padded_rows)
                cursor = 0
                for node, begin, end, padded in pieces:
                    self.assertEqual(begin, cursor)
                    self.assertLess(begin, end)
                    self.assertEqual(padded * (dim // 32) % 4096, 0)
                    cursor = end
                self.assertEqual(cursor, plan.total_rows)

    def test_shared_parser_and_cpuset_guard(self):
        self.assertEqual(placement.parse_numa(None), 0)
        self.assertEqual(placement.parse_numa("0-1"), (0, 1))
        self.assertEqual(placement.parse_numa("1,0"), (0, 1))
        for bad in ("", "-1", "65536", "0-9999999999", "1-0", "２", "0-8"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                placement.parse_numa(bad)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "has_memory").write_text("0-1\n")
            (root / "status").write_text("Mems_allowed_list:\t0-1\n")
            for n in (0, 1):
                d = root / f"node{n}"
                d.mkdir()
                (d / "meminfo").write_text(f"Node {n} MemTotal: 1048576 kB\n")
                (d / "cpulist").write_text(str(n))
            args = dict(sysfs=root, status=root / "status")
            native = NS(
                device_numa_nodes=mock.Mock(return_value=[1, 0, 1]),
                host_numa_granularity=mock.Mock(return_value=4096),
            )
            with patch.object(
                placement,
                "resolve_numas",
                wraps=partial(placement.resolve_numas, **args),
            ):
                backing = placement.FabricBacking.resolve(
                    native,
                    (1, 2, 3),
                    memory_kind="host",
                    device=0,
                    numa="auto",
                    gpu_devices=(),
                )
            self.assertEqual(backing.nodes, (0, 1))
            native.device_numa_nodes.assert_called_once_with()
            self.assertEqual(
                native.host_numa_granularity.call_args_list,
                [mock.call(0, 0), mock.call(1, 0)],
            )
            for nodes, error in (
                ([], "NO_VISIBLE_GPU"),
                ([0, -1], "HOST_NUMA_UNAVAILABLE"),
            ):
                with self.subTest(nodes=nodes), self.assertRaisesRegex(
                    ValueError, error
                ):
                    native.device_numa_nodes.return_value = nodes
                    placement.resolve_numas("auto", native=native, **args)
            (root / "status").write_text("Mems_allowed_list:\t0\n")
            with self.assertRaisesRegex(ValueError, "OUTSIDE_CPUSET"):
                placement.resolve_numas((0, 1), **args)

    def test_segmented_wire_counts_storage_hosts_not_allocations(self):
        for p in (2, 3):
            parsed, _ = prepare(descriptors(p, locations=(0, 1)))
            manifest = parsed.manifest
            ShardConfig(p).validate_manifest(manifest)
            for tensor in manifest.tensors:
                self.assertEqual(len(tensor.shards), p * 2)
                self.assertEqual(len(placement.host_groups(tensor)), p)

    def test_startup_package_carries_every_physical_capability(self):
        from rtp_llm.models_py.modules.dsv41.engram_distributed.exchange import (
            _assemble_package,
            _prepare_package,
        )

        docs = descriptors(2, locations=(0, 1))
        group = dict(
            generation=docs[0]["generation"],
            model_revision=docs[0]["model_revision"],
            members=[
                dict(shards=[dict(index=i, host_id=f"host-{i}")]) for i in range(2)
            ],
        )
        package, prepared = _assemble_package(
            group,
            docs,
            "unused",
            sample_rows=lambda _, m: {t.name: {} for t in m.tensors},
        )
        self.assertEqual(_prepare_package(group, package), prepared)
        manifest = prepared["manifest"].manifest
        self.assertEqual(len(prepared["capabilities"]), 16)
        self.assertEqual(len(placement.host_groups(manifest.tensors[0])), 2)
        package["documents"][0]["layers"][0]["segments"][0]["weight"]["handle"] = ""
        with self.assertRaisesRegex(ValueError, "invalid capability encoding"):
            _prepare_package(group, package)


class GpuBackingTest(unittest.TestCase):
    def test_gpu_queries_only_selected_devices_once_across_layers(self):
        native = NS(
            check_fabric_devices=mock.Mock(),
            device_granularity=mock.Mock(return_value=4096),
        )
        options = dict(memory_kind="gpu", device=0, numa=0, gpu_devices=(2, 0))
        backing = placement.FabricBacking.resolve(native, (1, 2, 3), **options)
        native.check_fabric_devices.assert_called_once_with((2, 0))
        self.assertEqual(
            native.device_granularity.call_args_list, [mock.call(2), mock.call(0)]
        )
        self.assertEqual(backing.nodes, (2, 0))
        self.assertEqual(backing.target_device(NS(location=2)), 2)
        self.assertEqual(backing.granularity, 4096)

    def test_gpu_rejects_invalid_devices_before_granularity_queries(self):
        native = NS(
            check_fabric_devices=mock.Mock(),
            device_granularity=mock.Mock(),
        )
        options = dict(memory_kind="gpu", device=0, numa=0, gpu_devices=(2,))
        native.check_fabric_devices.side_effect = IndexError()
        with self.assertRaisesRegex(ValueError, "GPU_OUTSIDE_VISIBLE_DEVICES"):
            placement.FabricBacking.resolve(native, (1,), **options)
        native.check_fabric_devices.side_effect = ValueError()
        with self.assertRaisesRegex(ValueError, "GPU_FABRIC_UNSUPPORTED"):
            placement.FabricBacking.resolve(native, (1,), **options)
        native.device_granularity.assert_not_called()

    def test_capacity_checks_every_selected_gpu(self):
        query = NS(mem_get_info=lambda d: ((100 if d == 0 else 20), 200))
        with patch.dict(sys.modules, {"torch": NS(cuda=query)}):
            check_gpu_capacity({0: 60, 1: 10}, 10)
            with self.assertRaisesRegex(ValueError, "device=1"):
                check_gpu_capacity({0: 60, 1: 11}, 10)
            for bad in (0, True, -1, 2**63):
                with self.assertRaises(ValueError):
                    check_gpu_capacity({0: 1}, bad)

    def test_publisher_distributes_without_replicas_and_rolls_back(self):
        for failure in (None, "capacity", "load"):
            owners, loads = [], []
            tensors = [
                NS(
                    name=f"layers.1.engram.embed.{role}",
                    shape=(65536, width),
                    relative_path="weights",
                )
                for role, width in (("weight", 64), ("scale", 2))
            ]
            checkpoint = NS(
                inspect_checkpoint=lambda *_: NS(
                    layers=[NS(layer_id=1, tensors=tensors)],
                    files=[NS(relative_path="weights")],
                    index=None,
                ),
                _cache_page_size=lambda _: None,
                _discard_consumed_pages=None,
                _open_tensor_source=None,
                _same_snapshot=lambda *_, **kw: None,
            )

            class Owner:
                def __init__(self, size, numa, device, *, memory_kind):
                    self.size, self.numa, self.device, self.kind, self.closed = (
                        size,
                        numa,
                        device,
                        memory_kind,
                        False,
                    )
                    owners.append(self)

                def export_handle(self):
                    return uuid4().bytes

                def close(self):
                    self.closed = True

            native = NS(
                FabricAllocation=Owner,
                check_fabric_devices=lambda _: None,
                device_granularity=lambda _: 4096,
                max_allocation_bytes=1 << 30,
                max_load_chunk_bytes=1 << 20,
            )

            def load(owner, root, tensor, snapshot, begin, end, columns, **kwargs):
                loads.append((tensor.name, begin, end, owner.device))
                if failure == "load" and len(loads) == 5:
                    raise OSError("synthetic GPU load failure")

            def capacity(required, reserve):
                self.assertEqual(owners, [])
                self.assertEqual(required, {i: 8192 * 66 for i in range(4)})
                if failure == "capacity":
                    raise ValueError("synthetic capacity failure")

            with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
                Path(root, "config.json").write_text("{}")
                stack.enter_context(
                    patch.dict(
                        sys.modules,
                        {
                            "rtp_llm.models_py.modules.dsv41.engram": NS(
                                EngramLayout=lambda _: NS(
                                    layer_ids=[1], num_embeddings=[65536], head_dim=64
                                ),
                                NgramHashState=lambda *_: None,
                            ),
                            "rtp_llm.models_py.modules.dsv41.engram_distributed.checkpoint": checkpoint,
                        },
                    )
                )
                stack.enter_context(
                    patch.object(placement, "check_gpu_capacity", side_effect=capacity)
                )
                stack.enter_context(
                    patch.multiple(
                        publish,
                        **vars(checkpoint),
                        _load_native=lambda: native,
                        hash_state_digest=lambda _: "sha256:" + "b" * 64,
                    )
                )
                stack.enter_context(patch.object(publish, "_reclaim_legacy_tables"))
                stack.enter_context(
                    patch.object(publish, "load_rows", side_effect=load)
                )
                generation = str(uuid4())

                def invoke():
                    return publish.shard_publisher.publish(
                        root,
                        "revision",
                        "host-0",
                        generation,
                        0,
                        2,
                        memory_kind="gpu",
                        gpu_devices=(0, 1, 2, 3),
                    )

                if failure:
                    with self.assertRaises((ValueError, OSError)):
                        invoke()
                    self.assertTrue(all(o.closed for o in owners))
                    self.assertNotIn(generation, publish.shard_publisher._owners)
                else:
                    doc = invoke()
                    self.assertEqual(doc["memory_kind"], "gpu")
                    self.assertEqual(
                        [
                            (s["row_begin"], s["row_end"], s["location"])
                            for s in doc["layers"][0]["segments"]
                        ],
                        [(8192 * d, 8192 * (d + 1), d) for d in range(4)],
                    )
                    self.assertEqual(sum(o.size for o in owners), 32768 * 66)
                    self.assertTrue(
                        all(o.numa == -1 and o.kind == "gpu" for o in owners)
                    )
                    for t in tensors:
                        self.assertEqual(
                            [(a, b, d) for name, a, b, d in loads if name == t.name],
                            [(8192 * d, 8192 * (d + 1), d) for d in range(4)],
                        )
                    for owner in publish.shard_publisher._owners.pop(generation):
                        owner.close()


class Tensor:
    def __init__(self, name, shape):
        self.name = name
        self.shape = tuple(shape)


class LoadRowsBoundsTest(unittest.TestCase):
    def test_row_range_and_chunk_bounds_are_checked_before_any_read(self):
        tensor = Tensor("layers.1.engram.embed.weight", (100, 64))
        for start, end, columns, chunk in [
            (1, 1, 64, 4096),
            (0, 101, 64, 4096),
            (0, 10, 32, 4096),
            (0, 10, 64, 0),
            (0, 10, 64, (64 << 20) + 1),
        ]:
            with self.subTest(case=(start, end, columns, chunk)), self.assertRaises(
                ValueError
            ):
                publish.load_rows(
                    None, None, tensor, None, start, end, columns, chunk_bytes=chunk
                )


class LegacyCleanupDiagnosticTest(unittest.TestCase):
    def test_cleanup_failure_is_best_effort_and_redacted(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        cleanup = Mock(side_effect=OSError("private checkpoint path and capability"))
        with patch.dict(
            sys.modules,
            {
                "rtp_llm.models_py.modules.dsv41.engram": SimpleNamespace(
                    reclaim_stale_shared_tables=cleanup
                )
            },
        ), self.assertLogs(publish._LOGGER, level="WARNING") as logs:
            publish._reclaim_legacy_tables(None)
        cleanup.assert_called_once_with()
        self.assertIn("status=failed", logs.output[0])
        self.assertIn("error_class=OSError", logs.output[0])
        self.assertNotIn("private checkpoint", logs.output[0])
        self.assertNotIn("capability", logs.output[0])

    def test_cleanup_observability_cannot_fail_publication(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        cleanup = Mock()
        with patch.dict(
            sys.modules,
            {
                "rtp_llm.models_py.modules.dsv41.engram": SimpleNamespace(
                    reclaim_stale_shared_tables=cleanup
                )
            },
        ), patch.object(publish._LOGGER, "log", side_effect=RuntimeError("log failed")):
            publish._reclaim_legacy_tables(None)
        cleanup.assert_called_once_with()

    def test_cleanup_honors_the_existing_publication_deadline(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        now = [0.0]
        cleanup = Mock(side_effect=lambda: now.__setitem__(0, 2.0))
        with patch.dict(
            sys.modules,
            {
                "rtp_llm.models_py.modules.dsv41.engram": SimpleNamespace(
                    reclaim_stale_shared_tables=cleanup
                )
            },
        ), patch.object(publish.time, "monotonic", lambda: now[0]):
            with self.assertRaisesRegex(ValueError, "LOCAL_SHARD_PUBLISH_TIMEOUT"):
                publish._reclaim_legacy_tables(1.0)
            cleanup.assert_called_once_with()
            cleanup.reset_mock()
            with self.assertRaisesRegex(ValueError, "LOCAL_SHARD_PUBLISH_TIMEOUT"):
                publish._reclaim_legacy_tables(1.0)
            cleanup.assert_not_called()


class PublishFailureTest(unittest.TestCase):
    def test_chunk_deadline_checked_after_read_before_native_write(self):
        import io
        import sys
        import time
        from contextlib import contextmanager
        from types import SimpleNamespace as NS
        from unittest.mock import patch

        @contextmanager
        def opened(*args):
            yield io.BytesIO(bytes(512)), 0

        now = [0.0]

        def read(source, count):
            now[0] += 2
            return source.read(count)

        checkpoint = NS(
            _cache_page_size=lambda _: None,
            _discard_consumed_pages=lambda *args: None,
            _open_tensor_source=opened,
            _read_exact=read,
        )
        writes = []
        tensor = NS(shape=(8, 64), data_range=(0, 512))
        with patch.multiple(publish, **vars(checkpoint)), patch.object(
            publish.time, "monotonic", lambda: now[0]
        ):
            with self.assertRaisesRegex(ValueError, "LOCAL_SHARD_PUBLISH_TIMEOUT"):
                publish.load_rows(
                    NS(load=lambda *args: writes.append(args)),
                    None,
                    tensor,
                    None,
                    0,
                    8,
                    64,
                    chunk_bytes=64,
                    deadline=1,
                )
        self.assertEqual(writes, [])

    def test_local_budget_preflight_and_cleanup_retry(self):
        made, phases = [], []

        class Owner:
            def __init__(self, *args):
                phases.append("allocate")
                self.fail_close = True
                made.append(self)

            def close(self):
                if self.fail_close:
                    raise RuntimeError("injected close failure")

        native = NS(
            FabricAllocation=Owner,
            max_load_chunk_bytes=4096,
            max_allocation_bytes=1 << 30,
            host_numa_granularity=lambda *args: 4096,
        )
        layout = NS(layer_ids=(1,), num_embeddings=(100000,), head_dim=64)
        layers = [
            NS(
                layer_id=1,
                tensors=[
                    NS(
                        name="layers.1.engram.embed." + role,
                        shape=(100000, width),
                        relative_path="weights",
                    )
                    for role, width in [("weight", 64), ("scale", 2)]
                ],
            )
        ]
        checkpoint = NS(
            inspect_checkpoint=lambda *args, **kwargs: NS(
                layers=layers, files=[NS(relative_path="weights")]
            ),
            _cache_page_size=lambda _: None,
            _discard_consumed_pages=None,
            _open_tensor_source=None,
            _same_snapshot=lambda *a, **k: None,
        )
        model = NS(
            EngramLayout=lambda _: layout,
            NgramHashState=lambda *args: None,
            reclaim_stale_shared_tables=lambda: phases.append("reclaim"),
        )
        modules = {
            "rtp_llm.models_py.modules.dsv41.engram": model,
            "rtp_llm.models_py.modules.dsv41.engram_distributed.checkpoint": checkpoint,
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text("{}")
            with patch.multiple(publish, **vars(checkpoint)), patch.dict(
                sys.modules, modules
            ), patch.object(publish, "_load_native", lambda: native), patch.object(
                publish, "hash_state_digest", lambda _: "sha256:" + "0" * 64
            ):
                with self.assertRaisesRegex(ValueError, "host budget"):
                    publish.shard_publisher.publish(
                        directory, "rev", "host", "budget-test", 0, 2, budget_bytes=4096
                    )
                self.assertEqual(made, [])  # fail before even the first allocation
                self.assertEqual(phases, [])
                with patch.object(
                    publish, "load_rows", side_effect=OSError("injected I/O failure")
                ):
                    with self.assertRaisesRegex(RuntimeError, "CLEANUP_INCOMPLETE"):
                        publish.shard_publisher.publish(
                            directory, "rev", "host", "cleanup-test", 0, 2
                        )
                self.assertEqual(phases, ["reclaim", "allocate"])
                self.assertEqual(
                    len(publish.shard_publisher._owners["cleanup-test"]), 1
                )
                self.assertEqual(
                    len(publish.shard_publisher.release("cleanup-test")), 1
                )
                made[0].fail_close = False
                self.assertEqual(publish.shard_publisher.release("cleanup-test"), [])
                self.assertNotIn("cleanup-test", publish.shard_publisher._owners)


def sharded_fixture():
    layers = ((1, 8192), (2, 8185))
    parsed, handles = prepare(descriptors(2, 32, layers=layers))
    layout = m.ExpectedLayout(layers, 32, "revision", HASH)
    return parsed, handles, layout


class View:
    def __init__(self, mapping):
        self.mapping = mapping


class Native:
    def __init__(self, fail_import=None, fail_view=None, fail_close=False):
        self.fail_import, self.fail_view, self.fail_close = (
            fail_import,
            fail_view,
            fail_close,
        )
        self.imports, self.views, self.closed = [], 0, []
        self.attempts = 0

    def FabricMapping(self, device, memory_kind):
        mapping = Mapping(self, len(self.imports))
        self.imports.append(mapping)
        return mapping


class Mapping:
    def __init__(self, native, index):
        self.native, self.index = native, index
        self.closed = False
        self.segments = []
        self.refs = []
        self.view_arguments = []

    def attach(self, segments):
        self.native.attempts += 1
        if self.native.attempts == self.native.fail_import:
            error = RuntimeError("injected import failure")
            error.operation = "import_handle"
            error.segment = len(segments) - 1
            error.cuda_error = 1
            error.cuda_error_name = "CUDA_ERROR_INVALID_VALUE"
            raise error
        self.segments = segments

    def tensor_view(self, shape, offset):
        self.view_arguments.append((shape, offset))
        self.native.views += 1
        if self.native.views == self.native.fail_view:
            raise RuntimeError("injected view failure")
        result = View(self)
        self.refs.append(weakref.ref(result))
        return result

    def info(self):
        return dict(
            active_views=sum(r() is not None for r in self.refs),
            mapped_ranges=0 if self.closed else len(self.segments),
            closed=self.closed,
        )

    def close(self):
        if self.native.fail_close and self.segments:
            raise RuntimeError("injected drain failure")
        if self.info()["active_views"]:
            raise RuntimeError("live views")
        if not self.closed and self.segments:
            self.native.closed.append(self.index)
        self.closed = True


class StorageTest(unittest.TestCase):
    def attach(
        self, native=None, verify=True, *, handles=None, device=0, generation=GEN
    ):
        parsed, fixture_handles, layout = sharded_fixture()
        storage = s.FabricEngramStorage._attach_validated(
            m.bind_manifest(parsed, layout, generation, m.Limits()),
            handles=fixture_handles if handles is None else handles,
            device=device,
            native=native or Native(),
        )
        if verify:
            storage.verify(lambda *_: True)
        return storage

    def test_two_shard_attach_stitches_two_spans_per_layer(self):
        parsed, handles, _ = sharded_fixture()
        native = Native()
        storage = self.attach(native, verify=False)
        self.assertEqual(storage.info()["allocation_count"], 8)
        self.assertEqual(storage.info()["mapping_count"], 4)
        storage.verify(lambda *_: True)
        weight, scale = storage.table_views(1)
        self.assertEqual(weight.mapping.view_arguments, [((8192, 32), 0)])
        self.assertEqual(scale.mapping.view_arguments, [((8192, 1), 0)])
        tail = next(
            t
            for t in parsed.manifest.tensors
            if t.name == "layers.2.engram.embed.scale"
        )
        self.assertEqual(tail.shape[0], 8185)
        self.assertEqual(
            storage._views[tail.name].mapping.view_arguments, [((8192, 1), 0)]
        )
        self.assertEqual(
            set(storage.info()["allocations"]),
            {tensor.name for tensor in parsed.manifest.tensors},
        )
        for tensor in parsed.manifest.tensors:
            view = storage._views[tensor.name]
            self.assertEqual(
                view.mapping.segments,
                [
                    (
                        handles[part.allocation_id],
                        part.allocation_bytes,
                        part.allocation_granularity,
                    )
                    for part in tensor.shards
                ],
            )
        del weight, scale, view
        storage.close()
        self.assertEqual(storage.info()["state"], "CLOSED")
        self.assertEqual(native.closed, [3, 2, 1, 0])

    def test_sharded_attach_passes_validated_identity_to_mapping_failure(self):
        native = Native(fail_import=1)
        with self.assertLogs(s._LOGGER, level="ERROR") as logged:
            with self.assertRaisesRegex(RuntimeError, "injected import failure"):
                self.attach(native, verify=False)
        event = logged.output[0]
        for field in (
            f"generation={uid(1)}",
            "layer=1",
            "tensor_kind=weight",
            "segment=1",
            "device=0",
            "operation=import_handle",
            "cuda_error=1",
            "cuda_error_name=CUDA_ERROR_INVALID_VALUE",
            "owner_sha256=" + hashlib.sha256(b"host-1").hexdigest()[:16],
        ):
            self.assertIn(field, event)
        self.assertNotIn("1/1/0/weight", event)
        self.assertNotIn("injected import failure", event)

    def test_sharded_live_view_and_graph_pin_block_all_unmaps(self):
        native = Native()
        storage = self.attach(native, verify=False)
        storage.verify(lambda *_: True)
        weight, scale = storage.table_views(1)
        storage._pin_graph()
        before = storage.info()
        with self.assertRaises(s.StorageBusyError):
            storage.close()
        self.assertEqual(storage.info(), before)
        storage._unpin_graph()
        with self.assertRaises(s.StorageBusyError):
            storage.close()
        self.assertEqual(native.closed, [])
        del weight, scale
        storage.close()
        self.assertEqual(native.closed, [3, 2, 1, 0])

    def test_sharded_failed_event_keeps_mappings_and_fence_for_retry(self):
        native = Native()
        storage = self.attach(native, verify=False)
        storage.verify(lambda *_: True)
        event = mock.Mock()
        event.synchronize.side_effect = RuntimeError("event failed")
        storage.register_lookup_event(event)
        with self.assertRaisesRegex(RuntimeError, "event failed"):
            storage.close()
        self.assertEqual(storage.info()["mapping_count"], 4)
        self.assertEqual(native.closed, [])
        event.synchronize.side_effect = None
        storage.close()
        self.assertEqual(event.synchronize.call_count, 2)

    def test_sharded_failed_rollback_retains_all_mappings_for_retry(self):
        native = Native(fail_import=3, fail_close=True)
        with self.assertRaises(s.CleanupPendingError) as failure:
            self.attach(native, verify=False)
        storage = failure.exception.resource
        self.assertEqual(storage.info()["state"], "FAILED")
        self.assertEqual(storage.info()["mapping_count"], 3)
        self.assertEqual(
            sum(v["mapped_ranges"] for v in storage.info()["allocations"].values()), 4
        )
        native.fail_close = False
        storage.close()
        self.assertEqual(native.closed, [1, 0])
        self.assertEqual(storage.info()["state"], "CLOSED")

    def test_model_contract_requires_the_complete_validated_layout(self):
        storage = self.attach()
        storage.require_model_layout((2, 1), (8185, 8192), 32)
        for args in (
            ((1,), (8192,), 32),
            ((1, 2), (8192, 8192), 32),
            ((1, 2), (8192, 8185), 64),
            ((1, 2), (8192, 8192, 8192), 32),
        ):
            with self.assertRaisesRegex(ValueError, "complete model Engram layout"):
                storage.require_model_layout(*args)
        storage.close()

    def test_mapping_budget_counts_all_hosts_without_reusing_local_publisher_budget(
        self,
    ):
        parsed, _, layout = sharded_fixture()
        local_bytes = sum(t.shards[0].allocation_bytes for t in parsed.manifest.tensors)
        validated = m.bind_manifest(
            parsed,
            layout,
            GEN,
            runtime_module._MAPPING_LIMITS,
        )
        self.assertEqual(local_bytes, 264 << 10)
        self.assertEqual(validated.physical_bytes, 2 * local_bytes)
        runtime_module._validate_mapping_capacity(validated, 2)
        # Scale the hard ceiling down to the fixture: use the actual two hosts,
        # not MAX_SHARDS, when checking the generation's total mapping bytes.
        with mock.patch.object(
            runtime_module, "HARD_MAX_TOTAL_ALLOCATION_BYTES", local_bytes
        ):
            runtime_module._validate_mapping_capacity(validated, 2)
        with mock.patch.object(
            runtime_module, "HARD_MAX_TOTAL_ALLOCATION_BYTES", local_bytes - 1
        ):
            with self.assertRaisesRegex(ValueError, "TOTAL_ALLOCATION_LIMIT"):
                runtime_module._validate_mapping_capacity(validated, 2)

    def test_close_is_idempotent(self):
        native = Native()
        storage = self.attach(native)
        self.assertEqual(native.attempts, 4)
        self.assertEqual(native.views, 4)
        self.assertEqual(storage.info()["physical_bytes"], 540672)
        storage.close()
        storage.close()
        self.assertEqual(native.closed, [3, 2, 1, 0])
        self.assertEqual(storage.info()["state"], "CLOSED")

    def test_attach_and_verified_events_report_capacity_without_readiness_ambiguity(
        self,
    ):
        with self.assertLogs(s._LOGGER, level="INFO") as logged:
            storage = self.attach(verify=False)
            self.assertEqual(len(logged.output), 2)
            self.assertNotIn("storage_verified", " ".join(logged.output))
            storage.verify(lambda *_: True)
        self.assertEqual(len(logged.output), 3)
        for event, expected in zip(
            logged.output,
            ("storage_attach_started", "storage_attached", "storage_verified"),
        ):
            self.assertIn(f"event={expected}", event)
            for field in (
                f"generation={uid(1)}",
                "device=0",
                "shard_count=2",
                "physical_bytes=540672",
                "allocation_count=8",
                "duration_ms=",
            ):
                self.assertIn(field, event)
            self.assertNotIn("0/1/0/weight", event)
        self.assertIn("state=MAPPED", logged.output[1])
        self.assertIn("state=READY", logged.output[2])
        self.assertIn("mapping_count=4", logged.output[2])
        with mock.patch.object(s._LOGGER, "log") as emit:
            with storage.use():
                pass
            weight, scale = storage.table_views(1)
            del weight, scale
            storage.register_lookup_event(mock.Mock())
            storage._pin_graph()
            storage._unpin_graph()
            storage.close()
        emit.assert_not_called()

    def test_verification_failure_is_not_logged_as_ready_or_leaked(self):
        storage = self.attach(verify=False)
        error = RuntimeError("secret-capability /private/checkpoint")

        def fail(*_):
            raise error

        with self.assertLogs(s._LOGGER, level="ERROR") as logged:
            with self.assertRaises(RuntimeError) as caught:
                storage.verify(fail)
        self.assertIs(caught.exception, error)
        self.assertEqual(storage.info()["state"], "FAILED")
        event = logged.output[0]
        self.assertIn(
            "event=storage_verification_failed phase=verify state=FAILED", event
        )
        self.assertIn("error_class=RuntimeError", event)
        self.assertNotIn("secret-capability", event)
        self.assertNotIn("/private", event)
        storage.close()

    def test_logger_failure_neither_breaks_ready_nor_masks_original_cleanup(self):
        with mock.patch.object(
            s._LOGGER, "log", side_effect=RuntimeError("broken logger")
        ):
            storage = self.attach()
            self.assertEqual(storage.info()["state"], "READY")
            storage.close()
            native = Native(fail_import=2)
            with self.assertRaisesRegex(RuntimeError, "injected import failure"):
                self.attach(native)
            self.assertEqual(native.closed, [0])

    def test_wrong_identity_and_handle_set_fail_before_cuda(self):
        for change in (
            dict(generation=uid(99)),
            dict(handles={}),
            dict(handles={uid(11): b"one"}),
            dict(handles=dict(sharded_fixture()[1], **{"0/1/0/weight": "not-bytes"})),
            dict(device=True),
        ):
            native = Native()
            with self.assertRaises(ValueError):
                self.attach(native, **change)
            self.assertEqual(native.attempts, 0)

    def test_second_import_failure_rolls_back_first(self):
        native = Native(fail_import=2)
        with self.assertRaisesRegex(RuntimeError, "import failure"):
            self.attach(native)
        self.assertEqual(native.closed, [0])

    def test_partial_view_failure_drops_views_before_reverse_cleanup(self):
        native = Native(fail_view=3)
        with self.assertRaisesRegex(RuntimeError, "view failure"):
            self.attach(native)
        self.assertEqual(native.closed, [3, 2, 1, 0])
        self.assertTrue(all(m.info()["active_views"] == 0 for m in native.imports))

    def test_failed_rollback_is_not_reported_as_clean(self):
        native = Native(fail_view=1, fail_close=True)
        with self.assertRaises(s.CleanupPendingError) as error:
            self.attach(native)
        pending = error.exception.resource
        self.assertEqual(pending.info()["state"], "FAILED")
        native.fail_close = False
        pending.close()
        self.assertEqual(native.closed, [3, 2, 1, 0])

    def test_live_external_views_prevent_any_partial_unmap(self):
        native = Native()
        storage = self.attach(native)
        weight, scale = storage.table_views(1)
        with self.assertRaises(s.StorageBusyError):
            storage.close()
        self.assertEqual(native.closed, [])
        self.assertEqual(storage.info()["state"], "CLOSING")
        with self.assertRaises(RuntimeError):
            storage.table_views(2)
        del weight, scale
        storage.close()
        self.assertEqual(native.closed, [3, 2, 1, 0])

    def test_graph_pin_rejects_drain_without_changing_ready(self):
        storage = self.attach()
        storage._pin_graph()
        with self.assertRaises(s.StorageBusyError):
            storage.close()
        self.assertEqual(storage.info()["state"], "READY")
        storage._unpin_graph()
        storage.close()
        with self.assertRaises(RuntimeError):
            storage._unpin_graph()

    def test_external_lookup_event_drains_before_unmapping(self):
        native = Native()
        storage = self.attach(native)

        class Fence:
            drained = False

            def synchronize(inner):
                self.assertEqual(native.closed, [])
                inner.drained = True

        fence = Fence()
        storage.register_lookup_event(fence)
        storage.close()
        self.assertTrue(fence.drained)
        self.assertEqual(native.closed, [3, 2, 1, 0])
        with self.assertRaises(RuntimeError):
            storage.register_lookup_event(fence)

    def test_ready_model_process_pin_blocks_actual_storage_close_and_survives_model_gc(
        self,
    ):
        native = Native()
        storage = self.attach(native)
        retained = []
        model = types.SimpleNamespace(
            _v4_args=types.SimpleNamespace(
                v41_config={"checkpoint_path": "/local/consumer"}
            ),
            v4=types.SimpleNamespace(
                embed=types.SimpleNamespace(
                    weight=types.SimpleNamespace(device="cuda:0")
                ),
                layers=[types.SimpleNamespace(engram=None) for _ in range(3)],
            ),
            py_hw_kernel_config=types.SimpleNamespace(enable_cuda_graph=True),
        )

        class LayerViews:
            def __init__(self, layer_id):
                self.views = storage.table_views(layer_id)

            def release_external_views(self):
                self.views = None

        from rtp_llm.models_py.modules.dsv41.engram_distributed.config import (
            EngramOptions,
            ShardConfig,
            StoragePolicy,
        )

        runtime = runtime_module.EngramRuntime()
        policy = StoragePolicy.from_environ({"RTP_ENGRAM_STORAGE": "host_shared"})
        runtime.install(
            {"staged": True},
            EngramOptions(
                policy, ShardConfig(2), runtime_module.PROCESS_LIFETIME_GRAPH
            ),
        )
        attachment = runtime_module.EngramModelAttachment.bind(
            model,
            runtime,
            layout_factory=lambda config: types.SimpleNamespace(layer_ids=(1, 2)),
            hash_factory=lambda *args, **kwargs: object(),
            layer_factory=lambda config, layer_id, *args, **kwargs: LayerViews(
                layer_id
            ),
            attach_factory=lambda *args, **kwargs: storage,
            retain=retained.append,
            environ={
                runtime_module.EXTERNAL_GRAPH_MODE_ENV: runtime_module.PROCESS_LIFETIME_GRAPH
            },
        )
        self.assertEqual(attachment.phase, "READY")
        attachment.require_ready(model, is_cuda_graph=True)
        before = storage.info()
        self.assertEqual(before["graph_pins"], 1)
        # Even an explicit model-side view release cannot authorize unmapping
        # while the engine may hold/replay a captured raw pointer.
        for _, _, module in attachment.layers:
            module.release_external_views()
        with self.assertRaises(s.StorageBusyError):
            storage.close()
        self.assertEqual(storage.info(), before)
        self.assertEqual(native.closed, [])
        weak_storage = weakref.ref(storage)
        del model, attachment, storage
        gc.collect()
        self.assertIs(weak_storage(), retained[0])
        with self.assertRaises(s.StorageBusyError):
            retained[0].close()
        self.assertEqual(native.closed, [])
        self.assertEqual(retained[0].info()["graph_pins"], 1)

    def test_unknown_layer_and_use_after_close_fail(self):
        storage = self.attach()
        with self.assertRaises(KeyError):
            storage.table_views(99)
        storage.close()
        with self.assertRaises(RuntimeError):
            with storage.use():
                self.fail("closed storage admitted work")

    def test_close_waits_for_submitter_then_rejects_new_work(self):
        storage = self.attach()
        entered, release, attempted, closed = (threading.Event() for _ in range(4))
        failures = []

        def submit():
            try:
                with storage.use():
                    entered.set()
                    self.assertTrue(release.wait(3))
            except BaseException as error:
                failures.append(error)

        def drain():
            attempted.set()
            try:
                storage.close()
                closed.set()
            except BaseException as error:
                failures.append(error)

        a = threading.Thread(target=submit)
        b = threading.Thread(target=drain)
        a.start()
        self.assertTrue(entered.wait(3))
        b.start()
        self.assertTrue(attempted.wait(3))
        self.assertFalse(closed.is_set())
        release.set()
        a.join(3)
        b.join(3)
        self.assertFalse(a.is_alive() or b.is_alive())
        self.assertFalse(failures)
        self.assertTrue(closed.is_set())

    def test_mapped_is_not_ready_without_explicit_content_verification(self):
        storage = self.attach(verify=False)
        self.assertEqual(storage.info()["state"], "MAPPED")
        with self.assertRaises(RuntimeError):
            storage.table_views(1)
        with self.assertRaises(ValueError):
            storage.verify(lambda views, verified: None)
        self.assertEqual(storage.info()["state"], "FAILED")
        storage.close()

    def test_verifier_cannot_reenter_close_or_use(self):
        storage = self.attach(verify=False)

        def verify(views, manifest):
            with self.assertRaises(s.StorageBusyError):
                storage.close()
            with self.assertRaises(RuntimeError):
                storage.verify(lambda *_: True)
            with self.assertRaises(RuntimeError):
                storage.table_views(1)
            return True

        storage.verify(verify)
        self.assertEqual(storage.info()["state"], "READY")
        storage.close()

    def test_drain_failure_keeps_failure_state(self):
        native = Native(fail_close=True)
        storage = self.attach(native)
        with self.assertRaisesRegex(RuntimeError, "drain failure"):
            storage.close()
        self.assertEqual(storage.info()["state"], "FAILED")
        native.fail_close = False
        storage.close()


if __name__ == "__main__":
    unittest.main()
