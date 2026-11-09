# SPDX-License-Identifier: Apache-2.0
"""Model checkpoint, external scale and warmup contracts; CPU only."""

import fcntl
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.modules.dsv41 import _engram_decode_prefetch as decode_prefetch
from rtp_llm.models_py.modules.dsv41 import dsv41_kernel_jit_warmup as warmup
from rtp_llm.models_py.modules.dsv41 import engram

spec = importlib.util.spec_from_file_location(
    "engram_external_checkpoint_test", Path(__file__).resolve().parents[2] / "engram.py"
)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def config():
    return dict(
        engram_layer_ids=[1, 14],
        engram_num_embeddings=[1000, 1000],
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_compressed_vocab_size=32,
        engram_pad_token_id=2,
        engram_vocab_size=13,
        rms_norm_eps=1e-20,
    )


class Storage:
    def __init__(self):
        self.device = 0
        self.ready = True
        self.uses = 0
        tensors = []
        for layer in (1, 14):
            for role, width in [("weight", 32), ("scale", 1)]:
                tensors.append(
                    SimpleNamespace(
                        name=f"layers.{layer}.engram.embed.{role}", shape=(1000, width)
                    )
                )
        self.validated = SimpleNamespace(
            manifest=SimpleNamespace(
                model_revision="checkpoint",
                backend="fabric_egm",
                tensors=tensors,
                generation="generation-1",
            )
        )

    @contextmanager
    def use(self):
        if not self.ready:
            raise RuntimeError("storage is not READY")
        self.uses += 1
        try:
            yield
        finally:
            self.uses -= 1

    def require_model_layout(self, layer_ids, num_embeddings, head_dim):
        if not self.uses:
            raise AssertionError("layout validation must use storage admission")
        expected = {
            f"layers.{layer}.engram.embed.{role}": (rows, width)
            for layer, rows in zip(layer_ids, num_embeddings)
            for role, width in [("weight", head_dim), ("scale", head_dim // 32)]
        }
        manifest = self.validated.manifest
        actual = {tensor.name: tuple(tensor.shape) for tensor in manifest.tensors}
        if manifest.backend != "fabric_egm" or actual != expected:
            raise ValueError(
                "external generation does not match the complete model Engram layout"
            )


class Embedding:
    def __init__(self):
        self.releases = 0

    def release_external_views(self):
        self.releases += 1


class Projection(torch.nn.Module):
    def __init__(self, weight, scale):
        super().__init__()
        self.weight = weight
        self.scale = scale


class ExternalCheckpointTests(unittest.TestCase):
    def setUp(self):
        # The table adapter imports the production module through its public name.
        # Bind that name to the same real-source fixture used by this test.
        binding = patch.dict(sys.modules, {"rtp_llm.models_py.modules.dsv41.engram": e})
        binding.start()
        self.addCleanup(binding.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prefix = "layers.1.engram."
        self.tensors = {
            self.prefix + "wkv.weight": torch.ones(160, 192).to(torch.float8_e4m3fn),
            self.prefix + "wkv.scale": torch.full((5, 6), 127, dtype=torch.uint8),
            self.prefix + "q_weight": torch.ones(4, 32, dtype=torch.bfloat16),
            self.prefix + "k_weight": torch.ones(4, 32, dtype=torch.bfloat16),
        }
        save_file(self.tensors, self.root / "small.safetensors")
        index = {name: "small.safetensors" for name in self.tensors}
        # These table files intentionally do not exist in the consumer checkpoint.
        index.update(
            {
                self.prefix + "embed.weight": "absent-large.safetensors",
                self.prefix + "embed.scale": "absent-large.safetensors",
            }
        )
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": index})
        )
        self.storage = Storage()
        self.embedding = Embedding()

    @contextmanager
    def cpu_cuda_constructor(self):
        # Mock only CUDA placement/projection construction. safe_open and local
        # WKV/q/k reads are real; separate real-GPU integration tests cover compute.
        module = ModuleType("rtp_llm.models_py.modules.dsv41.utils")
        module.V41MXFP8Linear = Projection
        with patch.dict(sys.modules, {module.__name__: module}), patch.object(
            torch.Tensor, "to", lambda tensor, *args, **kwargs: tensor
        ), patch.object(
            torch.cuda, "device", lambda device: nullcontext()
        ), patch.object(
            torch.cuda,
            "_lazy_init",
            side_effect=AssertionError("CPU test initialized CUDA"),
        ), patch.object(
            e.HostEngramEmbedding, "from_storage", return_value=self.embedding
        ) as attach:
            yield attach

    def test_external_reads_only_small_weights_with_large_files_absent(self):
        import safetensors

        original = safetensors.safe_open
        opened = []

        def tracked(path, *args, **kwargs):
            opened.append(Path(path).name)
            return original(path, *args, **kwargs)

        with self.cpu_cuda_constructor() as attach, patch.object(
            safetensors, "safe_open", side_effect=tracked
        ):
            model = e.Engram.from_checkpoint(
                config(), 1, str(self.root), "cuda:0", external_storage=self.storage
            )
        attach.assert_called_once_with(
            self.storage, 1, scale_group=None, scale_identity="checkpoint|generation-1"
        )
        self.assertEqual(opened, ["small.safetensors"] * 4)
        self.assertIs(model.embed_tokens, self.embedding)
        self.assertEqual(self.embedding.releases, 0)
        self.assertEqual(self.storage.uses, 0)
        self.assertEqual(tuple(model.q_weight.shape), (4, 32))
        model.release_external_views()
        self.assertEqual(self.embedding.releases, 1)

    def test_projection_or_qk_failure_releases_external_views(self):
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        del index["weight_map"][self.prefix + "k_weight"]
        (self.root / "model.safetensors.index.json").write_text(json.dumps(index))
        with self.cpu_cuda_constructor(), self.assertRaises(KeyError):
            e.Engram.from_checkpoint(
                config(), 1, str(self.root), "cuda:0", external_storage=self.storage
            )
        self.assertEqual(self.embedding.releases, 1)

    def test_external_device_mismatch_fails_before_checkpoint_io(self):
        for device in ["cpu", "cuda:1"]:
            with self.subTest(device=device), self.assertRaises(ValueError):
                e.Engram.from_checkpoint(
                    config(),
                    1,
                    "/does-not-exist",
                    device,
                    external_storage=self.storage,
                )

    def test_complete_layout_checked_before_opening_checkpoint(self):
        # A valid requested layer cannot hide a missing or mismatched other layer.
        mutations = [
            lambda ts: ts[:-1],
            lambda ts: ts
            + [SimpleNamespace(name="layers.99.engram.embed.weight", shape=(1, 32))],
            lambda ts: [
                (
                    SimpleNamespace(name=t.name, shape=(7, 1))
                    if t.name.endswith("14.engram.embed.scale")
                    else t
                )
                for t in ts
            ],
        ]
        for mutation in mutations:
            storage = Storage()
            storage.validated.manifest.tensors = mutation(
                storage.validated.manifest.tensors
            )
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                ValueError, "complete model"
            ):
                e.Engram.from_checkpoint(
                    config(), 1, "/does-not-exist", "cuda:0", external_storage=storage
                )

    def test_nonready_storage_fails_before_checkpoint_io(self):
        self.storage.ready = False
        with self.assertRaisesRegex(RuntimeError, "READY"):
            e.Engram.from_checkpoint(
                config(), 1, "/does-not-exist", "cuda:0", external_storage=self.storage
            )

    def test_wrong_backend_never_falls_back_to_legacy(self):
        self.storage.validated.manifest.backend = "local_shm"
        with self.assertRaisesRegex(ValueError, "complete model"):
            e.Engram.from_checkpoint(
                config(), 1, "/does-not-exist", "cuda:0", external_storage=self.storage
            )

    def test_legacy_still_requires_real_table_files(self):
        with self.assertRaises(FileNotFoundError):
            e.Engram.from_checkpoint(config(), 1, str(self.root), "cpu")

    def test_table_shape_is_metadata_for_both_backends(self):
        legacy = e.HostEngramEmbedding(
            torch.zeros(3, 32, dtype=torch.float8_e4m3fn),
            torch.full((3, 1), 127, dtype=torch.uint8),
        )
        self.assertEqual(legacy.table_shape, (3, 32))
        external = e.HostEngramEmbedding.__new__(e.HostEngramEmbedding)
        external._uses_external_storage = True
        external._table_shape = (1000, 32)
        external.weight = None
        self.assertEqual(external.table_shape, (1000, 32))

    def external_embedding(self):
        embedding = e.HostEngramEmbedding.__new__(e.HostEngramEmbedding)
        embedding._initialize_lookup_state(storage=object())
        embedding.weight = embedding.scales = None
        return embedding

    def lookup_model(self, embedding):
        return SimpleNamespace(
            _is_decode_role=False,
            _engram_cpu_runtime=None,
            _engram_layers=((1, 0), (14, 1)),
            v4=SimpleNamespace(
                layers={
                    i: SimpleNamespace(engram=SimpleNamespace(embed_tokens=embedding))
                    for i in (1, 14)
                }
            ),
        )

    def test_cpu_and_prefetch_capabilities_distinguish_backends(self):
        local = e.HostEngramEmbedding(
            torch.zeros(3, 32, dtype=torch.uint8), torch.ones(3, 1, dtype=torch.uint8)
        )
        external = self.external_embedding()
        self.assertTrue(local.supports_cpu_lookup)
        self.assertTrue(local.supports_generic_prefetch)
        self.assertFalse(external.supports_cpu_lookup)
        self.assertFalse(external.supports_generic_prefetch)
        # Views are required before either prefetch path can run.
        self.assertFalse(external.supports_async_prefetch)
        self.assertFalse(external.supports_decode_prefetch)

    def test_local_default_cpu_lookup_still_prepares_once(self):
        local = e.HostEngramEmbedding(
            torch.zeros(3, 32, dtype=torch.uint8), torch.ones(3, 1, dtype=torch.uint8)
        )
        model = self.lookup_model(local)
        runtime = SimpleNamespace(capacity=16, max_rows=16, slots={})
        with patch.dict("os.environ", {}, clear=True), patch.object(
            e, "EngramCPULookupRuntime", return_value=runtime
        ) as create:
            self.assertTrue(e.engram_cpu_lookup_enabled())
            self.assertFalse(e.engram_cpu_lookup_enabled(is_decode_role=True))
            e.prepare_engram_cpu_lookup(model, max_m=16)
            e.prepare_engram_cpu_lookup(model, max_m=16)
        create.assert_called_once_with(model, 16)
        self.assertIs(model._engram_cpu_runtime, runtime)

    def test_external_cpu_flag_never_creates_runtime_or_reads_missing_host_views(self):
        for env in (
            {},
            {"DSV41_CPU_ENGRAM_LOOKUP": "1"},
            {"DSV41_CPU_ENGRAM_LOOKUP": "1", "DSV41_ASYNC_ENGRAM_LOOKUP": "1"},
        ):
            model = self.lookup_model(self.external_embedding())
            with self.subTest(env=env), patch.dict(
                "os.environ", env, clear=True
            ), patch.object(
                e,
                "EngramCPULookupRuntime",
                side_effect=AssertionError("CPU runtime for external views"),
            ) as create:
                e.prepare_engram_cpu_lookup(model, max_m=16)
                e.prepare_engram_cpu_lookup(model, max_m=16)
                create.assert_not_called()
                self.assertIsNone(model._engram_cpu_runtime)

    def test_mixed_model_does_not_submit_cpu_work_for_gpu_only_layer(self):
        local = e.HostEngramEmbedding(
            torch.zeros(3, 32, dtype=torch.uint8), torch.ones(3, 1, dtype=torch.uint8)
        )
        model = self.lookup_model(local)
        model.v4.layers[14].engram.embed_tokens = self.external_embedding()
        with patch.dict("os.environ", {"DSV41_CPU_ENGRAM_LOOKUP": "1"}), patch.object(
            e, "EngramCPULookupRuntime"
        ) as create:
            e.prepare_engram_cpu_lookup(model, max_m=16)
        create.assert_not_called()

    def test_external_async_flag_creates_no_stream_event_or_lookup(self):
        external = self.external_embedding()
        # Missing CUDA metadata deliberately makes any access beyond the
        # capability check fail, without initializing a CUDA context.
        module = SimpleNamespace(embed_tokens=external, _lookup_stream=None)
        with patch.dict(
            "os.environ",
            {"DSV41_CPU_ENGRAM_LOOKUP": "0", "DSV41_ASYNC_ENGRAM_LOOKUP": "1"},
        ), patch.object(
            torch.cuda, "Stream", side_effect=AssertionError("stream created")
        ) as stream, patch.object(
            torch.cuda, "Event", side_effect=AssertionError("event created")
        ) as event:
            e.Engram.prepare_lookup_prefetch(module)
            self.assertIsNone(e.Engram.prefetch_lookup(module, object()))
            self.assertIsNone(external.warmup_prefetch(object(), "cuda:0"))
            with self.assertRaisesRegex(
                RuntimeError, "asynchronous prefetch is not supported"
            ):
                external(object(), "cuda:0", prefetch=True)
        stream.assert_not_called()
        event.assert_not_called()


class LegacyTableMigrationTests(unittest.TestCase):
    """Use real legacy leases/files during the switch to distributed storage."""

    def setUp(self):
        # The table adapter imports the production module through its public name.
        # Bind that name to the same real-source fixture used by this test.
        binding = patch.dict(sys.modules, {"rtp_llm.models_py.modules.dsv41.engram": e})
        binding.start()
        self.addCleanup(binding.stop)
        from rtp_llm.models_py.modules.dsv41.engram_distributed import (
            checkpoint as publish,
        )

        self.publish = publish
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        patcher = patch.object(e, "_SHARED_TABLE_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.dict(sys.modules, {"rtp_llm.models_py.modules.dsv41.engram": e})
        patcher.start()
        self.addCleanup(patcher.stop)

    def table(self, char):
        path = self.root / (e._shared_table_prefix() + char * 24)
        path.write_bytes(b"old local table")
        return path

    def test_orphan_removed_but_live_reader_and_other_uid_are_untouched(self):
        orphan, live = self.table("a"), self.table("b")
        other = self.root / (f"rtp_llm_engram_{os.getuid() + 1}_" + "c" * 24)
        malformed = self.root / (e._shared_table_prefix() + "not-a-table")
        for path in (other, malformed):
            path.write_bytes(b"unrelated")
        with live.open("rb") as lease:
            fcntl.flock(lease, fcntl.LOCK_SH)
            self.publish._reclaim_legacy_tables(None)
            self.assertFalse(orphan.exists())
            self.assertEqual(live.read_bytes(), b"old local table")
            self.assertEqual(other.read_bytes(), b"unrelated")
            self.assertEqual(malformed.read_bytes(), b"unrelated")
        self.publish._reclaim_legacy_tables(None)
        self.assertFalse(live.exists())

    def test_busy_creator_is_skipped_without_stalling_other_cleanup(self):
        busy, orphan = self.table("c"), self.table("d")
        with Path(str(busy) + ".lock").open("wb") as creation:
            fcntl.flock(creation, fcntl.LOCK_EX)
            self.publish._reclaim_legacy_tables(None)
            self.assertTrue(busy.exists())
            self.assertFalse(orphan.exists())
        self.publish._reclaim_legacy_tables(None)
        self.assertFalse(busy.exists())

    def test_live_legacy_writer_is_preserved_until_its_marker_retires(self):
        table = self.table("e")
        marker = Path(f"{table}.partial.{os.getpid()}")
        marker.touch()
        self.publish._reclaim_legacy_tables(None)
        self.assertTrue(table.exists())
        marker.unlink()
        self.publish._reclaim_legacy_tables(None)
        self.assertFalse(table.exists())

    def test_publisher_reclaims_real_orphan_before_first_native_allocation(self):
        from rtp_llm.models_py.modules.dsv41.engram_distributed import checkpoint
        from rtp_llm.models_py.modules.dsv41.engram_distributed import (
            runtime as bootstrap,
        )

        orphan = self.table("f")
        (self.root / "config.json").write_text("{}")
        allocation_calls = []

        def allocate(*args):
            allocation_calls.append(args)
            self.assertFalse(orphan.exists())
            raise RuntimeError("STOP_BEFORE_CUDA")

        native = SimpleNamespace(
            FabricAllocation=allocate,
            max_load_chunk_bytes=4096,
            max_allocation_bytes=1 << 30,
            host_numa_granularity=lambda *args: 4096,
        )
        layout = SimpleNamespace(layer_ids=(1,), num_embeddings=(100000,), head_dim=64)
        plan = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    layer_id=1,
                    tensors=[
                        SimpleNamespace(
                            name="layers.1.engram.embed." + role,
                            shape=(100000, width),
                            relative_path="weights",
                        )
                        for role, width in [("weight", 64), ("scale", 2)]
                    ],
                )
            ],
            files=[SimpleNamespace(relative_path="weights")],
        )
        # Only checkpoint metadata and CUDA allocation are replaced. The
        # publisher, migration hook, UID filtering and flock protocol are real.
        with patch.object(e, "EngramLayout", return_value=layout), patch.object(
            e, "NgramHashState", return_value=None
        ), patch.object(
            checkpoint, "hash_state_digest", return_value="sha256:" + "0" * 64
        ), patch.object(
            checkpoint, "_load_native", return_value=native
        ), patch.object(
            checkpoint, "inspect_checkpoint", return_value=plan
        ), patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialized")
        ), self.assertRaisesRegex(
            RuntimeError, "STOP_BEFORE_CUDA"
        ):
            self.publish.shard_publisher.publish(
                self.root, "rev", "host", "migration", 0, 2
            )
        self.assertEqual(len(allocation_calls), 1)
        self.assertNotIn("migration", self.publish.shard_publisher._owners)


spec = importlib.util.spec_from_file_location(
    "_engram_external_scale_source_under_test",
    Path(__file__).resolve().parents[2] / "_engram_scale_shards.py",
)
provider = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = provider
spec.loader.exec_module(provider)


class ScaleTensor:
    """Small independent byte rows with CUDA metadata and observable copies."""

    def __init__(
        self, data, *, device="cuda:0", dtype=torch.uint8, shape=None, pointer=4096
    ):
        self.data = data
        self.device = torch.device(device)
        self.dtype = dtype
        self.shape = (len(data), 8) if shape is None else shape
        self.ndim = len(self.shape)
        self.requires_grad = False
        self.pointer = pointer
        self.slices = []

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    def is_contiguous(self):
        return True

    def is_conj(self):
        return False

    def is_neg(self):
        return False

    def data_ptr(self):
        return self.pointer

    def numel(self):
        return len(self.data) * 8

    def cpu(self):
        raise AssertionError("external scales must never stage a full CPU table")

    def to(self, *_args, **_kwargs):
        raise AssertionError(
            "external singleton must copy rather than alias via to(device)"
        )

    def clone(self):
        return ScaleTensor([row[:] for row in self.data], device=self.device)

    def fill_(self, value):
        for row in self.data:
            row[:] = [value] * 8
        return self

    def __getitem__(self, selected):
        self.slices.append(selected)
        # Keep references to destination rows so copy_ writes into the parent.
        return ScaleTensor(self.data[selected], device=self.device)

    def copy_(self, source):
        for target, row in zip(self.data, source.data):
            target[:] = row
        return self


class ExternalScaleSourceTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            patch.object(
                torch.cuda,
                "_lazy_init",
                side_effect=AssertionError("CPU regression initialized CUDA"),
            )
        )
        self.stack.enter_context(
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False)
        )
        self.stack.enter_context(patch.object(torch, "Tensor", ScaleTensor))
        self.stack.enter_context(patch.object(provider, "_GENERATIONS", {}))

    @contextmanager
    def collective(self, rank, size=4):
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(provider.dist, "is_initialized", return_value=True)
            )
            stack.enter_context(
                patch.object(provider.dist, "get_world_size", return_value=size)
            )
            stack.enter_context(
                patch.object(provider.dist, "get_rank", return_value=rank)
            )
            stack.enter_context(
                patch.object(provider, "_members", return_value=tuple(range(size)))
            )
            stack.enter_context(
                patch.object(
                    provider,
                    "_local_metadata",
                    return_value={"uuid": f"gpu-{rank}", "eligible": True},
                )
            )
            stack.enter_context(
                patch.object(provider, "_eligible_members", return_value=True)
            )
            stack.enter_context(
                patch.object(provider, "_can_access_peers", return_value=True)
            )
            exchange = stack.enter_context(
                patch.object(
                    provider,
                    "_exchange",
                    side_effect=lambda _group, payload, _phase: [
                        dict(payload) for _ in range(size)
                    ],
                )
            )
            stack.enter_context(
                patch.object(torch.cuda, "device", return_value=nullcontext())
            )
            stack.enter_context(
                patch.object(torch.cuda, "current_stream", return_value=Mock())
            )
            stack.enter_context(
                patch.object(
                    torch,
                    "tensor",
                    side_effect=lambda data, **kw: ScaleTensor(
                        data, shape=(len(data),), **kw
                    ),
                )
            )
            symm = ModuleType("torch.distributed._symmetric_memory")
            symm.empty = Mock(
                side_effect=lambda shape, **kw: ScaleTensor(
                    [[0] * 8 for _ in range(shape[0])], pointer=4096 + rank * 256, **kw
                )
            )
            symm.rendezvous = Mock(
                side_effect=lambda local, **_kw: SimpleNamespace(
                    rank=rank,
                    world_size=size,
                    buffer_ptrs=[4096 + index * 256 for index in range(size)],
                    buffer_size=local.numel(),
                )
            )
            stack.enter_context(patch.dict(sys.modules, {symm.__name__: symm}))
            stack.enter_context(
                patch.object(torch.distributed, "_symmetric_memory", symm, create=True)
            )
            yield SimpleNamespace(group_name="external-scale-test"), symm, exchange

    def test_external_singleton_owns_copy_and_rejects_wrong_device(self):
        source = ScaleTensor([[index] * 8 for index in range(5)])
        for group in (None, object()):
            with self.subTest(group=group), patch.object(
                provider.dist, "is_initialized", return_value=True
            ), patch.object(provider.dist, "get_world_size", return_value=1):
                copied = provider.create_sharded_scales(
                    source, group, "cuda:0", None, external_source=True
                )
                self.assertIsNot(copied, source)
                self.assertEqual(copied.data, source.data)
                copied.data[0][0] = 99
                self.assertEqual(source.data[0][0], 0)
        with self.assertRaisesRegex(ValueError, "same-device CUDA"):
            provider.create_sharded_scales(
                source, None, "cuda:1", None, external_source=True
            )
        with self.assertRaisesRegex(ValueError, "CPU"):
            provider.create_sharded_scales(source, None, "cuda:0", None)

    def test_default_cpu_source_still_uses_existing_transfer(self):
        source = ScaleTensor([[42] * 8], device="cpu")
        transferred = object()
        with patch.object(
            source, "to", return_value=transferred
        ) as transfer, patch.object(
            source, "clone", side_effect=AssertionError("legacy CPU source cloned")
        ):
            self.assertIs(
                provider.create_sharded_scales(source, None, "cuda:0", None),
                transferred,
            )
            transfer.assert_called_once_with(torch.device("cuda:0"))

    def test_external_sources_require_explicit_same_device_and_geometry(self):
        for source in (
            ScaleTensor([[0] * 8], device="cpu"),
            ScaleTensor([[0] * 8], device="cuda:1"),
            ScaleTensor([[0] * 7], shape=(1, 7)),
            ScaleTensor([], shape=(0, 8)),
        ):
            with self.subTest(
                device=source.device, shape=source.shape
            ), self.collective(0) as (group, symm, _):
                with self.assertRaisesRegex(RuntimeError, "metadata disagree"):
                    provider.create_sharded_scales(
                        source, group, "cuda:0", "table", external_source=True
                    )
                symm.empty.assert_not_called()

    def test_rank_copy_covers_rows_once_with_padding_and_cache_hits_read_nothing(self):
        # Non-divisible and empty-tail cases detect accidental full copies and
        # off-by-one slices independently of the production partition formula.
        for rows, expected in (
            (7, ((0, 2), (2, 4), (4, 6), (6, 7))),
            (2, ((0, 1), (1, 2), (2, 2), (2, 2))),
        ):
            for rank, (begin, end) in enumerate(expected):
                with self.subTest(rows=rows, rank=rank), self.collective(rank) as (
                    group,
                    symm,
                    exchange,
                ):
                    source = ScaleTensor([[index] * 8 for index in range(rows)])
                    descriptor = provider.create_sharded_scales(
                        source, group, "cuda:0", f"rows-{rows}", external_source=True
                    )
                    capacity = 2 if rows == 7 else 1
                    wanted = source.data[begin:end] + [
                        [127] * 8 for _ in range(capacity - (end - begin))
                    ]
                    self.assertEqual(descriptor.local_scales.data, wanted)
                    self.assertEqual(
                        source.slices, [slice(begin, end)] if end > begin else []
                    )
                    self.assertEqual(symm.empty.call_count, 1)
                    self.assertEqual(symm.rendezvous.call_count, 1)
                    source.slices.clear()
                    cached = provider.create_sharded_scales(
                        source, group, "cuda:0", f"rows-{rows}", external_source=True
                    )
                    self.assertIs(cached, descriptor)
                    self.assertEqual(source.slices, [])
                    self.assertEqual(symm.empty.call_count, 1)
                    self.assertEqual(
                        [call.args[2] for call in exchange.call_args_list],
                        [
                            "table_metadata",
                            "peer_access",
                            "allocation",
                            "ready",
                            "table_metadata",
                            "peer_access",
                        ],
                    )


class ExternalEmbeddingForWarmup(engram.HostEngramEmbedding):
    """Keep the production metadata property; replace only the lookup kernel."""

    def __init__(self, table_shape):
        # Match from_storage's public/metadata state without a CUDA allocation.
        # There is deliberately no full host tensor, even for these huge shapes.
        self.weight = self.scales = None
        self._initialize_lookup_state(storage=object())
        self._table_shape = table_shape
        self._num_sms = 2
        self.lookups = []

    def __call__(self, indices, device):
        if indices.device.type != "cpu" or not indices.is_contiguous():
            raise AssertionError("test kernel boundary expects contiguous CPU IDs")
        self.lookups.append((indices.clone(), str(device)))


class LocalEmbeddingForWarmup(ExternalEmbeddingForWarmup):
    """Same kernel boundary with real small CPU tensors and local capabilities."""

    def __init__(self, table_shape):
        engram.HostEngramEmbedding.__init__(
            self,
            torch.empty(table_shape, dtype=torch.uint8),
            torch.empty((table_shape[0], table_shape[1] // 32), dtype=torch.uint8),
        )
        self._num_sms = 2
        self.lookups = []


class ExternalEngramWarmupTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            mock.patch.dict(
                os.environ,
                {
                    "WARM_UP": "1",
                    "MODEL_WARM_UP": "1",
                    "DSV41_CPU_ENGRAM_LOOKUP": "1",
                    "DSV41_ASYNC_ENGRAM_LOOKUP": "0",
                },
            )
        )
        self.stack.enter_context(mock.patch.object(warmup, "_ENGRAM_WARMED", set()))
        for name in ("_lazy_init", "init", "synchronize", "current_device"):
            self.stack.enter_context(
                mock.patch.object(
                    torch.cuda,
                    name,
                    side_effect=AssertionError("GPU call in CPU regression"),
                )
            )
        # This CPU test exercises orchestration with CUDA placement replaced at
        # tensor/kernel boundaries. Hidden GPUs must not skip that orchestration
        # at the hardware-availability guard before the real lookup loop runs.
        self.stack.enter_context(
            mock.patch.object(
                warmup.common,
                "_is_cuda_device",
                side_effect=lambda device: torch.device(device).type == "cuda",
            )
        )
        self.stack.enter_context(
            mock.patch.object(warmup.common, "_assert_not_capturing")
        )
        self.sync = self.stack.enter_context(
            mock.patch.object(warmup.common, "_sync_cuda")
        )
        self.stack.enter_context(
            mock.patch.object(
                warmup.common,
                "_run_triton_warmup_launch_with_retry",
                side_effect=lambda label, description, launch, **kwargs: launch(),
            )
        )
        self.gate = self.stack.enter_context(
            mock.patch.object(engram, "gated_engram_residual")
        )
        # Native decode scheduling is another GPU kernel boundary, exercised by
        # test_engram_decode_schedule. Keep this CPU orchestration regression
        # independent of Triton while checking that warmup reaches that boundary.
        self.decode_prepare = self.stack.enter_context(
            mock.patch.object(
                decode_prefetch,
                "prepare",
                return_value=None,
            )
        )
        self.cpu_runtime = self.stack.enter_context(
            mock.patch.object(
                engram,
                "EngramCPULookupRuntime",
                side_effect=AssertionError(
                    "CPU runtime must not use external host views"
                ),
            )
        )

        # Only CUDA placement is changed. Real torch tensors exercise slicing,
        # contiguity and masking in the production loop. Large allocations fail.
        self.allocation_numels = []
        for name in ("zeros", "ones"):
            factory = getattr(torch, name)

            def cpu_factory(*args, _factory=factory, **kwargs):
                kwargs.pop("device", None)
                size = args[0] if isinstance(args[0], (tuple, list)) else args
                numel = 1
                for value in size:
                    numel *= value
                if numel > 65536:
                    raise AssertionError("warmup attempted a full-table allocation")
                self.allocation_numels.append(numel)
                return _factory(*args, **kwargs)

            self.stack.enter_context(
                mock.patch.object(torch, name, side_effect=cpu_factory)
            )

        self.embeddings = [
            ExternalEmbeddingForWarmup((1_000_000_000, 32)),
            ExternalEmbeddingForWarmup((2_000_000_000, 32)),
        ]
        self.hash_outputs = []

        def hash_windows(windows, *, dead_mask):
            self.assertEqual(windows.shape, dead_mask.shape)
            self.assertEqual(dead_mask.dtype, torch.bool)
            hashes = torch.arange(windows.shape[0] * 2 * 4).reshape(
                windows.shape[0], 2, 4
            )
            self.hash_outputs.append(hashes)
            return hashes

        self.state = SimpleNamespace(
            layout=SimpleNamespace(
                layer_ids=(1, 14), n_hash_cols=4, head_dim=32, max_ngram_size=3
            ),
            hash_token_windows=mock.Mock(side_effect=hash_windows),
        )
        self.active_layers = [
            SimpleNamespace(
                engram=SimpleNamespace(
                    embed_tokens=embedding,
                    q_weight=torch.ones((4, 8)),
                    k_weight=torch.ones((4, 8)),
                    eps=1e-20,
                    layer_hash_index=index,
                    prepare_lookup_prefetch=mock.Mock(),
                    _lookup_stream=None,
                )
            )
            for index, embedding in enumerate(self.embeddings)
        ]
        layers = [SimpleNamespace(engram=None) for _ in range(15)]
        for layer_id, layer in zip(self.state.layout.layer_ids, self.active_layers):
            layers[layer_id] = layer
        self.model = SimpleNamespace(
            _engram_hash_state=self.state,
            v4=SimpleNamespace(layers=layers),
            _engram_layers=((1, 0), (14, 1)),
            _engram_cpu_runtime=None,
            _is_decode_role=False,
        )

    def run_warmup(self):
        warmup.warmup_v41_engram_jit(self.model, max_m=3, device="cuda:0")

    def test_external_none_weight_runs_real_loop_without_materializing_table(self):
        self.run_warmup()
        self.assertEqual(self.state.hash_token_windows.call_count, 3)
        for layer_index, embedding in enumerate(self.embeddings):
            self.assertIsNone(embedding.weight)
            self.assertIsNone(embedding.scales)
            self.assertEqual(len(embedding.lookups), 3)
            for (indices, device), hashes in zip(embedding.lookups, self.hash_outputs):
                torch.testing.assert_close(indices, hashes[:, layer_index])
                self.assertEqual(device, "cuda:0")
        self.assertEqual(self.gate.call_count, 4)  # Both masks, both layers at row=1.
        self.sync.assert_called_once_with("cuda:0")
        self.assertTrue(self.allocation_numels)
        self.assertLess(max(self.allocation_numels), 65536)
        self.assertEqual(len(warmup._ENGRAM_WARMED), 1)
        self.cpu_runtime.assert_not_called()
        self.assertEqual(
            self.decode_prepare.call_args_list,
            [mock.call(layer.engram) for layer in self.active_layers],
        )
        for layer in self.active_layers:
            layer.engram.prepare_lookup_prefetch.assert_not_called()

    def test_cache_uses_each_external_table_shape(self):
        self.run_warmup()
        self.run_warmup()
        self.assertEqual(self.state.hash_token_windows.call_count, 3)
        # Simulate another complete generation with a different second table.
        # Identical q/k shapes alone must not make this warmup key a cache hit.
        old_embedding = self.embeddings[1]
        replacement = ExternalEmbeddingForWarmup((2_000_000_001, 32))
        self.embeddings[1] = replacement
        self.model.v4.layers[14].engram.embed_tokens = replacement
        self.run_warmup()
        self.assertEqual(self.state.hash_token_windows.call_count, 6)
        self.assertEqual([len(e.lookups) for e in self.embeddings], [6, 3])
        self.assertEqual(len(old_embedding.lookups), 3)
        self.assertEqual(len(warmup._ENGRAM_WARMED), 2)
        self.assertEqual(self.sync.call_count, 3)

    def test_external_async_flag_skips_special_kernel_and_stream_on_cache_hit(self):
        with mock.patch.dict(
            os.environ,
            {
                "DSV41_CPU_ENGRAM_LOOKUP": "0",
                "DSV41_ASYNC_ENGRAM_LOOKUP": "1",
            },
        ), mock.patch.object(
            ExternalEmbeddingForWarmup,
            "warmup_prefetch",
            side_effect=AssertionError("external prefetch warmup must not run"),
        ) as prefetch:
            self.run_warmup()
            self.run_warmup()
        self.assertEqual(self.state.hash_token_windows.call_count, 3)
        prefetch.assert_not_called()
        self.cpu_runtime.assert_not_called()
        for layer in self.active_layers:
            layer.engram.prepare_lookup_prefetch.assert_not_called()

    def test_cache_distinguishes_same_geometry_backend_capabilities(self):
        for layer in self.active_layers:
            layer.engram.embed_tokens = ExternalEmbeddingForWarmup((3, 32))
        with mock.patch.dict(
            os.environ,
            {
                "DSV41_CPU_ENGRAM_LOOKUP": "0",
                "DSV41_ASYNC_ENGRAM_LOOKUP": "0",
            },
        ):
            self.run_warmup()
            for layer in self.active_layers:
                layer.engram.embed_tokens = LocalEmbeddingForWarmup((3, 32))
            self.run_warmup()
            self.run_warmup()
        self.assertEqual(self.state.hash_token_windows.call_count, 6)
        self.assertEqual(len(warmup._ENGRAM_WARMED), 2)
        for layer in self.active_layers:
            self.assertEqual(layer.engram.prepare_lookup_prefetch.call_count, 2)
        self.cpu_runtime.assert_not_called()

    def test_mixed_warmup_prefetches_only_capable_layer(self):
        local = LocalEmbeddingForWarmup((3, 32))
        local.warmup_prefetch = mock.Mock()
        self.model.v4.layers[1].engram.embed_tokens = local
        name = "rtp_llm.models_py.modules.dsv41._engram_triton"
        kernel = SimpleNamespace(
            PREFETCH_LOOKUP_CONFIG=(4, 4), PREFETCH_LOOKUP_CARVEOUT=100
        )
        with mock.patch.dict(
            os.environ,
            {
                "DSV41_CPU_ENGRAM_LOOKUP": "0",
                "DSV41_ASYNC_ENGRAM_LOOKUP": "1",
            },
        ), mock.patch.dict(sys.modules, {name: kernel}):
            self.run_warmup()
            self.run_warmup()
        self.assertEqual(local.warmup_prefetch.call_count, 3)
        self.assertEqual(
            self.model.v4.layers[1].engram.prepare_lookup_prefetch.call_count, 2
        )
        self.model.v4.layers[14].engram.prepare_lookup_prefetch.assert_not_called()
        self.assertEqual(len(self.embeddings[1].lookups), 3)
        self.cpu_runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
