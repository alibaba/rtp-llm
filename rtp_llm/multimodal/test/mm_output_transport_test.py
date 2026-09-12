import importlib
import sys
import threading
import time
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.py_config_modules import (
    MM_TRANSPORT_MODE_GRPC,
    MM_TRANSPORT_MODE_KVCM,
    MM_TRANSPORT_MODE_RDMA,
    MM_TRANSPORT_MODES,
    MMTransportConfig,
    VitConfig,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MMRdmaSlotPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
    ReleaseLeasePB,
    TensorDataTypePB,
)
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes
from rtp_llm.multimodal.transport import factory as transport_factory
from rtp_llm.multimodal.transport.base import (
    MMOutputTransport,
    MMTransportBackend,
    report_output_metrics,
)
from rtp_llm.multimodal.transport.factory import create_mm_output_transport
from rtp_llm.multimodal.transport.grpc.backend import (
    TRANSPORT_BYTES,
    GrpcInlineOutputBackend,
)
from rtp_llm.multimodal.transport.kvcm import backend as kvcm_backend
from rtp_llm.multimodal.transport.kvcm.backend import (
    _MAX_KVCM_KEY_BYTES,
    _MAX_LOGICAL_VALUES_PER_RECEIPT,
    _MAX_OBJECTS_PER_RECEIPT,
    TRANSPORT_KVCM,
    KvcmOutputBackend,
    _chunk_count,
    _chunk_tensor,
    _concatenated_layout,
)
from rtp_llm.multimodal.transport.rdma.backend import TRANSPORT_RDMA, RdmaOutputBackend


def _serialized_desc(handle: str, nbytes: int = 16) -> bytes:
    slot = MMRdmaSlotPB(roles=[MMRdmaSlotPB.EMBEDDING])
    slot.rdma_descriptor.lease_id = handle
    slot.rdma_descriptor.payload_bytes = nbytes
    slot.rdma_descriptor.tensors.add(
        shape=[1, nbytes // 4],
        nbytes=nbytes,
    )
    return slot.SerializeToString()


def _rows(rows: int, offset: float = 0.0) -> torch.Tensor:
    """[rows, 4] tensor whose values identify it, so concat order is observable."""
    return torch.arange(offset, offset + rows * 4, dtype=torch.float32).reshape(rows, 4)


@contextmanager
def _tensors_look_cuda():
    """Make CPU tensors pass the is_cuda gate without needing a GPU.

    Only the device predicate is faked: torch.cat/.contiguous/.to and the shapes they
    produce stay real, so the concat order and split_size this code derives from them are
    exercised rather than mocked away.
    """
    with patch.object(torch.Tensor, "is_cuda", property(lambda self: True)):
        yield


def _rdma_request() -> MultimodalInputsPB:
    return MultimodalInputsPB(support_rdma=True)


class MMOutputTransportFactoryTest(TestCase):
    def test_default_mode_is_grpc_and_auto_is_not_accepted(self):
        config = MMTransportConfig()

        self.assertEqual(config.mode, MM_TRANSPORT_MODE_GRPC)
        self.assertEqual(
            MM_TRANSPORT_MODES,
            (MM_TRANSPORT_MODE_GRPC, MM_TRANSPORT_MODE_RDMA, MM_TRANSPORT_MODE_KVCM),
        )
        self.assertNotIn("auto", MM_TRANSPORT_MODES)
        self.assertIsInstance(
            create_mm_output_transport(config)._backend, GrpcInlineOutputBackend
        )
        self.assertIsInstance(
            create_mm_output_transport()._backend, GrpcInlineOutputBackend
        )

    def test_default_grpc_path_does_not_import_optional_transport_modules(self):
        real_import = __import__

        def reject_optional_import(name, *args, **kwargs):
            if name.startswith(
                (
                    "kv_cache_manager",
                    "rtp_llm.multimodal.transport.kvcm",
                    "rtp_llm.multimodal.transport.rdma",
                )
            ):
                raise AssertionError(f"default gRPC path imported {name}")
            return real_import(name, *args, **kwargs)

        blocked_modules = {
            name: None
            for name in (
                "kv_cache_manager",
                "kv_cache_manager.client",
                "rtp_llm.multimodal.transport.kvcm",
                "rtp_llm.multimodal.transport.kvcm.backend",
                "rtp_llm.multimodal.transport.rdma",
                "rtp_llm.multimodal.transport.rdma.backend",
            )
        }
        with patch.dict(sys.modules, blocked_modules), patch(
            "builtins.__import__", side_effect=reject_optional_import
        ):
            reloaded_factory = importlib.reload(transport_factory)
            transport = reloaded_factory.create_mm_output_transport(MMTransportConfig())
        self.assertIsInstance(transport._backend, GrpcInlineOutputBackend)

    @patch("rtp_llm.multimodal.transport.rdma.backend.RdmaOutputBackend.create")
    def test_explicit_rdma_mode_selects_only_rdma_backend(self, create):
        backend = MagicMock(spec=MMTransportBackend)
        create.return_value = backend
        config = MMTransportConfig()
        config.mode = MM_TRANSPORT_MODE_RDMA

        transport = create_mm_output_transport(config, local_device_id=7)

        self.assertIs(transport._backend, backend)
        create.assert_called_once_with(config.rdma, 7)

    @patch("rtp_llm.multimodal.transport.kvcm.backend.KvcmOutputBackend.create")
    def test_explicit_kvcm_mode_selects_only_kvcm_backend(self, create):
        backend = MagicMock(spec=MMTransportBackend)
        create.return_value = backend
        config = MMTransportConfig()
        config.mode = MM_TRANSPORT_MODE_KVCM

        transport = create_mm_output_transport(config, local_device_id=7)

        self.assertIs(transport._backend, backend)
        create.assert_called_once_with(config.kvcm)

    def test_invalid_runtime_mode_is_rejected(self):
        config = MMTransportConfig()
        config.mode = "auto"

        with self.assertRaisesRegex(ValueError, "invalid mm_transport_mode.*auto"):
            create_mm_output_transport(config)

    def test_kvcm_gc_default_outlives_default_multimodal_request(self):
        config = MMTransportConfig()

        self.assertEqual(
            config.kvcm.object_gc_timeout_ms,
            VitConfig.DEFAULT_MM_TIMEOUT_MS + 60 * 1000,
        )

    @patch(
        "rtp_llm.multimodal.transport.rdma.backend.RdmaOutputBackend.create",
        side_effect=RuntimeError("provider init failed"),
    )
    def test_rdma_initialization_failure_is_propagated(self, create):
        config = MMTransportConfig()
        config.mode = MM_TRANSPORT_MODE_RDMA

        with self.assertRaisesRegex(RuntimeError, "provider init failed"):
            create_mm_output_transport(config)

        create.assert_called_once_with(config.rdma, 0)

    @patch(
        "rtp_llm.multimodal.transport.kvcm.backend.KvcmOutputBackend.create",
        side_effect=RuntimeError("kvcm init failed"),
    )
    def test_kvcm_initialization_failure_is_propagated(self, create):
        config = MMTransportConfig()
        config.mode = MM_TRANSPORT_MODE_KVCM

        with self.assertRaisesRegex(RuntimeError, "kvcm init failed"):
            create_mm_output_transport(config)

        create.assert_called_once_with(config.kvcm)


class RdmaOutputBackendTest(TestCase):
    def setUp(self):
        # The RDMA output exporter is the one boundary that needs hardware, so it stays a mock;
        # everything on this side of it runs for real.
        self.exporter = MagicMock()
        self.backend = RdmaOutputBackend(self.exporter)

    def test_invalid_descriptor_releases_parsed_slots_and_raises(self):
        self.exporter.export_embedding.return_value = [
            _serialized_desc("parsed"),
            b"\x80",
        ]

        with _tensors_look_cuda(), self.assertRaisesRegex(
            RuntimeError, "invalid RDMA descriptor"
        ):
            self.backend.transfer(_rdma_request(), MMEmbeddingRes([_rows(1)]))

        self.exporter.release.assert_called_once_with(["parsed"])

    def test_successful_transfer_preserves_order_and_shapes(self):
        embeddings = [_rows(2), _rows(3, offset=100.0)]
        positions = [_rows(2, offset=10.0), _rows(3, offset=20.0)]
        extras = [torch.ones(5), torch.zeros(6)]
        self.exporter.export_embedding.return_value = [
            _serialized_desc("one", nbytes=16),
            _serialized_desc("two", nbytes=8),
        ]

        with _tensors_look_cuda():
            result = self.backend.transfer(
                _rdma_request(),
                MMEmbeddingRes(embeddings, position_ids=positions, extra_input=extras),
            )

        args = self.exporter.export_embedding.call_args.args
        # Embedding and position ids are concatenated in list order; extras stay per-image.
        self.assertTrue(torch.equal(args[0], torch.cat(embeddings)))
        self.assertTrue(torch.equal(args[1], torch.cat(positions)))
        self.assertEqual(len(args[2]), 2)
        self.assertTrue(torch.equal(args[2][0], extras[0]))
        self.assertTrue(torch.equal(args[2][1], extras[1]))

        # Descriptor order is what lets the LLM re-concat the chunks.
        self.assertEqual(
            [
                slot.rdma_descriptor.lease_id
                for slot in result.receipt.output_rdma_slots
            ],
            ["one", "two"],
        )
        # split_size must describe the per-image row counts of the un-concatenated inputs.
        self.assertEqual(list(result.receipt.split_size), [2, 3])
        # The inline tensor fields stay empty on the RDMA path.
        self.assertFalse(result.receipt.HasField("multimodal_embedding"))
        self.assertEqual(result.transport, TRANSPORT_RDMA)

    def test_export_failure_and_empty_export_raise(self):
        with _tensors_look_cuda():
            self.exporter.export_embedding.side_effect = RuntimeError("mr full")
            with self.assertRaisesRegex(RuntimeError, "mr full"):
                self.backend.transfer(_rdma_request(), MMEmbeddingRes([_rows(1)]))
            self.exporter.export_embedding.side_effect = None
            self.exporter.export_embedding.return_value = []
            with self.assertRaisesRegex(RuntimeError, "returned no descriptors"):
                self.backend.transfer(_rdma_request(), MMEmbeddingRes([_rows(1)]))

    def test_invalid_rdma_request_inputs_raise_before_export(self):
        cuda_res = MMEmbeddingRes([_rows(1)])
        with _tensors_look_cuda():
            with self.assertRaisesRegex(RuntimeError, "did not advertise RDMA"):
                self.backend.transfer(MultimodalInputsPB(support_rdma=False), cuda_res)
            with self.assertRaisesRegex(RuntimeError, "no multimodal embeddings"):
                self.backend.transfer(_rdma_request(), MMEmbeddingRes([]))
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            self.backend.transfer(_rdma_request(), cuda_res)
        self.exporter.export_embedding.assert_not_called()


class _FakeKvcmWriter:
    def __init__(self):
        self.saved = []
        self.removed = []
        self.save_error = None
        self.remove_event = threading.Event()

    def save(self, keys, tensors):
        self.saved.append((list(keys), list(tensors)))
        if self.save_error is not None:
            raise self.save_error

    def remove(self, keys):
        self.removed.append(list(keys))
        self.remove_event.set()


def _kvcm_config(max_object_bytes=32, max_receipt_bytes=1024):
    config = MMTransportConfig().kvcm
    config.max_object_bytes = max_object_bytes
    config.max_receipt_bytes = max_receipt_bytes
    config.object_gc_timeout_ms = 60 * 1000
    return config


class KvcmOutputPipelineTest(TestCase):
    @patch("rtp_llm.multimodal.transport.base.kmonitor.report")
    def test_factory_to_release_pipeline_uses_packaged_client_contract(self, report):
        config = MMTransportConfig()
        config.mode = MM_TRANSPORT_MODE_KVCM
        config.kvcm.addresses = ["10.0.0.1:19001"]
        config.kvcm.instance_id = "rtp-emb-pipeline"
        config.kvcm.instance_group = "epd-emb"
        config.kvcm.user_data = "rtp"
        config.kvcm.transfer_client_config = '{"block_size": 1}'
        config.kvcm.max_object_bytes = 32
        config.kvcm.max_receipt_bytes = 1024
        config.kvcm.object_gc_timeout_ms = 60 * 1000

        writer = _FakeKvcmWriter()
        writer.close = MagicMock()
        config_type = MagicMock(side_effect=lambda **values: SimpleNamespace(**values))
        writer_type = MagicMock(return_value=writer)
        package = ModuleType("kv_cache_manager")
        package.__path__ = []
        client_module = ModuleType("kv_cache_manager.client")
        client_module.KvMetaObjectClientConfig = config_type
        client_module.KvMetaObjectClient = writer_type
        package.client = client_module

        with patch.dict(
            sys.modules,
            {
                "kv_cache_manager": package,
                "kv_cache_manager.client": client_module,
            },
        ):
            transport = create_mm_output_transport(config)

        try:
            receipt = transport.transfer(
                MultimodalInputsPB(support_kvcm=True),
                MMEmbeddingRes(
                    [_rows(2), _rows(1, offset=100.0)],
                    position_ids=[
                        torch.arange(2, dtype=torch.int32),
                        torch.arange(10, 11, dtype=torch.int32),
                    ],
                    extra_input=[
                        torch.tensor([1.0], dtype=torch.float16),
                        torch.tensor([2, 3], dtype=torch.int32),
                    ],
                ),
            )
            objects = list(receipt.output_kvcm_objects)
            keys = [obj.key for obj in objects]

            self.assertEqual(list(receipt.split_size), [2, 1])
            self.assertEqual([obj.value_size for obj in objects], [32, 16, 12, 2, 8])
            self.assertEqual(writer.saved[0][0], keys)
            self.assertEqual(report.call_count, 5)

            transport.release(ReleaseLeasePB(lease_id=keys))
            self.assertEqual(writer.removed, [keys])
            self.assertEqual(transport._backend._pending, {})
        finally:
            transport.close()

        config_type.assert_called_once()
        writer_type.assert_called_once()
        writer.close.assert_called_once_with()


class KvcmOutputBackendTest(TestCase):
    def setUp(self):
        self.writer = _FakeKvcmWriter()
        self.backend = KvcmOutputBackend(self.writer, _kvcm_config())
        self.addCleanup(self.backend.close)

    def test_variable_size_objects_are_chunked_and_described_exactly(self):
        embeddings = [_rows(2), _rows(3, offset=100.0)]
        positions = [
            torch.arange(2, dtype=torch.int32),
            torch.arange(3, dtype=torch.int32),
        ]
        extras = [
            torch.ones(3, dtype=torch.float16),
            torch.arange(2, dtype=torch.int32),
        ]
        result = self.backend.transfer(
            MultimodalInputsPB(support_kvcm=True),
            MMEmbeddingRes(
                embeddings,
                position_ids=positions,
                extra_input=extras,
            ),
        )

        self.assertEqual(result.transport, TRANSPORT_KVCM)
        self.assertEqual(list(result.receipt.split_size), [2, 3])
        objects = list(result.receipt.output_kvcm_objects)
        self.assertEqual([obj.value_size for obj in objects], [32, 32, 16, 20, 6, 8])
        self.assertEqual(
            [obj.role for obj in objects],
            [
                MMRdmaSlotPB.EMBEDDING,
                MMRdmaSlotPB.EMBEDDING,
                MMRdmaSlotPB.EMBEDDING,
                MMRdmaSlotPB.POS_ID,
                MMRdmaSlotPB.EXTRA_INPUT,
                MMRdmaSlotPB.EXTRA_INPUT,
            ],
        )
        self.assertEqual([obj.logical_index for obj in objects], [0, 0, 0, 0, 0, 1])
        self.assertEqual(
            [obj.tensor.nbytes for obj in objects], [obj.value_size for obj in objects]
        )
        self.assertEqual(
            [list(obj.tensor.shape) for obj in objects],
            [[2, 4], [2, 4], [1, 4], [5], [3], [2]],
        )
        self.assertEqual(
            [obj.tensor.data_type for obj in objects],
            [
                TensorDataTypePB.RDMA_TENSOR_FLOAT32,
                TensorDataTypePB.RDMA_TENSOR_FLOAT32,
                TensorDataTypePB.RDMA_TENSOR_FLOAT32,
                TensorDataTypePB.RDMA_TENSOR_INT32,
                TensorDataTypePB.RDMA_TENSOR_FLOAT16,
                TensorDataTypePB.RDMA_TENSOR_INT32,
            ],
        )
        self.assertTrue(all(obj.tensor.offset == 0 for obj in objects))
        self.assertEqual(len({obj.key for obj in objects}), len(objects))
        self.assertEqual(set(self.backend._pending), {obj.key for obj in objects})
        saved_keys, saved_tensors = self.writer.saved[0]
        self.assertEqual(saved_keys, [obj.key for obj in objects])
        self.assertEqual(
            [_tensor.numel() * _tensor.element_size() for _tensor in saved_tensors],
            [obj.value_size for obj in objects],
        )
        combined_embeddings = torch.cat(embeddings)
        expected_tensors = [
            combined_embeddings[:2],
            combined_embeddings[2:4],
            combined_embeddings[4:],
            torch.cat(positions),
            *extras,
        ]
        self.assertEqual(len(saved_tensors), len(expected_tensors))
        for actual, expected in zip(saved_tensors, expected_tensors):
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(result.payload_embedding_bytes, 80)
        self.assertEqual(result.payload_pos_bytes, 20)
        self.assertEqual(result.payload_extra_bytes, 14)

    def test_exact_object_and_receipt_boundaries_are_accepted(self):
        backend = KvcmOutputBackend(
            self.writer, _kvcm_config(max_object_bytes=32, max_receipt_bytes=32)
        )
        try:
            result = backend.transfer(
                MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([_rows(2)])
            )
            objects = list(result.receipt.output_kvcm_objects)
            self.assertEqual([obj.value_size for obj in objects], [32])
            self.assertEqual(result.payload_embedding_bytes, 32)
        finally:
            backend.close()

    def test_async_release_object_capacity_boundary_is_accepted(self):
        backend = KvcmOutputBackend(
            self.writer,
            _kvcm_config(
                max_object_bytes=4,
                max_receipt_bytes=_MAX_OBJECTS_PER_RECEIPT * 4,
            ),
        )
        try:
            tensor = torch.ones((_MAX_OBJECTS_PER_RECEIPT, 1), dtype=torch.int32)
            result = backend.transfer(
                MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([tensor])
            )
            keys = [obj.key for obj in result.receipt.output_kvcm_objects]

            self.assertEqual(len(keys), _MAX_OBJECTS_PER_RECEIPT)
            self.assertEqual(len(self.writer.saved), 1)
            self.assertEqual(len(self.writer.saved[0][0]), _MAX_OBJECTS_PER_RECEIPT)
            backend.release(keys)
            self.assertEqual(self.writer.removed, [keys])
            self.assertEqual(backend._pending, {})
        finally:
            backend.close()

    def test_mixed_supported_dtypes_are_promoted_before_storage(self):
        first = torch.ones((1, 2), dtype=torch.float16)
        second = torch.ones((2, 2), dtype=torch.bfloat16)

        result = self.backend.transfer(
            MultimodalInputsPB(support_kvcm=True),
            MMEmbeddingRes([first, second]),
        )

        saved = self.writer.saved[0][1]
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].dtype, torch.float32)
        self.assertEqual(
            result.receipt.output_kvcm_objects[0].tensor.data_type,
            TensorDataTypePB.RDMA_TENSOR_FLOAT32,
        )
        self.assertEqual(result.payload_embedding_bytes, 24)

    def test_chunk_helpers_reject_invalid_layouts_before_iteration(self):
        for nbytes, rows in ((0, 1), (4, 0), (5, 2)):
            with self.subTest(nbytes=nbytes, rows=rows):
                with self.assertRaisesRegex(ValueError, "stable byte width"):
                    _chunk_count(nbytes, rows, 32)

        cases = [
            (torch.tensor(1), "between 1 and 16 dimensions"),
            (torch.empty((1, 0)), "dimensions must all be positive"),
            (torch.ones((1, 1), dtype=torch.float64), "does not support tensor dtype"),
            (torch.ones((1, 1), device="meta"), "does not support tensor device meta"),
        ]
        for tensor, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    list(_chunk_tensor(tensor, 32))

        with patch.object(kvcm_backend, "_tensor_nbytes", return_value=0):
            with self.assertRaisesRegex(ValueError, "cannot store an empty tensor"):
                list(_chunk_tensor(torch.ones(1), 32))

        with patch.object(torch, "promote_types", return_value=torch.float64):
            with self.assertRaisesRegex(ValueError, "concatenated embeddings dtype"):
                _concatenated_layout([torch.ones((1, 1))], "embeddings")

    def test_transfer_defensive_checks_fail_closed_before_native_storage(self):
        tensor = _rows(1)
        prepared = [
            (tensor, MMRdmaSlotPB.EMBEDDING, 0),
            (tensor, MMRdmaSlotPB.EMBEDDING, 0),
        ]
        with patch.object(
            self.backend, "_prepare_tensors", return_value=(prepared, [2])
        ):
            with patch.object(kvcm_backend, "_MAX_OBJECTS_PER_RECEIPT", 1):
                with self.assertRaisesRegex(RuntimeError, "receipt object limit 1"):
                    self.backend.transfer(
                        MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([])
                    )

        with patch.object(self.backend, "_prepare_tensors", return_value=([], [])):
            with self.assertRaisesRegex(RuntimeError, "produced no exact-size objects"):
                self.backend.transfer(
                    MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([])
                )

        original_receipt_limit = self.backend._max_receipt_bytes
        self.backend._max_receipt_bytes = 8
        try:
            with patch.object(
                self.backend,
                "_prepare_tensors",
                return_value=([(tensor, MMRdmaSlotPB.EMBEDDING, 0)], [1]),
            ):
                with self.assertRaisesRegex(RuntimeError, "output size 16"):
                    self.backend.transfer(
                        MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([])
                    )
        finally:
            self.backend._max_receipt_bytes = original_receipt_limit

        self.assertEqual(self.writer.saved, [])

    def test_post_concat_position_row_check_fails_closed_before_storage(self):
        embedding = _rows(2)
        position = torch.arange(2, dtype=torch.int32)
        real_concat = torch.concat
        call_count = 0

        def truncate_second_concat(tensors, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            combined = real_concat(tensors, *args, **kwargs)
            return combined if call_count == 1 else combined[:-1]

        with patch.object(torch, "concat", side_effect=truncate_second_concat):
            with self.assertRaisesRegex(ValueError, "position rows do not match"):
                self.backend.transfer(
                    MultimodalInputsPB(support_kvcm=True),
                    MMEmbeddingRes([embedding], position_ids=[position]),
                )

        self.assertEqual(call_count, 2)
        self.assertEqual(self.writer.saved, [])

    def test_constructor_rejects_invalid_limits_before_starting_gc(self):
        cases = [
            {"max_object_bytes": 0},
            {"max_object_bytes": True},
            {"max_object_bytes": 1.0},
            {"max_object_bytes": 1024 * 1024 * 1024 + 1},
            {"max_object_bytes": 64, "max_receipt_bytes": 32},
            {"max_receipt_bytes": False},
            {"max_receipt_bytes": 1.0},
            {"max_receipt_bytes": sys.maxsize + 1},
            {"object_gc_timeout_ms": 0},
            {"object_gc_timeout_ms": True},
            {"object_gc_timeout_ms": 1.0},
            {"object_gc_timeout_ms": 1 << 63},
        ]
        with patch.object(threading.Thread, "start") as start:
            for overrides in cases:
                with self.subTest(overrides=overrides):
                    config = _kvcm_config()
                    for name, value in overrides.items():
                        setattr(config, name, value)
                    with self.assertRaises((TypeError, ValueError)):
                        KvcmOutputBackend(self.writer, config)
        start.assert_not_called()

    def test_constructor_rejects_incomplete_writer_before_starting_gc(self):
        with patch.object(threading.Thread, "start") as start:
            for writer in (None, object(), SimpleNamespace(save=lambda *_: None)):
                with self.subTest(writer=writer):
                    with self.assertRaisesRegex(TypeError, "callable save and remove"):
                        KvcmOutputBackend(writer, _kvcm_config())
        start.assert_not_called()

    def test_create_uses_kvcm_python_client_and_maps_config(self):
        config = _kvcm_config()
        config.addresses = ["10.0.0.1:19001", "10.0.0.2:19001"]
        config.instance_id = "rtp-emb-1"
        config.instance_group = "epd-emb"
        config.user_data = "rtp"
        config.transfer_client_config = '{"block_size": 1}'
        config.call_timeout_ms = 1234
        config.write_timeout_seconds = 45
        writer = _FakeKvcmWriter()
        writer.close = MagicMock()
        config_type = MagicMock(side_effect=lambda **values: SimpleNamespace(**values))
        writer_type = MagicMock(return_value=writer)
        package = ModuleType("kv_cache_manager")
        package.__path__ = []
        client_module = ModuleType("kv_cache_manager.client")
        client_module.KvMetaObjectClientConfig = config_type
        client_module.KvMetaObjectClient = writer_type
        package.client = client_module

        with patch.dict(
            sys.modules,
            {
                "kv_cache_manager": package,
                "kv_cache_manager.client": client_module,
            },
        ):
            created = KvcmOutputBackend.create(config)
        try:
            self.assertIs(created._writer, writer)
            self.assertTrue(created._owns_writer)
            config_type.assert_called_once_with(
                addresses=tuple(config.addresses),
                instance_id=config.instance_id,
                instance_group=config.instance_group,
                user_data=config.user_data,
                transfer_client_config=config.transfer_client_config,
                call_timeout_ms=config.call_timeout_ms,
                write_timeout_seconds=config.write_timeout_seconds,
                max_object_bytes=config.max_object_bytes,
            )
            writer_type.assert_called_once()
            self.assertIsInstance(writer_type.call_args.args[0], SimpleNamespace)
        finally:
            created.close()
        writer.close.assert_called_once_with()

    def test_create_reports_missing_kvcm_python_wheel(self):
        with patch.dict(
            sys.modules,
            {"kv_cache_manager": None, "kv_cache_manager.client": None},
        ):
            with self.assertRaisesRegex(RuntimeError, "kvcm_py_client wheel"):
                KvcmOutputBackend.create(_kvcm_config())

    def test_create_preserves_backend_error_when_writer_close_fails(self):
        config = _kvcm_config(max_object_bytes=64, max_receipt_bytes=32)
        writer = _FakeKvcmWriter()
        writer.close = MagicMock(side_effect=RuntimeError("provider detail"))
        config_type = MagicMock(side_effect=lambda **values: SimpleNamespace(**values))
        writer_type = MagicMock(return_value=writer)
        package = ModuleType("kv_cache_manager")
        package.__path__ = []
        client_module = ModuleType("kv_cache_manager.client")
        client_module.KvMetaObjectClientConfig = config_type
        client_module.KvMetaObjectClient = writer_type
        package.client = client_module

        with patch.dict(
            sys.modules,
            {
                "kv_cache_manager": package,
                "kv_cache_manager.client": client_module,
            },
        ), patch("rtp_llm.multimodal.transport.kvcm.backend.logging.warning") as logged:
            with self.assertRaisesRegex(ValueError, "max_receipt_bytes"):
                KvcmOutputBackend.create(config)

        writer.close.assert_called_once_with()
        logged.assert_called_once()
        self.assertIn("RuntimeError", repr(logged.call_args))
        self.assertNotIn("provider detail", repr(logged.call_args))

    def test_directly_injected_writer_is_not_closed_by_backend(self):
        writer = _FakeKvcmWriter()
        writer.close = MagicMock()
        backend = KvcmOutputBackend(writer, _kvcm_config())

        backend.close()

        writer.close.assert_not_called()

    def test_single_pass_logical_iterables_are_snapshotted_exactly_once(self):
        embeddings = [_rows(1), _rows(2, offset=20.0)]
        positions = [
            torch.arange(1, dtype=torch.int32),
            torch.arange(10, 12, dtype=torch.int32),
        ]
        extras = [
            torch.arange(2, dtype=torch.float16),
            torch.arange(3, dtype=torch.int32),
        ]
        response = MMEmbeddingRes([])
        response.embeddings = iter(embeddings)
        response.position_ids = iter(positions)
        response.extra_input = iter(extras)

        result = self.backend.transfer(MultimodalInputsPB(support_kvcm=True), response)

        self.assertEqual(list(result.receipt.split_size), [1, 2])
        saved_tensors = self.writer.saved[0][1]
        self.assertTrue(
            torch.equal(torch.cat(saved_tensors[:2]), torch.cat(embeddings))
        )
        self.assertTrue(torch.equal(saved_tensors[2], torch.cat(positions)))
        self.assertTrue(torch.equal(saved_tensors[3], extras[0]))
        self.assertTrue(torch.equal(saved_tensors[4], extras[1]))

    @patch("rtp_llm.multimodal.transport.kvcm.backend.torch.cuda.synchronize")
    def test_cuda_producer_work_is_synchronized_before_storage(self, synchronize):
        with _tensors_look_cuda():
            self.backend.transfer(
                MultimodalInputsPB(support_kvcm=True),
                MMEmbeddingRes([_rows(1)]),
            )

        synchronize.assert_called_once_with(torch.device("cpu"))
        self.assertEqual(len(self.writer.saved), 1)

    def test_release_removes_only_owned_keys_once(self):
        result = self.backend.transfer(
            MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([_rows(1)])
        )
        key = result.receipt.output_kvcm_objects[0].key

        self.backend.release([key, key, "not-owned"])
        self.backend.release([key])

        self.assertEqual(self.writer.removed, [[key]])

    def test_releasing_one_receipt_keeps_neighboring_receipt_live(self):
        first = self.backend.transfer(
            MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([_rows(1)])
        )
        second = self.backend.transfer(
            MultimodalInputsPB(support_kvcm=True),
            MMEmbeddingRes([_rows(3, offset=100.0)]),
        )
        first_keys = [obj.key for obj in first.receipt.output_kvcm_objects]
        second_keys = [obj.key for obj in second.receipt.output_kvcm_objects]

        self.assertTrue(set(first_keys).isdisjoint(second_keys))
        self.backend.release(first_keys)

        self.assertEqual(self.writer.removed, [first_keys])
        self.assertFalse(set(first_keys) & set(self.backend._pending))
        self.assertEqual(set(self.backend._pending), set(second_keys))

        self.backend.release(second_keys)
        self.assertEqual(self.writer.removed, [first_keys, second_keys])
        self.assertEqual(self.backend._pending, {})

    def test_release_rejects_a_scalar_string(self):
        result = self.backend.transfer(
            MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([_rows(1)])
        )
        key = result.receipt.output_kvcm_objects[0].key

        self.backend.release(key)

        self.assertEqual(self.writer.removed, [])
        self.assertIn(key, self.backend._pending)

    def test_release_validation_is_bounded_and_ignores_malformed_handles(self):
        result = self.backend.transfer(
            MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([_rows(1)])
        )
        key = result.receipt.output_kvcm_objects[0].key

        def oversized_handles():
            yield key
            yield None
            yield []
            yield "x" * (_MAX_KVCM_KEY_BYTES + 1)
            for index in range(_MAX_OBJECTS_PER_RECEIPT - 3):
                yield f"not-owned-{index}"
            # The bounded snapshot consumes exactly one item past the limit,
            # but must never advance into this sentinel.
            yield "first-tail"
            raise AssertionError("KVCM release validation drained its input")

        self.backend.release(oversized_handles())

        self.assertEqual(self.writer.removed, [[key]])

    def test_release_rejects_noniterable_and_invalid_utf8_handles(self):
        result = self.backend.transfer(
            MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([_rows(1)])
        )
        key = result.receipt.output_kvcm_objects[0].key

        self.backend.release(None)
        self.backend.release(["\ud800", b"bytes", "", key])

        self.assertEqual(self.writer.removed, [[key]])
        self.assertNotIn(key, self.backend._pending)

    def test_closed_backend_rejects_transfer_and_ignores_release(self):
        self.backend.close()
        self.backend.close()

        with self.assertRaisesRegex(RuntimeError, "backend is closed"):
            self.backend.transfer(
                MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([_rows(1)])
            )
        self.backend.release(["not-owned"])
        self.assertEqual(self.writer.saved, [])
        self.assertEqual(self.writer.removed, [])

    def test_save_failure_rolls_back_generated_keys(self):
        self.writer.save_error = RuntimeError("store failed")

        with self.assertRaisesRegex(RuntimeError, "store failed"):
            self.backend.transfer(
                MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([_rows(1)])
            )

        self.assertEqual(len(self.writer.saved), 1)
        self.assertEqual(self.writer.removed, [self.writer.saved[0][0]])

    @patch(
        "rtp_llm.multimodal.transport.kvcm.backend.MMOutputResult",
        side_effect=RuntimeError("result construction failed"),
    )
    def test_result_construction_failure_rolls_back_committed_keys(self, _result):
        with self.assertRaisesRegex(RuntimeError, "result construction failed"):
            self.backend.transfer(
                MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([_rows(1)])
            )

        self.assertEqual(len(self.writer.saved), 1)
        self.assertEqual(self.writer.removed, [self.writer.saved[0][0]])

    def test_close_waits_for_inflight_transfer_and_rolls_back_its_objects(self):
        save_entered = threading.Event()
        allow_save = threading.Event()

        class BlockingWriter(_FakeKvcmWriter):
            def save(self, keys, tensors):
                self.saved.append((list(keys), list(tensors)))
                save_entered.set()
                allow_save.wait(timeout=5.0)

        writer = BlockingWriter()
        backend = KvcmOutputBackend(writer, _kvcm_config())
        transfer_errors = []
        transfer_thread = threading.Thread(
            target=lambda: self._record_transfer_error(backend, transfer_errors)
        )
        close_thread = threading.Thread(target=backend.close)

        transfer_thread.start()
        self.assertTrue(save_entered.wait(timeout=1.0))
        close_thread.start()
        close_thread.join(timeout=0.05)
        self.assertTrue(close_thread.is_alive())

        allow_save.set()
        transfer_thread.join(timeout=1.0)
        close_thread.join(timeout=1.0)

        self.assertFalse(transfer_thread.is_alive())
        self.assertFalse(close_thread.is_alive())
        self.assertEqual(len(transfer_errors), 1)
        self.assertRegex(str(transfer_errors[0]), "closed")
        self.assertEqual(writer.removed, [writer.saved[0][0]])

    @staticmethod
    def _record_transfer_error(backend, errors):
        try:
            backend.transfer(
                MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([_rows(1)])
            )
        except Exception as error:  # noqa: BLE001 - asserted by the caller
            errors.append(error)

    def test_close_waits_for_inflight_release_before_finishing(self):
        remove_entered = threading.Event()
        allow_remove = threading.Event()

        class BlockingRemoveWriter(_FakeKvcmWriter):
            def __init__(self):
                super().__init__()
                self.remove_attempts = 0

            def remove(self, keys):
                self.remove_attempts += 1
                self.removed.append(list(keys))
                if self.remove_attempts == 1:
                    remove_entered.set()
                    allow_remove.wait(timeout=5.0)
                    raise RuntimeError("temporary remove failure")

        writer = BlockingRemoveWriter()
        backend = KvcmOutputBackend(writer, _kvcm_config())
        result = backend.transfer(
            MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([_rows(1)])
        )
        key = result.receipt.output_kvcm_objects[0].key
        release_thread = threading.Thread(target=backend.release, args=([key],))
        close_thread = threading.Thread(target=backend.close)

        release_thread.start()
        self.assertTrue(remove_entered.wait(timeout=1.0))
        close_thread.start()
        close_thread.join(timeout=0.05)
        self.assertTrue(close_thread.is_alive())

        allow_remove.set()
        release_thread.join(timeout=1.0)
        close_thread.join(timeout=1.0)

        self.assertFalse(release_thread.is_alive())
        self.assertFalse(close_thread.is_alive())
        # The failed in-flight release is requeued before close takes its
        # final snapshot, so shutdown cleanup makes a second attempt.
        self.assertEqual(writer.removed, [[key], [key]])

    def test_concurrent_close_callers_wait_for_one_shutdown_cleanup(self):
        remove_entered = threading.Event()
        allow_remove = threading.Event()

        class BlockingShutdownWriter(_FakeKvcmWriter):
            def remove(self, keys):
                self.removed.append(list(keys))
                remove_entered.set()
                allow_remove.wait(timeout=5.0)

        writer = BlockingShutdownWriter()
        backend = KvcmOutputBackend(writer, _kvcm_config())
        result = backend.transfer(
            MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([_rows(1)])
        )
        key = result.receipt.output_kvcm_objects[0].key
        first = threading.Thread(target=backend.close)
        second = threading.Thread(target=backend.close)

        first.start()
        self.assertTrue(remove_entered.wait(timeout=1.0))
        second.start()
        second.join(timeout=0.05)
        self.assertTrue(second.is_alive())

        allow_remove.set()
        first.join(timeout=1.0)
        second.join(timeout=1.0)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(writer.removed, [[key]])

    def test_shutdown_cleanup_failure_does_not_resurrect_pending_objects(self):
        result = self.backend.transfer(
            MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([_rows(1)])
        )
        key = result.receipt.output_kvcm_objects[0].key

        def fail_remove(_keys):
            raise RuntimeError("storage unavailable")

        self.writer.remove = fail_remove
        with patch(
            "rtp_llm.multimodal.transport.kvcm.backend.logging.warning"
        ) as logged:
            self.backend.close()
        logged.assert_called_once()
        self.assertIn("shutdown cleanup", repr(logged.call_args))
        self.assertIn("RuntimeError", repr(logged.call_args))
        self.assertNotIn("storage unavailable", repr(logged.call_args))
        self.assertNotIn(key, repr(logged.call_args))
        self.assertEqual(self.backend._pending, {})

        with patch(
            "rtp_llm.multimodal.transport.kvcm.backend.logging.warning"
        ) as warning:
            self.backend._remove_or_retry([key], "post-close probe")
        warning.assert_not_called()
        self.assertEqual(self.backend._pending, {})

    def test_retry_log_omits_native_error_text_and_object_keys(self):
        result = self.backend.transfer(
            MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([_rows(1)])
        )
        key = result.receipt.output_kvcm_objects[0].key

        def fail_remove(_keys):
            raise RuntimeError("secret-provider-detail")

        self.writer.remove = fail_remove
        with patch(
            "rtp_llm.multimodal.transport.kvcm.backend.logging.warning"
        ) as logged:
            self.backend.release([key])

        logged.assert_called_once()
        self.assertIn("release", repr(logged.call_args))
        self.assertIn("object_count", repr(logged.call_args))
        self.assertIn("RuntimeError", repr(logged.call_args))
        self.assertNotIn("secret-provider-detail", repr(logged.call_args))
        self.assertNotIn(key, repr(logged.call_args))
        self.assertIn(key, self.backend._pending)

        # Let addCleanup close the backend without emitting a second injected
        # failure or leaving a pending object in this unit test.
        self.writer.remove = lambda _keys: None

    def test_empty_internal_cleanup_is_a_noop(self):
        self.backend._remove_or_retry([], "empty retry")
        self.backend._best_effort_remove([], "empty cleanup")

        self.assertEqual(self.writer.removed, [])

    def test_gc_wait_is_capped_for_large_valid_deadlines(self):
        config = _kvcm_config()
        config.object_gc_timeout_ms = (1 << 63) - 1
        with patch.object(threading.Thread, "start"):
            backend = KvcmOutputBackend(self.writer, config)
        backend._pending["key"] = time.monotonic() + 10_000.0
        observed_timeouts = []

        def stop_after_wait(timeout=None):
            observed_timeouts.append(timeout)
            backend._closing = True

        with patch.object(backend._condition, "wait", side_effect=stop_after_wait):
            backend._gc_loop()
        backend._closed = True

        self.assertEqual(observed_timeouts, [kvcm_backend._MAX_GC_WAIT_SECONDS])

    def test_operation_accounting_detects_underflow_and_notifies_only_at_zero(self):
        with self.assertRaisesRegex(RuntimeError, "accounting underflow"):
            self.backend._end_operation()

        self.backend._begin_operation()
        self.backend._begin_operation()
        self.assertEqual(self.backend._active_operations, 2)
        self.backend._end_operation()
        self.assertEqual(self.backend._active_operations, 1)
        self.backend._end_operation()
        self.assertEqual(self.backend._active_operations, 0)

    def test_failed_save_rollback_is_retried_by_background_gc(self):
        class RetryWriter(_FakeKvcmWriter):
            def __init__(self):
                super().__init__()
                self.save_error = RuntimeError("store failed")
                self.remove_attempts = 0

            def remove(self, keys):
                self.remove_attempts += 1
                self.removed.append(list(keys))
                if self.remove_attempts == 1:
                    raise RuntimeError("temporary remove failure")
                self.remove_event.set()

        writer = RetryWriter()
        config = _kvcm_config()
        config.object_gc_timeout_ms = 20
        backend = KvcmOutputBackend(writer, config)
        try:
            with self.assertRaisesRegex(RuntimeError, "store failed"):
                backend.transfer(
                    MultimodalInputsPB(support_kvcm=True),
                    MMEmbeddingRes([_rows(1)]),
                )

            self.assertTrue(writer.remove_event.wait(timeout=2.0))
            self.assertEqual(writer.remove_attempts, 2)
            self.assertEqual(writer.removed[0], writer.saved[0][0])
            self.assertEqual(writer.removed[1], writer.saved[0][0])
        finally:
            backend.close()

    def test_expired_objects_are_removed_by_background_gc(self):
        config = _kvcm_config()
        config.object_gc_timeout_ms = 20
        backend = KvcmOutputBackend(self.writer, config)
        try:
            result = backend.transfer(
                MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([_rows(1)])
            )
            key = result.receipt.output_kvcm_objects[0].key

            self.assertTrue(self.writer.remove_event.wait(timeout=1.0))
            self.assertIn([key], self.writer.removed)
        finally:
            backend.close()

    def test_validation_happens_before_storage(self):
        with self.assertRaisesRegex(RuntimeError, "did not advertise KVCM"):
            self.backend.transfer(MultimodalInputsPB(), MMEmbeddingRes([_rows(1)]))
        with self.assertRaisesRegex(ValueError, "one KVCM tensor row"):
            small_backend = KvcmOutputBackend(
                self.writer, _kvcm_config(max_object_bytes=8)
            )
            try:
                small_backend.transfer(
                    MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([_rows(1)])
                )
            finally:
                small_backend.close()

        self.assertEqual(self.writer.saved, [])

    def test_receipt_byte_limit_is_rejected_before_concatenation_or_storage(self):
        backend = KvcmOutputBackend(
            self.writer, _kvcm_config(max_object_bytes=16, max_receipt_bytes=32)
        )
        try:
            with patch(
                "rtp_llm.multimodal.transport.kvcm.backend.torch.concat"
            ) as concat:
                with self.assertRaisesRegex(
                    RuntimeError, "output size 48 exceeds max_receipt_bytes 32"
                ):
                    backend.transfer(
                        MultimodalInputsPB(support_kvcm=True),
                        MMEmbeddingRes([_rows(3)]),
                    )
            concat.assert_not_called()
            self.assertEqual(self.writer.saved, [])
        finally:
            backend.close()

    def test_object_count_limit_is_rejected_before_concatenation_or_storage(self):
        backend = KvcmOutputBackend(
            self.writer,
            _kvcm_config(
                max_object_bytes=4,
                max_receipt_bytes=(_MAX_OBJECTS_PER_RECEIPT + 1) * 4,
            ),
        )
        try:
            tensor = torch.ones((_MAX_OBJECTS_PER_RECEIPT + 1, 1), dtype=torch.int32)
            with patch(
                "rtp_llm.multimodal.transport.kvcm.backend.torch.concat"
            ) as concat:
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"requires {_MAX_OBJECTS_PER_RECEIPT + 1} objects",
                ):
                    backend.transfer(
                        MultimodalInputsPB(support_kvcm=True),
                        MMEmbeddingRes([tensor]),
                    )
            concat.assert_not_called()
            self.assertEqual(self.writer.saved, [])
        finally:
            backend.close()

    def test_logical_value_count_is_bounded_before_concatenation_or_storage(self):
        tiny = torch.ones((1, 1), dtype=torch.float16)

        def guarded_embeddings():
            for _ in range(_MAX_LOGICAL_VALUES_PER_RECEIPT + 1):
                yield tiny
            raise AssertionError("KVCM validation consumed beyond its bounded sentinel")

        with patch("rtp_llm.multimodal.transport.kvcm.backend.torch.concat") as concat:
            with self.assertRaisesRegex(RuntimeError, "logical values"):
                result = MMEmbeddingRes([])
                result.embeddings = guarded_embeddings()
                self.backend.transfer(
                    MultimodalInputsPB(support_kvcm=True),
                    result,
                )
        concat.assert_not_called()
        self.assertEqual(self.writer.saved, [])

    def test_manifest_incompatibilities_are_rejected_before_storage(self):
        with self.assertRaisesRegex(RuntimeError, "no multimodal embeddings"):
            self.backend.transfer(
                MultimodalInputsPB(support_kvcm=True), MMEmbeddingRes([])
            )

        cases = [
            (
                "embedding must be iterable",
                MMEmbeddingRes(None),
            ),
            (
                "embeddings must be torch tensors",
                MMEmbeddingRes(["not-a-tensor"]),
            ),
            (
                "between 1 and 16 dimensions",
                MMEmbeddingRes([torch.tensor(1.0)]),
            ),
            (
                "between 1 and 16 dimensions",
                MMEmbeddingRes([torch.ones((1,) * 17)]),
            ),
            (
                "dimensions must all be positive",
                MMEmbeddingRes([torch.empty((1, 0))]),
            ),
            (
                "position_ids count",
                MMEmbeddingRes(
                    [_rows(2), _rows(3)],
                    position_ids=[torch.arange(5, dtype=torch.int32)],
                ),
            ),
            (
                r"position_ids\[0\] rows",
                MMEmbeddingRes(
                    [_rows(2), _rows(3)],
                    # The aggregate row count is valid, but the per-image
                    # boundaries are not. Accepting this would silently attach
                    # position rows to the wrong image after reconstruction.
                    position_ids=[
                        torch.arange(1, dtype=torch.int32),
                        torch.arange(4, dtype=torch.int32),
                    ],
                ),
            ),
            (
                "non-empty flat tensor",
                MMEmbeddingRes([_rows(1)], extra_input=[torch.ones((1, 1))]),
            ),
            (
                "extra_input values must be torch tensors",
                MMEmbeddingRes([_rows(1)], extra_input=["not-a-tensor"]),
            ),
            (
                "does not support embedding dtype",
                MMEmbeddingRes([torch.ones((1, 4), dtype=torch.float64)]),
            ),
            (
                "does not support embedding device meta",
                MMEmbeddingRes([torch.ones((1, 4), device="meta")]),
            ),
            (
                r"embeddings\[1\] shape is incompatible",
                MMEmbeddingRes([torch.ones((1, 2)), torch.ones((1, 3))]),
            ),
            (
                "position_ids must be iterable",
                MMEmbeddingRes([_rows(1)], position_ids=1),
            ),
            (
                "position_ids must be torch tensors",
                MMEmbeddingRes([_rows(1)], position_ids=["not-a-tensor"]),
            ),
            (
                "position_ids must have between 1 and 16 dimensions",
                MMEmbeddingRes([_rows(1)], position_ids=[torch.tensor(1)]),
            ),
            (
                "position_ids dimensions must all be positive",
                MMEmbeddingRes([_rows(1)], position_ids=[torch.empty((1, 0))]),
            ),
            (
                "does not support position_ids dtype",
                MMEmbeddingRes(
                    [_rows(1)], position_ids=[torch.ones(1, dtype=torch.float64)]
                ),
            ),
            (
                "does not support position_ids device meta",
                MMEmbeddingRes(
                    [_rows(1), _rows(1)],
                    position_ids=[
                        torch.ones(1, dtype=torch.int32),
                        torch.ones(1, dtype=torch.int32, device="meta"),
                    ],
                ),
            ),
            (
                "extra_input must be iterable",
                MMEmbeddingRes([_rows(1)], extra_input=1),
            ),
            (
                "extra_input count",
                MMEmbeddingRes([_rows(1), _rows(1)], extra_input=[torch.ones(1)]),
            ),
            (
                "does not support extra_input dtype",
                MMEmbeddingRes(
                    [_rows(1)], extra_input=[torch.ones(1, dtype=torch.float64)]
                ),
            ),
            (
                "does not support extra_input device meta",
                MMEmbeddingRes(
                    [_rows(1)], extra_input=[torch.ones(1, device="meta")]
                ),
            ),
        ]

        for expected_error, result in cases:
            with self.subTest(expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    self.backend.transfer(MultimodalInputsPB(support_kvcm=True), result)

        self.assertEqual(self.writer.saved, [])

    def test_embedding_row_count_is_checked_against_proto_int32(self):
        with patch.object(kvcm_backend, "_PROTO_INT32_MAX", 1):
            with self.assertRaisesRegex(ValueError, "rows exceed int32"):
                self.backend.transfer(
                    MultimodalInputsPB(support_kvcm=True),
                    MMEmbeddingRes([_rows(2)]),
                )
        self.assertEqual(self.writer.saved, [])


class GrpcInlineOutputBackendTest(TestCase):
    def test_payload_is_encoded_inline(self):
        terminal = GrpcInlineOutputBackend()

        result = terminal.transfer(
            MultimodalInputsPB(support_rdma=True),
            MMEmbeddingRes(
                [_rows(2), _rows(3, offset=100.0)],
                position_ids=[_rows(2, offset=10.0)],
                extra_input=[torch.ones(5)],
            ),
        )

        self.assertEqual(result.transport, TRANSPORT_BYTES)
        self.assertEqual(list(result.receipt.split_size), [2, 3])
        self.assertEqual(len(result.receipt.output_rdma_slots), 0)


class _FakeBackend(MMTransportBackend):
    name = "fake"

    def __init__(self, error):
        self._error = error
        self.transfer_calls = []

    def transfer(self, request, res):
        self.transfer_calls.append((request, res))
        raise self._error


class MMOutputTransportTest(TestCase):
    def setUp(self):
        self.terminal = GrpcInlineOutputBackend()

    @patch("rtp_llm.multimodal.transport.base.report_output_metrics")
    def test_failed_delivery_is_propagated(self, metrics):
        backend = _FakeBackend(RuntimeError("rdma send failed"))
        transport = MMOutputTransport(backend)

        with self.assertRaisesRegex(RuntimeError, "rdma send failed"):
            transport.transfer(_rdma_request(), MMEmbeddingRes([_rows(2)]))

        self.assertEqual(len(backend.transfer_calls), 1)
        metrics.assert_not_called()

    @patch("rtp_llm.multimodal.transport.base.kmonitor.report")
    def test_inline_output_metrics_preserve_payload_sizes(self, report):
        result = self.terminal.transfer(
            MultimodalInputsPB(),
            MMEmbeddingRes(
                [_rows(2)],
                position_ids=[torch.arange(2, dtype=torch.int32)],
                extra_input=[torch.ones(3, dtype=torch.float16)],
            ),
        )

        report_output_metrics(result)

        samples = {call.args[0]: call.args[1] for call in report.call_args_list}
        self.assertEqual(samples[GaugeMetrics.VIT_RESPONSE_EMBEDDING_BYTES_METRIC], 32)
        self.assertEqual(samples[GaugeMetrics.VIT_RESPONSE_POS_BYTES_METRIC], 8)
        self.assertEqual(samples[GaugeMetrics.VIT_RESPONSE_DEEPSTACK_BYTES_METRIC], 6)
        self.assertEqual(samples[GaugeMetrics.VIT_OUTPUT_TOKEN_COUNT_METRIC], 2)
        self.assertEqual(
            samples[GaugeMetrics.VIT_RPC_RESPONSE_BYTES_METRIC],
            result.receipt.ByteSize(),
        )

    @patch("rtp_llm.multimodal.transport.base.kmonitor.report")
    def test_rdma_output_metrics_use_descriptor_payload_sizes(self, report):
        self.exporter = MagicMock()
        self.exporter.export_embedding.return_value = [
            _serialized_desc("one", nbytes=24),
        ]
        backend = RdmaOutputBackend(self.exporter)
        with _tensors_look_cuda():
            result = backend.transfer(_rdma_request(), MMEmbeddingRes([_rows(2)]))

        report_output_metrics(result)

        samples = {call.args[0]: call.args[1] for call in report.call_args_list}
        self.assertEqual(samples[GaugeMetrics.VIT_RESPONSE_EMBEDDING_BYTES_METRIC], 24)
        self.assertEqual(samples[GaugeMetrics.VIT_RESPONSE_POS_BYTES_METRIC], 0)
        self.assertEqual(samples[GaugeMetrics.VIT_RESPONSE_DEEPSTACK_BYTES_METRIC], 0)
        self.assertEqual(samples[GaugeMetrics.VIT_OUTPUT_TOKEN_COUNT_METRIC], 2)
        self.assertEqual(
            samples[GaugeMetrics.VIT_RPC_RESPONSE_BYTES_METRIC],
            result.receipt.ByteSize(),
        )


if __name__ == "__main__":
    main()
