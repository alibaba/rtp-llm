from contextlib import contextmanager
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.py_config_modules import (
    MM_TRANSPORT_MODE_GRPC,
    MM_TRANSPORT_MODE_RDMA,
    MM_TRANSPORT_MODES,
    MMTransportConfig,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MMRdmaSlotPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
)
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
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
from rtp_llm.multimodal.transport.rdma.backend import (
    TRANSPORT_RDMA,
    RdmaOutputBackend,
)
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes


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
        self.assertEqual(MM_TRANSPORT_MODES, (MM_TRANSPORT_MODE_GRPC, MM_TRANSPORT_MODE_RDMA))
        self.assertNotIn("auto", MM_TRANSPORT_MODES)
        self.assertIsInstance(create_mm_output_transport(config)._backend, GrpcInlineOutputBackend)

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
            [slot.rdma_descriptor.lease_id for slot in result.receipt.output_rdma_slots],
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
                self.backend.transfer(
                    MultimodalInputsPB(support_rdma=False), cuda_res
                )
            with self.assertRaisesRegex(RuntimeError, "no multimodal embeddings"):
                self.backend.transfer(_rdma_request(), MMEmbeddingRes([]))
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            self.backend.transfer(_rdma_request(), cuda_res)
        self.exporter.export_embedding.assert_not_called()

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
        self.assertEqual(samples[GaugeMetrics.VIT_RPC_RESPONSE_BYTES_METRIC], result.receipt.ByteSize())

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
        self.assertEqual(samples[GaugeMetrics.VIT_RPC_RESPONSE_BYTES_METRIC], result.receipt.ByteSize())

if __name__ == "__main__":
    main()
