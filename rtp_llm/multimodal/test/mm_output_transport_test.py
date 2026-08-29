from contextlib import contextmanager
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MMRdmaSlotPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
)
from rtp_llm.multimodal.transport.base import MMTransportBackend
from rtp_llm.multimodal.transport.grpc.backend import (
    TRANSPORT_BYTES,
    GrpcInlineOutputBackend,
)
from rtp_llm.multimodal.transport.manager import MMOutputTransport
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


class RdmaOutputBackendTest(TestCase):
    def setUp(self):
        # The RDMA output exporter is the one boundary that needs hardware, so it stays a mock;
        # everything on this side of it runs for real.
        self.exporter = MagicMock()
        self.backend = RdmaOutputBackend(self.exporter)

    def test_invalid_descriptor_releases_parsed_slots_and_falls_back(self):
        # The slot we did parse must go back to the pool, or it leaks until exporter GC.
        self.exporter.export_embedding.return_value = [
            _serialized_desc("parsed"),
            b"\x80",
        ]

        with _tensors_look_cuda():
            self.assertIsNone(
                self.backend.try_transfer(_rdma_request(), MMEmbeddingRes([_rows(1)]))
            )

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
            result = self.backend.try_transfer(
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

    def test_export_failure_and_empty_export_fall_back(self):
        with _tensors_look_cuda():
            self.exporter.export_embedding.side_effect = RuntimeError("mr full")
            self.assertIsNone(
                self.backend.try_transfer(_rdma_request(), MMEmbeddingRes([_rows(1)]))
            )
            self.exporter.export_embedding.side_effect = None
            self.exporter.export_embedding.return_value = []
            self.assertIsNone(
                self.backend.try_transfer(_rdma_request(), MMEmbeddingRes([_rows(1)]))
            )

    def test_supports_rejects_without_doing_any_work(self):
        cuda_res = MMEmbeddingRes([_rows(1)])
        with _tensors_look_cuda():
            # Peer did not ask for RDMA (this also covers its circuit breaker being open).
            self.assertFalse(
                self.backend.supports(MultimodalInputsPB(support_rdma=False), cuda_res)
            )
            # No embeddings at all: the error path from mm_embedding_rpc.
            self.assertFalse(
                self.backend.supports(_rdma_request(), MMEmbeddingRes([]))
            )
            self.assertTrue(self.backend.supports(_rdma_request(), cuda_res))
        # No is_cuda patch: a genuinely host-resident embedding must not be exported, and must
        # be rejected before anything gets concatenated.
        self.assertFalse(self.backend.supports(_rdma_request(), cuda_res))
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
    """Lets the degradation logic be tested without mocking the hardware boundary."""

    def __init__(self, name: str, supports: bool, result):
        self.name = name
        self._supports = supports
        self._result = result
        self.transfer_calls = []

    def supports(self, request, res) -> bool:
        return self._supports

    def try_transfer(self, request, res):
        self.transfer_calls.append((request, res))
        return self._result

class MMOutputTransportTest(TestCase):
    def setUp(self):
        self.terminal = GrpcInlineOutputBackend()

    @patch("rtp_llm.multimodal.transport.manager.report_output_metrics")
    def test_failed_delivery_degrades_to_terminal(self, _metrics):
        candidate = _FakeBackend("fake", supports=True, result=None)
        transport = MMOutputTransport([candidate], terminal=self.terminal)

        receipt = transport.transfer(_rdma_request(), MMEmbeddingRes([_rows(2)]))

        self.assertEqual(len(candidate.transfer_calls), 1)
        # Fell through to inline bytes: payload in the tensor field, no descriptors.
        self.assertEqual(len(receipt.output_rdma_slots), 0)
        self.assertTrue(receipt.HasField("multimodal_embedding"))
        self.assertEqual(list(receipt.split_size), [2])

if __name__ == "__main__":
    main()
