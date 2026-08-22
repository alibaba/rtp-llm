from contextlib import contextmanager
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.py_config_modules import (
    MM_TRANSPORT_MODE_AUTO,
    MM_TRANSPORT_MODE_GRPC,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MMRdmaDescPB,
    MMRdmaTensorPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
    ReleaseEmbeddingPB,
)
from rtp_llm.multimodal.mm_output_transport.base import (
    MMOutputResult,
    MMTransportBackend,
)
from rtp_llm.multimodal.mm_output_transport.factory import create_mm_output_transport
from rtp_llm.multimodal.mm_output_transport.grpc.backend import (
    TRANSPORT_BYTES,
    GrpcInlineTerminal,
)
from rtp_llm.multimodal.mm_output_transport.manager import MMOutputTransport
from rtp_llm.multimodal.mm_output_transport.rdma.backend import (
    TRANSPORT_RDMA,
    RdmaTransportBackend,
)
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes


def _serialized_desc(handle: str, nbytes: int = 16) -> bytes:
    desc = MMRdmaDescPB(handle=handle, nbytes=nbytes)
    desc.tensors.add(
        role=MMRdmaTensorPB.EMBEDDING,
        shape=[1, nbytes // 4],
        nbytes=nbytes,
    )
    return desc.SerializeToString()


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


class RdmaTransportBackendTest(TestCase):
    def setUp(self):
        # The RDMA encoder op is the one boundary that needs hardware, so it stays a mock;
        # everything on this side of it runs for real.
        self.encoder = MagicMock()
        self.backend = RdmaTransportBackend(self.encoder)

    def test_invalid_descriptor_releases_parsed_slots_and_falls_back(self):
        # The slot we did parse must go back to the pool, or it leaks until encoder GC.
        self.encoder.export_embedding.return_value = [
            _serialized_desc("parsed"),
            b"\x80",
        ]

        with _tensors_look_cuda(), patch(
            "rtp_llm.multimodal.mm_output_transport.rdma.backend.kmonitor.report"
        ) as report:
            self.assertIsNone(
                self.backend.try_transfer(_rdma_request(), MMEmbeddingRes([_rows(1)]))
            )

        self.encoder.release.assert_called_once_with(["parsed"])
        reasons = [call.args[2]["reason"] for call in report.call_args_list]
        self.assertEqual(reasons, ["rdma_descriptor_invalid"])

    def test_descriptor_without_handle_falls_back(self):
        # A slot we could never release must not be handed to the LLM.
        self.encoder.export_embedding.return_value = [
            MMRdmaDescPB(nbytes=16).SerializeToString()
        ]

        with _tensors_look_cuda(), patch(
            "rtp_llm.multimodal.mm_output_transport.rdma.backend.kmonitor.report"
        ):
            self.assertIsNone(
                self.backend.try_transfer(_rdma_request(), MMEmbeddingRes([_rows(1)]))
            )
        self.encoder.release.assert_not_called()

    def test_successful_transfer_preserves_order_shapes_and_payload_bytes(self):
        embeddings = [_rows(2), _rows(3, offset=100.0)]
        positions = [_rows(2, offset=10.0), _rows(3, offset=20.0)]
        extras = [torch.ones(5), torch.zeros(6)]
        self.encoder.export_embedding.return_value = [
            _serialized_desc("one", nbytes=16),
            _serialized_desc("two", nbytes=8),
        ]

        with _tensors_look_cuda():
            result = self.backend.try_transfer(
                _rdma_request(),
                MMEmbeddingRes(embeddings, position_ids=positions, extra_input=extras),
            )

        args = self.encoder.export_embedding.call_args.args
        # Embedding and position ids are concatenated in list order; extras stay per-image.
        self.assertTrue(torch.equal(args[0], torch.cat(embeddings)))
        self.assertTrue(torch.equal(args[1], torch.cat(positions)))
        self.assertEqual(len(args[2]), 2)
        self.assertTrue(torch.equal(args[2][0], extras[0]))
        self.assertTrue(torch.equal(args[2][1], extras[1]))

        # Descriptor order is what lets the LLM re-concat the chunks.
        self.assertEqual(
            [desc.handle for desc in result.receipt.output_rdma_slots], ["one", "two"]
        )
        # split_size must describe the per-image row counts of the un-concatenated inputs.
        self.assertEqual(list(result.receipt.split_size), [2, 3])
        # The inline tensor fields stay empty on the RDMA path.
        self.assertFalse(result.receipt.HasField("multimodal_embedding"))
        # Payload bytes are aggregated per role across every slot manifest.
        self.assertEqual(result.transport, TRANSPORT_RDMA)
        self.assertEqual(result.payload_embedding_bytes, 24)
        self.assertEqual(result.payload_pos_bytes, 0)
        self.assertEqual(result.payload_extra_bytes, 0)

    def test_absent_position_ids_and_extras_are_passed_as_none(self):
        self.encoder.export_embedding.return_value = [_serialized_desc("one")]

        with _tensors_look_cuda():
            self.backend.try_transfer(_rdma_request(), MMEmbeddingRes([_rows(2)]))

        args = self.encoder.export_embedding.call_args.args
        self.assertIsNone(args[1])
        self.assertEqual(args[2], [])

    def test_export_failure_and_empty_export_fall_back(self):
        with _tensors_look_cuda(), patch(
            "rtp_llm.multimodal.mm_output_transport.rdma.backend.kmonitor.report"
        ) as report:
            self.encoder.export_embedding.side_effect = RuntimeError("mr full")
            self.assertIsNone(
                self.backend.try_transfer(_rdma_request(), MMEmbeddingRes([_rows(1)]))
            )
            self.encoder.export_embedding.side_effect = None
            self.encoder.export_embedding.return_value = []
            self.assertIsNone(
                self.backend.try_transfer(_rdma_request(), MMEmbeddingRes([_rows(1)]))
            )

        # Only the raising case is an anomaly worth a reason; an empty list is already logged.
        reasons = [call.args[2]["reason"] for call in report.call_args_list]
        self.assertEqual(reasons, ["rdma_export_error"])

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
        self.encoder.export_embedding.assert_not_called()

    def test_release_swallows_failure_and_reports_reason(self):
        self.encoder.release.side_effect = RuntimeError("encoder gone")

        with patch(
            "rtp_llm.multimodal.mm_output_transport.rdma.backend.kmonitor.report"
        ) as report:
            self.backend.release(["one"])  # must not raise: the release RPC stays a success

        self.assertEqual(
            [call.args[2]["reason"] for call in report.call_args_list],
            ["rdma_release_error"],
        )


class GrpcInlineTerminalTest(TestCase):
    def test_payload_is_encoded_inline_with_byte_counts(self):
        terminal = GrpcInlineTerminal()

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
        # 5 rows x 4 cols x fp32
        self.assertEqual(result.payload_embedding_bytes, 80)
        self.assertEqual(result.payload_pos_bytes, 32)
        self.assertEqual(result.payload_extra_bytes, 20)

    def test_empty_embeddings_still_yield_a_receipt(self):
        # The terminal may never signal give-up, even on the engine's error path.
        result = GrpcInlineTerminal().transfer(
            MultimodalInputsPB(), MMEmbeddingRes([])
        )
        self.assertEqual(result.receipt, MultimodalOutputPB())
        self.assertEqual(result.transport, TRANSPORT_BYTES)


class _FakeBackend(MMTransportBackend):
    """Lets the degradation logic be tested without mocking the hardware boundary."""

    def __init__(self, name: str, supports: bool, result):
        self.name = name
        self._supports = supports
        self._result = result
        self.released = []
        self.closed = False
        self.transfer_calls = []

    def supports(self, request, res) -> bool:
        return self._supports

    def try_transfer(self, request, res):
        self.transfer_calls.append((request, res))
        return self._result

    def release(self, handles):
        self.released.append(list(handles))

    def close(self):
        self.closed = True


def _fake_result(transport: str) -> MMOutputResult:
    return MMOutputResult(
        receipt=MultimodalOutputPB(split_size=[1]),
        transport=transport,
        payload_embedding_bytes=1,
        payload_pos_bytes=0,
        payload_extra_bytes=0,
    )


class MMOutputTransportTest(TestCase):
    def setUp(self):
        self.terminal = GrpcInlineTerminal()

    @patch("rtp_llm.multimodal.mm_output_transport.manager.report_output_metrics")
    def test_candidate_wins_when_it_delivers(self, _metrics):
        candidate = _FakeBackend("fake", supports=True, result=_fake_result("fake"))
        transport = MMOutputTransport([candidate], terminal=self.terminal)
        request = _rdma_request()

        receipt = transport.transfer(request, MMEmbeddingRes([_rows(1)]))

        self.assertEqual(list(receipt.split_size), [1])
        self.assertEqual(len(candidate.transfer_calls), 1)
        # Passed through verbatim, so a future data plane can read any field it needs without
        # the server or the manager growing a parameter for it.
        self.assertIs(candidate.transfer_calls[0][0], request)

    @patch("rtp_llm.multimodal.mm_output_transport.manager.report_output_metrics")
    def test_failed_delivery_degrades_to_terminal(self, _metrics):
        candidate = _FakeBackend("fake", supports=True, result=None)
        transport = MMOutputTransport([candidate], terminal=self.terminal)

        receipt = transport.transfer(_rdma_request(), MMEmbeddingRes([_rows(2)]))

        self.assertEqual(len(candidate.transfer_calls), 1)
        # Fell through to inline bytes: payload in the tensor field, no descriptors.
        self.assertEqual(len(receipt.output_rdma_slots), 0)
        self.assertTrue(receipt.HasField("multimodal_embedding"))
        self.assertEqual(list(receipt.split_size), [2])

    @patch("rtp_llm.multimodal.mm_output_transport.manager.report_output_metrics")
    def test_unsupported_candidate_is_not_attempted(self, _metrics):
        candidate = _FakeBackend("fake", supports=False, result=_fake_result("fake"))
        transport = MMOutputTransport([candidate], terminal=self.terminal)

        transport.transfer(_rdma_request(), MMEmbeddingRes([_rows(1)]))

        self.assertEqual(candidate.transfer_calls, [])

    def test_release_is_broadcast_to_every_backend(self):
        candidate = _FakeBackend("fake", supports=True, result=None)
        other = _FakeBackend("other", supports=True, result=None)
        transport = MMOutputTransport([candidate, other], terminal=self.terminal)

        transport.release(ReleaseEmbeddingPB(handle=["one", "two"]))

        self.assertEqual(candidate.released, [["one", "two"]])
        self.assertEqual(other.released, [["one", "two"]])

    def test_one_backend_raising_does_not_starve_the_others(self):
        # Today only the RDMA backend can raise and it swallows its own errors, so this pins the
        # manager-level guarantee before a second pull-based backend can rely on it.
        broken = _FakeBackend("broken", supports=True, result=None)
        broken.release = MagicMock(side_effect=RuntimeError("encoder gone"))
        healthy = _FakeBackend("healthy", supports=True, result=None)
        transport = MMOutputTransport([broken, healthy], terminal=self.terminal)

        transport.release(ReleaseEmbeddingPB(handle=["one"]))
        transport.close()

        self.assertEqual(healthy.released, [["one"]])
        self.assertTrue(healthy.closed)

    def test_empty_release_is_short_circuited(self):
        candidate = _FakeBackend("fake", supports=True, result=None)
        transport = MMOutputTransport([candidate], terminal=self.terminal)

        transport.release(ReleaseEmbeddingPB())

        self.assertEqual(candidate.released, [])

    def test_close_reaches_every_backend(self):
        candidate = _FakeBackend("fake", supports=True, result=None)
        transport = MMOutputTransport([candidate], terminal=self.terminal)

        transport.close()

        self.assertTrue(candidate.closed)


class CreateMMOutputTransportTest(TestCase):
    @patch("rtp_llm.multimodal.mm_output_transport.rdma.backend.MMRdmaEncoderOp")
    def test_auto_mode_enables_available_backend(self, encoder_op):
        encoder_op.return_value.enabled.return_value = True
        config = MagicMock(mode=MM_TRANSPORT_MODE_AUTO)

        transport = create_mm_output_transport(config)

        self.assertEqual(
            [b.name for b in transport._backends], [TRANSPORT_RDMA]
        )
        encoder_op.assert_called_once_with(config.rdma)

    @patch("rtp_llm.multimodal.mm_output_transport.rdma.backend.MMRdmaEncoderOp")
    def test_unavailable_backend_leaves_only_the_terminal(self, encoder_op):
        # In a build without the implementation linked the op constructs fine and only
        # answers enabled() == False, so this is the one signal that catches it.
        encoder_op.return_value.enabled.return_value = False

        transport = create_mm_output_transport(
            MagicMock(mode=MM_TRANSPORT_MODE_AUTO)
        )

        self.assertEqual(transport._backends, [])

    @patch(
        "rtp_llm.multimodal.mm_output_transport.rdma.backend.MMRdmaEncoderOp",
        side_effect=RuntimeError("init failed"),
    )
    def test_backend_init_exception_leaves_only_the_terminal(self, _encoder_op):
        transport = create_mm_output_transport(
            MagicMock(mode=MM_TRANSPORT_MODE_AUTO)
        )

        self.assertEqual(transport._backends, [])

    @patch("rtp_llm.multimodal.mm_output_transport.rdma.backend.MMRdmaEncoderOp")
    def test_grpc_mode_and_missing_config_do_not_construct_a_backend(self, encoder_op):
        for config in (None, MagicMock(mode=MM_TRANSPORT_MODE_GRPC)):
            transport = create_mm_output_transport(config)
            self.assertEqual(transport._backends, [])
        encoder_op.assert_not_called()

    @patch("rtp_llm.multimodal.mm_output_transport.rdma.backend.MMRdmaEncoderOp")
    def test_unknown_transport_mode_is_rejected(self, encoder_op):
        # Fail fast rather than silently reading an unknown mode as grpc.
        with self.assertRaises(ValueError):
            create_mm_output_transport(MagicMock(mode="rdma"))
        encoder_op.assert_not_called()


if __name__ == "__main__":
    main()
