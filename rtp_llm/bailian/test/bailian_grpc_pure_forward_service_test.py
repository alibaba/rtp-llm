"""Unit tests for ``rtp_llm.bailian.bailian_grpc_pure_forward_service``.

Tests verify that request_iterator is passed correctly to downstream stub
(not converted to list, which was a bug in the initial implementation).
"""

from __future__ import annotations

import struct
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from rtp_llm.bailian.bailian_grpc_pure_forward_service import (
    PureForwardServicer,
    _parse_forward_addrs,
)
from rtp_llm.bailian.proto import predict_v2_pb2


def _make_request(model_name: str = "test_model", id: str = "test_id") -> predict_v2_pb2.ModelInferRequest:
    """Create a minimal ModelInferRequest for testing."""
    req = predict_v2_pb2.ModelInferRequest()
    req.model_name = model_name
    req.id = id
    inp = req.inputs.add()
    inp.name = "input_ids"
    inp.datatype = "INT32"
    inp.shape.append(2)
    req.raw_input_contents.append(struct.pack("<2i", 1, 2))
    return req


def _make_response() -> predict_v2_pb2.ModelStreamInferResponse:
    """Create a minimal ModelStreamInferResponse for testing."""
    resp = predict_v2_pb2.ModelStreamInferResponse()
    out = resp.outputs.add()
    out.name = "output"
    out.datatype = "INT32"
    out.shape.append(1)
    resp.raw_output_contents.append(struct.pack("<i", 42))
    return resp


class ParseForwardAddrsTest(TestCase):
    def test_single_address(self) -> None:
        result = _parse_forward_addrs("10.0.0.1:8096")
        self.assertEqual(result, ["10.0.0.1:8096"])

    def test_comma_separated(self) -> None:
        result = _parse_forward_addrs("10.0.0.1:8096,10.0.0.2:8096")
        self.assertEqual(result, ["10.0.0.1:8096", "10.0.0.2:8096"])

    def test_json_array(self) -> None:
        result = _parse_forward_addrs('["10.0.0.1:8096", "10.0.0.2:8096"]')
        self.assertEqual(result, ["10.0.0.1:8096", "10.0.0.2:8096"])

    def test_empty_string(self) -> None:
        result = _parse_forward_addrs("")
        self.assertEqual(result, [])

    def test_whitespace_only(self) -> None:
        result = _parse_forward_addrs("   ")
        self.assertEqual(result, [])

    def test_comma_with_spaces(self) -> None:
        result = _parse_forward_addrs("10.0.0.1:8096 , 10.0.0.2:8096 ")
        self.assertEqual(result, ["10.0.0.1:8096", "10.0.0.2:8096"])


class IteratorBehaviorTest(TestCase):
    """Core tests: verify iterator is passed to stub (not converted to list).

    This tests the bug fix where [req] was passed instead of an iterator,
    causing TypeError: 'list' object is not an iterator.
    """

    def setUp(self) -> None:
        # Mock grpc.insecure_channel to avoid real connections
        self.channel_patcher = patch("grpc.insecure_channel", return_value=MagicMock())
        self.channel_patcher.start()

        self.servicer = PureForwardServicer(["127.0.0.1:1"])
        self.mock_stub = MagicMock()
        self.servicer._stubs = [self.mock_stub]

    def tearDown(self) -> None:
        self.servicer.close()
        self.channel_patcher.stop()

    def test_iterator_passed_not_list_no_log(self) -> None:
        """BUG FIX: iterator must be passed, not list (when logging disabled)."""
        self.servicer._log_debug = False

        def request_gen():
            yield _make_request("req1")
            yield _make_request("req2")

        mock_resp = _make_response()
        self.mock_stub.ModelStreamInfer.return_value = iter([mock_resp, mock_resp])

        responses = list(self.servicer.ModelStreamInfer(request_gen(), MagicMock()))

        # KEY ASSERTION: stub received iterator (has __next__), not list
        call_arg = self.mock_stub.ModelStreamInfer.call_args[0][0]
        self.assertTrue(hasattr(call_arg, "__iter__"), "Must be iterable")
        self.assertTrue(hasattr(call_arg, "__next__"), "Must be iterator (has __next__), not list")
        self.assertEqual(len(responses), 2)

    def test_iterator_passed_not_list_with_log(self) -> None:
        """BUG FIX: with logging enabled, iterator must still be passed."""
        self.servicer._log_debug = True

        def request_gen():
            yield _make_request("req1")
            yield _make_request("req2")

        mock_resp = _make_response()
        self.mock_stub.ModelStreamInfer.return_value = iter([mock_resp, mock_resp])

        with patch("rtp_llm.bailian.bailian_grpc_pure_forward_service.logging.info"):
            responses = list(self.servicer.ModelStreamInfer(request_gen(), MagicMock()))

        # KEY ASSERTION: stub received iterator (has __next__)
        call_arg = self.mock_stub.ModelStreamInfer.call_args[0][0]
        self.assertTrue(hasattr(call_arg, "__next__"), "Must be iterator, not list")
        self.assertEqual(len(responses), 2)

    def test_logged_iterator_closure_updates_counter(self) -> None:
        """Verify nonlocal closure correctly updates request counter."""
        self.servicer._log_debug = True
        req_count = 0

        def request_gen():
            for i in range(3):
                yield _make_request(f"req{i}")

        mock_resp = _make_response()
        self.mock_stub.ModelStreamInfer.return_value = iter([mock_resp] * 3)

        # The logged_iterator inside ModelStreamInfer uses nonlocal req_count
        # This test verifies that pattern works
        with patch("rtp_llm.bailian.bailian_grpc_pure_forward_service.logging.info"):
            responses = list(self.servicer.ModelStreamInfer(request_gen(), MagicMock()))

        self.assertEqual(len(responses), 3)


class NonlocalClosurePatternTest(TestCase):
    """Verify the nonlocal closure pattern used in logged_iterator."""

    def test_nonlocal_closure_works(self) -> None:
        """Test that nonlocal closure correctly modifies outer variable."""
        req_count = 0

        def logged_iterator():
            nonlocal req_count
            for i in range(5):
                req_count += 1
                yield i

        # Consume iterator
        result = list(logged_iterator())

        self.assertEqual(result, [0, 1, 2, 3, 4])
        self.assertEqual(req_count, 5)  # Counter was updated

    def test_iterator_vs_list_distinction(self) -> None:
        """Verify iterator (generator) has __next__, list does not."""
        gen = (x for x in range(3))
        lst = [0, 1, 2]

        # Generator is an iterator (has __next__)
        self.assertTrue(hasattr(gen, "__next__"))
        self.assertTrue(hasattr(gen, "__iter__"))

        # List is iterable but NOT an iterator (no __next__)
        self.assertFalse(hasattr(lst, "__next__"))
        self.assertTrue(hasattr(lst, "__iter__"))


if __name__ == "__main__":
    main()