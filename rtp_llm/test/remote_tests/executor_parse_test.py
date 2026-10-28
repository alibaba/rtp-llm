from types import SimpleNamespace

from rtp_llm.test.remote_tests import remote_execution_pb2
from rtp_llm.test.remote_tests.executor import RemoteExecutor


class _FakeCAS:
    grpc_uri = "grpc://cas.service:50051"

    def download_blob(self, digest):
        return b""


def _executor():
    executor = RemoteExecutor.__new__(RemoteExecutor)
    executor.grpc_uri = "grpc://scheduler.example.test:50052"
    executor.reapi_targets_combined = (
        "cas=cas.service:50051 | executor=scheduler.example.test:50052"
    )
    executor.cas = _FakeCAS()
    return executor


def _packed_response(exit_code=0):
    op = remote_execution_pb2.Operation(name="operations/done", done=True)
    response = remote_execution_pb2.ExecuteResponse(
        result=remote_execution_pb2.ActionResult(exit_code=exit_code)
    )
    op.response.Pack(response)
    return op.response


def test_parse_accepts_legacy_operation_without_error_field():
    legacy_op = SimpleNamespace(response=_packed_response())

    result = _executor()._parse(legacy_op)

    assert result.exit_code == 0


def test_parse_reads_lro_error_from_operation_result():
    op = remote_execution_pb2.Operation(
        name="operations/failed",
        done=True,
        error=remote_execution_pb2.Status(code=3, message="invalid action"),
    )

    result = _executor()._parse(op)

    assert result.exit_code == 1
    assert result.stderr_raw == (
        b"LRO operation failed: code=3, message='invalid action'"
    )


def test_parse_classifies_completed_operation_without_result_as_infra():
    result = _executor()._parse(SimpleNamespace())

    assert result.exit_code == -1
    assert result.infra_category == "executor_response"
    assert b"neither an error nor a response" in result.stderr_raw
