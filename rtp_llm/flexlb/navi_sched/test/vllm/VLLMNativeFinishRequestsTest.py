"""Exercise real adapter cancellation with both native vLLM return contracts."""

import ast
import copy
import inspect
import threading
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

SOURCE = Path(__file__).resolve().parents[2] / "adapters/vllm/VLLMEngineObserver.py"
tree = ast.parse(SOURCE.read_text())
wrapper = next(
    n
    for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name == "_NaviSchedScheduler"
)
method = next(
    n
    for n in wrapper.body
    if isinstance(n, ast.FunctionDef) and n.name == "finish_requests"
)
module = ast.parse("from __future__ import annotations\n")
module.body.append(copy.deepcopy(method))
namespace = {"inspect": inspect}
exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)


class PairScheduler:
    def finish_requests(self, request_ids, status) -> list[tuple[str, int]]:
        return self.result


class RequestScheduler:
    def finish_requests(self, request_ids, status) -> list[SimpleNamespace]:
        return self.result


class VLLMNativeFinishRequestsTest(unittest.TestCase):
    def check_contract(self, scheduler_type, with_native):
        native = SimpleNamespace(request_id="native", client_index=2)
        held = SimpleNamespace(request_id="held", client_index=7, status="waiting")
        scheduler = scheduler_type()
        pairs = scheduler_type is PairScheduler
        scheduler.requests = {"native": native} if with_native else {}
        scheduler.result = (
            ([("native", 2)] if pairs else [native]) if with_native else []
        )
        state = SimpleNamespace(external_request_ids={10: "held"}, requests={10: held})
        observer = SimpleNamespace(
            _matching_forward_batches=Mock(return_value=[state]),
            _abort_forward_requests=Mock(),
            _refresh_and_report_workload=Mock(),
            _workload_lock=threading.Lock(),
            _workload_contributions={},
        )
        target = SimpleNamespace(_scheduler=scheduler, _observer=observer)
        call = MethodType(namespace["finish_requests"], target)
        result = call(["native", "held"], "aborted")
        expected = list(scheduler.result) + ([("held", 7)] if pairs else [held])
        self.assertEqual(expected, result)
        if with_native:
            self.assertIs(scheduler.result[0], result[0])
        self.assertEqual("aborted", held.status)
        observer._abort_forward_requests.assert_called_once_with(state, {10})
        observer._refresh_and_report_workload.assert_called_once_with(
            scheduler, {"native", "held"}, "finish_requests"
        )

    def test_pair_results_with_native_and_held_requests(self):
        self.check_contract(PairScheduler, True)

    def test_pair_annotation_preserved_when_native_result_empty(self):
        self.check_contract(PairScheduler, False)

    def test_request_results_with_native_and_held_requests(self):
        self.check_contract(RequestScheduler, True)

    def test_request_annotation_preserved_when_native_result_empty(self):
        self.check_contract(RequestScheduler, False)


if __name__ == "__main__":
    unittest.main()
