import json
import unittest
from unittest.mock import Mock

from rtp_llm.access_logger.access_logger import AccessLogger
from rtp_llm.access_logger.json_util import dump_json
from rtp_llm.access_logger.py_access_log import PyAccessLog, RequestLog, ResponseLog
from rtp_llm.structure.request_constants import request_id_field_name


class PyAccessLogTest(unittest.TestCase):
    def test_http_path_is_emitted_at_top_level(self):
        log = PyAccessLog(
            request=RequestLog.from_request({"query": "q"}),
            response=ResponseLog(),
            id=7,
            path="/v1/reranker",
            log_time="2026-08-21 00:00:00.000",
        )

        payload = json.loads(dump_json(log))

        self.assertEqual("/v1/reranker", payload["path"])
        self.assertEqual({"query": "q"}, payload["request"]["request_json"])

    def test_absent_http_path_preserves_existing_schema(self):
        log = PyAccessLog(
            request=RequestLog.from_request("raw"),
            response=ResponseLog(),
            id=8,
            log_time="2026-08-21 00:00:00.000",
        )

        payload = json.loads(dump_json(log))

        self.assertNotIn("path", payload)

    def test_old_fourth_positional_argument_remains_log_time(self):
        log = PyAccessLog(
            RequestLog.from_request("raw"),
            ResponseLog(),
            9,
            "2026-08-21 00:00:00.000",
        )

        payload = json.loads(dump_json(log))

        self.assertEqual("2026-08-21 00:00:00.000", payload["log_time"])
        self.assertNotIn("path", payload)


class AccessLoggerPublicApiTest(unittest.TestCase):
    def setUp(self):
        self.access_logger = AccessLogger.__new__(AccessLogger)
        self.access_logger.logger = Mock()
        self.access_logger.query_logger = Mock()

    @staticmethod
    def request(**extra):
        return {request_id_field_name: 7, "query": "hello", **extra}

    @staticmethod
    def emitted_payload(logger):
        return json.loads(logger.info.call_args.args[0])

    def test_log_query_access_propagates_path_and_skips_private_requests(self):
        self.access_logger.log_query_access(self.request(), path="/v1/chat/completions")

        payload = self.emitted_payload(self.access_logger.query_logger)
        self.assertEqual("/v1/chat/completions", payload["path"])
        self.assertEqual("hello", payload["request"]["request_json"]["query"])

        self.access_logger.query_logger.reset_mock()
        self.access_logger.log_query_access(self.request(private_request=True))
        self.access_logger.query_logger.info.assert_not_called()

    def test_log_success_access_propagates_path_and_omits_absent_path(self):
        self.access_logger.log_success_access(
            self.request(), {"answer": "ok"}, path="/v1/completions"
        )

        payload = self.emitted_payload(self.access_logger.logger)
        self.assertEqual("/v1/completions", payload["path"])
        self.assertEqual([{"answer": "ok"}], payload["response"]["responses"])

        self.access_logger.logger.reset_mock()
        self.access_logger.log_success_access(self.request(), "plain response")
        payload = self.emitted_payload(self.access_logger.logger)
        self.assertNotIn("path", payload)

    def test_log_exception_access_propagates_path_and_redacts_private_request(self):
        self.access_logger.log_exception_access(
            self.request(private_request=True, secret="do-not-log"),
            ValueError("bad request"),
            path="/private",
        )

        payload = self.emitted_payload(self.access_logger.logger)
        self.assertEqual("/private", payload["path"])
        self.assertEqual({request_id_field_name: 7}, payload["request"]["request_json"])
        self.assertNotIn("secret", json.dumps(payload))
        self.assertEqual("bad request", payload["response"]["exception"])


if __name__ == "__main__":
    unittest.main()
