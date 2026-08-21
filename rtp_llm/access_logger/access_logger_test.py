import json
import unittest

from rtp_llm.access_logger.json_util import dump_json
from rtp_llm.access_logger.py_access_log import PyAccessLog, RequestLog, ResponseLog


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


if __name__ == "__main__":
    unittest.main()
