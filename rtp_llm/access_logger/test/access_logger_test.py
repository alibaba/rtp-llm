import asyncio
import json
import logging
import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from rtp_llm.access_logger import access_logger
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.server.backend_manager import BackendManager
from rtp_llm.server.server_args.misc_group_args import init_misc_group_args
from rtp_llm.server.server_args.server_args import EnvArgumentParser
from rtp_llm.structure.request_constants import request_id_field_name


class AccessLoggerTest(unittest.TestCase):
    def test_disabled_skips_handlers_objects_and_serialization(self):
        with ExitStack() as stack:
            for name in [
                "init_logger",
                "RequestLog",
                "ResponseLog",
                "PyAccessLog",
                "dump_json",
            ]:
                stack.enter_context(
                    patch.object(access_logger, name, side_effect=AssertionError(name))
                )
            logger = access_logger.AccessLogger("unused", 1, disable_access_log=True)
            # No request access is needed, even for malformed/error input.
            request = MagicMock()
            request.get.side_effect = AssertionError("request read")
            logger.log_query_access(request)
            logger.log_success_access(request, object())
            logger.log_exception_access(request, RuntimeError("failure"), object())
            logger.log_access(request, object())
            self.assertIsNone(logger.logger)
            self.assertIsNone(logger.query_logger)

    def make_enabled_logger(self):
        with patch.object(access_logger, "init_logger"):
            logger = access_logger.AccessLogger("unused", 1)
        logger.logger = MagicMock()
        logger.query_logger = MagicMock()
        return logger

    def test_enabled_keeps_query_success_and_error_records(self):
        logger = self.make_enabled_logger()
        request = {request_id_field_name: 42, "prompt": "test"}
        response = {"response": ["one", "two"], "finished": True}
        logger.log_query_access(request)
        logger.log_success_access(request, response)
        logger.log_exception_access(request, RuntimeError("failure"))
        query = json.loads(logger.query_logger.info.call_args.args[0])
        records = [json.loads(c.args[0]) for c in logger.logger.info.call_args_list]
        self.assertEqual(query["request"]["request_json"], request)
        self.assertEqual(records[0]["response"]["responses"], [response])
        self.assertEqual(records[1]["response"]["exception"], "failure")

    def test_enabled_preserves_private_request_semantics(self):
        logger = self.make_enabled_logger()
        request = {request_id_field_name: 42, "private_request": True, "prompt": "test"}
        logger.log_query_access(request)
        logger.log_success_access(request, {"response": "test"})
        logger.query_logger.info.assert_not_called()
        logger.logger.info.assert_not_called()
        logger.log_exception_access(request, RuntimeError("failure"))
        record = json.loads(logger.logger.info.call_args.args[0])
        self.assertEqual(record["request"]["request_json"], {request_id_field_name: 42})

    def test_env_and_cli_reach_frontend_and_backend_loggers(self):
        cases = [
            (None, [], False),
            ("0", [], False),
            ("1", [], True),
            ("true", [], True),
            ("false", [], False),
            ("1", ["--disable_access_log", "false"], False),
            ("0", ["--disable_access_log", "true"], True),
        ]
        for value, args, disabled in cases:
            with self.subTest(value=value, args=args), ExitStack() as stack:
                stack.enter_context(
                    patch.dict(
                        os.environ,
                        {} if value is None else {"DISABLE_ACCESS_LOG": value},
                        clear=True,
                    )
                )
                config = PyEnvConfigs()
                parser = EnvArgumentParser()
                parser.set_root_config(config)
                init_misc_group_args(parser, config.misc_config)
                parser.parse_args(args)
                self.assertEqual(config.misc_config.disable_access_log, disabled)
                self.assertEqual(config.vit_config.disable_access_log, disabled)
                handler = stack.enter_context(
                    patch.object(
                        access_logger,
                        "get_handler",
                        side_effect=lambda *a: logging.NullHandler(),
                    )
                )
                stack.enter_context(patch("rtp_llm.frontend.frontend_server.kmonitor"))
                stack.enter_context(patch("rtp_llm.server.backend_manager.kmonitor"))
                stack.enter_context(
                    patch(
                        "rtp_llm.frontend.frontend_server.get_log_path",
                        return_value="unused",
                    )
                )
                stack.enter_context(
                    patch(
                        "rtp_llm.server.backend_manager.get_log_path",
                        return_value="unused",
                    )
                )
                stack.enter_context(
                    patch("rtp_llm.server.backend_manager.DistributedServer")
                )
                for server in [FrontendServer(0, 0, config), BackendManager(config)]:
                    self.assertEqual(server._access_logger.logger is None, disabled)
                    self.assertEqual(
                        server._access_logger.query_logger is None, disabled
                    )
                self.assertEqual(handler.call_count, 0 if disabled else 4)

    def test_disabled_logging_preserves_completed_response(self):
        server = FrontendServer.__new__(FrontendServer)
        server._access_logger = access_logger.AccessLogger(
            "unused", 1, disable_access_log=True
        )
        response = {"response": ["one", "two"], "finished": True}
        result = SimpleNamespace(
            gen_complete_response_once=AsyncMock(return_value=response)
        )
        actual = asyncio.run(
            server._collect_complete_response_and_record_access_log(
                {request_id_field_name: 42}, result
            )
        )
        self.assertEqual(actual, response)

    def test_disabled_logging_preserves_error_and_cancellation_reporting(self):
        server = FrontendServer.__new__(FrontendServer)
        server.rank_id = server.server_id = "0"
        server._access_logger = access_logger.AccessLogger(
            "unused", 1, disable_access_log=True
        )
        for error in [RuntimeError("failure"), asyncio.CancelledError("cancelled")]:
            with self.subTest(error=type(error).__name__), patch(
                "rtp_llm.frontend.frontend_server.kmonitor"
            ) as monitor:
                response = server._handle_exception({request_id_field_name: 42}, error)
                self.assertEqual(response.status_code, 500)
                self.assertIn(str(error), json.loads(response.body)["message"])
                monitor.report.assert_called_once()


if __name__ == "__main__":
    unittest.main()
