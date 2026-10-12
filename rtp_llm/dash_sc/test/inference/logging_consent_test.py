"""Exercise frontend privacy through a real gRPC stream and log files.

The backend emits deterministic tokens, so this test needs no model checkpoint.
Both response builders and the frontend enqueue/logging path run with DEBUG on.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
import unittest
from pathlib import Path

import grpc
import torch

from rtp_llm.access_logger.log_utils import get_process_log_filename
from rtp_llm.dash_sc.access_log import (
    DASH_SC_GRPC_ACCESS_LOG_FILENAME,
    DASH_SC_GRPC_ACCESS_LOGGER_NAME,
    DASH_SC_GRPC_QUERY_LOG_FILENAME,
    DASH_SC_GRPC_QUERY_LOGGER_NAME,
    init_dash_sc_grpc_access_logger,
    init_dash_sc_grpc_query_logger,
)
from rtp_llm.dash_sc.client import build_model_infer_request
from rtp_llm.dash_sc.codec import (
    SamplingParams,
    build_stream_response_from_generate_outputs,
)
from rtp_llm.dash_sc.inference.servicer import DashScInferenceServicer
from rtp_llm.dash_sc.proto import predict_v2_pb2_grpc
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs


class _Backend:
    def __init__(self) -> None:
        self.requests = []

    async def enqueue(self, request):
        self.requests.append(request)

        async def chunks():
            for token_id, finished in ((83741, False), (83742, True)):
                yield GenerateOutputs(
                    generate_outputs=[
                        GenerateOutput(
                            output_ids=torch.tensor([token_id], dtype=torch.int32),
                            finished=finished,
                            aux_info=AuxInfo(input_len=3),
                        )
                    ]
                )

        return chunks()


class DashScLoggingConsentTest(unittest.IsolatedAsyncioTestCase):
    async def test_missing_consent_headers_allow_token_logging(self) -> None:
        await self._check_token_logging(allowed=True)

    async def test_all_consent_headers_allow_token_logging(self) -> None:
        await self._check_token_logging(
            allowed=True,
            attributes={
                "X-DashScope-LoggingConsent": "ALL",
                "x-dashscope-inner-loggingconsent": "All",
            },
            metadata=(("x-dashscope-inner-loggingconsent", "ALL"),),
        )

    async def test_inner_consent_refuses_token_logging(self) -> None:
        await self._check_token_logging(
            allowed=False,
            attributes={
                "X-DashScope-LoggingConsent": "all",
                "X-DashScope-Inner-LoggingConsent": "none",
            },
        )

    async def test_external_consent_refuses_token_logging(self) -> None:
        await self._check_token_logging(
            allowed=False,
            attributes={
                "X-DashScope-LoggingConsent": "none",
            },
        )

    async def test_metadata_consent_refuses_token_logging(self) -> None:
        await self._check_token_logging(
            allowed=False,
            attributes={"X-DashScope-LoggingConsent": "all"},
            metadata=(("x-dashscope-inner-loggingconsent", "none"),),
        )

    async def _check_token_logging(
        self, *, allowed: bool, attributes=None, metadata=()
    ) -> None:
        backend = _Backend()
        servicer = DashScInferenceServicer(backend, rank_id=0, server_id="1")
        server = grpc.aio.server()
        predict_v2_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(servicer, server)
        port = server.add_insecure_port("127.0.0.1:0")
        self.assertGreater(port, 0)
        root = logging.getLogger()
        old_level = root.level
        with tempfile.TemporaryDirectory(prefix="dash-sc-logging-consent-") as log_dir:
            main_log = Path(log_dir) / "main.log"
            handler = logging.FileHandler(main_log)
            root.addHandler(handler)
            root.setLevel(logging.DEBUG)
            try:
                init_dash_sc_grpc_access_logger(log_dir, 1, 0, 1)
                init_dash_sc_grpc_query_logger(log_dir, 1, 0, 1)
                await server.start()
                request = build_model_infer_request(
                    request_id="logging-consent",
                    model_name="default",
                    input_ids=[93817, 93818, 93819],
                    sampling=SamplingParams(max_new_tokens=2),
                )
                if attributes is not None:
                    request.parameters["ds_header_attributes"].string_param = (
                        json.dumps(attributes)
                    )

                async def requests():
                    yield request

                async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                    stub = predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)
                    responses = [
                        response
                        async for response in stub.ModelStreamInfer(
                            requests(),
                            metadata=metadata,
                            timeout=10,
                        )
                    ]
                self.assertEqual(len(backend.requests), 1)
                self.assertEqual(
                    backend.requests[0].token_ids.tolist(), [93817, 93818, 93819]
                )
                self.assertTrue(responses)
                self.assertTrue(
                    all(not response.error_message for response in responses)
                )
                self.assertEqual(
                    b"".join(
                        response.infer_response.raw_output_contents[index]
                        for response in responses
                        for index, output in enumerate(response.infer_response.outputs)
                        if output.name == "generated_ids"
                    ),
                    torch.tensor([83741, 83742], dtype=torch.int32).numpy().tobytes(),
                )
                # Cover the standalone (non-cached) response builder as well.
                build_stream_response_from_generate_outputs(
                    dash_sc_request_id="logging-consent-builder",
                    model_name="default",
                    request_log_tag="logging-consent-builder",
                    go=GenerateOutputs(
                        generate_outputs=[
                            GenerateOutput(
                                output_ids=torch.tensor([83743], dtype=torch.int32),
                                finished=True,
                            )
                        ]
                    ),
                    request_input_ids=[93817, 93818, 93819],
                    log_input_output=allowed,
                )
                access_log = Path(log_dir) / get_process_log_filename(
                    DASH_SC_GRPC_ACCESS_LOG_FILENAME, 0, 1
                )
                query_log = Path(log_dir) / get_process_log_filename(
                    DASH_SC_GRPC_QUERY_LOG_FILENAME, 0, 1
                )
                for name in (
                    DASH_SC_GRPC_ACCESS_LOGGER_NAME,
                    DASH_SC_GRPC_QUERY_LOGGER_NAME,
                ):
                    for log_handler in logging.getLogger(name).handlers:
                        log_handler.flush()
                records = [
                    json.loads(line) for line in access_log.read_text().splitlines()
                ]
                self.assertEqual(len(records), 1)
                record = records[0]
                self.assertEqual(record["status"], "OK")
                self.assertEqual(record["input_token_len"], 3)
                self.assertEqual(record["output_token_len"], 2)
                self.assertEqual(record["request_id"], "logging-consent")
                self.assertEqual(
                    record["input_ids"], [93817, 93818, 93819] if allowed else None
                )
                self.assertEqual(
                    record["generated_ids"], [83741, 83742] if allowed else None
                )
                self.assertEqual(len(query_log.read_text().splitlines()), 1)
                main_log_text = main_log.read_text()
                self.assertIn("generate_input: ", main_log_text)
                if not allowed:
                    self.assertIn("generate_input: None", main_log_text)
                    self.assertEqual(main_log_text.count("generated_ids: None"), 3)
                logs = main_log_text + access_log.read_text() + query_log.read_text()
                for token_id in (93817, 93818, 93819, 83741, 83742, 83743):
                    match = re.search(rf"(?<!\d){token_id}(?!\d)", logs)
                    if allowed:
                        self.assertIsNotNone(match)
                    else:
                        self.assertIsNone(match)
            finally:
                await server.stop(0)
                await servicer.close()
                root.removeHandler(handler)
                handler.close()
                root.setLevel(old_level)
                for name in (
                    DASH_SC_GRPC_ACCESS_LOGGER_NAME,
                    DASH_SC_GRPC_QUERY_LOGGER_NAME,
                ):
                    logger = logging.getLogger(name)
                    for log_handler in logger.handlers[:]:
                        logger.removeHandler(log_handler)
                        log_handler.close()


if __name__ == "__main__":
    unittest.main()
