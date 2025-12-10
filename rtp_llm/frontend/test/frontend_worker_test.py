import asyncio
import logging
import os
from unittest import TestCase, main

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.config.log_config import LOGGING_CONFIG
from rtp_llm.distribute.worker_info import g_worker_info, update_master_info
from rtp_llm.frontend.frontend_worker import BatchGenerationResponse, FrontendWorker
from rtp_llm.structure.request_extractor import request_id_field_name
from rtp_llm.test.model_test.test_util.fake_model_loader import FakeModelLoader
from rtp_llm.test.utils.port_util import PortManager


class FakeFrontendWorker(FrontendWorker):
    def __init__(self, engine):
        self.engine = engine
        # Call parent __init__ with the config, tokenizer, and backend_rpc_server_visitor
        # Since this is a test mock, we pass None for backend_rpc_server_visitor
        from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor

        self.backend_rpc_server_visitor = BackendRPCServerVisitor(engine.config, False)
        super().__init__(
            engine.config, engine.model.tokenizer, self.backend_rpc_server_visitor
        )


class FrontendWorkerTest(TestCase):
    def setUp(self):
        os.environ["KV_CACHE_MEM_MB"] = "100"
        os.environ["RESERVER_RUNTIME_MEM_MB"] = "1"
        os.environ["DEVICE_RESERVE_MEMORY_BYTES"] = str(64 * 1024 * 1024)
        self.tokenizer_path = os.path.join(
            os.getcwd(),
            "rtp_llm/test/model_test/fake_test/testdata/llama/fake/hf_source",
        )
        self.ckpt_path = os.path.join(
            os.getcwd(),
            "rtp_llm/test/model_test/fake_test/testdata/llama/fake/hf_source",
        )
        self.frontend_worker = self.create_frontend_worker()

    def tearDown(self):
        self.frontend_worker.engine.stop()

    def create_frontend_worker(self):
        port_list, _ = PortManager().get_consecutive_ports(1)
        os.environ["START_PORT"] = str(port_list[0])
        update_master_info("0.0.0.0", int(port_list[0]))
        g_worker_info.reload()
        self.fake_model_loader = FakeModelLoader(
            model_type="llama",
            tokenizer_path=self.tokenizer_path,
            ckpt_path=self.ckpt_path,
            max_seq_len=2048,
        )
        engine: BaseEngine = self.fake_model_loader.init_engine()
        return FakeFrontendWorker(engine)

    async def _run(self, frontend_worker, **kwargs):
        count = 0
        kwargs[request_id_field_name] = 1
        gen = frontend_worker.handle_request(**kwargs)
        result = []
        aux_info = []
        finished = []
        async for response in gen:
            if isinstance(response, BatchGenerationResponse):
                self.assertTrue("prompt_batch" in kwargs)
                logging.info(f"batch stream reponse: {response.response_batch}")
            else:
                logging.info(f"stream response:{response.response}")

        # collect log response
        log_response = await gen.gen_complete_response_once()
        if isinstance(log_response, BatchGenerationResponse):
            result = [_.response for _ in log_response.response_batch]
            aux_info = [_.aux_info for _ in log_response.response_batch]
            finished = [_.finished for _ in log_response.response_batch]
        else:
            result = log_response.response
            aux_info = log_response.aux_info
            finished = log_response.finished
        return result, aux_info, finished

    def test_simple(self):
        def func():
            return asyncio.run(
                self._run(
                    self.frontend_worker,
                    prompt="please write a story about dog",
                    generate_config={"top_k": 1, "max_new_tokens": 3, "top_p": 1},
                )
            )

        # just ensure every input has result
        result_text, aux_info, finished = func()
        logging.info(
            f"result_text: {result_text}, aux_info: {aux_info}, finished:{finished}"
        )
        self.assertTrue(len(result_text) > 0)

    def test_text_input(self):
        result_text, aux_info, finished = asyncio.run(
            self._run(
                self.frontend_worker,
                text="please write a story about dog",
                generate_config={"top_k": 1, "max_new_tokens": 3, "top_p": 1},
            )
        )
        # logging.info(f"test text input : {result_text}, aux_info: {aux_info}, finished:{finished}")
        self.assertTrue(len(result_text) > 0)

    def test_num_batch(self):
        def func():
            return asyncio.run(
                self._run(
                    self.frontend_worker,
                    prompt_batch=[
                        "please write a story about dog",
                        "please write a story about dog",
                    ],
                    yield_generator=True,
                    generate_config={
                        "top_k": 1,
                        "max_new_tokens": 3,
                        "top_p": 1,
                        "return_incremental": True,
                        "num_return_sequences": 3,
                    },
                )
            )

        result_text, aux_info, finished = func()
        logging.info(
            f"batch * num: {result_text}, aux_info: {aux_info}, finished:{finished}"
        )
        self.assertEqual(2, len(result_text))
        self.assertEqual(3, len(result_text[0]))
        self.assertEqual(3, len(result_text[1]))
        self.assertEqual(2, len(aux_info))
        self.assertEqual(3, len(aux_info[0]))
        self.assertEqual(3, len(aux_info[1]))
        self.assertEqual([True, True], finished)
        self.assertTrue(len(result_text) > 0)

    def test_num_return_sequences_1(self):
        result_text, aux_info, finished = asyncio.run(
            self._run(
                self.frontend_worker,
                prompt="please write a story about dog",
                yield_generator=True,
                generate_config={
                    "top_k": 1,
                    "max_new_tokens": 3,
                    "top_p": 1,
                    "return_incremental": True,
                    "num_return_sequences": 1,
                },
            )
        )
        # logging.info(f"no batch * num_return_sequences 1 : {result_text}, aux_info: {aux_info}, finished:{finished}")
        self.assertEqual(1, len(result_text))
        self.assertEqual(1, len(aux_info))
        self.assertEqual(True, finished)

    def test_batch_num_return_sequences_1(self):
        result_text, aux_info, finished = asyncio.run(
            self._run(
                self.frontend_worker,
                prompt_batch=[
                    "please write a story about dog",
                    "please write a story about dog",
                ],
                yield_generator=True,
                generate_config={
                    "top_k": 1,
                    "max_new_tokens": 3,
                    "top_p": 1,
                    "return_incremental": True,
                    "num_return_sequences": 1,
                },
            )
        )
        # logging.info(f"batch 2 * num_return_sequences 1: {result_text}, aux_info: {aux_info}, finished:{finished}")
        self.assertEqual(2, len(result_text))
        self.assertEqual(1, len(result_text[0]))
        self.assertEqual(1, len(result_text[1]))
        self.assertEqual(2, len(aux_info))
        self.assertEqual(1, len(aux_info[0]))
        self.assertEqual(1, len(aux_info[1]))
        self.assertEqual([True, True], finished)

    def test_incremental(self):
        def func():
            return asyncio.run(
                self._run(
                    self.frontend_worker,
                    prompt="please write a story about dog",
                    yield_generator=True,
                    generate_config={
                        "top_k": 1,
                        "max_new_tokens": 3,
                        "top_p": 1,
                        "return_incremental": True,
                    },
                )
            )

        result = []
        result_text, aux_info, finished = func()
        self.assertEqual(3, aux_info.get("iter_count"))
        self.assertEqual(True, finished)
        logging.info(
            f"incremental : {result_text}, aux_info: {aux_info}, finished:{finished}"
        )
        self.assertTrue(len(result_text) > 0)

    def test_batch_incremental(self):
        def func():
            return asyncio.run(
                self._run(
                    self.frontend_worker,
                    prompt_batch=[
                        "please write a story about dog",
                        "please write a story about dog",
                    ],
                    yield_generator=True,
                    generate_config={
                        "top_k": 1,
                        "max_new_tokens": 3,
                        "top_p": 1,
                        "return_incremental": True,
                    },
                )
            )

        result_text, aux_info, finished = func()
        self.assertEqual(2, len(aux_info))
        self.assertEqual([True, True], finished)
        logging.info(
            f"batch incremental: {result_text}, aux_info: {aux_info}, finished:{finished}"
        )
        self.assertTrue(len(result_text) > 0)

    def test_num_return_incremental(self):
        def func():
            return asyncio.run(
                self._run(
                    self.frontend_worker,
                    prompt="please write a story about dog",
                    yield_generator=True,
                    generate_config={
                        "top_k": 1,
                        "max_new_tokens": 3,
                        "top_p": 1,
                        "return_incremental": True,
                        "num_return_sequences": 3,
                    },
                )
            )

        result_text, aux_info, finished = func()
        logging.info(
            f"incremental num return: {result_text}, aux_info: {aux_info}, finished:{finished}"
        )
        self.assertEqual(3, len(result_text))
        self.assertEqual(3, len(aux_info))
        self.assertEqual(True, finished)
        self.assertTrue(len(result_text) > 0)

    def test_batch_num_return_incremental(self):
        def func():
            return asyncio.run(
                self._run(
                    self.frontend_worker,
                    prompt_batch=[
                        "please write a story about dog",
                        "please write a story about dog",
                    ],
                    yield_generator=True,
                    generate_config={
                        "top_k": 1,
                        "max_new_tokens": 3,
                        "top_p": 1,
                        "return_incremental": True,
                        "num_return_sequences": 3,
                    },
                )
            )

        result_text, aux_info, finished = func()
        logging.info(
            f"incremental batch * num return: {result_text}, aux_info: {aux_info}, finished:{finished}"
        )
        self.assertEqual(2, len(result_text))
        self.assertEqual(3, len(result_text[0]))
        self.assertEqual(3, len(result_text[1]))
        self.assertEqual(2, len(aux_info))
        self.assertEqual(3, len(aux_info[0]))
        self.assertEqual(3, len(aux_info[1]))
        self.assertEqual([True, True], finished)
        self.assertTrue(len(result_text) > 0)


if __name__ == "__main__":
    if os.environ.get("FT_SERVER_TEST", None) is None:
        logging.config.dictConfig(LOGGING_CONFIG)
    main()
