import importlib
import json
import multiprocessing
import sys
import tempfile
import traceback
import unittest
from unittest import mock


class _FakeTokenizer:
    stop_words_id_list = []
    stop_words_str_list = []
    eos_token_id = 2
    chat_template = None

    def __len__(self):
        return 100


class _FakeFrontendWorker:
    def __init__(self, py_env_configs, model_config, special_tokens):
        self.tokenizer = _FakeTokenizer()
        self.backend_rpc_server_visitor = object()

    def stop(self):
        pass


class _FakeOpenaiEndpoint:
    def __init__(self, *args, **kwargs):
        pass


def _run_standalone_frontend_start(result_queue, ckpt_path):
    try:
        sys.modules["rtp_llm.models"] = None

        from rtp_llm.config.py_config_modules import PyEnvConfigs
        from rtp_llm.frontend import frontend_app, frontend_server
        from rtp_llm.openai import openai_endpoint
        from rtp_llm.ops import RoleType, TaskType
        from rtp_llm.start_frontend_server import start_frontend_server
        from rtp_llm.utils.concurrency_controller import ConcurrencyController

        py_env_configs = PyEnvConfigs()
        py_env_configs.role_config.role_type = RoleType.FRONTEND
        py_env_configs.model_args.ckpt_path = ckpt_path
        py_env_configs.model_args.tokenizer_path = ckpt_path
        py_env_configs.model_args.model_type = "standalone_frontend_test"
        py_env_configs.model_args.task_type = TaskType.LANGUAGE_MODEL
        py_env_configs.model_args.max_seq_len = 128
        py_env_configs.model_args.act_type = "fp16"
        py_env_configs.kv_cache_config.seq_size_per_block = 64
        py_env_configs.server_config.start_port = 0
        py_env_configs.server_config.frontend_server_count = 1
        py_env_configs.parallelism_config.world_size = 1
        py_env_configs.parallelism_config.local_world_size = 1
        py_env_configs.parallelism_config.world_rank = 0
        py_env_configs.parallelism_config.tp_size = 1
        py_env_configs.parallelism_config.dp_size = 1

        with (
            mock.patch.object(frontend_server, "FrontendWorker", _FakeFrontendWorker),
            mock.patch.object(openai_endpoint, "OpenaiEndpoint", _FakeOpenaiEndpoint),
            mock.patch.object(
                frontend_app.GracefulShutdownServer, "run", lambda self: None
            ),
        ):
            app = start_frontend_server(0, 0, ConcurrencyController(1), py_env_configs)

        result_queue.put(
            (
                "ok",
                app.frontend_server._frontend_worker is not None,
                app.frontend_server._openai_endpoint is not None,
            )
        )
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))


class FrontendPackageImportTest(unittest.TestCase):
    def test_frontend_package_imports_without_async_model_dep(self):
        importlib.import_module("rtp_llm.frontend.frontend_app")

    def test_frontend_process_start_without_backend_models(self):
        with tempfile.TemporaryDirectory() as ckpt_path:
            with open(f"{ckpt_path}/config.json", "w", encoding="utf-8") as writer:
                json.dump({"architectures": ["StandaloneFrontendTest"]}, writer)

            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=_run_standalone_frontend_start,
                args=(result_queue, ckpt_path),
            )
            process.start()
            process.join(timeout=20)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                self.fail("standalone frontend start process timed out")

            self.assertEqual(process.exitcode, 0)
            status = result_queue.get(timeout=5)
            self.assertEqual(status[0], "ok", status[1])
            self.assertTrue(status[1], "frontend worker was not initialized")
            self.assertTrue(status[2], "OpenAI endpoint was not initialized")


if __name__ == "__main__":
    unittest.main()
