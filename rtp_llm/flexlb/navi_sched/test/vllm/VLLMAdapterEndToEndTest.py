import os
import socket
import tempfile
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest import mock

from navi_sched import NaviSchedMaster
from navi_sched.adapters.vllm import calculate_vllm_block_hashes
from navi_sched.adapters.vllm.VLLMEngineObserver import VLLMEngineObserver


class FullAttentionSpec:
    def __init__(self, block_size):
        self.block_size = block_size


class VLLMAdapterEndToEndTestBase(unittest.TestCase):
    MASTER_BIZ = "vllm_adapter_master"
    AGENT_BIZ = "vllm_adapter_prefill"
    BLOCK_SIZE = 16

    def setUp(self):
        self._original_dir = os.getcwd()
        self._runtime_dir = tempfile.TemporaryDirectory(
            prefix="navi-sched-vllm-adapter-"
        )
        os.chdir(self._runtime_dir.name)
        self._observers = []
        self._observers_by_model_port = {}
        self._master = None
        self._router = None
        self._ports = self._reserve_ports(4)
        self._environment = mock.patch.dict(
            os.environ,
            {"PYTHONHASHSEED": "0"},
            clear=False,
        )
        self._environment.start()

    def tearDown(self):
        try:
            if self._router is not None:
                self._router.close()
            if self._master is not None:
                self._master.stop()
            for observer in reversed(self._observers):
                observer.shutdown()
            self._environment.stop()
        finally:
            os.chdir(self._original_dir)
            self._runtime_dir.cleanup()

    def _make_observer(self, model_port, agent_port):
        os.environ.update(
            {
                "VLLM_NAVI_SCHED_BIZ_NAME": self.AGENT_BIZ,
                "VLLM_NAVI_SCHED_SERVER_IP": "127.0.0.1",
                "VLLM_NAVI_SCHED_SERVER_PORT": str(model_port),
                "VLLM_NAVI_SCHED_AGENT_GIG_PORT": str(agent_port),
            }
        )
        observer = VLLMEngineObserver(
            self._make_vllm_config(),
            SimpleNamespace(
                num_blocks=1024,
                kv_cache_groups=[
                    SimpleNamespace(kv_cache_spec=FullAttentionSpec(self.BLOCK_SIZE))
                ],
            ),
        )
        self._observers.append(observer)
        self._observers_by_model_port[model_port] = observer
        return observer

    def _start_master(self):
        local_nodes = [
            {
                "biz_name": self.AGENT_BIZ,
                "grpc_port": observer._agent.grpc_port,
                "support_heartbeat": True,
            }
            for observer in self._observers
        ]
        self._master = NaviSchedMaster(
            self.MASTER_BIZ,
            runtime_config={
                "thread_num": 16,
                "local_nodes": local_nodes,
            },
            max_wait_schedule_count=1,
            scheduler_name="max_cache_hit",
            name="vllm_adapter_master",
            enable_kmon=False,
            enable_snapshot=False,
            wait_no_query=False,
        ).init()

    def _restart_master_and_router(self):
        self._router.close()
        self._router = None
        self._master.stop()
        self._master = None
        self._start_master()
        self._start_router()

    @classmethod
    def _full_hashes(cls, token_ids):
        return [
            bytes(24) + value.to_bytes(8, "big")
            for value in calculate_vllm_block_hashes(token_ids, cls.BLOCK_SIZE)
        ]

    @staticmethod
    def _report_cache(observer, hashes, inserted):
        observer.on_block_lifecycle_event(
            "stored" if inserted else "removed",
            0,
            tuple(hashes),
            None,
            "test_fixture",
        )

    @staticmethod
    def _run_batch(observer, external_request_id, block_hashes):
        request = SimpleNamespace(
            request_id="internal-request",
            external_req_id=external_request_id,
            num_prompt_tokens=32,
            num_computed_tokens=32,
            num_cached_tokens=16,
            block_hashes=block_hashes,
        )
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 16})
        future = Future()
        wrapped = observer.wrap_execute_model(
            lambda unused_output, non_block: future,
            SimpleNamespace(requests={request.request_id: request}),
        )
        wrapped(output, non_block=True)
        future.set_result("model-output")

    @staticmethod
    def _make_vllm_config():
        return SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=16,
                enable_prefix_caching=True,
                prefix_caching_hash_algo="sha256_cbor",
            ),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            scheduler_config=SimpleNamespace(
                enable_chunked_prefill=False,
                max_num_partial_prefills=1,
                max_long_partial_prefills=1,
                long_prefill_token_threshold=0,
                max_num_scheduled_tokens=1024,
            ),
        )

    @staticmethod
    def _reserve_ports(count):
        sockets = []
        try:
            for _ in range(count):
                sock = socket.socket()
                sock.bind(("127.0.0.1", 0))
                sockets.append(sock)
            return [sock.getsockname()[1] for sock in sockets]
        finally:
            for sock in sockets:
                sock.close()
