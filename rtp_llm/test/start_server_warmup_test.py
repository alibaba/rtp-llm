import multiprocessing
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

from rtp_llm import start_backend_server, start_server
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.ops import RoleType, SpeculativeType
from rtp_llm.utils.backend_restart import RestartOutcome
from rtp_llm.utils.drain_flag import DrainFlag


class ServiceDrainingPropagationTest(unittest.TestCase):
    def test_each_launch_passes_a_fresh_event_to_all_process_groups(self):
        configs = PyEnvConfigs()
        configs.parallelism_config.dp_size = 1
        # Without in-place restart every process group shares the instance's event.
        configs.server_config.engine_restart_limit = 0
        with ExitStack() as stack:
            for name in (
                "init_controller",
                "_sync_server_shutdown_timeout",
                "_setup_startup_warmup_health_gate",
                "_maybe_run_startup_real_warmup",
                "_mark_startup_warmup_health_gate_ready",
            ):
                stack.enter_context(patch.object(start_server, name))
            manager_class = stack.enter_context(
                patch.object(start_server, "ProcessManager")
            )
            starters = [
                stack.enter_context(patch.object(start_server, name))
                for name in (
                    "start_backend_server_impl",
                    "start_frontend_server_impl",
                    "start_dash_sc_server_impl",
                )
            ]
            vit = stack.enter_context(
                patch.object(start_server, "start_vit_server_impl")
            )
            previous = None
            for role in (RoleType.PDFUSION, RoleType.PDFUSION, RoleType.VIT):
                configs.role_config.role_type = role
                start_server.start_server(configs)
                manager_class.return_value.request_failure_shutdown.assert_not_called()
                event = manager_class.call_args.kwargs["service_draining"]
                # multiprocessing.Event.set() hangs once a waiter was killed.
                self.assertIsInstance(event, DrainFlag)
                self.assertIsNot(event, previous)
                self.assertFalse(event.is_set())
                if role == RoleType.VIT:
                    self.assertIs(vit.call_args.args[2], event)
                else:
                    for starter in starters:
                        self.assertIs(starter.call_args.args[3], event)
                event.set()
                previous = event


class StartupRealWarmupTest(unittest.TestCase):
    def test_restart_warmup_uses_each_generations_resolved_length(self):
        config = SimpleNamespace(
            parallelism_config=SimpleNamespace(
                world_size=1, local_world_size=1, world_rank=0
            ),
            server_config=SimpleNamespace(
                engine_restart_limit=2,
                engine_restart_heartbeat_timeout_s=10,
                startup_health_check_timeout=30,
            ),
        )
        lengths = iter((4096, 8192))

        def spawn_backend(*args):
            status = {}

            def check_ready():
                status["max_seq_len"] = next(lengths)
                return True

            return Mock(), check_ready, status

        with patch.object(
            start_server, "_spawn_backend_process", side_effect=spawn_backend
        ), patch.object(start_server, "_maybe_run_startup_real_warmup") as warmup:
            restarter = start_server._create_backend_restarter(
                None, config, Mock(shutdown_requested=False), None, DrainFlag()
            )
            self.addCleanup(restarter.coordinator.close)
            for _ in range(2):
                outcome, _ = restarter._start_generation()
                self.assertEqual(outcome, RestartOutcome.RESTARTED)

        self.assertEqual(
            warmup.call_args_list, [call(config, 4096), call(config, 8192)]
        )

    def test_backend_ranks_report_resolved_model_length(self):
        from unittest.mock import Mock

        readers = []
        for _ in range(2):
            reader, writer = multiprocessing.Pipe(duplex=False)
            self.addCleanup(reader.close)
            readers.append(reader)
            start_backend_server._send_pipe_status(
                writer, "success", "ready", max_seq_len=4096
            )
        process = Mock(exitcode=None)
        process.is_alive.return_value = True
        self.assertEqual(
            start_backend_server._wait_for_ranks_startup(
                [process, process], readers, 2
            ),
            4096,
        )

    def test_warmup_uses_resolved_model_length(self):
        config = SimpleNamespace(
            model_args=SimpleNamespace(max_seq_len=None),
            sp_config=SimpleNamespace(type=SpeculativeType.MTP, gen_num_per_cycle=3),
            kv_cache_config=SimpleNamespace(
                seq_size_per_block=128, reserve_block_ratio=5
            ),
            grpc_config=None,
            server_config=SimpleNamespace(rpc_server_port=12345),
        )
        requests = []

        async def enqueue(request):
            requests.append(request)
            yield SimpleNamespace(generate_outputs=[])

        client = SimpleNamespace(enqueue=enqueue, close=AsyncMock())
        cache_client = SimpleNamespace(
            get_cache_status=AsyncMock(
                return_value={
                    "results": [
                        {
                            "address": "localhost:12345",
                            "total_kv_cache": "8192",
                            "block_size": "128",
                        }
                    ]
                }
            ),
            close=AsyncMock(),
        )
        with patch.dict("os.environ", {"RTP_LLM_STREAM_ASYNC": "0"}), patch(
            "rtp_llm.model_factory.ModelFactory.create_model_config",
            side_effect=AssertionError("warmup must reuse the backend model length"),
        ), patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.ModelRpcClient", return_value=client
        ), patch(
            "rtp_llm.utils.grpc_client_wrapper.GrpcClientWrapper",
            return_value=cache_client,
        ), patch.object(
            start_server,
            "_get_startup_real_warmup_grpc_addresses",
            return_value=["localhost:12345"],
        ):
            # Leave four tokens for MTP and one for the generated token.
            power_of_two_lengths = [2**power for power in range(1, 12)]
            for cli_length, resolved_length, expected in (
                (None, 2053, power_of_two_lengths),
                (8192, 2053, power_of_two_lengths),
                (None, 6, [1]),
            ):
                with self.subTest(
                    cli_length=cli_length, resolved_length=resolved_length
                ):
                    config.model_args.max_seq_len = cli_length
                    requests.clear()
                    start_server._run_startup_real_warmup_async(
                        start_server._run_startup_real_warmup_grpc(
                            config, resolved_length
                        )
                    )
                    self.assertEqual(
                        [request.token_ids.numel() for request in requests],
                        expected,
                    )
        self.assertEqual(client.close.await_count, 3)
        self.assertEqual(cache_client.close.await_count, 3)

    def test_warmup_fits_smallest_backend_with_speculative_margin(self):
        config = SimpleNamespace(
            sp_config=SimpleNamespace(type=SpeculativeType.MTP, gen_num_per_cycle=3),
            kv_cache_config=SimpleNamespace(
                seq_size_per_block=128, reserve_block_ratio=5
            ),
            grpc_config=None,
            server_config=SimpleNamespace(rpc_server_port=12345),
        )
        addresses = ["localhost:12345", "localhost:12346"]
        requests = []

        async def enqueue(request):
            # The smaller backend has 20 blocks, one reserved for admission.
            self.assertLessEqual(request.token_ids.numel() + 1 + 7, 19 * 128)
            requests.append(request)
            yield SimpleNamespace(generate_outputs=[])

        client = SimpleNamespace(enqueue=enqueue, close=AsyncMock())
        cache_client = SimpleNamespace(
            get_cache_status=AsyncMock(
                return_value={
                    "results": [
                        {
                            "address": addresses[0],
                            "total_kv_cache": "8192",
                            "block_size": "128",
                        },
                        {
                            "address": addresses[1],
                            "total_kv_cache": "2560",
                            "block_size": "128",
                        },
                    ]
                }
            ),
            close=AsyncMock(),
        )
        with patch.dict("os.environ", {"RTP_LLM_STREAM_ASYNC": "1"}), patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.ModelRpcClient", return_value=client
        ), patch(
            "rtp_llm.utils.grpc_client_wrapper.GrpcClientWrapper",
            return_value=cache_client,
        ), patch.object(
            start_server,
            "_get_startup_real_warmup_grpc_addresses",
            return_value=addresses,
        ):
            start_server._run_startup_real_warmup_async(
                start_server._run_startup_real_warmup_grpc(config, 4096)
            )
        expected = [2**power for power in range(1, 12)] + [2424]
        self.assertEqual(
            [request.token_ids.numel() for request in requests],
            expected * len(addresses),
        )
        cache_client.get_cache_status.assert_awaited_once_with(
            {"need_cache_keys": False}
        )
        cache_client.close.assert_awaited_once()
        self.assertEqual(client.close.await_count, len(addresses))

    def test_cp_warmup_leaves_a_local_block_for_mtp_with_no_pool_reserve(self):
        prompt = start_server._get_startup_real_warmup_kv_prompt_limit(
            total_tokens=20 * 8192,
            block_size=8192,
            local_block_size=4096,
            reserve_block_ratio=0,
            reserve_step=4,
        )
        self.assertEqual(prompt, 19 * 8192 - 1)

    def test_warmup_rejects_missing_or_unusable_backend_capacity(self):
        config = SimpleNamespace(
            sp_config=SimpleNamespace(type=SpeculativeType.NONE, gen_num_per_cycle=0),
            kv_cache_config=SimpleNamespace(
                seq_size_per_block=128, reserve_block_ratio=5
            ),
            grpc_config=None,
            server_config=SimpleNamespace(rpc_server_port=12345),
        )
        for results, error in (
            ([], RuntimeError),
            (
                [
                    {
                        "address": "localhost:12345",
                        "total_kv_cache": "0",
                        "block_size": "128",
                    }
                ],
                ValueError,
            ),
        ):
            with self.subTest(results=results):
                cache_client = SimpleNamespace(
                    get_cache_status=AsyncMock(return_value={"results": results}),
                    close=AsyncMock(),
                )
                with patch(
                    "rtp_llm.utils.grpc_client_wrapper.GrpcClientWrapper",
                    return_value=cache_client,
                ), patch(
                    "rtp_llm.cpp.model_rpc.model_rpc_client.ModelRpcClient"
                ) as client, patch.object(
                    start_server,
                    "_get_startup_real_warmup_grpc_addresses",
                    return_value=["localhost:12345"],
                ):
                    with self.assertRaises(error):
                        start_server._run_startup_real_warmup_async(
                            start_server._run_startup_real_warmup_grpc(config, 4096)
                        )
                cache_client.close.assert_awaited_once()
                client.assert_not_called()

    def test_warmup_rejects_invalid_resolved_length(self):
        for length in (1, 0, -1):
            with self.subTest(length=length):
                with self.assertRaisesRegex(ValueError, "should be at least"):
                    start_server._run_startup_real_warmup_async(
                        start_server._run_startup_real_warmup_grpc(None, length)
                    )

    def test_warmup_rejects_model_length_with_no_room_for_prompt(self):
        config = SimpleNamespace(
            sp_config=SimpleNamespace(type=SpeculativeType.MTP, gen_num_per_cycle=3),
        )
        with patch.dict("os.environ", {"RTP_LLM_STREAM_ASYNC": "0"}), patch.object(
            start_server, "_get_startup_real_warmup_grpc_addresses"
        ) as resolve_addresses:
            with self.assertRaisesRegex(
                ValueError, r"max_seq_len=5, reserve_step=4, max_new_tokens=1"
            ):
                start_server._run_startup_real_warmup_async(
                    start_server._run_startup_real_warmup_grpc(config, 5)
                )
        resolve_addresses.assert_not_called()

    def test_single_stage_dp_ranks_pass_startup_warmup_gate(self):
        for world_size in (2, 3, 4):
            for world_rank in range(world_size):
                with self.subTest(world_size=world_size, world_rank=world_rank):
                    config = SimpleNamespace(
                        parallelism_config=SimpleNamespace(
                            pp_size=1,
                            dp_size=world_size,
                            tp_size=1,
                            world_size=world_size,
                            world_rank=world_rank,
                        ),
                        runtime_config=SimpleNamespace(
                            warm_up=True, model_warm_up=True
                        ),
                        role_config=SimpleNamespace(role_type=RoleType.PREFILL),
                        model_args=SimpleNamespace(model_type="qwen_3"),
                    )
                    self.assertTrue(
                        start_server._is_startup_real_warmup_entry_rank(config)
                    )
                    self.assertFalse(
                        start_server._should_run_startup_real_warmup(config)
                    )

    def test_warmup_entry_rank_preserves_pp_and_tp_boundaries(self):
        for pp_size, world_rank, expected in (
            (1, 0, True),
            (1, 1, False),
            (1, 2, True),
            (2, 0, True),
            (2, 2, True),
            (2, 4, False),
            (2, 5, False),
            (2, 6, False),
        ):
            with self.subTest(pp_size=pp_size, world_rank=world_rank):
                config = SimpleNamespace(
                    parallelism_config=SimpleNamespace(
                        pp_size=pp_size,
                        dp_size=2,
                        tp_size=2,
                        world_size=pp_size * 4,
                        world_rank=world_rank,
                    )
                )
                self.assertEqual(
                    start_server._is_startup_real_warmup_entry_rank(config),
                    expected,
                )

    @staticmethod
    def _configs(sp_type, gamma):
        return SimpleNamespace(
            sp_config=SimpleNamespace(
                type=sp_type,
                gen_num_per_cycle=gamma,
            )
        )

    def test_speculative_reserve_matches_engine(self):
        mtp = self._configs(SpeculativeType.MTP, 3)
        dspark = self._configs(SpeculativeType.DSPARK, 3)
        dflash = self._configs(SpeculativeType.DFLASH, 3)

        with patch.dict("os.environ", {"RTP_LLM_STREAM_ASYNC": "0"}):
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(mtp),
                4,
            )
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(dspark),
                9,
            )
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(dflash),
                9,
            )
        with patch.dict("os.environ", {"RTP_LLM_STREAM_ASYNC": "1"}):
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(mtp),
                7,
            )
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(dspark),
                9,
            )
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(dflash),
                9,
            )


if __name__ == "__main__":
    unittest.main()
