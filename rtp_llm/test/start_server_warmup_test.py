import unittest
from contextlib import ExitStack
from multiprocessing import Pipe
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm import start_backend_server, start_server
from rtp_llm.ops import RoleType, SpeculativeType


class StartupRealWarmupTest(unittest.TestCase):
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

        with patch.dict("os.environ", {"RTP_LLM_STREAM_ASYNC": "0"}):
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(mtp),
                4,
            )
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(dspark),
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
            start_server._get_startup_real_warmup_request_token_len(
                token_len=1048576,
                max_len=1048576,
                reserve_step=8,
            ),
            1048567,
        )
        self.assertEqual(
            start_server._get_startup_real_warmup_request_token_len(
                token_len=1048576,
                max_len=1048576,
                reserve_step=4,
            ),
            1048571,
        )

    def test_k3_prefill_entry_only_and_both_switches_required(self):
        config = SimpleNamespace(
            runtime_config=SimpleNamespace(warm_up=True, model_warm_up=True),
            role_config=SimpleNamespace(role_type=RoleType.PREFILL),
            parallelism_config=SimpleNamespace(world_rank=0, world_size=8, tp_size=4),
            model_args=SimpleNamespace(model_type="kimi_k3"),
        )
        self.assertTrue(start_server._should_run_startup_real_warmup(config))
        config.parallelism_config.world_rank = 1
        self.assertFalse(start_server._should_run_startup_real_warmup(config))
        config.parallelism_config.world_rank = 0
        config.role_config.role_type = RoleType.DECODE
        self.assertFalse(start_server._should_run_startup_real_warmup(config))
        config.role_config.role_type = RoleType.PREFILL
        config.runtime_config.model_warm_up = False
        self.assertFalse(start_server._should_run_startup_real_warmup(config))

    def test_real_warmup_only_enables_dsv4_and_k3_targets(self):
        config = SimpleNamespace(
            runtime_config=SimpleNamespace(warm_up=True, model_warm_up=True),
            role_config=SimpleNamespace(role_type=RoleType.PREFILL),
            parallelism_config=SimpleNamespace(world_rank=0, world_size=1, tp_size=1),
            model_args=SimpleNamespace(model_type="deepseek_v4"),
        )
        for model_type, expected in (
            ("deepseek_v4", True),
            ("kimi_k3", True),
            ("deepseek_v4_mtp", False),
            ("kimi_k3_mtp", False),
            ("custom_model", False),
            ("", False),
        ):
            with self.subTest(model_type=model_type):
                config.model_args.model_type = model_type
                self.assertEqual(
                    start_server._should_run_startup_real_warmup(config), expected
                )


class StartupModelLengthTest(unittest.TestCase):
    def _start_single_rank(self, requested_length, model_length):
        configs = SimpleNamespace(
            model_args=SimpleNamespace(max_seq_len=requested_length),
            profiling_debug_logging_config=SimpleNamespace(debug_load_server=False),
            parallelism_config=SimpleNamespace(local_rank=0, world_size=1),
            server_config=Mock(shutdown_timeout=30, monitor_interval=1),
            distribute_config=Mock(fake_gang_env=False),
            ffn_disaggregate_config=None,
            prefill_cp_config=None,
        )
        backend = Mock(
            engine=SimpleNamespace(config=SimpleNamespace(max_seq_len=model_length))
        )
        process_manager = Mock()
        with ExitStack() as stack:
            stack.enter_context(patch.dict("os.environ", {}))
            # Keep model/GPU initialization and process creation outside this test.
            for name in (
                "_install_hot_hook_runtime",
                "copy_gemm_config",
                "set_parallelism_config",
                "configure_kv_cache_event_host_ip_port",
                "setup_cuda_device_and_accl_env",
                "set_global_controller",
                "install_oom_dump",
                "signal.signal",
            ):
                stack.enter_context(patch(f"rtp_llm.start_backend_server.{name}"))
            stack.enter_context(
                patch.dict(
                    "sys.modules",
                    {
                        "rtp_llm.server.backend_manager": SimpleNamespace(
                            BackendManager=Mock(return_value=backend)
                        )
                    },
                )
            )
            factory = stack.enter_context(
                patch("rtp_llm.start_server.torch.multiprocessing.Process")
            )

            def start_child():
                controller, child_configs, writer, _ = factory.call_args.kwargs["args"]
                start_backend_server.local_rank_start(
                    controller, child_configs, pipe_writer=writer
                )

            factory.return_value.start.side_effect = start_child
            start_server.start_backend_server_impl(None, configs, process_manager)
            ready = process_manager.register_health_check.call_args.kwargs[
                "check_ready_fn"
            ]
            self.assertTrue(ready())
        return configs

    @staticmethod
    def _rank_statuses(lengths):
        processes, readers = [], []
        for length in lengths:
            reader, writer = Pipe(duplex=False)
            writer.send(
                {
                    "status": "success",
                    "message": "ready",
                    "traceback": "",
                    "max_seq_len": length,
                }
            )
            writer.close()
            readers.append(reader)
            processes.append(Mock(exitcode=None, is_alive=Mock(return_value=True)))
        return processes, readers

    def test_omitted_length_reaches_warmup_from_loaded_model(self):
        configs = self._start_single_rank(None, 32768)
        self.assertEqual(configs.model_args.max_seq_len, 32768)
        self.assertEqual(start_server._get_startup_real_warmup_max_len(configs), 32768)

    def test_explicit_length_is_preserved(self):
        configs = self._start_single_rank(16384, 16384)
        self.assertEqual(start_server._get_startup_real_warmup_max_len(configs), 16384)

    def test_invalid_explicit_length_still_fails_warmup(self):
        for length in (0, -1):
            with self.subTest(length=length):
                configs = self._start_single_rank(length, 32768)
                with self.assertRaises(ValueError):
                    start_server._get_startup_real_warmup_max_len(configs)

    def test_multi_rank_startup_returns_resolved_length(self):
        processes, readers = self._rank_statuses([32768, 32768])
        self.assertEqual(
            start_backend_server._wait_for_ranks_startup(processes, readers, 2), 32768
        )

    def test_multi_rank_rejects_inconsistent_lengths(self):
        processes, readers = self._rank_statuses([32768, 65536])
        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            start_backend_server._wait_for_ranks_startup(processes, readers, 2)

    def test_failed_rank_status_still_aborts_startup(self):
        reader, writer = Pipe(duplex=False)
        start_backend_server._send_pipe_status(writer, "failed", "model load failed")
        process = Mock(exitcode=None, is_alive=Mock(return_value=True))
        with self.assertRaisesRegex(
            Exception, "Rank 0 startup failed: model load failed"
        ):
            start_backend_server._wait_for_ranks_startup([process], [reader], 1)

    def test_multi_rank_manager_forwards_resolved_length(self):
        processes, readers = self._rank_statuses([32768, 32768])
        reader, writer = Pipe(duplex=False)
        configs = SimpleNamespace(
            distribute_config=SimpleNamespace(fake_gang_env=False),
            server_config=SimpleNamespace(shutdown_timeout=30, monitor_interval=1),
        )

        def create_ranks(controller, configs, ctx, child_processes, child_readers):
            child_processes.extend(processes)
            child_readers.extend(readers)

        try:
            with (
                patch(
                    "rtp_llm.start_backend_server._create_rank_processes",
                    side_effect=create_ranks,
                ),
                patch("rtp_llm.start_backend_server.ProcessManager"),
            ):
                start_backend_server.multi_rank_start(None, configs, writer)
            self.assertTrue(reader.poll(0))
            status = reader.recv()
            self.assertEqual(status["status"], "success")
            self.assertEqual(status.get("max_seq_len"), 32768)
        finally:
            reader.close()
            writer.close()
            for rank_reader in readers:
                rank_reader.close()


if __name__ == "__main__":
    unittest.main()
