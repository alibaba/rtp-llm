import asyncio
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from rtp_llm import start_server
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.ops import CPRotateMethod, RoleType, SpeculativeType


def service_configs(model="deepseek_v41", *, batch_size=2, token_budget=0):
    config = PyEnvConfigs()
    config.model_args.model_type = model
    config.model_args.max_seq_len = 4096
    config.role_config.role_type = RoleType.PREFILL
    config.runtime_config.warm_up = True
    config.runtime_config.model_warm_up = True
    config.runtime_config.fifo_scheduler_config.max_context_batch_size = batch_size
    config.runtime_config.fifo_scheduler_config.max_batch_tokens_size = token_budget
    config.concurrency_config.concurrency_limit = 4
    config.parallelism_config.world_size = 4
    config.parallelism_config.world_rank = 0
    config.parallelism_config.tp_size = 4
    config.kv_cache_config.seq_size_per_block = 256
    config.kv_cache_config.reuse_cache = True
    config.kv_cache_config.enable_device_cache = True
    config.server_config.start_port = 31980
    return config


def completed_output(token_len=1, reuse_len=0, finished=True):
    return SimpleNamespace(
        generate_outputs=[
            SimpleNamespace(
                finished=finished,
                aux_info=SimpleNamespace(input_len=token_len, reuse_len=reuse_len),
            )
        ]
    )


class StartupRealWarmupTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"DSV4_STARTUP_REAL_WARMUP": "1"})
        env.start()
        self.addCleanup(env.stop)

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

    def test_maximum_request_leaves_room_for_output_after_speculative_reserve(self):
        # Match GenerateStream::maxTokenNum(): the reserve is subtracted from
        # the total sequence cap, not from the input cap alone.
        for max_len, reserve in ((524288, 15), (4096, 0), (17, 15), (33, 8)):
            for target in (2, max_len // 2, max_len):
                with self.subTest(max_len=max_len, reserve=reserve, target=target):
                    length = start_server._get_startup_real_warmup_request_token_len(
                        target, max_len, reserve
                    )
                    output_tokens = start_server.STARTUP_REAL_WARMUP_MAX_NEW_TOKENS
                    engine_token_cap = min(max_len - reserve, length + output_tokens)
                    self.assertEqual(engine_token_cap - length, output_tokens)
                    self.assertEqual(
                        length, min(target, max_len - reserve - output_tokens)
                    )
        self.assertEqual(
            start_server._get_startup_real_warmup_request_token_len(524288, 524288, 15),
            524272,
        )

    def test_request_rejects_capacity_without_room_for_input_and_output(self):
        for max_len, reserve in ((1, 0), (15, 15), (16, 15)):
            with self.subTest(max_len=max_len, reserve=reserve):
                with self.assertRaisesRegex(
                    ValueError, "request token len should be positive"
                ):
                    start_server._get_startup_real_warmup_request_token_len(
                        max_len, max_len, reserve
                    )

    def test_prefill_v4_and_v41_default_on(self):
        with patch.dict(os.environ):
            os.environ.pop("DSV4_STARTUP_REAL_WARMUP", None)
            for model in ("deepseek_v4", "deepseek_v41"):
                with self.subTest(model=model):
                    self.assertTrue(
                        start_server._should_run_startup_real_warmup(
                            service_configs(model)
                        )
                    )

    def test_optout_and_existing_warmup_switches(self):
        config = service_configs()
        for value in ("0", "false", "no"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"DSV4_STARTUP_REAL_WARMUP": value}
            ):
                self.assertFalse(start_server._should_run_startup_real_warmup(config))
        for field in ("warm_up", "model_warm_up"):
            with self.subTest(field=field):
                config = service_configs()
                setattr(config.runtime_config, field, False)
                self.assertFalse(start_server._should_run_startup_real_warmup(config))

    def test_other_roles_models_and_non_entry_rank_are_unchanged(self):
        for role in (
            RoleType.DECODE,
            RoleType.PDFUSION,
            RoleType.FRONTEND,
            RoleType.VIT,
        ):
            with self.subTest(role=role):
                config = service_configs()
                config.role_config.role_type = role
                self.assertFalse(start_server._should_run_startup_real_warmup(config))
        self.assertFalse(
            start_server._should_run_startup_real_warmup(service_configs("qwen_2"))
        )
        config = service_configs()
        config.parallelism_config.world_rank = 1
        self.assertFalse(start_server._should_run_startup_real_warmup(config))

    def test_invalid_override_fails_instead_of_silently_disabling(self):
        with patch.dict(os.environ, {"DSV4_STARTUP_REAL_WARMUP": "invalid"}):
            with self.assertRaises(ValueError):
                start_server._should_run_startup_real_warmup(service_configs())

    def test_optout_removes_inherited_health_gate(self):
        with patch.dict(
            os.environ,
            {
                "DSV4_STARTUP_REAL_WARMUP": "0",
                start_server.STARTUP_WARMUP_HEALTH_GATE_FILE_ENV: "/tmp/stale-gate",
            },
        ), patch.object(start_server, "_run_startup_real_warmup_grpc") as rpc:
            self.assertIsNone(
                start_server._setup_startup_warmup_health_gate(service_configs())
            )
            self.assertNotIn(
                start_server.STARTUP_WARMUP_HEALTH_GATE_FILE_ENV, os.environ
            )
            self.assertFalse(
                start_server._maybe_run_startup_real_warmup(service_configs())
            )
            rpc.assert_not_called()

    def test_ready_marker_is_published_atomically_and_write_failure_keeps_gate_closed(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            gate = os.path.join(directory, "ready")
            with patch.object(
                start_server.os, "replace", side_effect=OSError("write failed")
            ):
                with self.assertRaises(OSError):
                    start_server._mark_startup_warmup_health_gate_ready(gate)
            self.assertFalse(os.path.exists(gate))
            self.assertFalse(os.path.exists(gate + ".tmp"))
            start_server._mark_startup_warmup_health_gate_ready(gate)
            with open(gate) as marker:
                self.assertEqual(marker.read(), "ready\n")

    def test_v4_has_no_extra_path_probes(self):
        self.assertEqual(
            start_server._get_v41_startup_real_warmup_cases(
                service_configs("deepseek_v4")
            ),
            [],
        )

    def test_page_derived_ragged_reuse_plan(self):
        cases = start_server._get_v41_startup_real_warmup_cases(service_configs())
        self.assertEqual(cases[0], ("reuse_prime", [(512, 100, True)]))
        self.assertEqual(
            cases[1], ("ragged_probe", [(769, 100, True), (259, 101, False)])
        )

    def test_cp4_probe_uses_virtual_cache_block_without_changing_physical_page(self):
        config = service_configs()
        config.kv_cache_config.seq_size_per_block = 128
        config.prefill_cp_config.kv_cache_sharded = True
        config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        config.prefill_cp_config.prefill_cp_size = 32
        cases = start_server._get_v41_startup_real_warmup_cases(config)
        self.assertEqual(cases[0], ("reuse_prime", [(1024, 100, True)]))
        self.assertEqual(cases[1][1], [(1537, 100, True), (515, 101, False)])
        self.assertEqual(config.kv_cache_config.seq_size_per_block, 128)

    def test_cp4_budget_smaller_than_virtual_block_disables_reuse_probe(self):
        config = service_configs(token_budget=1024)
        config.kv_cache_config.seq_size_per_block = 128
        config.prefill_cp_config.kv_cache_sharded = True
        config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        cases = start_server._get_v41_startup_real_warmup_cases(config)
        self.assertEqual(len(cases), 1)
        self.assertTrue(all(not spec[2] for _, specs in cases for spec in specs))
        self.assertLessEqual(sum(spec[0] for spec in cases[0][1]), 1024)

    def test_inactive_cp_or_unsharded_cache_keeps_physical_cache_block(self):
        for sharded, method in (
            (True, CPRotateMethod.DISABLED),
            (False, CPRotateMethod.ALL_GATHER),
        ):
            config = service_configs()
            config.kv_cache_config.seq_size_per_block = 128
            config.prefill_cp_config.kv_cache_sharded = sharded
            config.prefill_cp_config.method = method
            self.assertEqual(
                start_server._get_v41_startup_real_warmup_cases(config)[0][1][0][0],
                256,
            )

    def test_page_and_capacity_boundaries(self):
        for page in (64, 128, 256, 512):
            for max_len in (17, 257, 8192):
                for token_budget in (0, 1, 7, 512):
                    with self.subTest(page=page, max_len=max_len, budget=token_budget):
                        config = service_configs(token_budget=token_budget)
                        config.kv_cache_config.seq_size_per_block = page
                        config.model_args.max_seq_len = max_len
                        cases = start_server._get_v41_startup_real_warmup_cases(config)
                        for phase, specs in cases:
                            self.assertLessEqual(len(specs), 2)
                            for token_len, _, _ in specs:
                                self.assertGreater(token_len, 0)
                                self.assertLess(token_len, max_len)
                            if token_budget:
                                self.assertLessEqual(
                                    sum(s[0] for s in specs), token_budget
                                )
                            if phase == "reuse_prime":
                                self.assertEqual(specs[0][0] % page, 0)

    def test_serial_service_never_forces_a_batch(self):
        for limited_field in ("scheduler", "concurrency"):
            config = service_configs()
            if limited_field == "scheduler":
                config.runtime_config.fifo_scheduler_config.max_context_batch_size = 1
            else:
                config.concurrency_config.concurrency_limit = 1
            cases = start_server._get_v41_startup_real_warmup_cases(config)
            self.assertEqual([len(specs) for _, specs in cases], [1, 1, 1])

    def test_disabled_device_reuse_does_not_prime_or_require_cache(self):
        for field in ("reuse_cache", "enable_device_cache"):
            config = service_configs()
            setattr(config.kv_cache_config, field, False)
            cases = start_server._get_v41_startup_real_warmup_cases(config)
            self.assertEqual(len(cases), 1)
            self.assertTrue(all(not s[2] for _, specs in cases for s in specs))

    def test_request_budget_reserves_dspark_tokens(self):
        config = service_configs()
        config.model_args.max_seq_len = 32
        config.sp_config.type = SpeculativeType.DSPARK
        config.sp_config.gen_num_per_cycle = 5
        cases = start_server._get_v41_startup_real_warmup_cases(config)
        self.assertTrue(all(s[0] <= 16 for _, specs in cases for s in specs))

    def test_incomplete_or_empty_response_fails_closed(self):
        for outputs in (
            None,
            SimpleNamespace(generate_outputs=[]),
            completed_output(finished=False),
        ):
            with self.subTest(outputs=outputs), self.assertRaises(RuntimeError):
                start_server._validate_startup_real_warmup_output(outputs)
        with self.assertRaisesRegex(RuntimeError, "did not reuse"):
            start_server._validate_startup_real_warmup_output(completed_output(), True)

    def test_request_flags_keep_probe_local_and_cache_tiers_explicit(self):
        for reuse in (False, True):
            inp = start_server._new_startup_real_warmup_input(
                123, 33, 1000, reuse=reuse
            )
            self.assertEqual(inp.token_ids.numel(), 33)
            self.assertEqual(inp.token_ids.device.type, "cpu")
            config = inp.generate_config
            self.assertEqual(config.max_new_tokens, 1)
            self.assertFalse(config.can_use_pd_separation)
            self.assertEqual(config.reuse_cache, reuse)
            self.assertEqual(config.enable_device_cache, reuse)
            self.assertFalse(config.enable_memory_cache)
            self.assertFalse(config.enable_disk_cache)
            self.assertFalse(config.enable_remote_cache)

    def test_health_gate_waits_for_all_backend_checks_and_real_requests(self):
        self._check_health_gate_failure(None)

    def test_request_failure_keeps_health_gate_closed(self):
        self._check_health_gate_failure("warmup")

    def test_backend_readiness_failure_never_runs_real_requests(self):
        self._check_health_gate_failure("backend")

    def _check_health_gate_failure(self, fail_stage):
        config = service_configs()
        manager = MagicMock(shutdown_requested=False)
        phases = []
        gate_paths = []

        def spawn_frontend(*_):
            gate = os.environ[start_server.STARTUP_WARMUP_HEALTH_GATE_FILE_ENV]
            gate_paths.append(gate)
            self.assertFalse(os.path.exists(gate))
            phases.append("frontend_inherited_gate")
            return []

        def ready():
            phases.append("all_backend_checks")
            return fail_stage != "backend"

        async def warmup(_):
            self.assertEqual(phases[-1], "all_backend_checks")
            self.assertFalse(os.path.exists(gate_paths[0]))
            await asyncio.sleep(0)
            if fail_stage == "warmup":
                raise RuntimeError("mock warmup failure")
            phases.append("real_requests_complete")

        manager.run_health_checks.side_effect = ready
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ))
            for name in (
                "init_controller",
                "_sync_server_shutdown_timeout",
                "start_backend_server_impl",
            ):
                stack.enter_context(patch.object(start_server, name))
            stack.enter_context(
                patch.object(start_server, "ProcessManager", return_value=manager)
            )
            stack.enter_context(
                patch.object(
                    start_server,
                    "start_frontend_server_impl",
                    side_effect=spawn_frontend,
                )
            )
            stack.enter_context(
                patch.object(start_server, "start_dash_sc_server_impl", return_value=[])
            )
            call = stack.enter_context(
                patch.object(
                    start_server, "_run_startup_real_warmup_grpc", side_effect=warmup
                )
            )
            try:
                start_server.start_server(config)
                self.assertEqual(os.path.exists(gate_paths[0]), fail_stage is None)
                if fail_stage:
                    manager.request_failure_shutdown.assert_called_once()
                else:
                    manager.request_failure_shutdown.assert_not_called()
                    self.assertEqual(phases[-1], "real_requests_complete")
                self.assertEqual(call.call_count, int(fail_stage != "backend"))
            finally:
                for gate in gate_paths:
                    if os.path.exists(gate):
                        os.remove(gate)


class StartupRealWarmupRpcTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Load lazy request dependencies before patch.dict restores sys.modules;
        # native Python extensions cannot be unloaded and initialized again.
        start_server._new_startup_real_warmup_input(0, 1, 1000)

    async def test_prefix_prime_finishes_before_ragged_probe(self):
        cases = start_server._get_v41_startup_real_warmup_cases(service_configs())
        primed = False
        calls = []

        async def batch(inputs):
            nonlocal primed
            calls.append(inputs)
            if len(calls) == 1:
                await asyncio.sleep(0)
                primed = True
                return [completed_output(512)]
            self.assertTrue(primed)
            self.assertEqual([inp.token_ids.numel() for inp in inputs], [769, 259])
            return [completed_output(769, 512), completed_output(259)]

        client = SimpleNamespace(batch_generate=AsyncMock(side_effect=batch))
        count = await start_server._run_v41_startup_real_warmup_probes(
            client, cases, 1000, "local"
        )
        self.assertEqual(count, 3)
        self.assertEqual(len(calls), 2)

    async def test_failed_prime_does_not_issue_dependent_probe(self):
        cases = start_server._get_v41_startup_real_warmup_cases(service_configs())
        client = SimpleNamespace(
            batch_generate=AsyncMock(side_effect=RuntimeError("prime failed"))
        )
        with self.assertRaisesRegex(RuntimeError, "prime failed"):
            await start_server._run_v41_startup_real_warmup_probes(
                client, cases, 1000, "local"
            )
        self.assertEqual(client.batch_generate.call_count, 1)

    async def test_missing_probe_result_fails(self):
        cases = [("ragged_probe", [(9, 100, False), (3, 101, False)])]
        client = SimpleNamespace(
            batch_generate=AsyncMock(return_value=[completed_output(9)])
        )
        with self.assertRaisesRegex(RuntimeError, "count mismatch"):
            await start_server._run_v41_startup_real_warmup_probes(
                client, cases, 1000, "local"
            )

    async def test_every_address_finishes_sweep_and_probes_before_client_close(self):
        calls = []

        class Client:
            def __init__(self, addresses, **_):
                self.addr = addresses[0]

            async def enqueue(self, inp):
                calls.append((self.addr, "sweep", inp.token_ids.numel()))
                self_config = inp.generate_config
                assert (
                    not self_config.reuse_cache and not self_config.enable_device_cache
                )
                yield completed_output(inp.token_ids.numel())

            async def batch_generate(self, inputs):
                calls.append((self.addr, "batch", len(inputs)))
                return [completed_output(inp.token_ids.numel(), 512) for inp in inputs]

            async def close(self):
                calls.append((self.addr, "close"))

        with (
            patch.object(
                start_server,
                "_get_startup_real_warmup_token_lens",
                return_value=[2, 17],
            ),
            patch.object(
                start_server,
                "_get_startup_real_warmup_grpc_addresses",
                return_value=["rank0", "rank4"],
            ),
            patch.dict(
                sys.modules,
                {
                    "rtp_llm.cpp.model_rpc.model_rpc_client": SimpleNamespace(
                        ModelRpcClient=Client
                    )
                },
            ),
        ):
            await start_server._run_startup_real_warmup_grpc(service_configs())
        for addr in ("rank0", "rank4"):
            self.assertEqual(
                [call[1:] for call in calls if call[0] == addr],
                [("sweep", 2), ("sweep", 17), ("batch", 1), ("batch", 2), ("close",)],
            )

    async def test_incomplete_sweep_response_closes_client_and_stops(self):
        client = SimpleNamespace(close=AsyncMock(), batch_generate=AsyncMock())

        async def enqueue(_):
            yield completed_output(finished=False)

        client.enqueue = enqueue
        with (
            patch.object(
                start_server, "_get_startup_real_warmup_token_lens", return_value=[2]
            ),
            patch.object(
                start_server,
                "_get_startup_real_warmup_grpc_addresses",
                return_value=["local"],
            ),
            patch.dict(
                sys.modules,
                {
                    "rtp_llm.cpp.model_rpc.model_rpc_client": SimpleNamespace(
                        ModelRpcClient=lambda **_: client
                    )
                },
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "completed response"):
                await start_server._run_startup_real_warmup_grpc(service_configs())
        client.close.assert_awaited_once()
        client.batch_generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
