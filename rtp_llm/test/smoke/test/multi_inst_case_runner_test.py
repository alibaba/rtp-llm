"""Lifecycle contracts; server interfaces are mocked, Native imports are not."""

import os
import unittest
from unittest.mock import Mock, patch

from rtp_llm.test.smoke import multi_inst_case_runner as module
from rtp_llm.test.smoke.task_info import TaskInfo, TaskStates


class MultiInstanceLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.gpus = patch.object(module, "get_gpu_ids", return_value=list(range(8)))
        self.gpu_ids = self.gpus.start()
        self.addCleanup(self.gpus.stop)
        self.ports = patch.object(
            module.MagaServerManager, "get_free_port", side_effect=range(10000, 10100)
        )
        self.ports.start()
        self.addCleanup(self.ports.stop)
        self.rpc_ready = patch.object(module, "_wait_tcp_port_ready")
        self.wait_rpc = self.rpc_ready.start()
        self.addCleanup(self.rpc_ready.stop)

    def manager(self, role):
        manager = Mock(spec=["stop_server", "address", "copy_logs"])
        manager.stop_server.side_effect = lambda: self.events.append(role)
        manager.address.return_value = "unit-cache-address"
        manager.copy_logs.side_effect = lambda: self.events.append(role + ":logs")
        return manager

    def runner(
        self,
        cls=module.SeparatedCaseRunner,
        frontend=False,
        remote_cache=False,
        vit_envs=None,
    ):
        envs = {"prefill": ["WORLD_SIZE=1"], "decode": ["WORLD_SIZE=1"]}
        if cls is module.FrontAppSeperationCaseRunner:
            envs = {"pd_fusion": ["WORLD_SIZE=1"], "frontend": []}
        elif frontend:
            envs["frontend"] = []
        envs.update(vit_envs or {})
        smoke_args = {
            role: "--enable_remote_cache true" if remote_cache else "" for role in envs
        }
        runner = cls(
            TaskInfo(model_type="unit", model_path="/unused", query_result=[]),
            envs,
            "H20_CU13",
            smoke_args,
        )
        self.owned = {}
        self.configs = []

        def parallel(configs):
            self.configs = configs
            managers = []
            for config in configs:
                role = config["role_name"]
                self.owned[role] = self.manager(role)
                managers.append(self.owned[role])
            return managers, [TaskStates() for _ in configs]

        def start(
            env, state, task, *, role_name, server_manager_callback=None, **kwargs
        ):
            manager = self.manager(role_name)
            self.owned[role_name] = manager
            if server_manager_callback is not None:
                server_manager_callback(manager)
            return manager

        runner.start_servers_parallel = Mock(side_effect=parallel)
        runner.start_server = Mock(side_effect=start)
        runner.curl_server = Mock(return_value=TaskStates())
        runner._keep_servers_alive = Mock(return_value=TaskStates())
        runner._start_remote_kvcm_server = Mock(return_value=self.manager("remote"))
        return runner

    def test_rpc_readiness_uses_cli_world_size_and_stride(self):
        runner = self.runner()
        runner._wait_model_rpc_ports(
            [
                {
                    "role_name": "prefill",
                    "port": 12000,
                    "env_dict": {"WORLD_SIZE": "1", "WORKER_INFO_PORT_NUM": "10"},
                    "smoke_args_str": "--world_size 2 --worker_info_port_num 20",
                },
                {"role_name": "vit", "port": 15000, "env_dict": {"WORLD_SIZE": "2"}},
            ]
        )
        self.assertEqual(
            [call.args[1] for call in self.wait_rpc.call_args_list], [12001, 12021]
        )

    def test_rpc_readiness_failure_cleans_servers_before_request(self):
        for cls in (module.SeparatedCaseRunner, module.DpSeperationCaseRunner):
            with self.subTest(cls=cls.__name__):
                runner = self.runner(cls, frontend=True)
                self.wait_rpc.side_effect = TimeoutError("model rpc not ready")
                with self.assertRaisesRegex(TimeoutError, "model rpc not ready"):
                    runner._run_impl()
                runner.curl_server.assert_not_called()
                for manager in self.owned.values():
                    manager.stop_server.assert_called_once()

    def test_shared_pd_still_requires_sum_resources(self):
        for cls in (module.SeparatedCaseRunner, module.DpSeperationCaseRunner):
            for shared in ("0", "1"):
                with self.subTest(cls=cls.__name__, shared=shared):
                    runner = self.runner(cls)
                    runner.env_args["prefill"].append("SMOKE_PD_SHARE_GPU=" + shared)
                    self.gpu_ids.return_value = [0]
                    with self.assertRaises(RuntimeError):
                        runner._run_impl()
                    runner.start_servers_parallel.assert_not_called()
                    runner.start_server.assert_not_called()

    def test_shared_visibility_preserves_two_reserved_roles(self):
        for cls in (module.SeparatedCaseRunner, module.DpSeperationCaseRunner):
            with self.subTest(cls=cls.__name__):
                runner = self.runner(cls)
                runner.env_args["prefill"].append("SMOKE_PD_SHARE_GPU=1")
                self.gpu_ids.return_value = [0, 1]
                self.assertTrue(runner._run_impl().ret)
                self.assertEqual(
                    [x["env_dict"]["CUDA_VISIBLE_DEVICES"] for x in self.configs],
                    ["0", "0"],
                )
                self.assertTrue(
                    all("SMOKE_PD_SHARE_GPU" not in x["env_dict"] for x in self.configs)
                )

    def test_vit_resources_remain_in_sum_and_keep_distinct_assignment(self):
        runner = self.runner(vit_envs={"vit": ["WORLD_SIZE=2"]})
        self.gpu_ids.return_value = [0, 1, 2]
        with self.assertRaises(RuntimeError):
            runner._run_impl()
        self.gpu_ids.return_value = [0, 1, 2, 3]
        self.assertTrue(runner._run_impl().ret)
        self.assertEqual(
            [x["env_dict"]["CUDA_VISIBLE_DEVICES"] for x in self.configs],
            ["0", "1", "2,3"],
        )

    def test_normal_return_stops_owned_servers_in_reverse_order(self):
        for cls, expected in (
            (module.SeparatedCaseRunner, ["frontend", "prefill", "decode"]),
            (module.DpSeperationCaseRunner, ["prefill", "decode", "frontend"]),
            (module.FrontAppSeperationCaseRunner, ["pd_fusion", "frontend"]),
        ):
            with self.subTest(cls=cls.__name__):
                self.events.clear()
                runner = self.runner(cls, frontend=True)
                self.assertTrue(runner._run_impl().ret)
                self.assertEqual(self.events, expected)

    def test_request_exception_still_cleans_owned_servers(self):
        for cls in (
            module.SeparatedCaseRunner,
            module.DpSeperationCaseRunner,
            module.FrontAppSeperationCaseRunner,
        ):
            with self.subTest(cls=cls.__name__):
                runner = self.runner(cls, frontend=True)
                runner.curl_server.side_effect = ValueError("request failed")
                with self.assertRaisesRegex(ValueError, "request failed"):
                    runner._run_impl()
                for manager in self.owned.values():
                    manager.stop_server.assert_called_once()

    def test_parallel_failure_cleans_successful_sibling_and_remote(self):
        for cls in (module.SeparatedCaseRunner, module.DpSeperationCaseRunner):
            with self.subTest(cls=cls.__name__):
                self.events.clear()
                runner = self.runner(cls, remote_cache=True)
                sibling = self.manager("sibling")
                runner.start_servers_parallel.side_effect = None
                runner.start_servers_parallel.return_value = (
                    [None, sibling],
                    [TaskStates(ret=False, err_msg="failed"), TaskStates()],
                )
                self.assertFalse(runner._run_impl().ret)
                self.assertEqual(self.events, ["sibling", "remote", "remote:logs"])
                runner.curl_server.assert_not_called()

    def test_parallel_exception_cleans_already_owned_remote(self):
        for cls in (module.SeparatedCaseRunner, module.DpSeperationCaseRunner):
            with self.subTest(cls=cls.__name__):
                self.events.clear()
                runner = self.runner(cls, remote_cache=True)
                runner.start_servers_parallel.side_effect = ValueError(
                    "startup exception"
                )
                with self.assertRaisesRegex(ValueError, "startup exception"):
                    runner._run_impl()
                self.assertEqual(self.events, ["remote", "remote:logs"])

    def test_failed_sequential_frontend_is_owned_even_when_start_returns_none(self):
        for cls in (
            module.SeparatedCaseRunner,
            module.DpSeperationCaseRunner,
            module.FrontAppSeperationCaseRunner,
        ):
            with self.subTest(cls=cls.__name__):
                runner = self.runner(cls, frontend=True)
                failed = self.manager("failed-frontend")

                def start(env, state, task, *, server_manager_callback, **kwargs):
                    server_manager_callback(failed)
                    state.ret = False
                    return None

                runner.start_server.side_effect = start
                self.assertFalse(runner._run_impl().ret)
                failed.stop_server.assert_called_once()
                runner.curl_server.assert_not_called()

    def test_frontapp_failed_backend_cleans_frontend_and_failed_backend(self):
        runner = self.runner(module.FrontAppSeperationCaseRunner)
        original = runner.start_server.side_effect

        def start(env, state, task, *, role_name, **kwargs):
            manager = original(env, state, task, role_name=role_name, **kwargs)
            if role_name == "pd_fusion":
                state.ret = False
                return None
            return manager

        runner.start_server.side_effect = start
        self.assertFalse(runner._run_impl().ret)
        self.assertEqual(self.events, ["pd_fusion", "frontend"])

    def test_stop_exception_does_not_block_other_owned_cleanup(self):
        runner = self.runner()
        original = runner.start_servers_parallel.side_effect

        def parallel(configs):
            managers, states = original(configs)
            managers[-1].stop_server.side_effect = ValueError("stop failed")
            return managers, states

        runner.start_servers_parallel.side_effect = parallel
        self.assertTrue(runner._run_impl().ret)
        self.owned["decode"].stop_server.assert_called_once()

    def test_keepalive_before_curl_delegates_without_double_stop(self):
        for cls in (
            module.SeparatedCaseRunner,
            module.DpSeperationCaseRunner,
            module.FrontAppSeperationCaseRunner,
        ):
            with (
                self.subTest(cls=cls.__name__),
                patch.dict(os.environ, {"SMOKE_KEEP_SERVER_ALIVE": "1"}),
            ):
                runner = self.runner(cls, frontend=True)
                self.assertTrue(runner._run_impl().ret)
                runner.curl_server.assert_not_called()
                runner._keep_servers_alive.assert_called_once()
                for manager in self.owned.values():
                    manager.stop_server.assert_not_called()

    def test_separated_after_curl_keepalive_delegates_reverse_order(self):
        with patch.dict(os.environ, {"SMOKE_KEEP_SERVER_ALIVE_AFTER_CURL": "1"}):
            runner = self.runner(frontend=True)
            self.assertTrue(runner._run_impl().ret)
            runner.curl_server.assert_called_once()
            roles = list(runner._keep_servers_alive.call_args.args[0])
            self.assertEqual(roles, ["frontend", "prefill", "decode"])
            for manager in self.owned.values():
                manager.stop_server.assert_not_called()

    def test_keepalive_failure_still_cleans_owned_servers(self):
        for cls in (
            module.SeparatedCaseRunner,
            module.DpSeperationCaseRunner,
            module.FrontAppSeperationCaseRunner,
        ):
            with (
                self.subTest(cls=cls.__name__),
                patch.dict(os.environ, {"SMOKE_KEEP_SERVER_ALIVE": "1"}),
            ):
                runner = self.runner(cls, frontend=True)
                runner._keep_servers_alive.side_effect = ValueError("keepalive failed")
                with self.assertRaisesRegex(ValueError, "keepalive failed"):
                    runner._run_impl()
                for manager in self.owned.values():
                    manager.stop_server.assert_called_once()

    def test_main_flexlb_cleanup_keeps_topology_and_reverse_ownership(self):
        runner = self.runner(
            frontend=True,
            vit_envs={"vit_0": ["WORLD_SIZE=1"], "vit_1": ["WORLD_SIZE=1"]},
        )
        master = Mock(spec=["terminate", "wait", "kill"])
        master.terminate.side_effect = lambda: self.events.append("master")
        stream = Mock(spec=["close"])
        stream.close.side_effect = lambda: self.events.append("master-log")
        lock = Mock(spec=["__exit__"])
        lock.__exit__ = Mock(
            side_effect=lambda *args: self.events.append("master-lock")
        )
        runner._start_master = Mock(return_value=(master, stream))
        with (
            patch.dict(os.environ, {"SMOKE_START_FLEXLB_MASTER": "1"}),
            patch.object(module, "PortManager") as port_manager,
        ):
            port_manager.return_value.get_consecutive_ports.return_value = (
                [20000, 20001, 20002],
                [lock],
            )
            self.assertTrue(runner._run_impl().ret)
        self.assertEqual(
            [item["role_name"] for item in self.configs],
            ["decode", "prefill", "vit_0", "vit_1"],
        )
        self.assertEqual(
            self.events,
            [
                "frontend",
                "master",
                "master-log",
                "master-lock",
                "vit_1",
                "vit_0",
                "prefill",
                "decode",
            ],
        )
        runner._start_master.assert_called_once()

    def test_flexlb_start_failure_still_stops_backends_and_releases_lock(self):
        runner = self.runner(
            frontend=True,
            vit_envs={"vit_0": ["WORLD_SIZE=1"], "vit_1": ["WORLD_SIZE=1"]},
        )
        lock = Mock(spec=["__exit__"])
        lock.__exit__ = Mock()
        runner._start_master = Mock(side_effect=ValueError("master failed"))
        with (
            patch.dict(os.environ, {"SMOKE_START_FLEXLB_MASTER": "1"}),
            patch.object(module, "PortManager") as port_manager,
        ):
            port_manager.return_value.get_consecutive_ports.return_value = (
                [20000, 20001, 20002],
                [lock],
            )
            with self.assertRaisesRegex(ValueError, "master failed"):
                runner._run_impl()
        lock.__exit__.assert_called_once_with(None, None, None)
        self.assertEqual(self.events, ["vit_1", "vit_0", "prefill", "decode"])

    def test_remote_stop_error_still_copies_logs(self):
        owned = self.manager("owned")
        remote = self.manager("remote")
        remote.stop_server.side_effect = ValueError("stop failed")
        module._cleanup_servers([owned], remote)
        self.assertEqual(self.events, ["owned", "remote:logs"])

    def test_cleanup_never_stops_unregistered_manager(self):
        owned = self.manager("owned")
        unrelated = self.manager("unrelated")
        module._cleanup_servers([None, owned])
        owned.stop_server.assert_called_once()
        unrelated.stop_server.assert_not_called()

    def test_main_alias_classes_and_local_two_part_topology_remain(self):
        self.assertIs(
            module.PdSeperationCaseRunner.__bases__[0], module.SeparatedCaseRunner
        )
        self.assertIs(
            module.VitSeperationCaseRunner.__bases__[0], module.SeparatedCaseRunner
        )
        self.assertEqual(
            module._build_local_dp_addresses(10000, 4, 2, 9),
            ["127.0.0.1:10000", "127.0.0.1:10018"],
        )
        runner = object.__new__(module.LocalTwoPartPdfusionCaseRunner)
        first, second = self.manager("first"), self.manager("second")
        runner._stop_servers([first, second])
        self.assertEqual(self.events, ["second", "first"])


if __name__ == "__main__":
    unittest.main()
