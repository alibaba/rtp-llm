import asyncio
import json
import socket
import sys
import unittest
from types import ModuleType
from types import SimpleNamespace as Obj
from unittest import mock

import navi_sched._navi_sched_core as core
from navi_sched.adapters.vllm import V6DPPCollectorRuntime as runtime
from V6DCollectorObserverTest import Peer


def environment(port="19114"):
    return {
        "NAVI_SCHED_PP_CACHE_COLLECTOR": "1",
        "NAVI_SCHED_PP_SIZE": "4",
        "NAVI_SCHED_PP_SHARD_ID": "0",
        "NAVI_SCHED_PP_CACHE_COLLECTOR_GROUP": "pp-test",
        "NAVI_SCHED_PP_CACHE_COLLECTOR_PORT": port,
    }


def spec():
    return {
        "node_count": 4,
        "groups": [
            {"group_id": 0, "spec_type": "FullAttentionSpec"},
            {"group_id": 1, "spec_type": "MambaSpec"},
        ],
    }


class V6DPPCollectorRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.config = Obj(
            parallel_config=Obj(
                pipeline_parallel_size=4, tensor_parallel_size=1, nnodes_within_dp=4
            ),
            cache_config=Obj(
                mamba_cache_mode="align", prefix_cache_retention_interval=8192
            ),
            speculative_config=Obj(use_eagle=lambda: True),
        )
        self.source = Obj(
            _pp_size=4,
            vllm_config=self.config,
            hash_block_size=4096,
            lcm_block_size=8192,
            mamba_group_ids={1},
            v6d_urls=(
                "http://127.0.0.1:9113",
                "http://10.0.0.2:9113",
                "http://10.0.0.3:9113",
                "http://10.0.0.4:9113",
            ),
        )

        class Connector:
            pass

        self.connector_type = Connector
        connector = Connector()
        connector.scheduler = self.source
        self.scheduler = Obj(connector=Obj(_connectors=[connector]))
        self.patch = mock.patch.dict(
            sys.modules,
            {"kvs_connector.v6d_object_connector": Obj(V6dObjectConnector=Connector)},
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def make_config(self, env=None):
        return runtime.make_pp_cache_agent_config(
            self.scheduler,
            self.config,
            core,
            environ=environment() if env is None else env,
        )

    def test_disabled_does_not_read_native_configuration(self):
        self.assertIsNone(
            runtime.make_pp_cache_agent_config(None, None, None, environ={})
        )

    def test_non_pp_does_not_parse_or_enable_pp_settings(self):
        self.config.parallel_config.pipeline_parallel_size = 1
        self.assertIsNone(
            self.make_config({"NAVI_SCHED_PP_CACHE_COLLECTOR": "invalid"})
        )

    def test_uses_all_native_nodes_and_independent_v6d_policy(self):
        config = self.make_config()
        self.assertTrue(config.valid())
        self.assertEqual(4, config.pipeline_parallel_size)
        self.assertEqual([0, 1, 2, 3], [node.shard_id for node in config.shards])
        self.assertEqual(
            ["127.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
            [node.host for node in config.shards],
        )
        self.assertEqual([19114] * 4, [node.grpc_port for node in config.shards])
        self.assertEqual(4096, config.policy.block_size)
        self.assertEqual(2, config.policy.linear_step)
        self.assertTrue(config.policy.hybrid_tail_required)
        self.assertEqual(1, config.policy.full_attention_query_drop_tokens)
        self.assertTrue(config.policy.mamba_replay_stop_shift)
        self.assertEqual(8192, config.policy.mamba_retention_interval)
        self.assertTrue(config.policy.mamba_align)

    def test_incomplete_handshake_is_rejected(self):
        self.source.v6d_urls = self.source.v6d_urls[:3]
        with self.assertRaisesRegex(ValueError, "handshake"):
            self.make_config()

    def test_explicit_rollout_hosts_are_verified_after_handshake(self):
        hosts = ["127.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]
        env = environment()
        env["NAVI_SCHED_PP_CACHE_COLLECTOR_HOSTS"] = json.dumps(hosts)
        self.source.v6d_urls = ()
        config = self.make_config(env)
        self.assertEqual(hosts, [node.host for node in config.shards])
        with self.assertRaisesRegex(ValueError, "completed native handshake"):
            runtime.verify_pp_cache_handshake(config, self.scheduler)
        self.source.v6d_urls = tuple("http://" + host + ":9113" for host in hosts)
        runtime.verify_pp_cache_handshake(config, self.scheduler)
        self.source.v6d_urls = self.source.v6d_urls[:3] + ("http://10.0.0.9:9113",)
        with self.assertRaisesRegex(ValueError, "completed native handshake"):
            runtime.verify_pp_cache_handshake(config, self.scheduler)

    def test_no_implicit_topology_is_guessed_before_native_handshake(self):
        self.source.v6d_urls = ()
        with self.assertRaisesRegex(ValueError, "explicit Collector hosts"):
            self.make_config()

    def test_wrong_pp_layout_and_native_port_collision_are_rejected(self):
        self.config.parallel_config.tensor_parallel_size = 2
        with self.assertRaisesRegex(ValueError, "TP=1"):
            self.make_config()
        self.config.parallel_config.tensor_parallel_size = 1
        with self.assertRaisesRegex(ValueError, "colliding"):
            self.make_config(environment("9113"))

    def test_wrong_group_stage_count_and_nonzero_agent_stage_are_rejected(self):
        env = environment()
        env["NAVI_SCHED_PP_SIZE"] = "2"
        with self.assertRaisesRegex(ValueError, "match pipeline"):
            self.make_config(env)
        env = environment()
        env["NAVI_SCHED_PP_SHARD_ID"] = "1"
        with self.assertRaisesRegex(ValueError, "stage zero"):
            self.make_config(env)

    def test_native_connector_is_required_only_when_enabled(self):
        self.scheduler.connector = None
        with self.assertRaisesRegex(ValueError, "native PP V6D"):
            self.make_config()
        self.assertIsNone(self.make_config({}))

    def test_invalid_explicit_settings_are_rejected(self):
        for key, value in (
            ("NAVI_SCHED_PP_CACHE_COLLECTOR_PORT", "0"),
            ("NAVI_SCHED_PP_CACHE_COLLECTOR_GROUP", ""),
            ("NAVI_SCHED_PP_SHARD_ID", "4"),
        ):
            env = environment()
            env[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.make_config(env)

    @staticmethod
    def daemon_type():
        class Daemon:
            def __init__(self):
                self.peer = Peer()
                self._kv_cache_spec = None
                self.result = object()
                self.error = None
                self.calls = 0

            async def push_kv_cache_spec(self, payload):
                self.calls += 1
                if self.error is not None:
                    raise self.error
                self._kv_cache_spec = payload
                return self.result

        return Daemon

    def test_disabled_daemon_hook_preserves_native_method(self):
        daemon_type = self.daemon_type()
        original = daemon_type.push_kv_cache_spec
        self.assertFalse(runtime.install_pp_daemon_hook(daemon_type, environ={}))
        self.assertIs(original, daemon_type.push_kv_cache_spec)

    def test_native_spec_starts_real_collector_once_and_preserves_result(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        daemon_type = self.daemon_type()
        self.assertTrue(
            runtime.install_pp_daemon_hook(
                daemon_type,
                environ=environment(str(port)),
                core=core,
                invalid_object_id=0,
            )
        )
        daemon = daemon_type()
        try:
            self.assertIs(daemon.result, asyncio.run(daemon.push_kv_cache_spec(spec())))
            observer = daemon._navi_sched_pp_collector
            self.assertEqual(port, observer._collector.get_grpc_port())
            self.assertIs(daemon.result, asyncio.run(daemon.push_kv_cache_spec(spec())))
            self.assertIs(observer, daemon._navi_sched_pp_collector)
            self.assertEqual(2, daemon.calls)
            self.assertFalse(daemon._navi_sched_pp_collector_error)
            self.assertIs(daemon.peer.seal_result, daemon.peer.stop())
            self.assertTrue(observer._stopped)
        finally:
            observer = getattr(daemon, "_navi_sched_pp_collector", None)
            if observer is not None:
                observer.stop()

    def test_default_object_id_uses_native_v6d_lite_without_vineyard_package(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        native_type = ModuleType("v6d.lite.common.type")
        native_type.ObjectID = Obj(
            invalid_object_id=mock.Mock(return_value=Obj(id=(1 << 64) - 1))
        )
        modules = {
            name: ModuleType(name) for name in ("v6d", "v6d.lite", "v6d.lite.common")
        }
        modules.update({"v6d.lite.common.type": native_type, "vineyard": None})
        daemon_type = self.daemon_type()
        runtime.install_pp_daemon_hook(
            daemon_type, environ=environment(str(port)), core=core
        )
        daemon = daemon_type()
        try:
            with mock.patch.dict(sys.modules, modules):
                result = asyncio.run(daemon.push_kv_cache_spec(spec()))
            self.assertIs(daemon.result, result)
            self.assertFalse(daemon._navi_sched_pp_collector_error)
            self.assertEqual(
                port, daemon._navi_sched_pp_collector._collector.get_grpc_port()
            )
            self.assertEqual(
                (1 << 64) - 1, daemon._navi_sched_pp_collector._invalid_object_id
            )
            native_type.ObjectID.invalid_object_id.assert_called_once_with()
        finally:
            observer = getattr(daemon, "_navi_sched_pp_collector", None)
            if observer is not None:
                observer.stop()

    def test_native_spec_exception_is_preserved_without_starting_collector(self):
        daemon_type = self.daemon_type()
        runtime.install_pp_daemon_hook(
            daemon_type, environ=environment(), core=core, invalid_object_id=0
        )
        daemon = daemon_type()
        daemon.error = ValueError("native spec failure")
        with self.assertRaises(ValueError) as caught:
            asyncio.run(daemon.push_kv_cache_spec(spec()))
        self.assertIs(daemon.error, caught.exception)
        self.assertFalse(hasattr(daemon, "_navi_sched_pp_collector"))

    def test_unsupported_spec_does_not_replace_native_result(self):
        daemon_type = self.daemon_type()
        runtime.install_pp_daemon_hook(
            daemon_type, environ=environment(), core=core, invalid_object_id=0
        )
        daemon = daemon_type()
        payload = spec()
        payload["groups"][0]["spec_type"] = "UnknownSpec"
        with self.assertLogs(runtime.logger, level="ERROR"):
            result = asyncio.run(daemon.push_kv_cache_spec(payload))
        self.assertIs(daemon.result, result)
        self.assertTrue(daemon._navi_sched_pp_collector_error)
        self.assertFalse(hasattr(daemon, "_navi_sched_pp_collector"))


class PPRequestCompletionObserverTest(unittest.TestCase):
    def setUp(self):
        self.agent = mock.Mock()
        self.agent.begin_pp_cache_request.side_effect = iter(range(1, 100))
        self.agent.finish_pp_cache_request.return_value = True
        self.reasons = Obj(STOP=0, LENGTH=1, ABORT=2, ERROR=3)
        self.observer = runtime.PPRequestCompletionObserver(
            self.agent, core, self.reasons
        )

    def test_all_output_outcomes_and_nonfinal_output(self):
        for index, (reason, status) in enumerate(
            (
                (0, core.PPCacheRequestStatus.SUCCEEDED),
                (1, core.PPCacheRequestStatus.SUCCEEDED),
                (2, core.PPCacheRequestStatus.CANCELLED),
                (3, core.PPCacheRequestStatus.FAILED),
            ),
            1,
        ):
            key = str(index)
            ticket = self.observer.begin(key, index + 100)
            self.observer.observe_outputs(
                Obj(outputs=[Obj(request_id=key, finished=False)], finished_requests=[])
            )
            self.assertEqual(index - 1, self.agent.finish_pp_cache_request.call_count)
            output = Obj(request_id=key, finished=True, finish_reason=reason)
            self.observer.observe_outputs(
                Obj(outputs=[output], finished_requests=[key])
            )
            self.agent.finish_pp_cache_request.assert_called_with(*ticket, status)
            self.observer.observe_outputs(
                Obj(outputs=[output], finished_requests=[key])
            )
            self.assertEqual(index, self.agent.finish_pp_cache_request.call_count)

    def test_reused_id_ignores_old_admission_failure(self):
        old = self.observer.begin("a", 10)
        self.assertIsNone(self.observer.begin("a", 10))
        self.observer.fail_admission("a", old)
        current = self.observer.begin("a", 10)
        self.assertNotEqual(old, current)
        self.observer.fail_admission("a", old)
        self.assertEqual(1, self.agent.finish_pp_cache_request.call_count)
        self.observer.cancel_requests(["a"])
        self.agent.finish_pp_cache_request.assert_called_with(
            *current, core.PPCacheRequestStatus.CANCELLED
        )

    def test_cancel_all_and_shutdown_prevent_new_registration(self):
        first = self.observer.begin("a", 10)
        second = self.observer.begin("b", 11)
        self.observer.finish_requests(None, Obj(name="FINISHED_ABORTED"))
        self.assertEqual(
            [
                mock.call(*first, core.PPCacheRequestStatus.CANCELLED),
                mock.call(*second, core.PPCacheRequestStatus.CANCELLED),
            ],
            self.agent.finish_pp_cache_request.call_args_list,
        )
        third = self.observer.begin("c", 12)
        self.observer.close()
        self.agent.finish_pp_cache_request.assert_called_with(
            *third, core.PPCacheRequestStatus.FAILED
        )
        self.assertIsNone(self.observer.begin("d", 13))
        self.observer.close()
        self.assertEqual(3, self.agent.finish_pp_cache_request.call_count)

    def test_native_free_without_result_does_not_assert_success(self):
        ticket = self.observer.begin("a", 10)
        self.observer.observe_outputs(Obj(outputs=[], finished_requests=["a"]))
        self.agent.finish_pp_cache_request.assert_called_with(
            *ticket, core.PPCacheRequestStatus.FAILED
        )

    def test_binding_failure_does_not_replace_native_error(self):
        self.agent.begin_pp_cache_request.side_effect = RuntimeError("register failed")
        with self.assertLogs(runtime.logger, level="ERROR"):
            self.assertIsNone(self.observer.begin("a", 10))
        self.agent.begin_pp_cache_request.side_effect = None
        self.agent.begin_pp_cache_request.return_value = 1
        ticket = self.observer.begin("a", 10)
        self.agent.finish_pp_cache_request.side_effect = RuntimeError("notify failed")
        with self.assertLogs(runtime.logger, level="ERROR"):
            self.observer.fail_admission("a", ticket)
        self.observer.fail_admission("a", ticket)
        self.assertEqual(1, self.agent.finish_pp_cache_request.call_count)

    def test_late_execution_error_cannot_finish_a_reused_request_id(self):
        old = self.observer.begin("a", 10)
        captured = self.observer.capture_requests(["a", "unknown"])
        self.assertEqual((("a", old),), captured)
        self.observer.cancel_requests(["a"])
        current = self.observer.begin("a", 10)
        self.observer.finish_execution_error(captured, cancelled=False)
        self.assertEqual(1, self.agent.finish_pp_cache_request.call_count)
        self.observer.finish_execution_error(
            self.observer.capture_requests(["a"]), cancelled=False
        )
        self.agent.finish_pp_cache_request.assert_called_with(
            *current, core.PPCacheRequestStatus.FAILED
        )
        self.assertEqual((), self.observer.capture_requests(["a"]))

    def test_cancelled_execution_reports_cancel_without_waiting_for_outputs(self):
        ticket = self.observer.begin("a", 10)
        captured = self.observer.capture_requests(["a"])
        self.observer.finish_execution_error(captured, cancelled=True)
        self.agent.finish_pp_cache_request.assert_called_once_with(
            *ticket, core.PPCacheRequestStatus.CANCELLED
        )
        self.observer.finish_execution_error(captured, cancelled=True)
        self.assertEqual(1, self.agent.finish_pp_cache_request.call_count)

    def test_actual_binding_default_agent_does_not_track_pp(self):
        config = core.NaviSchedAgentConfig()
        agent = core.NaviSchedAgent(config)
        self.assertEqual(0, agent.begin_pp_cache_request(10))
        self.assertFalse(
            agent.finish_pp_cache_request(10, 1, core.PPCacheRequestStatus.FAILED)
        )
        agent.stop()


if __name__ == "__main__":
    unittest.main()
