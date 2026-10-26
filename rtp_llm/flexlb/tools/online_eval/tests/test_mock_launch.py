"""Launch and control wire contracts shared by all Mock entrypoints."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from adapters.client_configuration import ClientConfiguration, ClientLimits, ClientTransport
from adapters.load_client import client_environment
from adapters.mock_engine import MockEngineControl
from adapters.mock_launch import MockLaunchSpec
from traffic.playback.configuration import PlaybackConfiguration


def test_launch_preserves_explicit_zero_false_and_empty_values_and_jvm_options():
    spec = MockLaunchSpec(
        "java",
        Path("mock.jar"),
        1,
        0,
        28001,
        jvm_options=("-Xmx1g", "-XX:+ExitOnOutOfMemoryError"),
        whale=False,
        auto_fetch=False,
        events_file="",
        rpc_threads=0,
        calibration="calibration.json",
        engine_config="engine-config.json",
    )
    argv = spec.argv()
    assert argv[:5] == ["java", "-Xmx1g", "-XX:+ExitOnOutOfMemoryError", "-jar", "mock.jar"]
    flags = dict(zip(argv[5::2], argv[6::2]))
    assert flags == {
        "--mock.n_prefill": "1",
        "--mock.n_decode": "0",
        "--mock.base_grpc_port": "28001",
        "--mock.whale": "false",
        "--mock.auto_fetch": "false",
        "--mock.events_file": "",
        "--mock.rpc_threads": "0",
        "--mock.calibration": "calibration.json",
        "--mock.engine_config": "engine-config.json",
    }
    argv.append("unrelated")
    assert "unrelated" not in spec.argv()


def test_optional_flags_preserve_java_defaults_and_discovery_follows_explicit_overrides():
    spec = MockLaunchSpec("java", "mock.jar", 1, 1, 28001, discovery_file="owned.json", extra_args=("--mock.auto_fetch", "false"))
    assert spec.argv()[3:] == [
        "--mock.n_prefill",
        "1",
        "--mock.n_decode",
        "1",
        "--mock.base_grpc_port",
        "28001",
        "--mock.auto_fetch",
        "false",
        "--mock.discovery_file",
        "owned.json",
    ]
    assert "--mock.discovery_file" not in replace(spec, discovery_file=None).argv()


@pytest.mark.parametrize(
    "action,fields", [("inject", {"config": {"enqueue_error": True}}), ("clear_inject", {}), ("start_engine", {}), ("stop_engine", {})]
)
def test_engine_commands_keep_wire_body_and_error_contract(action, fields):
    control = MockEngineControl()
    control.mock_http_port = 29000
    args = ("prefill-0", fields["config"]) if action == "inject" else ("prefill-0",)
    with patch("adapters.mock_engine.http_post_json", return_value=(200, None)) as send:
        assert getattr(control, action)(*args) == {}
    send.assert_called_once_with(f"http://127.0.0.1:29000/{action}", {"engine": "prefill-0", **({"mock": {"fault": fields["config"]}} if action == "inject" else fields)})
    with patch("adapters.mock_engine.http_post_json", return_value=(404, {"error": "gone"})):
        with pytest.raises(RuntimeError, match=action + r"\(prefill-0\) failed: 404"):
            getattr(control, action)(*args)


@pytest.mark.parametrize("status,expected", [(200, True), (400, False), (500, False)])
def test_perf_commands_keep_boolean_results_and_do_not_confuse_body_fields_with_action(status, expected):
    control = MockEngineControl()
    control.mock_http_port = 29000
    with patch("adapters.mock_engine.http_post_json", return_value=(status, {})) as send:
        assert control.set_perf("prefill-0", forward={"decode": {"scale": 2}}) is expected
    send.assert_called_once_with("http://127.0.0.1:29000/set_perf", dict(engine="prefill-0", mock={"forward": {"decode": {"scale": 2}}}))


@pytest.mark.parametrize("mode,timeout", [("graceful", 95.0), ("abrupt", 5.0)])
def test_topology_commands_keep_target_selection_and_drain_budget(mode, timeout):
    control = MockEngineControl()
    control.mock_http_port = 29000
    with patch("adapters.mock_engine.http_post_json", return_value=(404, None)) as send:
        assert control.remove_engine(port=28001, mode=mode, drain_timeout_ms=300) == (404, None)
    send.assert_called_once_with("http://127.0.0.1:29000/remove_engine", dict(port=28001, mode=mode, drain_timeout_ms=300), timeout=timeout)


def test_typed_client_configuration_is_reused_and_projects_the_same_wire_environment():
    data = dict(playback=dict(duration_s=30), limits=dict(max_concurrency=100))
    configuration = ClientConfiguration.read(data)
    assert ClientConfiguration.read(configuration) is configuration
    assert client_environment(configuration, priority=75) == client_environment(data, priority=75)
    data["limits"]["max_concurrency"] = 1
    assert configuration.limits.max_concurrency == 100


def test_flow_admission_retains_one_typed_client_configuration():
    from adapters.java_flow_spec import JavaFlowSpec
    from program_fixtures import config, settings

    data = config("master_performance")["parameters"]["traffic"]
    data = {key: value for key, value in data.items() if key != "kind"}
    data["client"] = settings()
    admitted = JavaFlowSpec.read(data)
    assert isinstance(admitted.client, ClientConfiguration)
    assert admitted.client.limits.max_concurrency > 0


def test_typed_shared_defaults_still_require_bounded_scenario_duration():
    defaults = ClientConfiguration(PlaybackConfiguration(), ClientLimits(), ClientTransport())
    with pytest.raises(ValueError, match="duration_s"):
        client_environment(defaults)
