import sys
from dataclasses import replace

import pytest
from mock_framework import case as case_cli
from mock_framework.config import load_yaml
from mock_framework.execution import ExecutionError
from mock_framework.execution.launch import local_launch, prepare_launch


@pytest.fixture
def case(case_data, compile_data):
    from mock_framework.execution.test.test_master import configuration

    case_data["environment"]["master"]["config"] = configuration()
    case_data["environment"]["decode"] = case_data["environment"]["prefill"]
    case_data["environment"]["client"] = {"config": {}}
    return compile_data(case_data)


@pytest.fixture
def artifacts(tmp_path):
    framework = tmp_path / "framework with spaces"
    for relative in (
        "java/master/target/classes/org/flexlb/MasterMain.class",
        "java/zookeeper/target/classes/org/flexlb/mock/zookeeper/ZooKeeperMain.class",
        "java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar",
        "java/client/target/mock-client-1.0-SNAPSHOT-all.jar",
        "dependency.jar",
    ):
        path = framework / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (framework / "java/zookeeper/target/dependencies.txt").write_text(
        str(framework / "dependency.jar") + "\n"
    )
    (framework / "java/master/target/dependencies.txt").write_text(
        str(framework / "dependency.jar") + "\n"
    )
    java, prometheus = tmp_path / "java", tmp_path / "prometheus"
    for path in (java, prometheus):
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    return framework, java, prometheus


def test_local_recipes_round_trip_without_shell_or_cwd_assumptions(case, artifacts):
    framework, java, prometheus = artifacts
    launch = local_launch(case, java, prometheus, framework=framework)
    assert launch == prepare_launch(launch, case, framework)
    assert set(launch["components"]) == set(case.environment)
    assert launch["components"]["prefill"] == launch["components"]["decode"]
    assert (
        launch["components"]["prefill"]["command"][-1]
        == "org.flexlb.mock.engine.EngineMain"
    )
    assert launch["components"]["master"]["command"][-1] == "org.flexlb.MasterMain"
    assert (
        launch["zookeeper"]["command"][-1] == "org.flexlb.mock.zookeeper.ZooKeeperMain"
    )
    assert launch["components"]["client"]["command"][-1].endswith(
        "mock-client-1.0-SNAPSHOT-all.jar"
    )
    assert all(
        recipe["command"][0] == str(java) and recipe["cwd"] == str(framework)
        for recipe in launch["components"].values()
    )


@pytest.mark.parametrize(
    "damage", ["java", "prometheus", "engine", "master", "classpath"]
)
def test_local_recipes_reject_missing_or_unusable_inputs(case, artifacts, damage):
    framework, java, prometheus = artifacts
    if damage in ("java", "prometheus"):
        (java if damage == "java" else prometheus).chmod(0o644)
    elif damage == "engine":
        (
            framework / "java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar"
        ).unlink()
    elif damage == "master":
        (framework / "java/master/target/classes/org/flexlb/MasterMain.class").unlink()
    else:
        (framework / "java/master/target/dependencies.txt").write_text(
            "relative-or-missing.jar\n"
        )
    with pytest.raises(ExecutionError):
        local_launch(case, java, prometheus, framework=framework)


@pytest.mark.parametrize("configuration", ["unsupported-role", "non-native-master"])
def test_local_recipes_reject_roles_or_master_configuration_the_binaries_cannot_run(
    case, artifacts, configuration
):
    framework, java, prometheus = artifacts
    environment = (
        {"unknown": case.environment["prefill"]}
        if configuration == "unsupported-role"
        else {"master": {"config": {}}}
    )
    with pytest.raises(ExecutionError):
        local_launch(
            replace(case, environment=environment),
            java,
            prometheus,
            framework=framework,
        )


def test_cli_generates_a_readable_launch_file_and_refuses_overwrite(
    case, artifacts, tmp_path, monkeypatch
):
    framework, java, prometheus = artifacts
    monkeypatch.setattr(
        case_cli, "__file__", str(framework / "python/mock_framework/case.py")
    )
    output = tmp_path / "launch.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mock-case",
            "launch",
            str(case.directory / "case.yaml"),
            "--java",
            str(java),
            "--prometheus",
            str(prometheus),
            "--output",
            str(output),
        ],
    )
    case_cli.main()
    content = output.read_text()
    assert "&id" not in content
    launch = prepare_launch(load_yaml(output), case, output.parent)
    commands = launch["components"]
    assert commands["master"]["command"][1:5] == [
        "-XX:ActiveProcessorCount=32",
        "-Xmx96g",
        "-XX:MaxDirectMemorySize=16g",
        "-XX:MaxGCPauseMillis=150",
    ]
    assert commands["client"]["command"][1:3] == [
        "-XX:ActiveProcessorCount=16",
        "-Xmx16g",
    ]
    for role in ("prefill", "decode"):
        assert commands[role]["command"][1:3] == [
            "-XX:ActiveProcessorCount=2",
            "-Xmx2g",
        ]
    assert launch["zookeeper"]["command"][1] == "-XX:ActiveProcessorCount=2"
    with pytest.raises(SystemExit) as error:
        case_cli.main()
    assert error.value.code == 2 and output.read_text() == content


def test_vision_uses_the_same_engine_entry_without_a_backend_tag(case, artifacts):
    framework, java, prometheus = artifacts
    environment = dict(case.environment)
    environment["vit"] = {
        "deployment": {"replicas": 1},
        "config": {"mock.compute": {"forward": {"expression": "1"}}},
    }
    case = replace(case, environment=environment)
    launch = local_launch(case, java, prometheus, framework=framework)
    assert launch["components"]["vit"] == launch["components"]["prefill"]


def test_launch_rejects_duplicate_master_business_setting(case, artifacts, tmp_path):
    framework, java, prometheus = artifacts
    launch = local_launch(case, java, prometheus, framework=framework)
    launch["components"]["master"]["env"] = {
        "FLEXLB_CONFIG": case.environment["master"]["config"]["FLEXLB_CONFIG"]
    }
    with pytest.raises(
        ValueError, match="environment.master.config and launch.components.master.env"
    ):
        prepare_launch(launch, case, tmp_path)


def test_launch_rejects_explicit_locally_assigned_master_identity(
    case, artifacts, tmp_path
):
    framework, java, prometheus = artifacts
    launch = local_launch(case, java, prometheus, framework=framework)
    launch["components"]["master"]["env"] = {"FLEXLB_ADVERTISED_IP": "127.0.0.1"}
    with pytest.raises(ValueError, match="FLEXLB_ADVERTISED_IP is assigned"):
        prepare_launch(launch, case, tmp_path)
