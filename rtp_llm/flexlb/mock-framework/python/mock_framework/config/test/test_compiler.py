from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import pytest
from mock_framework.config import ConfigError, compile_case
from mock_framework.config.compiler import expand_config
from mock_framework.config.validation import Bounds, schema
from mock_framework.execution.step.step import Reference, Step
from mock_framework.execution.step.step_registry import StepRegistry, register_step
from mock_framework.execution.test.helpers import FRAMEWORK


@pytest.mark.parametrize(
    "path",
    sorted((FRAMEWORK / "cases").glob("*/case.yaml")),
    ids=lambda path: path.parent.name,
)
def test_shipped_cases_compile_and_reference_existing_inputs(path, registry):
    case = compile_case(path, registry)
    for phase in case.phases:
        for step in phase.steps:
            if step.definition.name == "client/start/playback":
                assert (case.shared_root / step.args["requests_file"]).is_file()


def test_compiles_without_java_catalog_and_read_can_have_no_assertion(
    case_data, compile_data
):
    case = compile_data(case_data)
    assert case.environment["prefill"]["config"] == {
        "a_b_c": 128,
        "cache": {"sync": {"delay_ms": 10}},
    }
    steps = case.phases[0].steps
    assert all(isinstance(s, Step) for s in steps)
    assert steps[1].expect is None
    assert steps[1].args["targets"] == Reference("leader", ())
    assert steps[3].expect.comparisons["gte"] == Reference("tps", ())
    assert str(case.view).endswith("shared/views/default/dashboard.json")


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(schema_version=3),
        lambda d: d["environment"]["prefill"]["deployment"].update(replicas=True),
        lambda d: d["phases"][0].update(parallel=True),
        lambda d: d["phases"][0]["steps"][0].update(run="background"),
        lambda d: d["phases"][0]["steps"][0].update(interface="client/traffic/run"),
        lambda d: d["phases"][0]["steps"][0].update(within="-1s"),
        lambda d: d["phases"][0]["steps"][1]["args"].update(
            targets={"ref": "leader.nodes"}
        ),
        lambda d: d["phases"][0]["steps"][1]["args"].update(
            targets={"ref": "steps.tps"}
        ),
        lambda d: d["phases"][0]["steps"][1]["args"].update(targets=[]),
        lambda d: d["phases"][0]["steps"][1]["args"].update(unknown=1),
        lambda d: d["phases"][0]["steps"][2]["args"].update(range=["0s", "0s"]),
        lambda d: d["phases"][0]["steps"][2]["args"].update(range="all"),
        lambda d: d["phases"][0]["steps"][2].update(expect={"count": 1}),
        lambda d: d["phases"][0]["steps"][2].update(expect={"gte": 2, "lte": 1}),
        lambda d: d["phases"][0]["steps"][3].update(
            expect={"gte": {"ref": "steps.leader"}}
        ),
        lambda d: d["phases"][0]["steps"][3].update(
            expect={"gte": {"ref": "steps.tps.value"}}
        ),
    ],
)
def test_rejects_invalid_case_before_execution(case_data, compile_data, change):
    case_data["environment"]["prefill"]["deployment"] = {}
    change(case_data)
    with pytest.raises(ConfigError):
        compile_data(case_data)


def test_old_call_check_wrappers_are_rejected(case_data, compile_data):
    case_data["phases"][0]["steps"][0] = {"check": case_data["phases"][0]["steps"][0]}
    with pytest.raises(ConfigError):
        compile_data(case_data)


def test_global_reference_is_typed(case_data, compile_data):
    case_data["phases"][0]["steps"][0]["args"] = {"targets": {"ref": "globals.masters"}}
    assert compile_data(case_data).phases[0].steps[0].args["targets"] == Reference(
        "globals", ("masters",)
    )


@dataclass(frozen=True)
class ExtensionArgs:
    threshold: Annotated[int, Bounds(1, 8)]


@register_step("extension/queue", effect="read")
async def queue(ctx, args: ExtensionArgs) -> int:
    return args.threshold


def test_function_registration_defines_contract_without_compiler_changes(
    case_data, compile_data
):
    registry = StepRegistry((queue,))
    case_data["phases"][0]["steps"] = [
        {
            "id": "extension",
            "describe": "Python extension",
            "interface": "extension/queue",
            "args": {"threshold": 4},
            "expect": {"lte": 8},
        }
    ]
    assert compile_data(case_data, interfaces=registry).phases[0].steps[0].args == {
        "threshold": 4
    }
    for bad in ("4", True, 0, 9):
        case_data["phases"][0]["steps"][0]["args"]["threshold"] = bad
        with pytest.raises(ConfigError):
            compile_data(case_data, interfaces=registry)
    with pytest.raises(ConfigError, match="duplicate"):
        registry.register(queue)


@pytest.mark.parametrize(
    "raw",
    [
        {"cache": 1, "cache.sync.enabled": True},
        {"cache.sync.enabled": True, "cache": 1},
        {"cache": {"sync": {}}, "cache.sync.enabled": True},
        {"cache..enabled": True},
    ],
)
def test_config_collisions(raw):
    with pytest.raises(ConfigError):
        expand_config(raw)


def test_mock_nested_and_dotted_input_share_one_namespace():
    nested = {"mock": {"compute": {"forward": {"expression": "1"}, "cp": {"size": 4}}}}
    mixed = {"mock.compute": {"forward.expression": "1"}, "mock.compute.cp.size": 4}
    assert expand_config(nested) == expand_config(mixed)
    for raw in (
        {
            "mock.compute": {"forward": {"expression": "1"}},
            "mock.compute.forward.expression": "2",
        },
        {
            "mock.compute.forward.expression": "1",
            "mock": {"compute": {"forward": {"expression": "2"}}},
        },
    ):
        with pytest.raises(ConfigError, match="collision"):
            expand_config(raw)


def test_native_payload_and_duplicate_keys(case_data, compile_data):
    assert expand_config({"raw_json": {"original.key": "untouched"}}) == {
        "raw_json": {"original.key": "untouched"}
    }
    case_data["environment"]["master"]["config"][
        "FLEXLB_CONFIG"
    ] = '{"scheduler":{},"scheduler":{}}'
    with pytest.raises(ConfigError, match="Duplicate native JSON"):
        compile_data(case_data)


@pytest.mark.parametrize("role", ["master", "prefill", "decode", "vit"])
def test_component_config_file_matches_inline_config_and_resolves_from_case_directory(
    role, case_data, compile_data, tmp_path, monkeypatch
):
    import yaml

    if role not in case_data["environment"]:
        case_data["environment"][role] = {
            "config": {"mock.compute.forward.expression": "1"}
        }
        if role == "decode":
            case_data["environment"][role]["engine"] = "rtp"
    expected = compile_data(case_data).environment
    folder = tmp_path / "configs"
    folder.mkdir()
    (folder / f"{role}.yaml").write_text(
        yaml.safe_dump(case_data["environment"][role]["config"])
    )
    case_data["environment"][role]["config"] = f"configs/{role}.yaml"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert compile_data(case_data).environment == expected


@pytest.mark.parametrize(
    "content, message",
    [
        ("[]", "mapping"),
        ("a: 1\na: 2\n", "Duplicate key"),
        ("cache: 1\ncache.enabled: true\n", "collision"),
        ('FLEXLB_CONFIG: \'{"a":1,"a":2}\'\n', "Duplicate native JSON"),
        ("FLEXLB_CONFIG: 'invalid'\n", "Invalid FLEXLB_CONFIG"),
    ],
)
def test_component_config_files_use_existing_strict_validation(
    content, message, case_data, compile_data, tmp_path
):
    (tmp_path / "master.yaml").write_text(content)
    case_data["environment"]["master"]["config"] = "master.yaml"
    with pytest.raises(ConfigError, match=message):
        compile_data(case_data)


@pytest.mark.parametrize(
    "path", ["missing.yaml", "", "https://example.com/config.yaml"]
)
def test_component_config_file_rejects_missing_empty_or_remote_paths(
    path, case_data, compile_data
):
    case_data["environment"]["prefill"]["config"] = path
    with pytest.raises(ConfigError):
        compile_data(case_data)


def test_nullable_reference_preserves_both_possible_types():
    from mock_framework.config.validation import compatible

    assert compatible(schema(list[str] | None), schema(list[str] | None))
    assert not compatible(schema(list[str] | None), schema(list[str]))


def test_typed_records_inside_result_collections_remain_referenceable_values():
    import asyncio

    @register_step("extension/records", effect="read")
    async def records(ctx, args: ExtensionArgs) -> dict[str, list[ExtensionArgs]]:
        return {"records": [args]}

    registry = StepRegistry((records,))
    assert asyncio.run(
        registry.get("extension/records").invoke(None, {"threshold": 4})
    ) == {"records": [{"threshold": 4}]}


@pytest.mark.parametrize("wait", ["0s", "20ms", "1.5s"])
def test_wait_after_is_a_step_duration(case_data, compile_data, wait):
    assert compile_data(case_data).phases[0].steps[0].wait_after == "0s"
    case_data["phases"][0]["steps"][0]["wait_after"] = wait
    assert compile_data(case_data).phases[0].steps[0].wait_after == wait


@pytest.mark.parametrize("wait", ["-1s", "1", 1, True, None, {"ref": "steps.leader"}])
def test_invalid_wait_after_rejected_before_execution(case_data, compile_data, wait):
    case_data["phases"][0]["steps"][0]["wait_after"] = wait
    with pytest.raises(ConfigError):
        compile_data(case_data)


@pytest.mark.parametrize("path", [".", "/"])
def test_view_requires_a_file_path_with_shared_root(case_data, compile_data, path):
    case_data["view"] = path
    with pytest.raises(ConfigError, match="view: expected a dashboard file path"):
        compile_data(case_data)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_master_json_rejects_nonfinite_constants(case_data, compile_data, constant):
    case_data["environment"]["master"]["config"]["FLEXLB_CONFIG"] = (
        '{"nested":{"value":' + constant + "}}"
    )
    with pytest.raises(ConfigError, match="Invalid FLEXLB_CONFIG JSON.*Nonfinite"):
        compile_data(case_data)


def test_nested_references_keep_static_sibling_constraints_and_error_paths(
    case_data, compile_data
):
    @dataclass(frozen=True)
    class NestedArgs:
        items: list[ExtensionArgs]
        targets: list[str] | None = None

    @register_step("extension/nested", effect="read")
    async def nested(ctx, args: NestedArgs) -> int:
        return len(args.items)

    registry = StepRegistry((queue, nested))
    steps = [
        {
            "id": "source",
            "describe": "value",
            "interface": "extension/queue",
            "args": {"threshold": 4},
        },
        {
            "id": "nested",
            "describe": "nested refs",
            "interface": "extension/nested",
            "args": {
                "items": [{"threshold": {"ref": "steps.source"}}, {"threshold": 8}]
            },
        },
    ]
    case_data["phases"][0]["steps"] = steps
    case = compile_data(case_data, interfaces=registry)
    assert case.phases[0].steps[1].args["items"][0]["threshold"] == Reference(
        "source", ()
    )
    steps[1]["args"]["items"][1]["threshold"] = 9
    with pytest.raises(
        ConfigError, match=r"nested\.args\.items\[1\]\.threshold: above maximum"
    ):
        compile_data(case_data, interfaces=registry)


def test_union_preserves_nested_constructor_failure():
    @dataclass(frozen=True)
    class Positive:
        value: int

        def __post_init__(self):
            if self.value <= 0:
                raise ValueError("value must be positive")

    @dataclass(frozen=True)
    class Choice:
        item: Positive | None

    @register_step("extension/choice", effect="read")
    async def choice(ctx, args: Choice) -> int:
        return 1

    interface = StepRegistry((choice,)).get("extension/choice")
    with pytest.raises(
        ConfigError, match=r"extension/choice\.args\.item: value must be positive"
    ):
        interface.arguments({"item": {"value": 0}})


def test_registered_actions_are_functions_and_compiled_steps_share_only_the_definition(
    case_data, compile_data
):
    from inspect import iscoroutinefunction

    from mock_framework.execution.step.step_registry import StepDefinition

    registry = StepRegistry((queue,))
    definition = registry.get("extension/queue")
    assert isinstance(definition, StepDefinition)
    assert definition.handler is queue and iscoroutinefunction(queue)
    case_data["phases"][0]["steps"] = [
        {
            "id": "one",
            "describe": "first call",
            "interface": "extension/queue",
            "args": {"threshold": 1},
        },
        {
            "id": "two",
            "describe": "second call",
            "interface": "extension/queue",
            "args": {"threshold": 2},
        },
    ]
    case = compile_data(case_data, interfaces=registry)
    first, second = case.phases[0].steps
    assert type(first) is Step and type(second) is Step
    assert first.definition is second.definition is definition
    assert first.args == {"threshold": 1} and second.args == {"threshold": 2}
    assert case.to_dict()["phases"][0]["steps"][0] == {
        "id": "one",
        "describe": "first call",
        "interface": "extension/queue",
        "args": {"threshold": 1},
        "expect": None,
        "within": None,
        "wait_after": "0s",
        "background": False,
    }


def test_action_registration_rejects_classes_sync_functions_and_wrong_signatures():
    class LegacyAction:
        @staticmethod
        async def execute(ctx, args: ExtensionArgs) -> int:
            return args.threshold

    def synchronous(ctx, args: ExtensionArgs) -> int:
        return args.threshold

    async def missing_types(ctx, args):
        return 1

    async def keyword_only(ctx, *, args: ExtensionArgs) -> int:
        return args.threshold

    for invalid in (LegacyAction, synchronous, missing_types, keyword_only):
        with pytest.raises(ConfigError):
            register_step("extension/invalid", effect="read")(invalid)
    with pytest.raises(ConfigError):
        StepRegistry((synchronous,))
