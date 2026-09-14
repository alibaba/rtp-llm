import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from rtp_llm.test.amd_coverage import (
    AMD_GTEST_ADAPTER,
    AMD_GTEST_CASES,
    AMD_TARGETS,
    route_amd_items,
    validate_amd_coverage,
)


def _baseline_nodeids():
    return [
        target.path + "::" + (target.case or "test_case") for target in AMD_TARGETS
    ] + [
        AMD_GTEST_ADAPTER + f"::test_rocm_beam_search_gtest[{case}]"
        for case in AMD_GTEST_CASES
    ]


@pytest.mark.parametrize("missing", range(38))
def test_missing_baseline_target_cannot_be_replaced_by_unrelated_cases(missing):
    nodeids = _baseline_nodeids()
    validate_amd_coverage(nodeids)
    nodeids.pop(missing)
    nodeids.extend(f"other.py::test_other[{i}]" for i in range(300))
    with pytest.raises(ValueError, match="AMD baseline coverage missing"):
        validate_amd_coverage(nodeids)


@pytest.mark.parametrize("target", AMD_TARGETS, ids=lambda target: target.name)
def test_amd_route_preserves_multicard_requirements(target):
    markers = [pytest.mark.gpu(type="H20")]
    item = SimpleNamespace(
        nodeid=target.path + "::" + (target.case or "test_case"),
        add_marker=lambda marker, append: (
            markers.append(marker) if append else markers.insert(0, marker)
        ),
    )
    route_amd_items([item])
    assert markers[0].kwargs == {"type": "MI308X", "count": target.gpu_count}
    assert markers[1].kwargs == {"type": "H20"}


@pytest.mark.parametrize(
    "xml,exitcode",
    [
        ("<testsuites/>", 0),
        (
            '<testsuites><testcase classname="RocmBeamSearchOpTest" name="simpleTest" status="notrun" result="suppressed"/></testsuites>',
            0,
        ),
        (
            '<testsuites><testcase classname="RocmBeamSearchOpTest" name="simpleTest" status="run" result="completed"><skipped/></testcase></testsuites>',
            0,
        ),
        (
            '<testsuites><testcase classname="RocmBeamSearchOpTest" name="wrongCase" status="run" result="completed"/></testsuites>',
            0,
        ),
        (
            '<testsuites><testcase classname="RocmBeamSearchOpTest" name="simpleTest" status="run" result="completed"/></testsuites>',
            1,
        ),
    ],
)
def test_gtest_adapter_rejects_zero_skipped_wrong_or_failed_case(
    tmp_path, monkeypatch, xml, exitcode
):
    adapter = _load_adapter()
    binary = tmp_path / "binary"
    binary.touch()
    monkeypatch.setattr(adapter, "_BINARY", binary)

    def run(args, **kwargs):
        assert args[1] == "--gtest_filter=RocmBeamSearchOpTest.simpleTest"
        Path(args[2].removeprefix("--gtest_output=xml:")).write_text(xml)
        return SimpleNamespace(returncode=exitcode, stdout="gtest output", stderr="")

    monkeypatch.setattr(adapter.subprocess, "run", run)
    with pytest.raises(AssertionError):
        adapter.test_rocm_beam_search_gtest(AMD_GTEST_CASES[0], tmp_path)


def _load_adapter():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "beam_search_adapter", root / AMD_GTEST_ADAPTER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
