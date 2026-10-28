from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from rtp_llm.test.remote_tests.junit_merge import merge_reports


@pytest.mark.parametrize(
    "bad_report",
    [
        None,
        "<broken",
        "<not-junit/>",
        "<testsuite tests='4'/>",
        "<testsuite><testcase name='skipped'><skipped/></testcase></testsuite>",
        "<testsuite><testcase name='failed'><failure/></testcase></testsuite>",
    ],
)
def test_required_report_failure_survives_successful_sibling(tmp_path, bad_report):
    good = tmp_path / "good.xml"
    good.write_text("<testsuite><testcase name='passed'/></testsuite>")
    bad = tmp_path / "bad.xml"
    if bad_report is not None:
        bad.write_text(bad_report)
    output = tmp_path / "merged.xml"
    with pytest.raises(ValueError):
        merge_reports([good, bad], output, required=[good, bad], forbid_skips=True)
    assert ET.parse(output).find(".//error") is not None
    assert ET.parse(output).find(".//testcase[@name='passed']") is not None


def test_optional_empty_phase_does_not_hide_required_cases(tmp_path):
    good = tmp_path / "good.xml"
    good.write_text(
        "<testsuites><testsuite><testcase name='one'/><testcase name='two'/></testsuite></testsuites>"
    )
    empty = tmp_path / "empty.xml"
    empty.write_text("<testsuite tests='0'/>")
    assert (
        merge_reports(
            [good, empty], tmp_path / "merged.xml", required=[good], forbid_skips=True
        )
        == 2
    )


def test_entirely_empty_session_fails(tmp_path):
    with pytest.raises(ValueError, match="zero testcases"):
        merge_reports([], tmp_path / "merged.xml")

@pytest.mark.parametrize("with_real_failure", [False, True])
def test_integrity_error_has_explicit_non_execution_category(tmp_path, with_real_failure):
    from rtp_llm.test.remote_tests.junit_merge import count_report_integrity_errors

    original = tmp_path / "original.xml"
    failure = "<failure message='original assertion'/>" if with_real_failure else ""
    original.write_text(
        "<testsuite><testcase name='real'><properties>"
        "<property name='nodeid' value='test_original.py::test_real'/>"
        "</properties>" + failure + "</testcase></testsuite>"
    )
    missing = tmp_path / "missing.xml"
    merged = tmp_path / "merged.xml"
    with pytest.raises(ValueError):
        merge_reports([original, missing], merged, required=[original, missing])
    root = ET.parse(merged).getroot()
    assert len(list(root.iter("testcase"))) == 2
    assert count_report_integrity_errors(root) == 1
    assert len(list(root.iter("error"))) == 1
    assert len(list(root.iter("failure"))) == int(with_real_failure)


@pytest.mark.parametrize(
    "mutation",
    ["legacy_unmarked", "real_nodeid", "wrong_kind", "duplicate_kind",
     "no_error", "wrong_error", "ordinary_suite", "different_name", "failure"],
)
def test_non_owned_or_invalid_integrity_records_are_not_excluded(tmp_path, mutation):
    from rtp_llm.test.remote_tests.junit_merge import count_report_integrity_errors

    merged = tmp_path / "merged.xml"
    with pytest.raises(ValueError):
        merge_reports([], merged)
    root = ET.parse(merged).getroot()
    suite = root.find("testsuite")
    case = suite.find("testcase")
    properties = case.find("properties")
    marker = properties.find("property")
    if mutation == "legacy_unmarked":
        case.remove(properties)
    elif mutation == "real_nodeid":
        ET.SubElement(properties, "property", name="nodeid", value="real.py::test_real")
    elif mutation == "wrong_kind":
        marker.set("value", "ordinary")
    elif mutation == "duplicate_kind":
        ET.SubElement(properties, "property", **marker.attrib)
    elif mutation == "no_error":
        case.remove(case.find("error"))
    elif mutation == "wrong_error":
        case.find("error").set("message", "a real test error")
    elif mutation == "ordinary_suite":
        suite.set("name", "pytest")
    elif mutation == "different_name":
        case.set("name", "test_real")
    elif mutation == "failure":
        ET.SubElement(case, "failure", message="real assertion")
    assert count_report_integrity_errors(root) == 0
