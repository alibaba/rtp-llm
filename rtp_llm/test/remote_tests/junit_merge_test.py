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
