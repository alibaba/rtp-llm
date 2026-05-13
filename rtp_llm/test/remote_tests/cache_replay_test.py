from types import SimpleNamespace

from rtp_llm.test.remote_tests.plugin import RemoteREAPIPlugin
from rtp_llm.test.remote_tests.test_cache import CacheEntry, DigestRef


def _entry(
    nodeid: str,
    *,
    gpu_type: str = "H20",
    profile: str = "py_ut_sm9x",
    scope: str = "profile=py_ut_sm9x|markexpr=H20|keyword=|pytest_args=",
    junit: str | None = None,
) -> CacheEntry:
    return CacheEntry(
        test_nodeid=nodeid,
        gpu_type=gpu_type,
        gpu_count=1,
        result="passed",
        exit_code=0,
        timestamp=9999999999.0,
        duration_s=1.25,
        stdout_digest=DigestRef("stdout", 1),
        stderr_digest=DigestRef("stderr", 1),
        output_files={},
        junit_testcase_xml=junit,
        session_profile=profile,
        session_scope=scope,
    )


def test_cache_entry_v3_roundtrip_preserves_replay_metadata():
    entry = _entry(
        "pkg/test_mod.py::test_a",
        junit=(
            '<testcase classname="pkg.test_mod" name="test_a" time="1.25">'
            '<properties><property name="nodeid" value="pkg/test_mod.py::test_a" /></properties>'
            "</testcase>"
        ),
    )

    restored = CacheEntry.from_dict(entry.to_dict())

    assert restored.test_nodeid == "pkg/test_mod.py::test_a"
    assert restored.junit_testcase_xml == entry.junit_testcase_xml
    assert restored.session_profile == "py_ut_sm9x"
    assert restored.session_scope == entry.session_scope
    assert restored.stdout_digest.hash == "stdout"


def test_session_junit_merge_replays_cached_cases_and_deduplicates_fresh():
    plugin = RemoteREAPIPlugin.__new__(RemoteREAPIPlugin)
    plugin._session_cached_entries = {
        "pkg/test_mod.py::test_cached": _entry(
            "pkg/test_mod.py::test_cached",
            junit=(
                '<testcase classname="pkg.test_mod" name="test_cached" time="1.25">'
                '<properties><property name="nodeid" value="pkg/test_mod.py::test_cached" /></properties>'
                "</testcase>"
            ),
        ),
        "pkg/test_mod.py::test_fresh": _entry(
            "pkg/test_mod.py::test_fresh",
            junit=(
                '<testcase classname="pkg.test_mod" name="test_fresh" time="9.99">'
                '<properties><property name="nodeid" value="pkg/test_mod.py::test_fresh" /></properties>'
                "</testcase>"
            ),
        ),
    }
    fresh = (
        '<testsuites><testsuite name="fresh" tests="1">'
        '<testcase classname="pkg.test_mod" name="test_fresh" time="0.5">'
        '<properties><property name="nodeid" value="pkg/test_mod.py::test_fresh" /></properties>'
        "</testcase></testsuite></testsuites>"
    )

    merged, tests, cached = plugin._merge_session_junit(fresh)

    assert tests == 2
    assert cached == 1
    assert "test_cached" in merged
    assert merged.count('name="test_fresh"') == 1
    assert 'tests="2"' in merged


def test_session_deselect_uses_only_complete_matching_cache_entries():
    plugin = RemoteREAPIPlugin.__new__(RemoteREAPIPlugin)
    plugin._gpu_request = SimpleNamespace(gpu_type="H20")
    plugin.ci_profile = "py_ut_sm9x"
    plugin.pytest_args = ""
    plugin._test_cache_ttl = 7
    plugin._collect_outputs = False
    plugin.config = SimpleNamespace(
        option=SimpleNamespace(markexpr="H20", keyword="")
    )
    plugin._reconstruct_result_from_cache = lambda entry, require_outputs=False: object()

    good = _entry(
        "pkg/test_mod.py::test_good",
        junit=(
            '<testcase classname="pkg.test_mod" name="test_good" time="1.25">'
            '<properties><property name="nodeid" value="pkg/test_mod.py::test_good" /></properties>'
            "</testcase>"
        ),
    )
    no_junit = _entry("pkg/test_mod.py::test_no_junit")
    wrong_profile = _entry(
        "pkg/test_mod.py::test_perf",
        profile="perf_sm9x",
        scope="profile=perf_sm9x|markexpr=H20|keyword=|pytest_args=",
        junit=(
            '<testcase classname="pkg.test_mod" name="test_perf" time="1.25">'
            '<properties><property name="nodeid" value="pkg/test_mod.py::test_perf" /></properties>'
            "</testcase>"
        ),
    )

    deselect = plugin._build_session_deselect_args(
        {
            "tests": {
                "good": good.to_dict(),
                "no_junit": no_junit.to_dict(),
                "wrong_profile": wrong_profile.to_dict(),
            }
        }
    )

    assert deselect == ["pkg/test_mod.py::test_good"]
    assert list(plugin._session_cached_entries) == ["pkg/test_mod.py::test_good"]
