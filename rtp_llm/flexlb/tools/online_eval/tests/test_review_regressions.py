"""Regression coverage for confirmed PR review findings."""
from pathlib import Path

import pytest

from analysis.statistics import percentile_nr as report_percentile
from flexlb_cfg import ConfigOverride, _retype_ordering, render_env

ROOT = Path(__file__).resolve().parents[1]


def test_redundant_priority_override_preserves_existing_policy():
    ordering = {"type": "PRIORITY", "defaultPriority": 75,
                "preemption": {"allowedVictimStages": ["DECODE_ENGINE_OWNED"],
                               "timeoutMs": 900}}
    doc = {"scheduler": {"ordering": ordering.copy()}}
    _retype_ordering(doc, ConfigOverride(ordering="priority"))
    assert doc["scheduler"]["ordering"] == ordering
    _retype_ordering(doc, ConfigOverride(ordering="priority", default_priority=20,
                                       strip_preemption=True))
    assert doc["scheduler"]["ordering"] == {"type": "PRIORITY", "defaultPriority": 20}


@pytest.mark.parametrize("value", [-1, 0.5, float("nan"), float("inf"), True, "2"])
def test_invalid_decision_lifetime_rejected_before_rendering(value):
    with pytest.raises(ValueError, match="decision_lifetime"):
        render_env("stress-na130", ConfigOverride(decision_lifetime=value))


@pytest.mark.parametrize("values,p,expected", [([10, 20], .5, 10),
    (list(range(1, 101)), .95, 95), ([], .99, None)])
def test_percentiles_agree_at_rank_boundaries(values, p, expected):
    assert report_percentile(values, p) == expected
