"""Evaluate priority/time and batch weighting with the real Prometheus query engine."""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from mock_framework.execution.step import _metrics
from mock_framework.execution.test.helpers import FRAMEWORK


@pytest.mark.skipif(
    not os.environ.get("PROMETHEUS_BIN"), reason="Real Prometheus/promtool required"
)
def test_tps_queries_preserve_sparse_priorities_and_unequal_batch_weights(
    tmp_path, monkeypatch
):
    tool = Path(os.environ["PROMETHEUS_BIN"]).with_name("promtool")
    if not tool.is_file():
        pytest.skip("promtool required")
    expressions = []

    async def instant(ctx, expression, at):
        expressions.append(expression)
        if expression.startswith("count_over_time"):
            return [_metrics.QuerySample({"instance": i}, 8, at) for i in ("a", "b")]
        return [_metrics.QuerySample({}, 0, at)]

    monkeypatch.setattr(_metrics, "instant", instant)
    ctx = SimpleNamespace(
        query_time="1970-01-01T00:00:17.500Z",
        runtime=SimpleNamespace(
            monitoring=SimpleNamespace(interval=1),
            topology=SimpleNamespace(
                active=lambda role: [{"rpc": i} for i in ("a", "b")]
            ),
        ),
        records=SimpleNamespace(emit=lambda **event: None),
    )
    asyncio.run(
        _metrics.gauge(
            ctx,
            _metrics.MetricArgs(range=["-8s", "0s"]),
            "prefill",
            "rtp_llm_context_tps",
        )
    )
    queries = [
        {
            "expr": expressions[-1],
            "eval_time": "17500ms",
            "exp_samples": [{"labels": "{}", "value": 25}],
        }
    ]
    dashboard = json.loads((FRAMEWORK / "views/default.json").read_text())
    panel = next(
        p for row in dashboard["panels"] for p in row["panels"] if p["id"] == 52
    )
    expression = (
        panel["targets"][0]["expr"]
        .replace("${prefill_group}", "role")
        .replace("${prefill_instance}", ".*")
        .replace("$__rate_interval", "8s")
    )
    queries.append(
        {
            "expr": expression,
            "eval_time": "17500ms",
            "exp_samples": [{"labels": '{role="prefill"}', "value": 2000 / 101}],
        }
    )
    inputs = [
        {
            "series": 'rtp_llm_context_tps{role="prefill",instance="a",priority="0"}',
            "values": ("10 stale " * 16).strip(),
        },
        {
            "series": 'rtp_llm_context_tps{role="prefill",instance="a",priority="7"}',
            "values": ("stale 30 " * 16).strip(),
        },
        {
            "series": 'rtp_llm_context_tps{role="prefill",instance="b",priority="0"}',
            "values": "5+0x31",
        },
        {
            "series": 'rtp_llm_forward_tokens_total{role="prefill",instance="a",kind="input"}',
            "values": "0+1000x31",
        },
        {
            "series": 'rtp_llm_forward_tokens_total{role="prefill",instance="b",kind="input"}',
            "values": "0+1000x31",
        },
        {
            "series": 'rtp_llm_model_forward_seconds_count{role="prefill",instance="a"}',
            "values": "0+100x31",
        },
        {
            "series": 'rtp_llm_model_forward_seconds_count{role="prefill",instance="b"}',
            "values": "0+1x31",
        },
    ]
    spec = {
        "evaluation_interval": "1s",
        "tests": [
            {"interval": "1s", "input_series": inputs, "promql_expr_test": queries}
        ],
    }
    path = tmp_path / "tps.yaml"
    path.write_text(yaml.safe_dump(spec))
    result = subprocess.run(
        [str(tool), "test", "rules", str(path)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
