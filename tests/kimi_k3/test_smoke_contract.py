"""Check the acceptance harness fails closed without running model inference."""

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "example/k3/main_migration"


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


long_prefix = load("long_prefix_case")
smoke = load("text_smoke")


def cli(tmp_path, monkeypatch, *, layers=93, extra=()):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"text_config": {"num_hidden_layers": layers}})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "text_smoke.py",
            "--base-url",
            "http://127.0.0.1:9000",
            "--decode-health-url",
            "http://127.0.0.1:9001/health",
            "--decode-role-addr",
            "127.0.0.1:9001:9002",
            "--namespace",
            "unit-test",
            "--output",
            str(tmp_path / "run/result.json"),
            "--block-size",
            "4096",
            "--long-prefix-checkpoint",
            str(checkpoint),
            "--require-mtp",
            *extra,
        ],
    )
    return smoke.parse_args()


def test_ordinary_layout_is_not_tp_times_block(tmp_path, monkeypatch):
    args = cli(tmp_path, monkeypatch)
    assert args.suite == "main-text"
    assert args.decode_dp_size == 1
    assert smoke.Runner(args).reuse_unit_tokens == 4096
    assert smoke.cache_block_boundaries(4096, 4096, 65536) == (
        4096,
        8192,
        65536,
        131072,
    )


@pytest.mark.parametrize(
    "extra",
    [
        ["--reuse-unit-tokens", "32768"],
        ["--long-prefix-target-tokens", "70000"],
        ["--chunk-tokens", "32768"],
        ["--decode-dp-size", "2"],
        ["--suite", "all"],
    ],
)
def test_reject_weakened_or_incompatible_full_profile(tmp_path, monkeypatch, extra):
    with pytest.raises(SystemExit):
        cli(tmp_path, monkeypatch, extra=extra)


def test_four_layers_cannot_claim_full_smoke(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        cli(tmp_path, monkeypatch, layers=4)


@pytest.mark.parametrize(
    "answer",
    [
        "```json\n" + json.dumps(long_prefix.EXPECTED) + "\n```",
        json.dumps(long_prefix.EXPECTED).replace(
            '"square": 1369', '"square": 1369, "square": 1369'
        ),
        json.dumps({**long_prefix.EXPECTED, "square": "1369"}),
        json.dumps({**long_prefix.EXPECTED, "unexpected": 1}),
    ],
)
def test_long_prefix_requires_strict_json(answer):
    with pytest.raises(ValueError):
        long_prefix.check_answer(answer)


def test_long_prefix_answer_and_checkpoint_frontier():
    long_prefix.check_answer(json.dumps(long_prefix.EXPECTED))
    long_prefix.check_prefix_reuse(106496, 108000, 4096)
    with pytest.raises(ValueError):
        long_prefix.check_prefix_reuse(110592, 108000, 4096)
    with pytest.raises(ValueError):
        long_prefix.check_prefix_reuse(4096, 108000, 4096)


class FailingTransport:
    def __init__(self):
        self.calls = 0

    def open(self, request, timeout):
        self.calls += 1
        raise HTTPError(
            request.full_url, 503, "unavailable", {}, io.BytesIO(b"raw-failure")
        )


def test_formal_http_failure_is_saved_and_never_retried(tmp_path, monkeypatch):
    args = cli(tmp_path, monkeypatch)
    runner = smoke.Runner(args)
    runner.opener = FailingTransport()
    with pytest.raises(smoke.TransportFailure):
        runner.request(smoke.Case("formal-failure", "exact input", "answer", "miss"))
    assert runner.opener.calls == 1
    artifacts = list((args.output.parent / "requests").glob("*.json"))
    assert len(artifacts) == 1
    audit = json.loads(artifacts[0].read_text())
    assert audit["request"]["messages"][0]["content"] == "exact input"
    assert audit["response_body"] == "raw-failure"
    assert audit["phase"] == "formal"
    assert audit["passed"] is False
    runner.save(False, "transport failure")
    result = json.loads(args.output.read_text())
    assert "PageRR owner" in result["not_applicable"]
    assert result["formal_request_retries"] == 0
    assert result["passed"] is False


def test_long_prefix_http_failure_retains_wire_response(tmp_path):
    case = long_prefix.LongPrefixCase(
        "http://127.0.0.1:9000",
        tmp_path,
        "test",
        timeout=1,
        budget=1024,
        page_size=64,
        bytes_per_token=8,
    )
    case.opener = FailingTransport()
    with pytest.raises(HTTPError):
        case.request(
            "failed-seed", [{"role": "user", "content": "exact input"}], [1, 2]
        )
    assert case.opener.calls == 1
    assert (tmp_path / "failed-seed-response.raw").read_bytes() == b"raw-failure"
    assert (
        json.loads((tmp_path / "failed-seed-http-error.json").read_text())["status"]
        == 503
    )
