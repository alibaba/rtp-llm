import os
from pathlib import Path

import pytest

from rtp_llm.test.smoke_framework import runner


@pytest.mark.parametrize("fail", [False, True])
def test_native_smoke_owns_temporary_directory(monkeypatch, fail):
    monkeypatch.delenv("TEST_TMPDIR", raising=False)
    used = []

    def run_case(name, config):
        directory = Path(os.environ["TEST_TMPDIR"])
        assert directory.is_dir()
        (directory / "deep_gemm_pd_cache").mkdir()
        used.append(directory)
        if fail:
            raise RuntimeError("case failed")

    monkeypatch.setattr(runner, "_run_smoke_test", run_case)
    if fail:
        with pytest.raises(RuntimeError, match="case failed"):
            runner.run_smoke_test("case", {})
    else:
        runner.run_smoke_test("case", {})
    assert "TEST_TMPDIR" not in os.environ
    assert len(used) == 1 and not used[0].exists()


def test_native_smoke_preserves_caller_temporary_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_TMPDIR", str(tmp_path))

    def run_case(name, config):
        assert Path(os.environ["TEST_TMPDIR"]) == tmp_path
        (tmp_path / "caller-cache").write_text("preserved")

    monkeypatch.setattr(runner, "_run_smoke_test", run_case)
    runner.run_smoke_test("case", {})
    assert os.environ["TEST_TMPDIR"] == str(tmp_path)
    assert (tmp_path / "caller-cache").read_text() == "preserved"
