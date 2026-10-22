"""Captured child-output contracts; no Native module or compiler is executed."""

import importlib.util
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_setup_module():
    spec = importlib.util.spec_from_file_location(
        "_rtp_setup_stub_observability_test", PROJECT_ROOT / "setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, {"RTP_BAZEL_CONFIG": "--config=cuda13_arm"}):
        spec.loader.exec_module(module)
    return module


def _child(returncode=0, stderr=""):
    return SimpleNamespace(returncode=returncode, stderr=stderr)


def test_skipped_driver_stderr_keeps_exact_soname_traceback_and_exit(tmp_path, capsys):
    module = _load_setup_module()
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "<string>", line 1, in <module>\n'
        "ImportError: libcuda.so.1: cannot open shared object file: "
        "No such file or directory\n"
    )
    with patch.object(
        module.subprocess, "run", side_effect=[_child(), _child(23, stderr), _child()]
    ) as run:
        module.generate_pyi_stubs(tmp_path)
    output = capsys.readouterr().out
    assert stderr in output
    assert "SKIP: pybind11_stubgen for libth_transformer" in output
    assert "--- skipped stderr (full, exit=23) ---" in output
    assert "--- end skipped stderr ---" in output
    assert "WARNING: pybind11_stubgen skipped for 1 module(s)" in output
    assert run.call_count == 3
    for call, name in zip(
        run.call_args_list,
        ("libth_transformer_config", "libth_transformer", "librtp_compute_ops"),
    ):
        assert name in call.args[0][-1]
        assert call.kwargs["timeout"] == 60
        assert call.kwargs["capture_output"] is True
        assert call.kwargs["text"] is True


def test_skipped_project_soname_is_reported_without_reclassification(tmp_path, capsys):
    module = _load_setup_module()
    stderr = (
        "ImportError: kv_cache_manager_client.so: cannot open shared object file: "
        "No such file or directory\n"
    )
    with patch.object(
        module.subprocess, "run", side_effect=[_child(), _child(1, stderr), _child()]
    ):
        module.generate_pyi_stubs(tmp_path)
    output = capsys.readouterr().out
    assert stderr in output
    assert "SKIP: pybind11_stubgen for libth_transformer" in output
    assert "--- skipped stderr (full, exit=1) ---" in output
    assert "--- stderr (full) ---" not in output


def test_skipped_stderr_cannot_mask_other_module_failure(tmp_path, capsys):
    module = _load_setup_module()
    skipped = "ImportError: libcuda.so.1: cannot open shared object file: No such file or directory\n"
    failed = "ImportError: librtp_compute_ops.so: undefined symbol: actual_symbol\n"
    with patch.object(
        module.subprocess,
        "run",
        side_effect=[_child(), _child(1, skipped), _child(7, failed)],
    ):
        with pytest.raises(
            RuntimeError, match=r"failed for 1 module\(s\): librtp_compute_ops"
        ):
            module.generate_pyi_stubs(tmp_path)
    output = capsys.readouterr().out
    assert skipped in output and failed in output
    assert "failed for librtp_compute_ops (exit=7)" in output
    assert "WARNING: pybind11_stubgen skipped" not in output


def test_unclassified_failure_keeps_existing_failure_path(tmp_path, capsys):
    module = _load_setup_module()
    stderr = "ImportError: libth_transformer.so: undefined symbol: actual_symbol\n"
    with patch.object(
        module.subprocess, "run", side_effect=[_child(), _child(4, stderr), _child()]
    ):
        with pytest.raises(
            RuntimeError, match=r"failed for 1 module\(s\): libth_transformer"
        ):
            module.generate_pyi_stubs(tmp_path)
    output = capsys.readouterr().out
    assert stderr in output
    assert "--- skipped stderr" not in output
    assert "failed for libth_transformer (exit=4)" in output


def test_successful_children_keep_success_output_without_stderr_dump(tmp_path, capsys):
    module = _load_setup_module()
    with patch.object(module.subprocess, "run", return_value=_child()) as run:
        module.generate_pyi_stubs(tmp_path)
    output = capsys.readouterr().out
    assert run.call_count == 3
    assert output.count("Generated .pyi for ") == 3
    assert "stderr" not in output


def test_timeout_is_still_failure_and_later_children_are_not_dropped(tmp_path, capsys):
    module = _load_setup_module()
    timeout = subprocess.TimeoutExpired(["existing-stub-child"], 60)
    with patch.object(
        module.subprocess, "run", side_effect=[_child(), timeout, _child()]
    ) as run:
        with pytest.raises(
            RuntimeError, match=r"failed for 1 module\(s\): libth_transformer"
        ):
            module.generate_pyi_stubs(tmp_path)
    output = capsys.readouterr().out
    assert run.call_count == 3
    assert "could not run for libth_transformer" in output
    assert "--- skipped stderr" not in output
