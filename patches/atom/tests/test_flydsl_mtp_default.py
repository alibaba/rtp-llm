"""CPU-only policy regression against the patched ATOM wheel source."""
import ast
import importlib.metadata
import os
import types
from pathlib import Path
from unittest.mock import patch


def test_default_policy():
    source = Path(importlib.metadata.distribution("atom").locate_file(
        "atom/plugin/rtpllm/utils/qwen4_exp_context.py"))
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_flydsl_mtp_enabled")
    torch = types.SimpleNamespace(version=types.SimpleNamespace(hip=None))
    scope = {"os": os, "torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    for hip, flag, replay, expected in [
        ("7", None, None, True), ("7", "0", None, False),
        ("7", None, "1", False), (None, None, None, False),
        ("7", "1", "0", True), ("7", "1", "1", "error"),
    ]:
        torch.version.hip = hip
        env = {}
        if flag is not None:
            env["ATOM_RTP_FLYDSL_GDN_MTP"] = flag
        if replay is not None:
            env["ATOM_RTP_ENABLE_REPLAYSSM"] = replay
        with patch.dict(os.environ, env, clear=True):
            try:
                actual = scope["_flydsl_mtp_enabled"]()
            except ValueError:
                actual = "error"
        assert actual == expected, (hip, flag, replay, actual)


if __name__ == "__main__":
    test_default_policy()
    print("ROCm default policy PASS")
