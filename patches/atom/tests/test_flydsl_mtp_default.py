"""CPU-only unified ReplaySSM policy regression against the patched wheel."""
import importlib.metadata
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch


def test_default_policy():
    source = Path(importlib.metadata.distribution("atom").locate_file(
        "atom/plugin/rtpllm/utils/replay_policy.py"))
    torch = types.SimpleNamespace(version=types.SimpleNamespace(hip="7.0"))
    with patch.dict(sys.modules, {"torch": torch}):
        spec = importlib.util.spec_from_file_location("policy_test", source)
        policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy)
    for hip, value, expected, flydsl in [
        ("7", None, "1", True), ("7", "1", "1", True),
        ("7", "0", "0", True), ("7", "legacy", "legacy", False),
        (None, None, "0", False),
    ]:
        torch.version.hip = hip
        env = {"ATOM_RTP_REPLAY_HYBRID": "1", "ATOM_RTP_FLYDSL_GDN_MTP": "0"}
        if value is not None:
            env["ATOM_RTP_ENABLE_REPLAYSSM"] = value
        with patch.dict(os.environ, env, clear=True):
            assert policy.replay_mode() == expected
            assert policy.flydsl_mtp_enabled() == flydsl
    with patch.dict(os.environ, {"ATOM_RTP_ENABLE_REPLAYSSM": "invalid"}, clear=True):
        try:
            policy.replay_mode()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid replay mode was accepted")


if __name__ == "__main__":
    test_default_policy()
    print("Unified ReplaySSM policy PASS")
