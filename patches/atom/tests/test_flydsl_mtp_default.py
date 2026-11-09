"""RTP must consume the existing ATOM ReplaySSM environment setting."""
import os
from unittest.mock import patch
import torch
from atom.utils import envs
import importlib.metadata
import importlib.util

# Load only the policy module; importing the RTP utils package initializes GPU kernels.
source = importlib.metadata.distribution("atom").locate_file(
    "atom/plugin/rtpllm/utils/replay_policy.py")
spec = importlib.util.spec_from_file_location("replay_policy_test", source)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)
replay_mode = policy.replay_mode
flydsl_mtp_enabled = policy.flydsl_mtp_enabled


def test_default_policy():
    for hip, value, override, mode in [
        ("7", None, None, "1"), ("7", "1", True, "1"),
        ("7", "0", False, "0"), (None, None, None, "0"),
    ]:
        old = {"ATOM_RTP_ENABLE_REPLAYSSM": "legacy",
               "ATOM_RTP_REPLAY_HYBRID": "0", "ATOM_RTP_FLYDSL_GDN_MTP": "0"}
        if value is not None:
            old["ATOM_ENABLE_REPLAYSSM"] = value
        with patch.dict(os.environ, old, clear=True), patch.object(torch.version, "hip", hip):
            assert envs.ATOM_ENABLE_REPLAYSSM is override
            assert replay_mode() == mode
            assert flydsl_mtp_enabled() == bool(hip)


if __name__ == "__main__":
    test_default_policy()
    print("Native ATOM ReplaySSM policy PASS")
