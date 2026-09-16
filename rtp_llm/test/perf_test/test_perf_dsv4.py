"""Mainline DSV4 workloads, executed by the native perf framework."""

import json
from pathlib import Path

import pytest

from rtp_llm.test.perf_test.perf_runner import build_perf_params, run_perf_test


_ROOT = Path(__file__).resolve().parent
_PRESETS = json.loads((_ROOT / "perf_presets.json").read_text())
PERF_TESTS = {
    name: {
        "argv": preset["args"],
        "envs": preset["env"],
        "gpu_type": "L20D_TEST",
        "gpu_count": int(preset["env"]["WORLD_SIZE"]),
        "markers": ["perf", "cuda"],
    }
    for name, preset in _PRESETS.items()
    if name.startswith("v4_")
}


@pytest.mark.timeout(14400)
@pytest.mark.parametrize("test_name,test_config", build_perf_params(pytest, PERF_TESTS))
def test_perf_dsv4(test_name, test_config):
    from rtp_llm.test.cuda13_preflight import prepare_cuda13_runtime

    prepare_cuda13_runtime(test_config["gpu_count"])
    run_perf_test(test_name, test_config, _ROOT)
