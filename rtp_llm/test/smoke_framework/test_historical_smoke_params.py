"""CPU checks of actual pytest parameters and runner environment forwarding.

These tests do not start a server or prove GPU chunking/remote cache execution.
"""

import importlib
import shlex

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params, get_gpu_count
from rtp_llm.test.smoke_framework.runner import _build_env_args


def suite(name):
    return importlib.import_module("rtp_llm.test.smoke.suites.test_smoke_" + name)


def options(arguments):
    tokens = shlex.split(arguments)
    assert len(tokens) % 2 == 0
    names = tokens[::2]
    assert len(names) == len(set(names)), "ambiguous duplicate startup option"
    return dict(zip(names, tokens[1::2]))


@pytest.mark.parametrize(
    "name,case,gpus",
    [
        ("h20_dense", "dense_fp8pb_dynamic", 1),
        ("h20_mla", "mla_glm4_moe_lite", 1),
        ("h20_eagle", "eagle_mtp_tp2", 2),
    ],
)
def test_chunked_parameter_reaches_pytest_and_runner_environment(name, case, gpus):
    module = suite(name)
    config = module.SMOKE_CASES[case]
    params = build_smoke_params(pytest, {module.SUITE_NAME: {case: config}}, {})
    assert len(params) == 1
    assert params[0].values == (case, config)
    args = options(params[0].values[1]["smoke_args"])
    assert args["--prefill_chunk_size"] == "64"
    assert args["--seq_size_per_block"] == "64"
    if case == "eagle_mtp_tp2":
        assert args["--max_seq_len"] == "320"
    assert get_gpu_count(config) == gpus
    assert f"WORLD_SIZE={gpus}" in _build_env_args(config["smoke_args"], [])


@pytest.mark.parametrize(
    "name,case",
    [
        *(
            ("cuda_remote_cache", n)
            for n in (
                "remote_cache_basic",
                "remote_cache_basic_async",
                "remote_cache_kill",
                "remote_cache_tp2",
                "remote_cache_pd",
                "remote_cache_match_fail",
                "remote_cache_write_start_fail",
                "remote_cache_write_finish_fail",
                "remote_cache_edge",
            )
        ),
        ("h20_eagle", "eagle_remote_cache_tp2"),
        ("h20_next", "next_long_reuse_remote"),
    ],
)
def test_remote_upload_roles_keep_device_store_and_fault_environment(name, case):
    config = suite(name).SMOKE_CASES[case]
    arguments = config["smoke_args"]
    roles = arguments if isinstance(arguments, dict) else {"single": arguments}
    envs = _build_env_args(arguments, config.get("envs", []))
    for role, text in roles.items():
        # Eagle's historical duplicated --warm_up is unrelated and retained.
        if case == "eagle_remote_cache_tp2":
            text = text.replace("--warm_up 0 ", "", 1)
        args = options(text)
        for key, expected in {
            "--enable_device_cache": "1",
            "--enable_memory_cache": "0",
            "--enable_disk_cache": "0",
            "--test_block_num": "500",
            "--block_tree_device_evict_low_watermark_ratio": "0.001",
            "--block_tree_device_evict_high_watermark_ratio": "0.002",
        }.items():
            assert args[key] == expected
        assert not any(
            k.startswith("--reco_") or k == "--write_cache_sync" for k in args
        )
        role_envs = envs[role] if isinstance(envs, dict) else envs
        assert "KVCM_LOG_LEVEL=DEBUG" in role_envs
        for env in config.get("envs", []):
            assert env in role_envs
    assert config["sleep_time_qr"] >= 10
