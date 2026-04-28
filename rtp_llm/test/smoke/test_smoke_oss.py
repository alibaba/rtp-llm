"""Pytest entry for OSS (open-source) smoke tests.

Independent from internal_source — drives smoke_defs_oss.py via the
case_runner / multi_inst_case_runner framework.

Run:
    pytest rtp_llm/test/smoke/test_smoke_oss.py --rtp-ci-profile=smoke_h20_oss
    pytest rtp_llm/test/smoke/test_smoke_oss.py -k "mla_fp8"
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Type

logging.basicConfig(
    level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Add the parent of "smoke/" to sys.path so the runner modules import as `smoke.*`.
smoke_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).parent)
if smoke_dir not in sys.path:
    sys.path.insert(0, smoke_dir)

# Tell common_def to resolve REL_PATH against the OSS data tree
# (rtp_llm/test/smoke/data/) even when internal_source/ is also present.
# Must be set BEFORE the first `from smoke.common_def import ...`.
os.environ.setdefault("SMOKE_REL_PATH_PREFER", "oss")

import pytest  # noqa: E402
from smoke.case_runner import CaseRunner  # noqa: E402
from smoke.common_def import REL_PATH  # noqa: E402
from smoke.multi_inst_case_runner import (  # noqa: E402
    DpSeperationCaseRunner,
    FrontAppSeperationCaseRunner,
    PdSeperationCaseRunner,
    VitSeperationCaseRunner,
)
from smoke.smoke_defs_oss import build_smoke_params, get_gpu_count  # noqa: E402
from smoke.task_info import TaskInfo  # noqa: E402
from smoke.utils import resolve_prompt_refs  # noqa: E402


def check_use_prompt_batch(task_info: TaskInfo) -> bool:
    for query_result in task_info.query_result:
        if query_result.get("query", {}).get("prompt_batch", False):
            return True
    return False


def get_runner_type(smoke_args, envs) -> Type[CaseRunner]:
    """Determine runner type from smoke_args and envs structure."""
    if isinstance(smoke_args, dict):
        if "prefill" in smoke_args:
            if "--role_type DECODE" in smoke_args.get(
                "prefill", ""
            ) and "DECODE_ENTRANCE=1" in str(envs):
                return DpSeperationCaseRunner
            return PdSeperationCaseRunner
        elif "frontend" in smoke_args:
            return FrontAppSeperationCaseRunner
        elif "vit" in smoke_args:
            return VitSeperationCaseRunner
    if isinstance(envs, dict):
        if "prefill" in envs:
            return PdSeperationCaseRunner
        elif "frontend" in envs:
            return FrontAppSeperationCaseRunner
        elif "vit" in envs:
            return VitSeperationCaseRunner
    return CaseRunner


def _build_env_args(smoke_args, envs):
    """Build env_args (list or dict) for CaseRunner from smoke_args + envs."""
    from smoke.smoke_defs_oss import _parse_world_size

    if isinstance(smoke_args, dict):
        env_args: Dict[str, list] = {}
        envs_dict = envs if isinstance(envs, dict) else {}
        for role, args_str in smoke_args.items():
            role_envs = list(envs_dict.get(role, []))
            ws = _parse_world_size(args_str)
            role_envs.append(f"WORLD_SIZE={ws}")
            role_envs.append("DETERMINISTIC_GEMM=1")
            env_args[role] = role_envs
        return env_args
    env_list = list(envs) if isinstance(envs, list) else []
    ws = _parse_world_size(smoke_args)
    env_list.append(f"WORLD_SIZE={ws}")
    env_list.append("ENABLE_STABLE_SCATTER_ADD=ON")
    env_list.append("DETERMINISTIC_GEMM=1")
    return env_list


def run_smoke_test(test_name: str, test_config: dict):
    smoke_args = test_config.get("smoke_args", "")
    envs = test_config.get("envs", [])
    task_info_path = test_config["task_info"]
    gpu_card = test_config["gpu_type"]

    env_args = _build_env_args(smoke_args, envs)

    _env_keys_set: list = []
    if isinstance(env_args, list):
        for env_str in env_args:
            if "=" in env_str:
                key, value = env_str.split("=", 1)
                _env_keys_set.append((key, os.environ.get(key)))
                os.environ[key] = value
    elif isinstance(env_args, dict):
        for role_envs in env_args.values():
            for env_str in role_envs:
                if "=" in env_str:
                    key, value = env_str.split("=", 1)
                    _env_keys_set.append((key, os.environ.get(key)))
                    os.environ[key] = value

    gpu_count = get_gpu_count(test_config)
    for k in ("GPU_COUNT", "WORLD_SIZE"):
        if k not in os.environ:
            _env_keys_set.append((k, None))
    os.environ.setdefault("GPU_COUNT", str(gpu_count))
    os.environ.setdefault("WORLD_SIZE", str(gpu_count))

    logging.info("cwd: %s test_name: %s envs: %s", os.getcwd(), test_name, env_args)

    with open(os.path.join(REL_PATH, task_info_path), "r") as f:
        x = json.load(f)
    # Resolve $prompt:xxx refs against rtp_llm/test/smoke/data/prompt_candidates.json
    # (entry.py does the same — pytest entry must too, otherwise the literal
    # "$prompt:s2" gets sent to the server instead of the real prompt text).
    if "query_result" in x:
        x["query_result"] = [resolve_prompt_refs(qr) for qr in x["query_result"]]
    task_info = TaskInfo(
        **x, taskinfo_rel_path=os.path.join(REL_PATH, task_info_path)
    )

    runner_class = get_runner_type(smoke_args, envs)
    logging.info("runner_class: %s", str(runner_class))

    runner_params: Dict[str, Any] = {
        "task_info": task_info,
        "env_args": env_args,
        "gpu_card": gpu_card,
        "smoke_args": smoke_args,
    }

    for param in ["sleep_time_qr", "kill_remote", "concurrency_test"]:
        if param in test_config:
            runner_params[param] = test_config[param]

    if check_use_prompt_batch(task_info) and isinstance(env_args, list):
        env_args.append("USE_GATHER_BATCH_SCHEDULER=1")
        runner_params["batch_infer"] = True
        logging.info("use gather batch scheduler")

    runner = runner_class(**runner_params)
    try:
        task_states = runner.run()
        logging.info("raw info: %s", str(task_states))
        assert task_states.ret is True, f"smoke task run failed: {test_name}"
    finally:
        for key, old_val in reversed(_env_keys_set):
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val


_test_params = build_smoke_params(pytest)


# 7200s matches the --remote --timeout=7200 ceiling; a tighter local mark
# kills cases that the remote worker is still actively executing (mistakenly
# reporting them as FAILED while remote later returns PASS).
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_oss(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
