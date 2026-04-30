"""Smoke test runner — single canonical implementation used by both OSS and internal entries.

Replaces the byte-for-byte duplicated `run_smoke_test` / `get_runner_type` /
`_build_env_args` / `check_use_prompt_batch` blocks in `test_smoke_oss.py` and
`test_smoke_internal.py`. Internal smoke entries reduce to ~30 lines (data + parametrize).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Mapping, Type, Union

from smoke.case_runner import CaseRunner
from smoke.common_def import REL_PATH
from smoke.multi_inst_case_runner import (
    DpSeperationCaseRunner,
    FrontAppSeperationCaseRunner,
    PdSeperationCaseRunner,
    VitSeperationCaseRunner,
)
from smoke.task_info import TaskInfo
from smoke.utils import resolve_prompt_refs

from rtp_llm.test.smoke_framework.manifest import _parse_world_size, get_gpu_count


def check_use_prompt_batch(task_info: TaskInfo) -> bool:
    for query_result in task_info.query_result:
        if query_result.get("query", {}).get("prompt_batch", False):
            return True
    return False


def get_runner_type(
    smoke_args: Union[str, Mapping[str, str]],
    envs: Union[List[str], Mapping[str, List[str]]],
) -> Type[CaseRunner]:
    """Determine runner class from smoke_args / envs structure (multi-role aware)."""
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


def _build_env_args(
    smoke_args: Union[str, Mapping[str, str]],
    envs: Union[List[str], Mapping[str, List[str]]],
):
    """Build env_args (list or dict) for CaseRunner.

    Single-role: returns flat list of "KEY=VAL" strings.
    Multi-role: returns dict {role: [env strings]} — each role's WORLD_SIZE comes
    from its own smoke_args.
    """
    if isinstance(smoke_args, dict):
        env_args: Dict[str, List[str]] = {}
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


def run_smoke_test(test_name: str, test_config: Mapping[str, Any]) -> None:
    """Drive a single smoke test case end-to-end.

    1. Build env_args (single-role list or multi-role dict).
    2. Inject single-role envs into the parent process — multi-role envs reach
       per-role subprocesses via MagaServerManager(env_args=...) (don't pollute
       parent: see PR4 / A5 in the plan).
    3. Restore parent env in `finally` so cross-case state doesn't leak.
    """
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

    gpu_count = get_gpu_count(test_config)
    for k in ("GPU_COUNT", "WORLD_SIZE"):
        if k not in os.environ:
            _env_keys_set.append((k, None))
    os.environ.setdefault("GPU_COUNT", str(gpu_count))
    os.environ.setdefault("WORLD_SIZE", str(gpu_count))

    logging.info(
        "cwd: %s test_name: %s envs: %s", os.getcwd(), test_name, env_args
    )

    with open(os.path.join(REL_PATH, task_info_path), "r") as f:
        x = json.load(f)
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

    for param in ("sleep_time_qr", "kill_remote", "concurrency_test"):
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
