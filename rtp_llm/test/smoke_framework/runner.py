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

from rtp_llm.test.smoke.case_runner import CaseRunner
from rtp_llm.test.smoke import common_def
from rtp_llm.test.smoke.multi_inst_case_runner import (
    DpSeperationCaseRunner,
    FrontAppSeperationCaseRunner,
    PdSeperationCaseRunner,
    VitSeperationCaseRunner,
)
from rtp_llm.test.smoke.task_info import TaskInfo
from rtp_llm.test.smoke.utils import resolve_prompt_refs
from rtp_llm.test.smoke_framework.manifest import _parse_world_size, get_gpu_count
from rtp_llm.test.smoke.rel_path_config import compute_smoke_rel_path
from rtp_llm.utils.import_util import has_internal_source


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
        # Shared env list applies to every role unless overridden per-role.
        shared_envs = list(envs) if isinstance(envs, list) else []
        envs_dict = envs if isinstance(envs, dict) else {}
        for role, args_str in smoke_args.items():
            role_envs = list(shared_envs)
            role_envs.extend(envs_dict.get(role, []))
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


def _configure_optional_internal_env(test_config: Mapping[str, Any]) -> None:
    if test_config.get("platform") != "cuda":
        return
    gpu_type = str(test_config.get("gpu_type", "")).upper()
    if "ROCM" in gpu_type or gpu_type.startswith("MI"):
        return
    if not has_internal_source():
        return
    try:
        from rtp_llm.test.util.set_internal_env import configure_optional_env
    except ImportError:
        return
    configure_optional_env()


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

    # Compute per-test REL_PATH from data_root instead of using the
    # import-time captured value.  This ensures task_info and prompt
    # references resolve against the correct data tree (oss/internal).
    data_root = test_config.get("data_root")
    if data_root is not None:
        local_rel_path = compute_smoke_rel_path(common_def.ABS_PATH, prefer=data_root)
    else:
        local_rel_path = common_def.REL_PATH

    env_args = _build_env_args(smoke_args, envs)
    _configure_optional_internal_env(test_config)

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

    logging.info("cwd: %s test_name: %s envs: %s", os.getcwd(), test_name, env_args)

    # 12 of 207 smoke task_info files are JSONC (start with `//` header comments).
    # Match legacy entry.py behavior: prefer json5 (handles comments), fall back
    # to stdlib json. Without this, parsing fails with "Expecting value: line 1
    # column 1" because stdlib json doesn't accept `//` comments.
    with open(os.path.join(local_rel_path, task_info_path), "r") as f:
        try:
            import json5

            x = json5.load(f)
        except ImportError:
            x = json.load(f)
    # Temporarily update common_def.REL_PATH so resolve_prompt_refs (and
    # any other code that reads it) uses the per-test data root.
    saved_rel_path = common_def.REL_PATH
    common_def.REL_PATH = local_rel_path
    # Clear prompt cache so it reloads from the correct data tree.
    import rtp_llm.test.smoke.utils as _smoke_utils
    _smoke_utils._PROMPT_CACHE = None
    try:
        if "query_result" in x:
            x["query_result"] = [resolve_prompt_refs(qr) for qr in x["query_result"]]
    finally:
        common_def.REL_PATH = saved_rel_path
    task_info = TaskInfo(**x, taskinfo_rel_path=os.path.join(local_rel_path, task_info_path))

    runner_class = get_runner_type(smoke_args, envs)
    logging.info("runner_class: %s", str(runner_class))

    # kvcm_config: legacy `entry.py` had a separate `--kvcm-envs` arg; under
    # pytest dispatch, parse the same KVCM-control keys out of `envs` and
    # forward to CaseRunner so RemoteKVCMServer.start_server() sees
    # ENABLE_DEBUG_SERVICE / KVCM_LOG_LEVEL / STORAGE_CONFIG /
    # INSTANCE_GROUP_CONFIG / TEST_*_FAILURE etc. Without this the kvcm spawns
    # with enable_debug_service=false and fault-injection cases
    # (remote_cache_match_fail / write_start_fail / write_finish_fail) can't
    # trigger their failure paths → COMPARE_FAILED on PR 537 run 39175306.
    kvcm_config: Dict[str, str] = {}
    if isinstance(envs, list):
        for env_str in envs:
            if "=" in env_str:
                k, v = env_str.split("=", 1)
                kvcm_config[k] = v

    runner_params: Dict[str, Any] = {
        "task_info": task_info,
        "env_args": env_args,
        "gpu_card": gpu_card,
        "smoke_args": smoke_args,
        "kvcm_config": kvcm_config,
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
