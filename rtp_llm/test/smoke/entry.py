import argparse
import glob
import json
import logging
import os
import shutil
from typing import Dict, List, Type, Union, Any

logging.basicConfig(
    level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from smoke.case_runner import CaseRunner
from smoke.common_def import REL_PATH
from smoke.utils import resolve_prompt_refs
from smoke.gpu_diagnostics import (
    ExceptionType,
    classify_exception,
    dump_gpu_state,
    snapshot_dmesg,
)
from smoke.multi_inst_case_runner import (
    DpSeperationCaseRunner,
    FrontAppSeperationCaseRunner,
    PdSeperationCaseRunner,
    VitSeperationCaseRunner,
)
from smoke.task_info import TaskInfo
from rtp_llm.utils.util import str_to_bool

def get_runner_type(
    env_args: Union[List[str], Dict[str, List[str]]]
) -> Type[CaseRunner]:
    if isinstance(env_args, list):
        return CaseRunner
    else:
        if "prefill" in env_args:
            if "DECODE_ENTRANCE=1" in env_args["prefill"]:
                return DpSeperationCaseRunner
            else:
                return PdSeperationCaseRunner
        elif "frontend" in env_args:
            return FrontAppSeperationCaseRunner
        elif "vit" in env_args:
            return VitSeperationCaseRunner
        else:
            raise Exception(f"unknow env_args for runner selection: {env_args}")


def _parse_kv_list(raw: str) -> Dict[str, str]:
    """Parse a JSON list of 'KEY=VALUE' strings into a dict."""
    items = json.loads(raw)
    result = {}
    for item in items:
        k, v = item.split("=", 1)
        result[k] = v
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="smoke runner")
    parser.add_argument("--suite_name", type=str, required=True, help="suite_name")
    parser.add_argument("--task_info", type=str, required=True, help="task_info")
    parser.add_argument("--envs", type=str, required=False, default="", help="envs")
    parser.add_argument("--smoke_args", type=str, required=False, default="", help="smoke_args")
    parser.add_argument("--gpu_card", type=str, required=True, default="", help="gpu_card")
    parser.add_argument("--kvcm_envs", type=str, required=False, default="[]", help="KVCM server config")
    parser.add_argument("--sleep_time_qr", type=int, default=0, help="sleep seconds between queries")
    parser.add_argument("--kill_remote", type=str, default="False", help="kill KVCM server mid-test")
    parser.add_argument("--concurrency_test", type=str, default="False", help="concurrent request mode")
    args, _ = parser.parse_known_args()

    logging.info(
        "cwd: %s envs: %s, smoke_args: %s", os.getcwd(), args.envs, args.smoke_args
    )
    with open(os.path.join(REL_PATH, args.task_info), "r") as f:
        try:
            import json5
            x = json5.load(f)
        except ImportError:
            x = json.load(f)
        if "query_result" in x:
            x["query_result"] = [resolve_prompt_refs(qr) for qr in x["query_result"]]
        task_info = TaskInfo(
            **x, taskinfo_rel_path=os.path.join(REL_PATH, args.task_info)
        )

    env_args = json.loads(args.envs)
    smoke_args = args.smoke_args
    if smoke_args:
        smoke_args_clean = smoke_args.strip().strip("'\"")
        if smoke_args_clean.startswith("{") and smoke_args_clean.endswith("}"):
            try:
                smoke_args = json.loads(smoke_args_clean)
            except json.JSONDecodeError:
                pass
        elif smoke_args.startswith('"') and smoke_args.endswith('"'):
            smoke_args = smoke_args[1:-1]

    kvcm_config = _parse_kv_list(args.kvcm_envs)

    runner_class = get_runner_type(env_args)
    logging.info("runner_class: %s", str(runner_class))
    runner_params: Dict[str, Any] = {
        "task_info": task_info,
        "env_args": env_args,
        "gpu_card": args.gpu_card,
        "smoke_args": smoke_args,
        "kvcm_config": kvcm_config,
        "sleep_time_qr": args.sleep_time_qr,
        "kill_remote": str_to_bool(args.kill_remote),
        "concurrency_test": str_to_bool(args.concurrency_test),
    }
    # prompt_batch queries are now routed to /batch_infer per-query in case_runner
    # (see CaseRunner._resolve_endpoint), no env-level switch needed.

    runner = runner_class(**runner_params)

    dmesg_baseline = snapshot_dmesg()

    try:
        task_states = runner.run()
    except Exception as e:
        exc_type = classify_exception(e)
        if exc_type != ExceptionType.NOT_GPU_ERROR:
            output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())
            dump_gpu_state(
                exc=e,
                failure_context=f"runner exception ({exc_type.value})",
                log_path=os.path.join(output_dir, "gpu_state_runner_crash.log"),
                dmesg_baseline=dmesg_baseline,
            )
        raise
    finally:
        output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())
        rocm_debug_dir = os.path.join(output_dir, "rocm_debug")
        for pattern in ["/tmp/rocm_debug_agent*", "/tmp/rocm_code_objects/*", "/tmp/gpucore.*"]:
            for src in glob.glob(pattern):
                os.makedirs(rocm_debug_dir, exist_ok=True)
                dst = os.path.join(rocm_debug_dir, os.path.basename(src))
                try:
                    shutil.copy2(src, dst)
                    logging.info("copied rocm debug artifact: %s -> %s", src, dst)
                except Exception:
                    pass

    logging.info("raw info: %s", str(task_states))
    assert task_states.ret == True, f"smoke task run failed\n{task_states}"
