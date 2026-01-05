import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel
from tqdm import tqdm

from rtp_llm.test.perf_test.batch_perf_impl import BatchPerfImpl
from rtp_llm.test.perf_test.dataclass import (
    MetricState,
    TableType,
    create_metrics_table,
)
from rtp_llm.test.perf_test.test_util import create_query, write_odps
from rtp_llm.test.utils.maga_server_manager import MagaServerManager
from rtp_llm.utils.util import check_with_info


class RunningConfig(BaseModel):
    batch_size_list: List[int]
    input_len_list: List[int]
    input_query_dict: Dict[int, str]
    env: Dict[str, Any]
    result_dir: str
    decode_test_length: int
    is_speculative: bool
    propose_step: int = 0
    generate_config: Dict[str, Any]


def write_odps_wrapper(
    device_name: str,
    model_name: str,
    model_size: float,
    prec: str,
    dp_size: int,
    tp_size: int,
    metrics_list: List[MetricState],
):
    table_name = os.environ.get("ODPS_TABLE", "perf_test_2")
    fields = [
        "model",
        "size",
        "weight_type",
        "device",
        "framework",
        "commit",
        "batch_size",
        "seq_len",
        "context_time",
        "generate_time",
        "tp_size",
        "dp_size",
    ]
    records: List[Any] = []
    for metrics_item in metrics_list:
        metrics = metrics_item.metrics
        batch_size = metrics_item.batch_size
        input_len = metrics_item.input_len
        if metrics.success_requests != metrics.total_requests:
            logging.warning(
                f"batch {batch_size} seq {input_len} not all success, {metrics.success_requests}/{metrics.total_requests}"
            )
            continue
        records.append(
            [
                model_name,
                model_size,
                prec,
                device_name,
                "",
                "",
                batch_size,
                input_len,
                metrics.avg_prefill_time,
                metrics.avg_decode_time,
                tp_size,
                dp_size,
            ]
        )
    write_odps(table_name, records, fields)


def run_single(
    base_port: int,
    dp_size: int,
    tp_size: int,
    batch_size_list: List[int],
    input_len_list: List[int],
    input_query_dict: Dict[int, str],
    gang_config_string: Optional[str] = None,
    local_world_size: int = 0,
    request_tpot: int = 100,
    connection_timeout: int = 10,
    retry_times: int = 3,
    retry_interval: float = 0.5,
    is_decode: bool = True,
    dump_json_path: str = ".",
    decode_test_length: int = 10,
    is_speculative: bool = False,
    propose_step: int = 0,
    generate_config: Dict[str, Any] = {},
) -> List[MetricState]:
    if not local_world_size:
        local_world_size = int(
            os.environ.get("LOCAL_WORLD_SIZE", str(dp_size * tp_size))
        )
    if not gang_config_string:
        gang_config_string = os.environ.get("GANG_CONFIG_STRING", "")
    if not gang_config_string:
        gang_config_string = f"name:perf_part0,ip:127.0.0.1,port:{base_port}"

    title_prefix = f"Speculative(step={propose_step}) " if is_speculative else ""
    title = "Decode Result" if is_decode else "Prefill Result"
    title = f"{title_prefix}{title}"
    batch_size_list = [1] if not is_decode else batch_size_list

    metrics_list: List[MetricState] = []

    total_tests = len(batch_size_list) * len(input_len_list)
    with tqdm(total=total_tests, desc=f"Running {title}", unit="test") as pbar:
        for batch_size in batch_size_list:
            for input_len in input_len_list:
                # pbar.set_description(
                #     f"Running {title} - batch_size: {batch_size}, input_len: {input_len}"
                # )
                logging.info(
                    f"Running {title} - batch_size: {batch_size}, input_len: {input_len}"
                )
                # Execute a perf test for specific batch size and input length
                metric = BatchPerfImpl(
                    base_port=base_port,
                    dp_size=dp_size,
                    tp_size=tp_size,
                    local_world_size=local_world_size,
                    batch_size=batch_size * dp_size,
                    input_len=input_len,
                    query=input_query_dict[input_len],
                    gang_config_string=gang_config_string,
                    request_tpot=request_tpot,
                    connection_timeout=connection_timeout,
                    retry_times=retry_times,
                    retry_interval=retry_interval,
                    is_decode=is_decode,
                    decode_test_length=decode_test_length,
                    generate_config=generate_config,
                ).run()
                metrics_list.append(MetricState(input_len, batch_size, metric))

                # 更新进度条
                pbar.update(1)

    metrics_table = create_metrics_table(
        TableType.Decode if is_decode else TableType.Prefill,
        metrics_list,
        dump_json_path,
        {"dp_size": dp_size, "tp_size": tp_size},
        title,
        generate_config,
    )
    logging.info("metrics_table: \n" + str(metrics_table))
    return metrics_list


def start_server(
    args: argparse.Namespace,
    model_env: Dict[str, Any],
    log_name: str,
):
    current_env = os.environ.copy()
    current_env.update(model_env)
    server = MagaServerManager(env_args=current_env, process_file_name=log_name)
    server.start_server(
        model_path=args.ckpt_path,
        model_type=args.model_type,
        tokenizer_path=args.tokenizer_path,
    )
    server.wait_sever_done()
    return server


def parse_args():
    parser = argparse.ArgumentParser(description="batch decode runner")
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--dp_size", type=int, required=True)
    parser.add_argument("--tp_size", type=int, required=True)
    parser.add_argument("--model_size", type=float, default=0)
    parser.add_argument("--batch_size", type=str, default="1,2,4,8,16")
    parser.add_argument("--input_len", type=str, default="128,1024,2048,4096,8192")
    parser.add_argument("--test_name", type=str, default="batch_decode_test")
    parser.add_argument("--prec", type=str, default="fp16")
    parser.add_argument("--world_size", type=int, default=0)
    parser.add_argument("--disaggregate", type=int, default=0)
    # partial test, 0: test all, 1: test decode only, 2: test prefill only
    parser.add_argument("--partial", type=int, default=0)
    parser.add_argument("--result_dir", type=str, default=".")
    parser.add_argument("--generate_config", type=str, default="{}")
    args = parser.parse_args()
    return args


def merge_state(
    decode_result: List[MetricState], prefill_result: List[MetricState]
) -> List[MetricState]:
    prefill_result_dict = {}
    for prefill_item in prefill_result:
        if prefill_item.metrics.success_requests == prefill_item.metrics.total_requests:
            prefill_result_dict[prefill_item.input_len] = (
                prefill_item.metrics.avg_prefill_time
            )
        else:
            prefill_result_dict[prefill_item.input_len] = -1
    for decode_item in decode_result:
        if decode_item.input_len in prefill_result_dict:
            decode_item.metrics.avg_prefill_time = prefill_result_dict[
                decode_item.input_len
            ]
        else:
            decode_item.metrics.avg_prefill_time = -1
    return decode_result


def run_normal_test(args: argparse.Namespace, running_config: RunningConfig):
    server = start_server(args, running_config.env, "process.log")
    decode_result = None
    prefill_result = None
    if args.partial == 0 or args.partial == 1:
        decode_result = run_single(
            base_port=server.port,
            dp_size=args.dp_size,
            tp_size=args.tp_size,
            batch_size_list=running_config.batch_size_list,
            input_len_list=running_config.input_len_list,
            input_query_dict=running_config.input_query_dict,
            is_decode=True,
            dump_json_path=running_config.result_dir,
            decode_test_length=running_config.decode_test_length,
            is_speculative=running_config.is_speculative,
            propose_step=running_config.propose_step,
            generate_config=running_config.generate_config,
        )
    if args.partial == 0 or args.partial == 2:
        prefill_result = run_single(
            base_port=server.port,
            dp_size=args.dp_size,
            tp_size=args.tp_size,
            batch_size_list=[1],
            input_len_list=running_config.input_len_list,
            input_query_dict=running_config.input_query_dict,
            is_decode=False,
            dump_json_path=running_config.result_dir,
            decode_test_length=running_config.decode_test_length,
            is_speculative=running_config.is_speculative,
            propose_step=running_config.propose_step,
            generate_config=running_config.generate_config,
        )
    server.stop_server()
    return decode_result, prefill_result


def run_disaggregate_test(args: argparse.Namespace, running_config: RunningConfig):
    assert args.partial == 0, "disaggregate test only support test all"
    decode_env = json.loads(os.environ.get("DECODE_CONFIG", "{}"))
    decode_env.update(running_config.env)
    decode_env["BATCH_DECODE_SCHEDULER_WARMUP_TYPE"] = "0"
    decode_server = start_server(args, decode_env, "decode.log")
    decode_result = run_single(
        base_port=decode_server.port,
        dp_size=args.dp_size,
        tp_size=args.tp_size,
        batch_size_list=running_config.batch_size_list,
        input_len_list=running_config.input_len_list,
        input_query_dict=running_config.input_query_dict,
        is_decode=True,
        dump_json_path=running_config.result_dir,
        decode_test_length=running_config.decode_test_length,
        is_speculative=running_config.is_speculative,
        propose_step=running_config.propose_step,
        generate_config=running_config.generate_config,
    )
    decode_server.stop_server()
    prefill_env = json.loads(os.environ.get("PREFILL_CONFIG", "{}"))
    prefill_env.update(running_config.env)
    prefill_env["BATCH_DECODE_SCHEDULER_WARMUP_TYPE"] = "1"
    prefill_server = start_server(
        args,
        prefill_env,
        "prefill.log",
    )
    prefill_result = run_single(
        base_port=prefill_server.port,
        dp_size=args.dp_size,
        tp_size=args.tp_size,
        batch_size_list=[1],
        input_len_list=running_config.input_len_list,
        input_query_dict=running_config.input_query_dict,
        is_decode=False,
        dump_json_path=running_config.result_dir,
        decode_test_length=running_config.decode_test_length,
        is_speculative=running_config.is_speculative,
        propose_step=running_config.propose_step,
        generate_config=running_config.generate_config,
    )
    prefill_server.stop_server()
    return decode_result, prefill_result


def create_test_env(max_len: int, max_concurrency: int, partial: int):
    return {
        "USE_BATCH_DECODE_SCHEDULER": "1",
        "FAKE_BALANCE_EXPERT": "1",
        "MAX_SEQ_LEN": str(max_len + 20),
        "CONCURRENCY_LIMIT": str(max_concurrency),
        "TORCH_CUDA_PROFILER_DIR": os.environ.get(
            "TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd()
        ),
        "BATCH_DECODE_SCHEDULER_WARMUP_TYPE": (
            "0" if (partial == 0 or partial == 1) else "1"
        ),
    }


def main():
    print("current path: ", os.getcwd(), flush=True)

    args = parse_args()
    if args.world_size != 0:
        check_with_info(
            args.dp_size * args.tp_size == args.world_size,
            "dp_size * tp_size must be equal to world_size",
        )
    if args.partial not in [0, 1, 2]:
        raise ValueError("partial must be 0, 1, or 2")
    batch_size_list = [int(x) for x in args.batch_size.split(",")]
    input_len_list = [int(x) for x in args.input_len.split(",")]

    is_speculative = bool(os.environ.get("SP_TYPE", ""))
    decode_test_length = int(os.environ.get("DECODE_TEST_LENGTH", 10))
    propose_step = int(os.environ.get("GEN_NUM_PER_CIRCLE", 1))

    test_env = create_test_env(
        max(input_len_list) + decode_test_length, max(batch_size_list), args.partial
    )

    input_query_dict = create_query(
        args.model_type, args.tokenizer_path, input_len_list
    )

    running_config = RunningConfig(
        batch_size_list=batch_size_list,
        input_len_list=input_len_list,
        input_query_dict=input_query_dict,
        env=test_env,
        result_dir=args.result_dir,
        decode_test_length=decode_test_length,
        is_speculative=is_speculative,
        propose_step=propose_step,
        generate_config=json.loads(args.generate_config),
    )

    if args.disaggregate == 0:
        decode_result, prefill_result = run_normal_test(args, running_config)
    else:
        decode_result, prefill_result = run_disaggregate_test(args, running_config)
    if args.partial != 0:
        return
    metrics_list = merge_state(decode_result, prefill_result)
    device_name = os.environ.get("DEVICE_NAME", torch.cuda.get_device_name(0))
    # use decode parallel config as odps column for current
    write_odps_wrapper(
        device_name,
        args.model_type,
        args.model_size,
        args.prec,
        args.dp_size,
        args.tp_size,
        metrics_list,
    )
