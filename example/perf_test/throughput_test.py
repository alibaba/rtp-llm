import os
import json
from pydantic import BaseModel
import multiprocessing
import argparse
import requests
import time
import aiohttp
import asyncio
import logging
import copy
import torch
import sys
import threading
from typing import List, Any, Tuple, Dict
from filelock import FileLock

os.environ['TP_SIZE'] = os.environ.get('TP_SIZE', '1')
os.environ['WORLD_SIZE'] = os.environ['TP_SIZE']
os.environ['LOCAL_WORLD_SIZE'] = os.environ['TP_SIZE']

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
from example.perf_test.test_util import (
    origin_prompt, write_odps, get_prompt)
from maga_transformer.model_factory import ModelFactory
from maga_transformer.start_server import main


class ThroughputTestResult(BaseModel):
    max_concurrent_num: int
    query_num: int
    avg_query_time_s: float
    total_time_s: float
    tokens_per_sec: float


def test_throughput(prompts: List[Tuple[str, int]], max_concurrent_num: int,
                    **kwargs: Any):
    query_time = 0.0
    total_time = 0.0
    exception = None
    tasks: List[asyncio.Task[Any]] = []
    sem = asyncio.Semaphore(max_concurrent_num)
    generate_config = kwargs.get('generate_config', {})
    if 'generate_config' in kwargs:
        del kwargs['generate_config']
    port = os.environ['START_PORT']
    api_url = f'http://127.0.0.1:{port}/'

    async def run_one_query(sem, **kwargs):
        nonlocal query_time
        nonlocal exception
        async with sem:
            begin_time = time.time()
            timeout = aiohttp.ClientTimeout(total=3 * 3600)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    async with session.post(api_url, json=kwargs) as response:
                        chunks = []
                        async for chunk, _ in response.content.iter_chunks():
                            chunks.append(chunk)
                        output = b"".join(chunks).decode("utf-8")
                        output = json.loads(output)
                        if "error" in output or "error_code" in output:
                            raise Exception(f'query error: {str(output)}')
                    query_time += time.time() - begin_time
                except Exception as e:
                    logging.error(f'except: {e}')
                    exception = e

    async def generator():
        nonlocal total_time
        nonlocal exception
        try:
            begin_time = time.time()
            for prompt, output_len in prompts:
                new_generate_config = copy.deepcopy(generate_config)
                new_generate_config.update({
                    'max_new_tokens': output_len,
                    'min_new_tokens': output_len
                })
                task = asyncio.create_task(
                    run_one_query(sem=sem,
                                  prompt=prompt,
                                  generate_config=new_generate_config,
                                  **kwargs))
                tasks.append(task)
            await asyncio.gather(*tasks)
            total_time = time.time() - begin_time
        except Exception as e:
            logging.error(f'except: {e}')
            exception = e

    def start_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generator())

    backgroud_thread = threading.Thread(target=start_loop)
    backgroud_thread.start()
    backgroud_thread.join()
    return exception is None, total_time, query_time


def wait_server_ready(proc):
    port = os.environ['START_PORT']
    api_url = f'http://127.0.0.1:{port}/'
    while proc.is_alive():
        try:
            response = requests.get(api_url)
            if response.status_code == 200 and response.json().get(
                    "status") == "home":
                print("Server is ready.")
                break
        except requests.RequestException as e:
            time.sleep(10)
            print("Server not ready, wait 10 second")
    if not proc.is_alive():
        raise Exception("Server start failed")


def get_tokenizer():
    model_config = ModelFactory.create_normal_model_config()
    model_cls = ModelFactory.get_model_cls(model_config.model_type)
    params = model_cls.create_config(model_config)
    tokenizer = model_cls.get_tokenizer(params)
    return tokenizer


def run_test(max_concurrent_nums: List[int], query_num: int,
                        model_type: str, model_size: float,
                        lora_infos: Dict[str, Any]):
    multiprocessing.set_start_method('spawn')
    proc = multiprocessing.Process(target=main)
    proc.start()
    wait_server_ready(proc)

    state = True
    test_lens: List[Tuple[int, int]] = []
    with open(os.path.join(CUR_PATH, 'ShareGPT_V3_test_data_lens.json')) as f:
        test_lens = json.load(f)
    total_tokens = sum([sum(lens) for lens in test_lens])
    assert query_num >= len(test_lens)
    test_lens = test_lens[:query_num]
    tokenizer = get_tokenizer()
    prompts = [(get_prompt(tokenizer, origin_prompt, input_len), output_len)
               for input_len, output_len in test_lens]
    generate_config = {}
    res: List[ThroughputTestResult] = []
    if len(lora_infos) > 1:
        name = list(lora_infos.keys())[0]
        adapter_name = [name]
        generate_config = {'adapter_name': adapter_name}
    for max_concurrent_num in max_concurrent_nums:
        cur_state, total_time_s, query_time_s = test_throughput(
            prompts=prompts,
            max_concurrent_num=max_concurrent_num,
            top_k=1,
            top_p=0,
            temperature=1,
            generate_config=generate_config)
        state = state and cur_state
        avg_query_time_s = query_time_s / query_num
        tokens_per_sec = total_tokens / total_time_s
        print(
            f"--- model_type={model_type} model_size={model_size} max_concurrent_num={max_concurrent_num}, query_num={query_num} ---",
            file=sys.stderr)
        print(
            f"total time: {total_time_s:.3f}s, avg query time: {avg_query_time_s:.3f}s, throughput {tokens_per_sec}tokens/s",
            file=sys.stderr)
        res.append(
            ThroughputTestResult(max_concurrent_num=max_concurrent_num,
                                 query_num=query_num,
                                 avg_query_time_s=avg_query_time_s,
                                 total_time_s=total_time_s,
                                 tokens_per_sec=tokens_per_sec))
    proc.terminate()
    proc.join()
    return state, res


def report_throughput_test_res(model_type: str, model_size: float, prec: str,
                               device: str, framework: str,
                               results: List[ThroughputTestResult]):
    lantency_records = []
    for res in results:
        lantency_records.append([
            model_type,
            model_size,
            prec,
            device,
            framework,
            str(os.environ.get('CIS_ENV_COMMIT_ID')),
            int(os.environ['TP_SIZE']),
            res.max_concurrent_num,
            res.query_num,
            res.avg_query_time_s,
            res.total_time_s,
            res.tokens_per_sec,
        ])
    table_name = os.environ.get('THROUGHPUT_ODPS_TABLE',
                                'perf_test_throughput')
    fields = [
        'model_type', 'model_size', 'weight_type', 'device', 'framework', 'commit',
        'tp_size', 'concurrency', 'query_num',
        'avg_query_time_s', 'total_time_s', 'tokens_per_sec'
    ]
    write_odps(table_name, lantency_records, fields)


if __name__ == '__main__':
    os.environ['LOAD_CKPT_NUM_PROCESS'] = '0'
    logging.basicConfig(
        level="INFO",
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='perf runner')
    parser.add_argument('--model_size', type=float, required=True)
    parser.add_argument('--prec', type=str, required=True)
    parser.add_argument('--query_num', type=int, default=1000)
    parser.add_argument('--max_concurrent_nums', type=str, default="64")
    parser.add_argument('--lora_infos', type=str, required=True)
    args, _ = parser.parse_known_args()
    lora_infos = {}
    for lora_info in args.lora_infos.split(','):
        logging.info(f"lora_info is {lora_info}")
        if lora_info != '{}':
            lora_infos[lora_info.split(':', 1)[0]] = lora_info.split(':', 1)[1]
    lock_path = '/tmp/maga_transformer/perf_test/gpu_status_lock'
    logging.info(f"env is {os.environ}")
    lock = FileLock(lock_path)

    device_name = os.environ.get('DEVICE_NAME')
    if device_name is None:
        device_name = torch.cuda.get_device_name(0)

    model_type = os.environ['MODEL_TYPE']
    if len(lora_infos) == 1:
        model_type += '_static_lora'
    elif len(lora_infos) > 1:
        model_type += '_dynamic_lora'

    while True:
        try:
            lock.acquire()
        except:
            logging.info("lock file failed")
            time.sleep(1)
            continue
        throughput_test_state, throughput_res = run_test(
            [int(x) for x in args.max_concurrent_nums.split(',')],
            args.query_num, model_type, args.model_size, lora_infos)
        assert throughput_test_state
        report_throughput_test_res(model_type, args.model_size, args.prec,
                                   device_name, 'maga_transformer',
                                   throughput_res)
        break
