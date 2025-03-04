import json
import asyncio
import logging
import argparse
import os
from pathlib import Path
import sys
import copy
import threading
import asyncio
import time
import multiprocessing
import torch
import torch.distributed as dist

from pydantic import BaseModel
from typing import Dict, Any, List
from filelock import FileLock

os.environ['TP_SIZE'] = os.environ.get('TP_SIZE', '1')
os.environ['WORLD_SIZE'] = os.environ['TP_SIZE']
os.environ['LOCAL_WORLD_SIZE'] = os.environ['TP_SIZE']

CUR_PATH = os.path.dirname(os.path.abspath(__file__))

from maga_transformer.utils.weight_type import WEIGHT_TYPE
from maga_transformer.pipeline.pipeline import Pipeline
from maga_transformer.model_factory import AsyncModel, ModelFactory
from maga_transformer.models.base_model import ModelConfig
from maga_transformer.distribute.worker_info import update_master_info
from example.perf_test.test_util import (
    origin_prompt, write_odps, get_prompt)


class LatencyTestResult(BaseModel):
    batch_size: int
    input_len: int
    context_decoder_time: float
    decoder_time: float


def test_pipeline_time(pipeline, prompt, **kwargs):
    context_time = 0.0
    acc_time = 0.0
    token_count = 0
    exception = None

    async def generator():
        nonlocal context_time
        nonlocal acc_time
        nonlocal exception
        nonlocal token_count
        try:
            begin_time = time.time()
            async for _ in pipeline.pipeline_async(prompt=prompt, **kwargs):
                end_time = time.time()
                cost_time = end_time - begin_time
                if token_count == 0:
                    context_time = cost_time
                else:
                    acc_time += cost_time
                begin_time = end_time
                token_count += 1
        except Exception as e:
            import traceback
            logging.error(f'except: {e} {traceback.format_exc()}')
            exception = e

    def start_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generator())

    backgroud_thread = threading.Thread(target=start_loop)
    backgroud_thread.start()
    backgroud_thread.join()

    if token_count > 1:
        avg_time = acc_time / (token_count - 1)
    else:
        avg_time = 0
    return exception is None, context_time, avg_time


def pipeline_main(model_args, barrier):
    port = 8088
    update_master_info('0.0.0.0', port)
    dist.init_process_group(backend=dist.Backend.NCCL,
                            init_method='tcp://0.0.0.0:' + str(port),
                            rank=int(os.environ['WORLD_RANK']),
                            world_size=int(os.environ['WORLD_SIZE']))
    model = ModelFactory.from_huggingface(
        model_args['ckpt_path'],
        model_config=ModelConfig(**model_args))
    tokenizer = model.model.tokenizer
    pipeline = Pipeline(model, model.config, model.tokenizer)
    barrier.wait()
    return pipeline, tokenizer


def pipeline_proc(model_args, barrier):
    pipeline, tokenizer = pipeline_main(model_args, barrier)
    time.sleep(100000)  # forever


def run_latency_test(model_args: Dict[str, Any], test_args: Dict[str, Any],
                     pipeline, tokenizer):
    state = True

    test_batchsize = test_args['test_batchsize']
    test_input_len = test_args['test_input_len']
    prompts = {
        x: get_prompt(tokenizer, origin_prompt, x)
        for x in set(test_input_len)
    }
    res = []
    for bs, seqlen in zip(test_batchsize, test_input_len):
        # find closest prompt input > seqlen
        prompt = prompts[seqlen]
        actual_seqlen = len(tokenizer.encode(prompt))
        generate_config = {
            'max_new_tokens': 10,
            'min_new_tokens': 10,
            'is_streaming': True,
            'num_return_sequences': bs
        }
        # dynamic lora
        if len(model_args['lora_infos']) > 1:
            name = list(model_args['lora_infos'].keys())[0]
            generate_config['adapter_name'] = name
        test_pipeline_time(pipeline=pipeline,
                           prompt=prompt,
                           top_k=1,
                           top_p=0,
                           temperature=1,
                           generate_config=generate_config)

        with torch.cuda.nvtx.range(f"batchsize={bs}, seqlen={actual_seqlen}"):
            cur_state, context_time, generate_time = test_pipeline_time(
                pipeline=pipeline,
                prompt=prompt,
                top_k=1,
                top_p=0,
                temperature=1,
                generate_config=generate_config)
        state = state and cur_state
        torch.cuda.nvtx.range_pop()
        print(
            f"--- model_type={model_args['model_type']} batchsize={bs}, seqlen={actual_seqlen} ---",
            file=sys.stderr)
        print(
            f"context time: {context_time*1000:.3f} ms, generate time: {generate_time*1000:.3f} ms",
            file=sys.stderr)
        res.append(
            LatencyTestResult(batch_size=bs,
                              input_len=actual_seqlen,
                              context_decoder_time=context_time,
                              decoder_time=generate_time))
    return state, res


def run_test(model_args: Dict[str, Any], test_args: Dict[str, Any]):
    logging.info(f"model_args is {model_args}")

    procs = []
    multiprocessing.set_start_method('spawn')
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    tp_size = int(os.environ['TP_SIZE'])
    if devices is not None:
        devices = devices.split(',')
        if tp_size > len(devices):
            raise Exception(os.environ['CUDA_VISIBLE_DEVICES'] +
                            ' not match ' + os.environ['TP_SIZE'])
    else:
        devices = [str(x) for x in range(tp_size)]

    barrier = multiprocessing.Barrier(tp_size)
    for i in range(1, tp_size):
        logging.info('launch tp_rank=%s', str(i))
        os.environ['WORLD_RANK'] = str(i)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(devices)
        proc = multiprocessing.Process(target=pipeline_proc,
                                       args=(model_args, barrier))
        proc.start()
        procs.append(proc)

    stop_event = threading.Event()

    def check():
        while not stop_event.is_set():
            if not all(proc.is_alive() for proc in procs):
                logging.error('some proc dead')
                [proc.terminate() for proc in procs]
                os._exit(-1)
            time.sleep(1)

    monitor_thread = threading.Thread(target=check)
    monitor_thread.start()

    os.environ['CUDA_VISIBLE_DEVICES'] = devices[0]
    os.environ['WORLD_RANK'] = '0'

    pipeline = None
    lantency_test_res: List[LatencyTestResult] = []
    lantency_test_state = False
    try:
        pipeline, tokenizer = pipeline_main(model_args, barrier)
        lantency_test_state, lantency_test_res = run_latency_test(
            model_args, test_args, pipeline, tokenizer)
    finally:
        if pipeline is not None:
            pipeline.model.decoder_engine_.stop()

        stop_event.set()
        monitor_thread.join()

        for proc in procs:
            proc.terminate()
        del pipeline
        torch.cuda.empty_cache()

    return lantency_test_state, lantency_test_res


def report_latency_test_res(model_type, model_size, prec, device, framework,
                            results: List[LatencyTestResult]):
    lantency_records = []
    for res in results:
        lantency_records.append([
            model_type,
            model_size,
            prec,
            device,
            framework,
            str(os.environ.get('CIS_ENV_COMMIT_ID')),
            res.batch_size,
            res.input_len,
            res.context_decoder_time * 1000,
            res.decoder_time * 1000,
            int(os.environ['TP_SIZE']),
        ])
    table_name = os.environ.get('ODPS_TABLE', 'perf_test_2')
    fields = [
        'model', 'size', 'weight_type', 'device', 'framework', 'commit',
        'batch_size', 'seq_len', 'context_time', 'generate_time', 'tp_size'
    ]
    write_odps(table_name, lantency_records, fields)


class testcase:

    def __init__(self, model_type, ckpt_path, tokenizer_path, prec,
                 test_batch_size, test_input_len, lora_infos):
        self.model_type = model_type
        self.ckpt_path = ckpt_path
        self.tokenizer_path = tokenizer_path
        self.test_batch_size = test_batch_size
        self.test_input_len = test_input_len
        self.prec = prec
        self.lora_infos = lora_infos

    def get_args(self):
        return {
            'model_args': {
                'model_type':
                self.model_type,
                'ckpt_path':
                self.ckpt_path,
                'tokenizer_path':
                self.tokenizer_path,
                'weight_type':
                WEIGHT_TYPE.INT8 if self.prec == 'int8' else WEIGHT_TYPE.FP16,
                'max_seq_len':
                int(os.environ['MAX_SEQ_LEN']),
                'lora_infos':
                self.lora_infos
            },
            'test_args': {
                'test_batchsize': self.test_batch_size,
                'test_input_len': self.test_input_len,
            }
        }


if __name__ == '__main__':
    os.environ['LOAD_CKPT_NUM_PROCESS'] = '0'
    logging.basicConfig(
        level="INFO",
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='perf runner')

    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_size', type=float, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--batch_size', type=str, required=True)
    parser.add_argument('--input_len', type=str, required=True)
    parser.add_argument('--prec', type=str, required=True)
    parser.add_argument('--lora_infos', type=str, required=True)

    args, _ = parser.parse_known_args()
    lora_infos = {}
    for lora_info in args.lora_infos.split(','):
        logging.info(f"lora_info is {lora_info}")
        if lora_info != '{}':
            lora_infos[lora_info.split(':', 1)[0]] = lora_info.split(':', 1)[1]
    logging.info(f"lora_infos is {lora_infos}")
    if len(lora_infos) != 0 and os.environ.get("DYNAMIC_LORA") == 'NO':
        sys.exit()

    t_case = testcase(args.model_type, args.ckpt_path, args.tokenizer_path,
                      args.prec, [int(x) for x in args.batch_size.split(',')],
                      [int(x) for x in args.input_len.split(',')], lora_infos)

    lock_path = '/tmp/maga_transformer/perf_test/gpu_status_lock'
    logging.info(f"env is {os.environ}")
    lock = FileLock(lock_path)

    device_name = os.environ.get('DEVICE_NAME')
    if device_name is None:
        device_name = torch.cuda.get_device_name(0)

    while True:
        try:
            lock.acquire()
        except:
            logging.info("lock file failed")
            time.sleep(1)
            continue
        logging.info(f"t_case.get_args() is {t_case.get_args()}")
        lantency_test_state, lantency_res = run_test(**t_case.get_args())
        model_type = args.model_type
        if len(lora_infos) == 1:
            model_type += '_static_lora'
        elif len(lora_infos) > 1:
            model_type += '_dynamic_lora'
        assert lantency_test_state
        report_latency_test_res(model_type, args.model_size, args.prec,
                                device_name, 'maga_transformer', lantency_res)

        break
