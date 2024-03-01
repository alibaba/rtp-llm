import os
import sys
import json
import time
import logging
import logging.config
import uvicorn
import traceback
import multiprocessing
from multiprocessing import Process
from typing import Generator, Union, Any, Dict, List
import torch

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), '..'))

from maga_transformer.config.log_config import LOGGING_CONFIG
from maga_transformer.distribute.worker_info import g_worker_info, g_parallel_info
from maga_transformer.server.inference_app import InferenceApp

def local_rank_start():
    app = None
    try:
        # avoid multiprocessing load failed
        if os.environ.get('FT_SERVER_TEST', None) is None:
            logging.config.dictConfig(LOGGING_CONFIG)
        # reload for multiprocessing.start_method == fork
        g_parallel_info.reload()
        g_worker_info.reload()
        logging.info(f'start local {g_worker_info}, {g_parallel_info}')
        app = InferenceApp()
        app.start()
    except BaseException as e:
        logging.error(f'start server error: {e}, trace: {traceback.format_exc()}')
        raise e
    return app

def multi_rank_start():
    local_world_size = min(torch.cuda.device_count(), g_parallel_info.world_size)
    os.environ['LOCAL_WORLD_SIZE'] = str(local_world_size)
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError as e:
        logging.warn(str(e))
        pass
    procs: List[Process] = []
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    cuda_device_list = cuda_devices.split(',') if cuda_devices is not None else \
            [str(i) for i in range(torch.cuda.device_count())]
    for idx, world_rank in enumerate(range(g_parallel_info.world_rank,
                                            g_parallel_info.world_rank + local_world_size)):
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device_list[idx]
        os.environ['WORLD_RANK'] = str(world_rank)
        proc = multiprocessing.Process(target=local_rank_start)
        proc.start()
        procs.append(proc)
    if os.environ.get('FAKE_GANG_ENV', None) is not None:
        return procs
    while any(proc.is_alive() for proc in procs):
        if not all(proc.is_alive() for proc in procs):
            [proc.terminate() for proc in procs]
            logging.error(f'some proc is not alive, exit!')
        time.sleep(1)
    [proc.join() for proc in procs]

def main():
    os.makedirs('logs', exist_ok=True)

    if g_parallel_info.world_size % torch.cuda.device_count() != 0 and g_parallel_info.world_size > torch.cuda.device_count():
        raise Exception(f'result: {g_parallel_info.world_size % torch.cuda.device_count()} \
            not support WORLD_SIZE {g_parallel_info.world_size} for {torch.cuda.device_count()} local gpu')
        
    if torch.cuda.device_count() > 1 and g_parallel_info.world_size > 1:
        return multi_rank_start()
    else:
        return local_rank_start()

if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    if os.environ.get('FT_SERVER_TEST', None) is None:
        logging.config.dictConfig(LOGGING_CONFIG)
    main()