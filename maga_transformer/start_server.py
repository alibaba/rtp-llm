import os
import sys
import json
import time
import logging
import logging.config
import uvicorn
import traceback
import subprocess
import multiprocessing
from multiprocessing import Process
from typing import Generator, Union, Any, Dict, List
import torch

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), '..'))

from maga_transformer.config.log_config import LOGGING_CONFIG
from maga_transformer.distribute.worker_info import g_worker_info, g_parallel_info
from maga_transformer.server.inference_app import InferenceApp
from maga_transformer.server.vit_rpc_server import vit_start_server

def local_rank_start():
    app = None
    try:
        # avoid multiprocessing load failed
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
    if g_parallel_info.dp_size > 1:
        # tp must on one device when dp
        assert g_parallel_info.world_rank % g_parallel_info.tp_size == 0
    for _, world_rank in enumerate(range(g_parallel_info.world_rank,
                                            g_parallel_info.world_rank + local_world_size)):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(cuda_device_list)
        os.environ['WORLD_RANK'] = str(world_rank)
        proc = Process(target=local_rank_start)
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

def load_gpu_nic_affinity():
    if os.environ.get("ACCL_NIC_GPU_AFFINITY") != None:
        return True
    # 检查 /usr/local/bin/run_affinity 是否存在
    run_affinity_path = "/usr/local/bin/run_affinity"
    if not os.path.exists(run_affinity_path):
        logging.info(f"get gpu nic affinity failed, {run_affinity_path} not exist")
        return False

    try:
        # 执行 run_affinity 文件
        result = subprocess.run(
            [run_affinity_path],
            check=True,  # 如果返回非零退出码则抛出异常
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except Exception as e:
        # 执行失败
        logging.info(f"get gpu nic affinity failed, run {run_affinity_path} failed, exception is {e}")
        return False

    # 检查当前目录是否存在 npu_nic_affinity.json
    json_path = "npu_nic_affinity.json"
    if not os.path.exists(json_path):
        logging.info(f"get gpu nic affinity failed, {json_path} 不存在")
        return False

    try:
        # 读取 JSON 文件内容
        with open(json_path, "r") as f:
            content = f.read().strip()  # 读取并去除首尾空白
        # 将内容存入环境变量
        os.environ["ACCL_NIC_GPU_AFFINITY"] = content
        logging.info(f"get gpu nic affinity success, set env ACCL_NIC_GPU_AFFINITY to {content}")
        return True
    except Exception as e:
        logging.info(f"get gpu nic affinity failed, load {json_path} failed, exception is {e}")
        return False

def main():
    os.makedirs('logs', exist_ok=True)
    load_gpu_nic_affinity()

    if int(os.environ.get('VIT_SEPARATION', 0)) == 1:
        return vit_start_server()

    if not torch.cuda.is_available():
        return local_rank_start()

    if g_parallel_info.world_size % torch.cuda.device_count() != 0 and g_parallel_info.world_size > torch.cuda.device_count():
        raise Exception(f'result: {g_parallel_info.world_size % torch.cuda.device_count()} \
            not support WORLD_SIZE {g_parallel_info.world_size} for {torch.cuda.device_count()} local gpu')

    if torch.cuda.device_count() > 1 and g_parallel_info.world_size > 1:
        return multi_rank_start()
    else:
        return local_rank_start()

if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    main()
