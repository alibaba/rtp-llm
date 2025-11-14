import copy
import datetime
import json
import logging
import os
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import timedelta
from threading import Thread
from typing import Any, Dict, List, Union

import requests
import torch.distributed
import uvicorn
from fastapi import FastAPI

from rtp_llm.config.py_config_modules import GangConfig, ServerConfig
from rtp_llm.config.uvicorn_config import get_uvicorn_logging_config
from rtp_llm.distribute.gang_info import GangInfo, get_gang_info

# for ut
from rtp_llm.distribute.gang_test_util import create_store, store_based_barrier
from rtp_llm.distribute.worker_info import (
    WorkerInfo,
    g_master_info,
    g_parallel_info,
    g_worker_info,
    update_master_info,
)


def http_post_with_retry(
    url: str, data: Dict[str, Any], retry_limit: int = 3, timeout: int = 150
):
    retry_time = 0
    ret = None
    while retry_time < retry_limit:
        retry_time += 1
        try:
            ret = requests.post(url, json=data, timeout=timeout)
            if ret.status_code == 200:
                return ret
        except:
            if retry_time == retry_limit:
                raise
    return ret


class FailedRankInfo:
    def __init__(
        self,
        failed_ip: str,
        failed_local_rank: int,
        failed_world_rank: int,
        timestamp: int,
        formatted_time: str,
        reporter: List[str],
    ):
        self.failed_ip = failed_ip
        self.failed_world_rank = failed_world_rank
        self.failed_local_rank = failed_local_rank
        self.timestamp = timestamp
        self.formatted_time = formatted_time
        self.reporter = reporter

    def __str__(self):
        return (
            f"FailedRankInfo(failed_ip='{self.failed_ip}', "
            f"failed_local_rank={self.failed_local_rank}, "
            f"failed_world_rank={self.failed_world_rank}, "
            f"timestamp={self.timestamp}, "
            f"formatted_time='{self.formatted_time}', "
            f"reporter={self.reporter})"
        )

    def __repr__(self):
        return self.__str__()


class GangServer:
    def __init__(self, gang_config: GangConfig, server_config: ServerConfig):
        self.gang_config = gang_config
        self.server_config = server_config
        self._initialized: bool = False
        self._gang_status: Dict[str, str] = {}
        self._gang_parts: Dict[str, str] = {}
        self._gang_info: GangInfo = None
        self._request_threadpool = ThreadPoolExecutor(g_parallel_info.world_size)
        self._failure_events: Dict[int, FailedRankInfo] = {}
        self._delay_exit_loops: int = 0
        self._max_delay_times: int = 3

    def _start_server(self):
        app = FastAPI()

        @app.post("/heartbeat")
        def heartbeat(req: Dict[str, str]):
            name = req["name"]
            ip = req["ip"]
            if not self._initialized:
                logging.debug("server member recv: %s %s", name, ip)
                self._gang_status[name] = ip
                last_underscore_index = name.rfind("_")
                if last_underscore_index != -1:
                    part_name = name[:last_underscore_index]
                else:
                    part_name = name
                self._gang_parts[part_name] = ip
            return {"initializing": not self._initialized}

        @app.post("/broadcast_parts")
        def broadcast_parts(req: Dict[str, Any]):
            logging.debug("broadcast parts recv: %s", json.dumps(req))
            self.gang_config.json_gang_parts = json.dumps(req)

        @app.post("/report_failure")
        def handle_failure_report(req: Dict[str, Any]):
            logging.info(f"receive handle_failure_report info {req}")
            try:
                failed_info = FailedRankInfo(
                    req["failed_ip"],
                    req["failed_local_rank"],
                    req["failed_world_rank"],
                    req["timestamp"],
                    req["formatted_time"],
                    [req["reporter"]],
                )
                self._update_failure_table(failed_info)
                return {"status": "ok"}
            except Exception as e:
                logging.error(f"Error while processing failure report: {e}")
                return {"status": "error", "message": str(e)}, 500

        try:
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=g_worker_info.gang_hb_port,
                log_config=get_uvicorn_logging_config(),
                loop="asyncio",
            )
        except:
            logging.error(traceback.format_exc())
            os._exit(-1)

    def _start(self):
        t = Thread(target=self._start_server)
        t.daemon = True
        t.start()

    def _update_failure_table(self, failed_info: FailedRankInfo):
        rank = failed_info.failed_world_rank
        if rank not in self._failure_events:
            logging.info(f"add rank {rank} for {failed_info}")
            self._failure_events[rank] = failed_info
        else:
            if failed_info.timestamp < self._failure_events[rank].timestamp:
                self._failure_events[rank].timestamp = failed_info.timestamp
                self._failure_events[rank].reporter.append(failed_info.reporter)

    def _check_gang_info(self, gang_info: GangInfo):
        if gang_info.master is None or gang_info.self is None:
            raise Exception(
                f"gang_info master {gang_info.master}, gang_info self {gang_info.self}"
            )
        if len(gang_info.members) != g_parallel_info.world_size:
            members_details = "\n".join(str(member) for member in gang_info.members)
            raise Exception(
                f"gang_info member : ({members_details}), "
                f"gang_info members size {len(gang_info.members)} != world_size({g_parallel_info.world_size})"
            )
        for member in gang_info.members:
            if not member.ip or not member.name:
                raise Exception(f"gang_info not complete: {gang_info}")

    def _exchange_gang_info(self, gang_info: GangInfo):
        for member in gang_info.members:
            url = f"http://{member.ip}:{member.gang_hb_port}/heartbeat"
            try:
                result = requests.post(
                    url,
                    json={"name": gang_info.self.name, "ip": gang_info.self.ip},
                    timeout=1,
                )
            except Exception as e:
                logging.debug("request %s failed, %s", url, str(e))
                continue
            if result.status_code == 200:
                response = result.json()
                if response["initializing"] == True:
                    logging.debug("client member recv: %s %s", member.name, member.ip)
                    self._gang_status[member.name] = member.ip
            self._broadcast_parts(gang_info)

    def _check_ready(self):
        miss: List[str] = []
        ready: List[str] = []
        assert self._gang_info is not None, "gang info should not be none"
        for member in self._gang_info.members:
            actual_ip = self._gang_status.get(member.name)
            if actual_ip != member.ip:
                miss.append(member.name)
            else:
                ready.append(member.name)
        if len(miss) > 0:
            raise Exception(
                f"worker rank: {g_parallel_info.tp_rank} not ready, collected workers: {ready}, missed workers: {miss}"
            )

    def _wait_ready(self):
        timeout_minutes = self.gang_config.gang_timeout_min
        sleep_time = self.gang_config.gang_sleep_time
        start_time = datetime.datetime.now()
        retry_time = 0
        while True:
            try:
                self._gang_info = get_gang_info(
                    start_port=self.server_config.start_port,
                    gang_config=self.gang_config,
                )
                if self._gang_info.only_leader:
                    self._exchange_gang_info(self._gang_info)
                    self._check_gang_info(self._gang_info)
                else:
                    self._check_gang_info(self._gang_info)
                    self._exchange_gang_info(self._gang_info)
                self._check_ready()
                return
            except Exception as e:
                logging.warning(
                    f"gang worker rank:{g_parallel_info.world_rank} is not complete, error_msg: {str(e)}, retry times: {retry_time}"
                )
                logging.warning(f"stack trace: {traceback.format_exc()}")
                cur_time = datetime.datetime.now()
                if cur_time - start_time > datetime.timedelta(minutes=timeout_minutes):
                    raise Exception("failed to start gang server")
                retry_time += 1
                time.sleep(sleep_time)

    def wait_infernece_server_ready(self):
        for member in self._gang_info.workers():
            url = f"http://{member.ip}:{member.backend_server_port}/health"
            try:
                resposne = requests.get(url, timeout=1)
            except:
                raise Exception(
                    f"failed to check member {member.ip}:{member.backend_server_port}/health"
                )
            if resposne.status_code != 200:
                raise Exception(
                    f"member {member.ip}:{member.backend_server_port} /health status_code: {resposne.status_code}, is not ready"
                )

    def request_workers(
        self,
        req: Dict[str, Any],
        uri: str = "inference_internal",
        is_wait: bool = False,
    ) -> Union[List[requests.Response], List[Future[requests.Response]]]:
        req = copy.deepcopy(req)

        def curl_impl(url: str) -> requests.Response:
            return requests.post(url, json=req)

        future_list: List[Future[requests.Response]] = []
        for member in self._gang_info.workers():
            url = f"http://{member.ip}:{member.backend_server_port}/{uri}"
            future_ = self._request_threadpool.submit(curl_impl, url)
            future_list.append(future_)
        if is_wait:
            return [x.result() for x in future_list]
        else:
            return future_list

    def _broadcast_parts(self, gang_info: GangInfo):
        if (
            gang_info.only_leader
            and gang_info.master.ip == gang_info.self.ip
            and gang_info.master.local_rank == gang_info.self.local_rank
        ):
            for member in gang_info.members:
                master_part_name = gang_info.master.name
                last_underscore_index = master_part_name.rfind("_")
                if last_underscore_index != -1:
                    master_part_name = master_part_name[:last_underscore_index]
                self._gang_parts[master_part_name] = gang_info.master.ip
                if (
                    len(self._gang_parts) * g_parallel_info.local_world_size
                    == g_parallel_info.world_size
                ):
                    broadcast_url = (
                        f"http://{member.ip}:{member.gang_hb_port}/broadcast_parts"
                    )
                    parts: Dict[str, Any] = {}
                    for name, address in self._gang_parts.items():
                        part_info = {}
                        part_info["name"] = name
                        part_info["ip"] = address
                        parts[name] = part_info
                    self.gang_config.json_gang_parts = json.dumps(parts)
                    try:
                        result = requests.post(broadcast_url, json=parts, timeout=1)
                    except:
                        logging.debug("request %s failed", broadcast_url)

    def _health_check_impl(self):
        failed_member = None
        for member in self._gang_info.members:
            url = f"http://{member.ip}:{member.gang_hb_port}/heartbeat"
            try:
                res = http_post_with_retry(url, {"name": "", "ip": ""})
            except:
                logging.error(
                    f"Gang server {member.ip}:{member.gang_hb_port} heartbeat loss"
                )
                failed_member = member
                break
            if res.status_code != 200:
                logging.error(f"Gang server {member.ip} status code is not 200")
                failed_member = member
                break
            try:
                if res.json()["initializing"] == True:
                    logging.error(
                        f"Gang server {member.ip}:{member.gang_hb_port} is restarted"
                    )
                    failed_member = member
                    break
            except Exception as e:
                logging.error(
                    f"Gang server {member.ip}:{member.gang_hb_port} hb failed {e}"
                )
                failed_member = member
                break

        if failed_member:
            logging.info(
                f"broadcast for failed member [world rank {failed_member.world_rank}]"
            )
            self.broadcast_failure(failed_member)
            self._delay_exit_loops += 1

        if self._delay_exit_loops > self._max_delay_times:
            logging.info(
                f"member [world rank {failed_member.world_rank}] heartbeat exception, do abort!"
            )
            os._exit(-1)

    # use base model type, python3
    def broadcast_failure(self, failed_member: WorkerInfo):
        for member in self._gang_info.members:
            if failed_member.world_rank != member.world_rank:
                report_url = f"http://{member.ip}:{member.gang_hb_port}/report_failure"
                try:
                    now = datetime.datetime.now()
                    formatted_time_ms = (
                        now.strftime("%Y-%m-%d %H:%M:%S")
                        + f".{now.microsecond // 1000:03}"
                    )
                    current_timestamp_ms = int(time.time() * 1000)
                    http_post_with_retry(
                        report_url,
                        {
                            "failed_ip": failed_member.ip,
                            "failed_world_rank": failed_member.world_rank,
                            "failed_local_rank": failed_member.local_rank,
                            "formatted_time": formatted_time_ms,
                            "timestamp": current_timestamp_ms,
                            "reporter": str(-1),
                        },
                        1,
                        30,
                    )
                except Exception as e:
                    logging.warning(
                        f"Failed to notify [world rank {member.world_rank}] {e}"
                    )
                    continue
        logging.info(f"broadcast success")

    def _check_failed(self):
        if len(self._failure_events) > 0:
            first_failure_rank, first_failure_info = min(
                self._failure_events.items(), key=lambda x: x[1].timestamp
            )
            logging.error(f"first failure node is {first_failure_info}")

    def start_health_check(self):
        sleep_time = self.gang_config.gang_sleep_time

        def wrapper():
            while True:
                time.sleep(sleep_time)
                try:
                    self._health_check_impl()
                    self._check_failed()
                except:
                    os._exit(-1)

        t = Thread(target=wrapper)
        t.daemon = True
        t.start()

    # c10d自带tcp_barrier依赖group已经创建，且timeout无法修改，所以需要重新实现一个
    def memory_barrier(self, master_url: str, timeout: int = 10):
        self.store = create_store(
            master_url, g_parallel_info.world_rank, g_parallel_info.world_size
        )
        store_based_barrier(
            g_parallel_info.world_rank,
            g_parallel_info.world_size,
            self.store,
            timeout=timedelta(seconds=timeout),
        )

    def start(self):
        if g_parallel_info.world_size == 1:
            logging.info("world_size==1, do not start gang_server")
            update_master_info("", self.server_config.start_port)
            self._gang_info = get_gang_info(
                start_port=self.server_config.start_port,
                gang_config=self.gang_config,
            )
            return

        self._start()
        self._wait_ready()
        self._initialized = True
        update_master_info(
            self._gang_info.master.ip, self._gang_info.master.server_port
        )
        master_url = (
            f"tcp://{g_master_info.ip}:{self._gang_info.master.server_port - 1}"
        )
        logging.info(f"gang worker {g_parallel_info} init_process_group {master_url}")
        init_process_timeout = self.gang_config.dist_barrier_timeout
        # Default value is 10 minutes for NCCL
        if init_process_timeout is not None:
            init_process_timeout = timedelta(seconds=init_process_timeout)
        os.environ["TORCH_DIST_INIT_BARRIER"] = "1"
        torch.distributed.init_process_group(
            backend=torch.distributed.Backend.NCCL,
            init_method=master_url,
            rank=g_parallel_info.world_rank,
            world_size=g_parallel_info.world_size,
            timeout=init_process_timeout,
        )

        from rtp_llm.models_py.distributed.symm_mem import get_symm_mem_communicator

        # TODO: remove this to all reduce for lately init
        # some not impl errors happened after engine start, need to fix later
        if g_parallel_info.tp_size > 1:
            get_symm_mem_communicator()

        logging.info(f"gang worker {g_parallel_info} start_health_check")
        self.start_health_check()
        logging.info(f"gang init done")
