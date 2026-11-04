import argparse
import json
import multiprocessing
import os
import resource
import statistics
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from navi_sched import NaviSchedAgent, NaviSchedMaster
from navi_sched._navi_sched_core import (
    get_simulated_execution_stats,
    report_simulated_batch,
)


AGENT_BIZ_NAME = "agent_biz"
MASTER_BIZ_NAME = "master_biz"


def percentile(sorted_values, ratio: float):
    if not sorted_values:
        return 0.0
    index = min(
        len(sorted_values) - 1,
        int((len(sorted_values) - 1) * ratio))
    return sorted_values[index]


def transport_stats(stats):
    values = sorted(stats.get("report_transport_latency_ms", []))
    return {
        "sample_count": len(values),
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": values[-1] if values else 0.0,
    }


def make_agent(index: int) -> NaviSchedAgent:
    agent = NaviSchedAgent(
        AGENT_BIZ_NAME,
        total_blocks=1024,
        block_size=256,
        linear_step=1,
        hybrid_tail_required=False,
        name=f"scale_agent_{index}",
        runtime_config={"thread_num": 1, "queue_size": 10000},
        enable_kmon=False,
        enable_snapshot=False,
        enable_vis=False,
        log_file_name=f"agent_{index}.log",
        log_level="WARN",
        gig={
            "grpc_io_thread_num": 1,
            "heartbeat_thread_num": 1,
            "heartbeat_queue_size": 10000,
        },
        wait_no_query=False,
    )
    agent.init()
    return agent


def start_agents(first_index: int, count: int, workers: int):
    begin = time.monotonic()
    with ThreadPoolExecutor(max_workers=min(workers, count)) as executor:
        agents = list(executor.map(
            make_agent, range(first_index, first_index + count)))
    return agents, time.monotonic() - begin


def report_round(agents, round_id: int, first_index: int, workers: int):
    begin = time.monotonic()

    def report(item):
        local_index, agent = item
        request_id = (round_id + 1) * 1000000 + first_index + local_index + 1
        if not report_simulated_batch(agent, request_id, 1024, 256):
            raise RuntimeError(
                f"Agent {first_index + local_index} rejected its report")

    with ThreadPoolExecutor(max_workers=min(workers, len(agents))) as executor:
        list(executor.map(report, enumerate(agents)))
    return time.monotonic() - begin


def stop_agents(agents, workers: int):
    begin = time.monotonic()
    with ThreadPoolExecutor(max_workers=min(workers, len(agents))) as executor:
        list(executor.map(lambda agent: agent.stop(), agents))
    return time.monotonic() - begin


def agent_worker(connection, first_index: int, count: int, workers: int,
                 runtime_dir: str):
    agents = []
    try:
        threading.stack_size(1024 * 1024)
        os.makedirs(runtime_dir, exist_ok=True)
        os.chdir(runtime_dir)
        agents, start_seconds = start_agents(first_index, count, workers)
        connection.send({
            "type": "ready",
            "ports": [agent.grpc_port for agent in agents],
            "start_seconds": start_seconds,
        })
        while True:
            command = connection.recv()
            if command["type"] == "report":
                elapsed = report_round(
                    agents, command["round"], first_index, workers)
                connection.send({"type": "reported", "seconds": elapsed})
            elif command["type"] == "stop":
                elapsed = stop_agents(agents, workers)
                connection.send({
                    "type": "stopped",
                    "seconds": elapsed,
                    "max_rss_kb": resource.getrusage(
                        resource.RUSAGE_SELF).ru_maxrss,
                })
                return
            else:
                raise RuntimeError(f"unknown worker command: {command}")
    except BaseException as error:
        try:
            connection.send({
                "type": "error",
                "message": str(error),
                "traceback": traceback.format_exc(),
            })
        except BaseException:
            pass
        for agent in agents:
            try:
                agent.stop()
            except BaseException:
                pass
        raise
    finally:
        connection.close()


def partition_agents(agent_count: int, process_count: int):
    process_count = min(agent_count, process_count)
    base, remainder = divmod(agent_count, process_count)
    first_index = 0
    partitions = []
    for process_index in range(process_count):
        count = base + (1 if process_index < remainder else 0)
        partitions.append((first_index, count))
        first_index += count
    return partitions


def wait_for_messages(workers, expected_type: str, timeout_seconds: int):
    deadline = time.monotonic() + timeout_seconds
    pending = set(range(len(workers)))
    messages = [None] * len(workers)
    while pending:
        progress = False
        for index in list(pending):
            process, connection = workers[index]
            if connection.poll(0):
                message = connection.recv()
                if message.get("type") == "error":
                    raise RuntimeError(
                        f"Agent worker {index} failed: {message['message']}\n"
                        f"{message['traceback']}")
                if message.get("type") != expected_type:
                    raise RuntimeError(
                        f"Agent worker {index} returned {message.get('type')}, "
                        f"expected {expected_type}")
                messages[index] = message
                pending.remove(index)
                progress = True
            elif not process.is_alive():
                raise RuntimeError(
                    f"Agent worker {index} exited with code {process.exitcode}")
        if pending and time.monotonic() >= deadline:
            raise RuntimeError(
                f"timed out waiting for {expected_type} from "
                f"{len(pending)} Agent workers")
        if pending and not progress:
            time.sleep(0.01)
    return messages


def start_agent_workers(args, runtime_dir: Path):
    context = multiprocessing.get_context("spawn")
    workers = []
    begin = time.monotonic()
    for worker_index, (first_index, count) in enumerate(
            partition_agents(args.agent_count, args.agent_processes)):
        parent_connection, child_connection = context.Pipe()
        process = context.Process(
            target=agent_worker,
            args=(child_connection, first_index, count,
                  args.start_workers,
                  str(runtime_dir / f"worker_{worker_index}")),
            name=f"agent-worker-{worker_index}",
        )
        process.start()
        child_connection.close()
        workers.append((process, parent_connection))
    messages = wait_for_messages(workers, "ready", args.timeout_seconds)
    ports = [port for message in messages for port in message["ports"]]
    return workers, ports, {
        "wall_seconds": time.monotonic() - begin,
        "worker_seconds": [message["start_seconds"] for message in messages],
    }


def start_master(agent_ports):
    local_nodes = [{
        "biz_name": AGENT_BIZ_NAME,
        "ip": "127.0.0.1",
        "grpc_port": port,
        "support_heartbeat": True,
    } for port in agent_ports]
    begin = time.monotonic()
    master = NaviSchedMaster(
        MASTER_BIZ_NAME,
        name="scale_master",
        runtime_config={
            "thread_num": 64,
            "queue_size": 100000,
            "processing_size": 100000000,
            "local_nodes": local_nodes,
        },
        enable_kmon=False,
        enable_snapshot=False,
        enable_vis=False,
        log_file_name="master.log",
        log_level="WARN",
        gig={
            "grpc_io_thread_num": 8,
            "heartbeat_thread_num": 1,
            "heartbeat_queue_size": 100000,
        },
        wait_no_query=False,
    )
    master.init()
    return master, time.monotonic() - begin


def wait_for_agent_count(master, count: int, timeout_seconds: int):
    begin = time.monotonic()
    deadline = begin + timeout_seconds
    while time.monotonic() < deadline:
        stats = get_simulated_execution_stats(master)
        if len(stats["latency_models"]) == count:
            return time.monotonic() - begin
        time.sleep(0.05)
    discovered = len(get_simulated_execution_stats(master)["latency_models"])
    raise RuntimeError(
        f"Master discovered {discovered} of {count} Agents")


def run_report_round(master, workers, agent_count: int, round_id: int,
                     timeout_seconds: int):
    before = get_simulated_execution_stats(master)
    expected = before["finished_request_count"] + agent_count
    begin = time.monotonic()
    for _, connection in workers:
        connection.send({"type": "report", "round": round_id})
    messages = wait_for_messages(workers, "reported", timeout_seconds)
    submitted_seconds = time.monotonic() - begin
    deadline = begin + timeout_seconds
    while time.monotonic() < deadline:
        stats = get_simulated_execution_stats(master)
        if stats["finished_request_count"] >= expected:
            return {
                "round": round_id,
                "worker_report_seconds": [
                    message["seconds"] for message in messages],
                "submitted_seconds": submitted_seconds,
                "completed_seconds": time.monotonic() - begin,
                "new_finished_request_count":
                    stats["finished_request_count"] -
                    before["finished_request_count"],
                "transport": transport_stats(stats),
            }
        time.sleep(0.01)
    raise RuntimeError(f"report round {round_id} did not complete")


def stop_agent_workers(workers, timeout_seconds: int):
    begin = time.monotonic()
    for process, connection in workers:
        if process.is_alive():
            connection.send({"type": "stop"})
    messages = wait_for_messages(workers, "stopped", timeout_seconds)
    for process, connection in workers:
        process.join(timeout=10)
        connection.close()
        if process.exitcode != 0:
            raise RuntimeError(
                f"Agent worker {process.name} exited with code {process.exitcode}")
    return {
        "wall_seconds": time.monotonic() - begin,
        "worker_seconds": [message["seconds"] for message in messages],
        "worker_max_rss_kb": [message["max_rss_kb"] for message in messages],
    }


def terminate_agent_workers(workers):
    for process, connection in workers:
        if process.is_alive():
            process.terminate()
    for process, connection in workers:
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join()
        connection.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-count", type=int, default=1000)
    parser.add_argument("--agent-processes", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--start-workers", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--output", default="master_scale_result.json")
    return parser.parse_args()


def save_result(output: Path, result, stage: str):
    result["stage"] = stage
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(f"master scale stage completed: {stage}", flush=True)


def main():
    args = parse_args()
    if min(args.agent_count, args.agent_processes, args.rounds,
           args.start_workers, args.timeout_seconds) <= 0:
        raise ValueError("all numeric arguments must be positive")
    threading.stack_size(1024 * 1024)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime_dir = output.parent / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output.parent)
    workers = []
    master = None
    workers_stopped = False
    result = {
        "agent_count": args.agent_count,
        "agent_process_count": min(args.agent_count, args.agent_processes),
        "round_count": args.rounds,
    }
    try:
        workers, ports, result["agent_start"] = start_agent_workers(
            args, runtime_dir)
        save_result(output, result, "agent_start")
        master, result["master_start_seconds"] = start_master(ports)
        save_result(output, result, "master_start")
        result["handshake_seconds"] = wait_for_agent_count(
            master, args.agent_count, args.timeout_seconds)
        result["transport_after_handshake"] = transport_stats(
            get_simulated_execution_stats(master))
        save_result(output, result, "handshake")
        result["rounds"] = []
        for index in range(args.rounds):
            result["rounds"].append(run_report_round(
                master, workers, args.agent_count, index,
                args.timeout_seconds))
            save_result(output, result, f"report_round_{index + 1}")
        result["agent_stop"] = stop_agent_workers(
            workers, args.timeout_seconds)
        workers_stopped = True
        save_result(output, result, "agent_stop")
        begin = time.monotonic()
        master.stop()
        result["master_stop_seconds"] = time.monotonic() - begin
        master = None
    finally:
        if workers and not workers_stopped:
            terminate_agent_workers(workers)
        if master is not None:
            master.stop()
    result["parent_max_rss_kb"] = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss
    result["total_worker_max_rss_kb"] = sum(
        result["agent_stop"]["worker_max_rss_kb"])
    save_result(output, result, "complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
