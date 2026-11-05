import argparse
import gc
import os
import shutil
import time
from contextlib import ExitStack
from pathlib import Path

from navi_sched import (
    NaviSchedAgent,
    NaviSchedClient,
    NaviSchedMaster,
    NaviSchedRequest,
)


def wait_for_metadata(client, request, biz_name, timeout_ms):
    deadline = time.monotonic() + timeout_ms / 1000.0
    while True:
        decision = client.schedule(
            request,
            biz_name,
            timeout_ms=timeout_ms,
        )
        if decision.target is not None:
            return decision
        if decision.reason not in {"meta_unavailable", "not_scheduled"}:
            raise RuntimeError(
                f"scheduling failed: {decision.reason or 'unknown'}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"agent was not scheduled: {decision.reason}"
            )
        time.sleep(0.01)


def make_runtime_config(thread_num=64, local_nodes=()):
    return {"thread_num": thread_num, "local_nodes": list(local_nodes)}


def _make_local_node(biz_name, grpc_port, support_heartbeat):
    return {
        "biz_name": biz_name,
        "grpc_port": grpc_port,
        "support_heartbeat": support_heartbeat,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-biz-name", default="master_biz")
    parser.add_argument("--agent-biz-name", default="agent_biz")
    parser.add_argument("--agent-count", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=1000)
    parser.add_argument("--keep-runtime", action="store_true")
    parser.add_argument("--vis-output-dir")
    args = parser.parse_args()

    if args.agent_count <= 0:
        parser.error("--agent-count must be positive")
    vis_output_dir = (
        Path(args.vis_output_dir).resolve()
        if args.vis_output_dir else None
    )
    options = {
        "enable_kmon": False,
        "enable_vis": bool(vis_output_dir or args.keep_runtime),
        "wait_no_query": False,
    }
    with ExitStack() as stack:
        work_dir = (
            Path(__file__).resolve().parents[1] /
            "tmp" /
            "embedded_cluster"
        )
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True)
        if not args.keep_runtime:
            stack.callback(shutil.rmtree, work_dir, True)
        original_dir = os.getcwd()
        os.chdir(work_dir)
        stack.callback(os.chdir, original_dir)
        agents = []
        for index in range(args.agent_count):
            agents.append(stack.enter_context(NaviSchedAgent(
                args.agent_biz_name,
                server_ip="127.0.0.1",
                server_port=18000 + index,
                total_blocks=1024,
                block_size=256,
                linear_step=1,
                hybrid_tail_required=False,
                name=f"host_{index}",
                runtime_config=make_runtime_config(thread_num=4),
                **options,
            )))
        for agent in agents:
            agent.init()
        master = stack.enter_context(NaviSchedMaster(
            args.master_biz_name,
            name="master",
            gig={"grpc_io_thread_num": 20},
            runtime_config=make_runtime_config(
                thread_num=64,
                local_nodes=[
                    _make_local_node(
                        args.agent_biz_name, agent.grpc_port, True)
                    for agent in agents
                ],
            ),
            **options,
        ))
        master.init()
        client = stack.enter_context(NaviSchedClient(
            "client_biz",
            args.master_biz_name,
            name="client",
            runtime_config=make_runtime_config(
                thread_num=4,
                local_nodes=[
                    _make_local_node(
                        args.master_biz_name, master.grpc_port, False)
                ],
            ),
            **options,
        ))
        client.init()
        request = NaviSchedRequest(
            request_id=1,
            prompt_token_count=512,
            block_hashes=[101, 202],
        )
        decision = wait_for_metadata(
            client,
            request,
            args.agent_biz_name,
            args.timeout_ms,
        )
        if decision.reason != "cold_start_inflight_selected":
            raise RuntimeError(
                f"unexpected scheduling result: {decision.reason}"
            )
        if decision.target is None or not decision.target.address.startswith(
            "127.0.0.1:"
        ):
            raise RuntimeError(
                f"unexpected target address: {decision.target}"
            )
        print(decision)
        for agent in agents:
            agent.stop()
        master.stop()
        vis_paths = []
        if vis_output_dir or args.keep_runtime:
            for vis_name in (
                "agent_graph.vis",
                "client_graph.vis",
                "master_graph.vis",
                "master_schedule_graph.vis",
            ):
                vis_path = work_dir / vis_name
                if not vis_path.is_file() or vis_path.stat().st_size == 0:
                    raise RuntimeError(
                        f"VIS output was not generated: {vis_path}"
                    )
                vis_paths.append(vis_path)
        if vis_output_dir:
            vis_output_dir.mkdir(parents=True, exist_ok=True)
            for vis_path in vis_paths:
                output_path = vis_output_dir / vis_path.name
                shutil.copy2(vis_path, output_path)
                print(output_path)
        elif args.keep_runtime:
            for vis_path in vis_paths:
                print(vis_path)
    if not args.keep_runtime:
        del client
        del master
        agents.clear()
        gc.collect()
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
