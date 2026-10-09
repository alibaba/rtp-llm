"""CI Master -> Python FE smoke; BE generation/status use test doubles."""

import argparse
import asyncio
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def post(port, path, body):
    request = Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Request-ID": "smoke"},
    )
    try:
        response = urlopen(request, timeout=10)
    except HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


def wait_for(check, description, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {description}")


def reserve_ports(offset):
    """Reserve an HTTP port and its RPC offset until their servers start."""
    for _ in range(100):
        first, second = socket.socket(), socket.socket()
        first.bind(("127.0.0.1", 0))
        try:
            second.bind(("127.0.0.1", first.getsockname()[1] + offset))
            return first, second
        except (OSError, OverflowError):
            first.close()
            second.close()
    raise RuntimeError("Cannot reserve local HTTP/RPC ports")


@contextmanager
def frontend():
    import grpc
    import uvicorn
    from fastapi.responses import JSONResponse

    from rtp_llm.config.py_config_modules import PyEnvConfigs
    from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
    from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2_grpc as rpc
    from rtp_llm.frontend.frontend_app import FrontendApp
    from rtp_llm.frontend.frontend_server import FrontendServer
    from rtp_llm.frontend.frontend_worker import (
        BatchPipelineResponse,
        FrontendWorker,
        PipelineResponse,
    )
    from rtp_llm.frontend.shutdown_manager import FrontendShutdownManager
    from rtp_llm.ops import RoleType
    from rtp_llm.utils.concurrency_controller import (
        ConcurrencyController,
        set_global_controller,
    )

    http_socket, rpc_socket = reserve_ports(1)
    port = http_socket.getsockname()[1]
    state = SimpleNamespace(
        port=port,
        calls=[],
        preassign=True,
        alive=True,
        active=0,
        peak=0,
        gate=None,
        atomic_calls=0,
        item_calls=0,
    )

    class StatusService(rpc.RpcServiceServicer):
        def GetWorkerStatus(self, request, context):
            return pb.WorkerStatusPB(
                role="PDFUSION",
                role_type=pb.ROLE_TYPE_PDFUSION,
                alive=state.alive,
                status_version=time.monotonic_ns(),
                dp_size=1,
                tp_size=1,
                available_kv_cache=4096,
                total_kv_cache=4096,
                max_seq_len=8192,
                max_batch_tokens_size=8192,
            )

        def GetCacheStatus(self, request, context):
            return pb.CacheStatusPB(
                version=1, available_kv_cache=4096, total_kv_cache=4096, block_size=16
            )

    async def generate(text, config):
        # Verify the real FE parsed Master's injected addresses without a token.
        if state.preassign:
            assert len(config.role_addrs) == 1
            addr = config.role_addrs[0]
            assert (addr.role, addr.ip, addr.http_port, addr.grpc_port) == (
                RoleType.PDFUSION,
                "127.0.0.1",
                port,
                port + 1,
            )
        else:
            assert not config.role_addrs
        assert config.max_new_tokens == 7
        if text == "fail":
            raise RuntimeError("smoke: injected BE generation failure")
        if text == "timeout":
            await asyncio.sleep(3)
        if text.startswith("parallel-"):
            if state.gate is None:
                state.gate = asyncio.Event()
            state.active += 1
            state.peak = max(state.peak, state.active)
            try:
                if state.active == 64:
                    state.gate.set()
                await asyncio.wait_for(state.gate.wait(), timeout=5)
                await asyncio.sleep(0.05)
            finally:
                state.active -= 1
        if text == "slow":
            await asyncio.sleep(0.15)
        return PipelineResponse(response="echo:" + text, finished=True)

    async def batch_generate(request, **kwargs):
        state.atomic_calls += 1
        yield BatchPipelineResponse(
            response_batch=[
                await generate(text, config)
                for text, config in zip(request.input_texts, request.generate_configs)
            ]
        )

    async def item_generate(request_id, text, urls, generate_config, **kwargs):
        state.item_calls += 1
        yield await generate(text, generate_config)

    worker = FrontendWorker.__new__(FrontendWorker)
    worker.backend_rpc_server_visitor = SimpleNamespace(
        pd_sep_config=SimpleNamespace(role_type=RoleType.FRONTEND),
        host_service=SimpleNamespace(service_available=True),
    )
    # Only generation is stubbed: production request extraction, topology
    # selection, aggregation, logging and concurrency handling still execute.
    worker._yield_batch_generate = batch_generate
    worker._yield_generate = item_generate
    set_global_controller(ConcurrencyController(128))
    server = FrontendServer(0, 0, PyEnvConfigs())
    server._frontend_worker = worker
    owner = FrontendApp.__new__(FrontendApp)
    owner.frontend_server = server
    owner.shutdown_manager = FrontendShutdownManager()
    owner.separated_frontend = True
    owner.server_config = SimpleNamespace(http_port=port)
    app = owner.create_app()

    async def fault_injection(scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/batch_infer":
            return await app(scope, receive, send)
        messages, body = [], b""
        while True:
            message = await receive()
            messages.append(message)
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        payload = json.loads(body)
        state.calls.append(payload)
        assert b"x-rtp-llm-dispatcher-routing-token" not in dict(scope["headers"])
        prompts = payload.get("prompt_batch", [])
        if prompts == ["malformed"]:
            return await JSONResponse({"response_batch": []})(scope, receive, send)
        if prompts == ["bad-request"]:
            return await JSONResponse({"error": "smoke"}, status_code=400)(
                scope, receive, send
            )

        async def replay():
            return messages.pop(0) if messages else await receive()

        await app(scope, replay, send)

    executor = ThreadPoolExecutor(max_workers=2)
    backend = grpc.server(executor)
    rpc.add_RpcServiceServicer_to_server(StatusService(), backend)
    rpc_socket.close()
    assert backend.add_insecure_port(f"127.0.0.1:{port + 1}") == port + 1
    backend.start()
    http_socket.listen(128)
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(
            fault_injection,
            log_level="warning",
            lifespan="on",
        )
    )
    thread = threading.Thread(
        target=uvicorn_server.run, kwargs={"sockets": [http_socket]}, daemon=True
    )
    thread.start()
    try:
        wait_for(lambda: uvicorn_server.started, "Python FE startup")
        yield state
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=10)
        backend.stop(0).wait()
        executor.shutdown(wait=True)
        http_socket.close()
        assert not thread.is_alive(), "Python FE did not stop"


@contextmanager
def master(args, state, preassign):
    state.preassign = preassign
    first, second = reserve_ports(2)
    port = first.getsockname()[1]
    mode = "preassign" if preassign else "independent-fe"
    log_dir = args.log_dir / mode
    log_dir.mkdir(parents=True)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("DISPATCH_")
    }
    env.update(
        {
            "DISPATCH_ENABLED": "true",
            "DISPATCH_SUB_BATCH": "size:1",
            "DISPATCH_BATCH_TIMEOUT_MS": "1000",
            "FLEXLB_SYNC_CONSISTENCY_CONFIG": '{"needConsistency":false}',
            "FLEXLB_CONFIG": json.dumps(
                {
                    "schemaVersion": 3,
                    "requestLifecycle": {"request": {"timeoutMs": 60000}},
                    "grpcServer": {"shutdownQuietPeriodMs": 1},
                }
            ),
            "MODEL_SERVICE_CONFIG": json.dumps(
                {
                    "service_id": "aigc.text-generation.generation.dispatcher-smoke",
                    "role_endpoints": [
                        {
                            "group": "default",
                            "pd_fusion_endpoint": {
                                "address": "smoke-be",
                                "protocol": "http",
                                "path": "/",
                            },
                        }
                    ],
                    "hosts": {
                        "smoke-be": [f"127.0.0.1:{state.port}"],
                        "smoke-fe": [f"127.0.0.1:{state.port}"],
                    },
                }
            ),
        }
    )
    if not preassign:
        env.update(
            DISPATCH_PRE_ASSIGN_BE="false", DISPATCH_FE_POOL_SERVICE_ID="smoke-fe"
        )
    command = [args.java]
    if args.master_jar:
        command += ["-jar", str(args.master_jar)]
    else:
        command += ["-cp", args.master_classpath, "org.flexlb.Application"]
    command += [
        f"--server.port={port}",
        "--management.server.port=0",
        f"--flexlb.log.path={log_dir}",
    ]
    first.close()
    second.close()
    with (log_dir / "stdout.log").open("w") as output:
        process = subprocess.Popen(
            command, env=env, stdout=output, stderr=subprocess.STDOUT
        )
        try:

            def ready():
                assert process.poll() is None, f"Master exited; see {log_dir}"
                try:
                    status, body = post(
                        port,
                        "/rtp_llm/batch_schedule",
                        {
                            "batch_count": 1,
                            "allocation_type": "BE" if preassign else "FE",
                        },
                    )
                    return status == 200 and body.get("success")
                except (URLError, TimeoutError):
                    return False

            wait_for(ready, f"{mode} Master allocation")
            yield port
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def check_cases(port, state):
    def request(prompts, path="/dispatcher/batch_infer"):
        return post(
            port,
            path,
            {"prompt_batch": prompts, "generate_config": {"max_new_tokens": 7}},
        )

    before = len(state.calls)
    status, body = request(["a", "b"], "/dispatcher/_dryrun/batch_infer")
    assert status == 200 and body["chunk_count"] == 2, (status, body)
    assert [chunk["prompt_batch"] for chunk in body["chunks"]] == [["a"], ["b"]]
    assert "role_addrs" not in json.dumps(body)
    empty = request([])
    assert empty == (200, {"response_batch": []}), empty
    invalid = post(port, "/dispatcher/batch_infer", ["not a JSON object"])
    assert invalid[0] == 400, invalid
    assert len(state.calls) == before, "dry-run/empty/invalid requests reached FE"

    for path in ("/dispatcher/", "/dispatcher/batch_infer"):
        status, body = request(["slow", "你好", "last"], path)
        assert status == 200 and "_partial_failure" not in body, (status, body)
        assert [item["response"] for item in body["response_batch"]] == [
            "echo:slow",
            "echo:你好",
            "echo:last",
        ], body

    for fault in ("fail", "malformed", "timeout"):
        status, body = request(["first", fault, "last"])
        assert status == 200, (fault, status, body)
        assert body["response_batch"][1] is None, (fault, body)
        assert [body["response_batch"][i]["response"] for i in (0, 2)] == [
            "echo:first",
            "echo:last",
        ], body
        assert body["_partial_failure"] == {
            "failed_count": 1,
            "total_count": 3,
            "failed_indices": [1],
        }, body

    for fault, reason, expected_status in (
        ("fail", "fe_server_error", 500),
        ("malformed", "malformed_sub_batch", 500),
        ("timeout", "fe_unavailable", 500),
        ("bad-request", "fe_client_error", 400),
    ):
        status, body = request([fault, fault])
        assert status == expected_status, (fault, status, body)
        assert body == {
            "error": "all_sub_batches_failed",
            "failed_count": 2,
            "total_count": 2,
            "total_chunks": 2,
            "failed_reasons": [reason],
        }, body

    state.gate, state.peak = None, 0
    prompts = [f"parallel-{i}" for i in range(65)]
    status, body = request(prompts)
    assert status == 200 and "_partial_failure" not in body, (status, body)
    assert [item["response"] for item in body["response_batch"]] == [
        "echo:" + prompt for prompt in prompts
    ], body
    assert state.peak == 64, state.peak
    print(
        f"PASS preassign={state.preassign}: split/order, failures, timeout, concurrency=64",
        flush=True,
    )


def main():
    runtime = Path(__file__).absolute().parent / "flexlb_runtime"
    parser = argparse.ArgumentParser(description=__doc__)
    master_input = parser.add_mutually_exclusive_group()
    master_input.add_argument("--master-jar", type=Path)
    master_input.add_argument("--master-classpath")
    parser.add_argument("--java", default=str(runtime / "java/bin/java"))
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    if not args.master_jar and not args.master_classpath:
        args.master_jar = runtime / "flexlb-api.jar"
    if args.master_jar and not args.master_jar.is_file():
        parser.error(f"CI Master JAR is missing: {args.master_jar}")
    args.log_dir = (
        args.log_dir
        or Path(
            os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
            or tempfile.mkdtemp(prefix="dispatcher-smoke-")
        )
    ).resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LOG_PATH"] = str(args.log_dir / "fe")
    print(f"Smoke logs: {args.log_dir}", flush=True)
    with frontend() as state:
        for preassign in (True, False):
            with master(args, state, preassign) as port:
                check_cases(port, state)
                if preassign:
                    assert state.atomic_calls > 0 and state.item_calls == 0
                    # A live FE with a dead BE must fail allocation before fanout.
                    state.alive = False
                    wait_for(
                        lambda: post(
                            port,
                            "/rtp_llm/batch_schedule",
                            {
                                "batch_count": 1,
                            },
                        )[0]
                        != 200,
                        "unavailable BE observation",
                    )
                    before = len(state.calls)
                    status, body = post(
                        port, "/dispatcher/batch_infer", {"prompt_batch": ["a"]}
                    )
                    assert status == 503 and body["error"] == "batch_schedule_failed", (
                        status,
                        body,
                    )
                    assert len(state.calls) == before
                    state.alive = True
                    print("PASS unavailable BE: 503 before FE fanout", flush=True)
                else:
                    assert state.item_calls > 0
    print(
        "PASS Master -> Python FE smoke (generation/status use test doubles)",
        flush=True,
    )


if __name__ == "__main__":
    main()
