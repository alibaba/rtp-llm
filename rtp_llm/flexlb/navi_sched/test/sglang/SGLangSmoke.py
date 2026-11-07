"""Run under the repository GPU lock with a local model and a native Master."""

import argparse
import asyncio
import ctypes
import dataclasses
import json
import os
import re
from pathlib import Path


async def wait_for_master_snapshot(directory, master_port, agent_port):
    """Wait for native snapshot writes without periodically polling files."""
    directory.mkdir(parents=True, exist_ok=True)
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1")
    changed = asyncio.Event()
    loop = asyncio.get_running_loop()

    def on_write():
        os.read(fd, 65536)
        changed.set()

    try:
        if libc.inotify_add_watch(fd, os.fsencode(directory), 0x8 | 0x80) < 0:
            raise OSError(ctypes.get_errno(), "inotify_add_watch")
        loop.add_reader(fd, on_write)
        async with asyncio.timeout(30):
            while True:
                changed.clear()
                for path in sorted(directory.glob("*_snapshot"), reverse=True):
                    content = path.read_text()
                    if f"  grpc_port: {master_port}\n" not in content:
                        continue
                    blocks = content.split("\n  Agent\n")[1:]
                    for block in blocks:
                        if f"    id: 127.0.0.1:{agent_port}\n" not in block:
                            continue
                        cache_line = re.search(r"^    cache_sync: (.+)$", block, re.M)
                        workload_line = re.search(r"^    workload: (.+)$", block, re.M)
                        if not cache_line or not workload_line:
                            continue
                        cache = dict(re.findall(r"(\w+)=([\w./-]+)", cache_line[1]))
                        workload = dict(
                            re.findall(r"(\w+)=([\w./-]+)", workload_line[1])
                        )
                        rounds = re.search(
                            r"^  schedule_round_count: (\d+)$", content, re.M
                        )
                        if (
                            not rounds
                            or int(rounds[1]) < 4
                            or int(cache["master_cached"]) == 0
                        ):
                            continue
                        if (
                            workload["request_count"] != "0"
                            or workload["running_capacity_slots"] != "0/8"
                        ):
                            continue
                        assert cache["agent_cached"] == cache["master_cached"], cache
                        assert (
                            cache["agent_checksum"] == cache["master_checksum"]
                        ), cache
                        assert cache["state_mismatches"] == "0", cache
                        assert workload["capacity_split"] == "true", workload
                        assert int(workload["revision"]) > 0, workload
                        requests_line = re.search(r"^    requests: (.+)$", block, re.M)
                        if not requests_line:
                            continue
                        requests = dict(
                            re.findall(r"(\w+)=([\w./-]+)", requests_line[1])
                        )
                        if any(
                            requests[key] != "0"
                            for key in (
                                "inflight",
                                "waiting_for_batch",
                                "queued_for_execution",
                                "running",
                            )
                        ):
                            continue
                        return {
                            "path": str(path.resolve()),
                            "cache": cache,
                            "workload": workload,
                            "schedule_round_count": int(rounds[1]),
                            "requests": requests,
                        }
                await changed.wait()
    finally:
        loop.remove_reader(fd)
        os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--disable-overlap", action="store_true")
    args = parser.parse_args()
    os.environ.update(
        SGLANG_NAVI_SCHED_ENABLED="1",
        SGLANG_NAVI_SCHED_TARGET_BIZ="sglang_smoke_worker",
        SGLANG_NAVI_SCHED_MASTER_BIZ="sglang_smoke_master",
        SGLANG_NAVI_SCHED_SERVER_IP="127.0.0.1",
        SGLANG_PLUGINS="navi_sched",
    )
    from navi_sched import NaviSchedMaster
    from sglang import Engine
    from sglang.srt.managers.io_struct import GenerateReqInput

    engine = None
    master = None
    try:
        engine = Engine(
            model_path=args.model,
            context_length=2048,
            mem_fraction_static=0.1,
            max_running_requests=8,
            attention_backend="triton",
            sampling_backend="pytorch",
            disable_cuda_graph=True,
            disable_overlap_schedule=args.disable_overlap,
            log_level="info",
        )
        manager = engine.tokenizer_manager
        runtime = manager.navi_sched
        loop = engine.loop

        async def wait_ready():
            manager.auto_create_handle_loop()
            await asyncio.wait_for(runtime.ready.wait(), 60)

        loop.run_until_complete(wait_ready())
        prompt = (
            "Use evidence first. Distinguish planned, running, observed, candidate cause, "
            "confirmed cause and complete. Label unverified claims. For fixes, reproduce "
            "the failure, test the fix, remove it to confirm failure, restore it and "
            "validate the real environment. Independently check counterexamples. "
            "Now answer with one short sentence: what is two plus two?"
        )
        sampling = {"temperature": 0, "max_new_tokens": 12}
        local = loop.run_until_complete(
            asyncio.wait_for(
                engine.async_generate(prompt, sampling_params=sampling), 90
            )
        )
        assert local["meta_info"]["completion_tokens"] > 0, local
        assert runtime.transport.snapshot["local_fallback_count"] == 1
        no_master_fallback_count = runtime.transport.snapshot["local_fallback_count"]
        runtime.transport.close()
        runtime.transport = None

        snapshot_dir = Path("master_snapshots") / f"sglang_smoke_{os.getpid()}"
        master = NaviSchedMaster(
            "sglang_smoke_master",
            runtime_config={
                "thread_num": 8,
                "local_nodes": [
                    {
                        "biz_name": "sglang_smoke_worker",
                        "ip": "127.0.0.1",
                        "grpc_port": runtime.agent.grpc_port,
                        "support_heartbeat": True,
                    }
                ],
            },
            scheduler_name="max_cache_hit",
            max_wait_schedule_count=1,
            enable_kmon=False,
            wait_no_query=False,
            enable_snapshot=True,
            snapshot_keep_count=8,
            snapshot_dir=str(snapshot_dir),
            latency_model_pool_state_file="",
        ).init()
        runtime.config = dataclasses.replace(
            runtime.config,
            runtime={
                "local_nodes": [
                    {
                        "biz_name": "sglang_smoke_master",
                        "ip": "127.0.0.1",
                        "grpc_port": master.grpc_port,
                    }
                ]
            },
        )
        remote = loop.run_until_complete(
            asyncio.wait_for(
                engine.async_generate(prompt, sampling_params=sampling), 90
            )
        )
        assert remote["text"] == local["text"], (local, remote)
        assert remote["meta_info"]["cached_tokens"] > 0, remote
        assert runtime.transport.client.request_received_count == 1
        assert runtime.snapshot["forwarded_request_count"] == 1, runtime.snapshot
        assert runtime.transport.snapshot["remote_request_count"] == 1, runtime.snapshot
        assert runtime.transport.snapshot["local_fallback_count"] == 0, runtime.snapshot
        assert (
            runtime.transport.snapshot["last_schedule"]["predicted_cache_hit_tokens"]
            == remote["meta_info"]["cached_tokens"]
        ), runtime.snapshot

        async def cancel_stream():
            request = GenerateReqInput(
                text=prompt,
                rid="smoke_cancel",
                stream=True,
                sampling_params={"temperature": 0, "max_new_tokens": 128},
            )
            stream = runtime.generate(request, None)
            first = await anext(stream)
            assert first["meta_info"]["completion_tokens"] > 0, first
            await stream.aclose()
            assert runtime.transport._streams == {}

        loop.run_until_complete(cancel_stream())

        async def check_http_entries():
            import httpx
            from sglang.srt.entrypoints import http_server

            http_server.set_global_state(
                http_server._GlobalState(manager, engine.template_manager, {})
            )
            http_server.app.state.openai_serving_chat = manager.serving_chat_class(
                manager, engine.template_manager
            )
            before = runtime.transport.snapshot["remote_request_count"]
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=http_server.app),
                base_url="http://sglang-smoke",
            ) as client:
                response = await client.post(
                    "/generate", json={"text": prompt, "sampling_params": sampling}
                )
                assert response.status_code == 200, response.text
                assert (
                    response.json()["output_ids"] == remote["output_ids"]
                ), response.text
                chat = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": manager.served_model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "max_tokens": 12,
                        "stream": True,
                    },
                )
                assert chat.status_code == 200, chat.text
                chunks = [
                    json.loads(line[6:])
                    for line in chat.text.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
                assert "data: [DONE]" in chat.text, chat.text
                assert any(
                    choice.get("finish_reason") in ("length", "stop")
                    for chunk in chunks
                    for choice in chunk["choices"]
                ), chunks
                text = "".join(
                    choice.get("delta", {}).get("content") or ""
                    for chunk in chunks
                    for choice in chunk["choices"]
                )
                assert text, chunks
            assert runtime.transport.snapshot["remote_request_count"] == before + 2
            assert runtime.transport.snapshot["local_fallback_count"] == 0
            return {
                "generate_output_matches_python": True,
                "openai_sse_text": text,
                "additional_remote_requests": 2,
                "transport": "ASGI in process",
            }

        http_entries = loop.run_until_complete(
            asyncio.wait_for(check_http_entries(), 90)
        )
        master_state = loop.run_until_complete(
            wait_for_master_snapshot(
                snapshot_dir / "master_snapshot",
                master.grpc_port,
                runtime.agent.grpc_port,
            )
        )
        assert runtime._execution_tasks == {}, runtime._execution_tasks
        assert runtime._batches == {}, runtime._batches
        assert manager.rid_to_state == {}, manager.rid_to_state
        assert runtime.transport._active_rids == set()
        assert runtime.transport._async_client.pending == {}
        assert not runtime.transport._async_client.notification.closed
        assert not runtime._notification.closed
        assert runtime.snapshot["scheduler"]["request_count"] == 0, runtime.snapshot
        assert runtime.snapshot["agent_workload_accepted"], runtime.snapshot
        assert runtime.snapshot["forwarded_request_count"] == 4, runtime.snapshot
        assert (
            runtime.snapshot["prefill_completed_request_count"] == 4
        ), runtime.snapshot
        assert runtime.transport.snapshot["remote_request_count"] == 4, runtime.snapshot
        print(
            json.dumps(
                {
                    "local": local,
                    "no_master_fallback_count": no_master_fallback_count,
                    "remote": remote,
                    "snapshot": runtime.snapshot,
                    "master_adopted_state": master_state,
                    "source_and_target_requests_drained": True,
                    "async_schedules_drained": True,
                    "native_event_notifications": True,
                    "http_entries": http_entries,
                    "disable_overlap": args.disable_overlap,
                    "stream_cancelled": True,
                    "native_master_grpc_port": master.grpc_port,
                },
                ensure_ascii=True,
            )
        )
    finally:
        if engine is not None:
            engine.shutdown()
        if master is not None:
            master.stop()


if __name__ == "__main__":
    main()
