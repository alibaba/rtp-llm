"""Profile a selected warmed Prefill wave of the reference workload.

The original reference driver still owns the prompts, batch gate, HTTP checks,
eight-device lease and service teardown. Warmup capture waits for complete
traces before the clean measurement waves; measured capture stays labelled.
"""

import asyncio
import json
from pathlib import Path
import re
import sys
import time

from rtp_llm.test.perf_test import dsv41_ppu_reference as reference


async def load(config):
    import httpx

    node, case = config["node"], config["case"]
    if case["phase"] != "prefill" or not (node.get("profile_last_wave") or node.get("profile_warmup_wave")):
        raise ValueError("profile entry requires an explicit Prefill profile deployment")
    if case.get("prefill_sampling") == "sglang-gist" and node.get("profile_last_wave"):
        raise ValueError("queued gist sampling supports a warmup profile only")
    expected = case["warmup_steps"] + case["measure_steps"]
    capture_wave = int(node.get("profile_warmup_wave", expected))
    if node.get("profile_warmup_wave") and not 1 <= capture_wave <= case["warmup_steps"]:
        raise ValueError("warmup profile must finish before measured waves")
    directory = Path(config["node"]["environment"]["TORCH_CUDA_PROFILER_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    original = httpx.AsyncClient.post
    owners = case.get("prefill_dp", 1)
    submitted = 0
    completed = 0
    configured = False
    arm_task = None
    trace_prefix = node.get("profile_trace_name", "dp8_prefill" if owners > 1 else "cp8_prefill")

    async def wait_traces():
        deadline = time.monotonic() + 180
        complete = {}
        while time.monotonic() < deadline:
            candidates = [path for path in directory.glob("*.json")
                          if path.name.startswith(trace_prefix + "_wr")]
            present = {int(match[1]) for path in candidates
                       if (match := re.search(r"_wr(\d+)_\d+\.json$", path.name))}
            if present != set(range(8)):
                await asyncio.sleep(1)
                continue
            if len(candidates) != 8:
                raise RuntimeError(f"expected one trace per rank for {trace_prefix}: {candidates}")
            for path in candidates:
                match = re.search(r"_wr(\d+)_\d+\.json$", path.name)
                if match is None or path.name in complete:
                    continue
                try:
                    data = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                events = data.get("traceEvents", ())
                if owners > 1:
                    marker = (f"executor.model_forward(ctx_batch={case['batch']},gen_batch=0,"
                              f"tokens={case['batch'] * reference.result.FRESH},max_seq={reference.result.CONTEXT})")
                    if sum(event.get("cat") == "cpu_op" and event.get("name") == marker for event in events) != 1:
                        raise RuntimeError(f"DP trace captured a different model step: {path.name}")
                if any(event.get("cat") == "kernel" for event in events):
                    complete[path.name] = {"rank": int(match[1]), "bytes": path.stat().st_size}
            if {item["rank"] for item in complete.values()} == set(range(8)):
                reference.save(Path(config["output"]) / "profile-traces.json", complete)
                return
            await asyncio.sleep(1)
        raise RuntimeError(f"missing complete all-rank GPU traces: {complete}")

    async def arm(client):
        nonlocal configured
        request = {
            "trace_name": node.get("profile_trace_name", "cp8_prefill"),
            "start_step": 0, "num_steps": 1, "enable_all_rank": True,
        }
        addresses = reference.owner_addresses(config["prefill_address"], owners)
        # The CP request broadcasts within its TP group. DP capture is tied
        # to the actual requests below, so idle work cannot consume its step.
        replies = await asyncio.gather(*(
            original(client, "http://" + address + "/start_profile", json=request)
            for address in addresses
        ))
        receipts = []
        for address, reply in zip(addresses, replies):
            reply.raise_for_status()
            receipt = reply.json()
            if receipt.get("status") != "ok":
                raise RuntimeError(f"native profiler rejected the wave on {address}: {receipt}")
            receipts.append(receipt)
        reference.save(Path(config["output"]) / "profile-control.json", {
            "request": request, "response": receipts[0] if owners == 1 else receipts,
            "addresses": addresses,
            "wave": capture_wave, "owners": owners, "profiled": True,
            "clean_timing": capture_wave <= case["warmup_steps"],
        })
        configured = True

    async def post(client, url, *args, **kwargs):
        nonlocal submitted, completed, arm_task, configured
        wave = None
        if str(url).endswith("/batch_infer"):
            wave = submitted // owners + 1
            submitted += 1
            if wave == capture_wave:
                if owners > 1:
                    payload = json.loads(kwargs["content"])
                    request = {
                        "gen_timeline": True, "profile_step": 1,
                        "profile_trace_name": node.get("profile_trace_name", "dp8_prefill"),
                    }
                    payload["generate_config"].update(request)
                    kwargs["content"] = json.dumps(payload).encode()
                    if submitted == capture_wave * owners:
                        reference.save(Path(config["output"]) / "profile-control.json", {
                            "request": request, "arming": "request",
                            "addresses": reference.owner_addresses(config["prefill_address"], owners),
                            "wave": capture_wave, "owners": owners, "profiled": True,
                            "clean_timing": capture_wave <= case["warmup_steps"],
                        })
                        configured = True
                else:
                    if arm_task is None:
                        arm_task = asyncio.create_task(arm(client))
                    await arm_task
        response = await original(client, url, *args, **kwargs)
        if wave == capture_wave:
            completed += 1
            if completed == owners:
                await wait_traces()
        return response

    httpx.AsyncClient.post = post
    try:
        await reference.load(config)
    finally:
        httpx.AsyncClient.post = original
    batches = case["warmup_steps"] + 1 if case.get("prefill_sampling") == "sglang-gist" else expected
    if submitted != batches * owners or not configured:
        raise RuntimeError(f"profile did not cover the expected warmed wave: {submitted}/{batches * owners}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "load":
        raise ValueError("profile entry is only a reference load subprocess")
    asyncio.run(load(json.loads(sys.argv[2])))
