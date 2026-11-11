"""Profile the last warmed Prefill wave of the existing reference workload.

The original reference driver still owns the prompts, batch gate, HTTP checks,
eight-device lease and service teardown. Profiled results are labelled and
must not be used as clean latency measurements.
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

    if config["case"]["phase"] != "prefill" or not config["node"].get("profile_last_wave"):
        raise ValueError("profile entry requires an explicit Prefill profile deployment")
    expected = config["case"]["warmup_steps"] + config["case"]["measure_steps"]
    directory = Path(config["node"]["environment"]["TORCH_CUDA_PROFILER_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    original = httpx.AsyncClient.post
    waves = 0
    configured = False

    async def post(client, url, *args, **kwargs):
        nonlocal waves, configured
        if str(url).endswith("/batch_infer"):
            waves += 1
            if waves == expected:
                request = {
                    "trace_name": config["node"].get("profile_trace_name", "cp8_prefill"),
                    "start_step": 0,
                    "num_steps": 1,
                    "enable_all_rank": True,
                }
                reply = await original(
                    client, "http://" + config["prefill_address"] + "/start_profile", json=request
                )
                reply.raise_for_status()
                receipt = reply.json()
                if receipt.get("status") != "ok":
                    raise RuntimeError(f"native profiler rejected the wave: {receipt}")
                reference.save(Path(config["output"]) / "profile-control.json", {
                    "request": request, "response": receipt,
                    "wave": waves, "profiled": True, "clean_timing": False,
                })
                configured = True
        return await original(client, url, *args, **kwargs)

    httpx.AsyncClient.post = post
    try:
        await reference.load(config)
    finally:
        httpx.AsyncClient.post = original
    if waves != expected or not configured:
        raise RuntimeError(f"profile did not cover the expected warmed wave: {waves}/{expected}")
    deadline = time.monotonic() + 180
    complete = {}
    while time.monotonic() < deadline:
        for path in directory.glob("*.json"):
            match = re.search(r"_wr(\d+)_\d+\.json$", path.name)
            if match is None or path.name in complete:
                continue
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if any(event.get("cat") == "kernel" for event in data.get("traceEvents", ())):
                complete[path.name] = {"rank": int(match[1]), "bytes": path.stat().st_size}
        if {item["rank"] for item in complete.values()} == set(range(8)):
            reference.save(Path(config["output"]) / "profile-traces.json", complete)
            return
        await asyncio.sleep(1)
    raise RuntimeError(f"missing complete all-rank GPU traces: {complete}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "load":
        raise ValueError("profile entry is only a reference load subprocess")
    asyncio.run(load(json.loads(sys.argv[2])))
