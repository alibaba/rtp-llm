"""Manual HTTP benchmark with the standard smoke service lifecycle.

Private checkpoint, requests and settings are supplied via MKT_FIXTURE_*.
MKT_E2E_MODE selects off/legacy/optimized. Metrics contain no request text;
responses for candidate-set comparisons stay in local mode-0600 files.
"""

import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import statistics
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import aiohttp

from rtp_llm.test.smoke.rocm_mkt_cascade_fixture_capture_test import (
    CONFIG_KEYS,
    _configure_local_service_proxy_environment,
    _json_hash,
    _private_manager_logs,
    _PrivateServerManager,
    _required_path,
    _sha_file,
    _write_json,
)


def _stats(values):
    if not values:
        return {}
    values = sorted(values)

    def quantile(p):
        at = (len(values) - 1) * p
        lo, hi = math.floor(at), math.ceil(at)
        return values[lo] + (values[hi] - values[lo]) * (at - lo)

    return dict(
        mean=statistics.mean(values),
        p50=quantile(0.5),
        p95=quantile(0.95),
        p99=quantile(0.99),
        max=values[-1],
    )


def _parse_response(raw):
    text = raw.decode()
    if text.startswith("data:"):
        records = [
            json.loads(line[5:].strip())
            for line in text.splitlines()
            if line.startswith("data:") and line[5:].strip() != "[DONE]"
        ]
        return records[-1] if records else {}
    return json.loads(text)


async def _phase(
    port,
    samples,
    output,
    name,
    *,
    concurrency=1,
    count=80,
    qps=0,
    duration=20,
    aux=False,
):
    directory = output / name
    directory.mkdir(mode=0o700)
    rows, saved = [], {}
    limit = asyncio.Semaphore(512)
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=512),
        timeout=aiohttp.ClientTimeout(total=90),
        trust_env=False,
    ) as session:

        async def request(index, scheduled=None):
            async with limit:
                start = time.perf_counter()
                payload = copy.deepcopy(samples[index % len(samples)])
                payload.pop("__request_id__", None)
                config = payload["generate_config"]
                for key in (
                    "traceId",
                    "send_engine_timestamp",
                    "gen_timeline",
                    "profile_step",
                ):
                    config.pop(key, None)
                config.update(aux_info=aux, timeout_ms=60000)
                status, error, obj, raw = 0, None, {}, b""
                try:
                    async with session.post(
                        f"http://127.0.0.1:{port}/", json=payload
                    ) as response:
                        status, raw = response.status, await response.read()
                    end = time.perf_counter()
                    obj = _parse_response(raw)
                    if (
                        status != 200
                        or obj.get("error")
                        or obj.get("error_code") not in (None, 0)
                    ):
                        # Keep response/error text private, including unexpected errors.
                        error = (
                            "admission"
                            if "CONCURRENCY_LIMIT" in raw.decode(errors="replace")
                            else "service_error"
                        )
                except Exception as exc:
                    end = time.perf_counter()
                    error = type(exc).__name__
                sequences = obj.get("response", [])
                finished = obj.get("finished")
                complete = finished is True or (
                    isinstance(finished, list)
                    and bool(finished)
                    and all(x is True for x in finished)
                )
                ok = (
                    status == 200
                    and error is None
                    and complete
                    and isinstance(sequences, list)
                    and len(sequences) == 1000
                )
                if not ok and error is None:
                    error = "invalid_completion"
                sample = index % len(samples)
                row = dict(
                    sample=sample,
                    ok=ok,
                    status=status,
                    error=error,
                    start=start,
                    end=end,
                    rt_ms=(end - start) * 1000,
                    schedule_lag_ms=max(0, start - (scheduled or start)) * 1000,
                    sequence_count=len(sequences),
                    response_bytes=len(raw),
                    output_hash=_json_hash(sequences),
                )
                rows.append(row)
                if ok and str(sample) not in saved:
                    saved[str(sample)] = obj
                elif not ok and "first_failure" not in saved:
                    saved["first_failure"] = obj

        wall_start, begin = time.time(), time.perf_counter()
        if qps:
            tasks = []
            for index in range(round(qps * duration)):
                scheduled = begin + index / qps
                await asyncio.sleep(max(0, scheduled - time.perf_counter()))
                tasks.append(asyncio.create_task(request(index, scheduled)))
            await asyncio.gather(*tasks)
        else:
            indices = iter(range(count))

            async def worker():
                for index in indices:
                    await request(index)

            await asyncio.gather(*(worker() for _ in range(concurrency)))
        end = time.perf_counter()
    successful = [row for row in rows if row["ok"]]
    summary = dict(
        name=name,
        concurrency=concurrency,
        offered_qps=qps,
        duration_s=duration if qps else None,
        count=len(rows),
        success=len(successful),
        fail=len(rows) - len(successful),
        errors=dict(Counter(row["error"] for row in rows if not row["ok"])),
        elapsed_s=end - begin,
        wall_start=wall_start,
        wall_end=time.time(),
        success_qps=len(successful) / (end - begin),
        rt_ms=_stats([row["rt_ms"] for row in successful]),
        schedule_lag_ms=_stats([row["schedule_lag_ms"] for row in rows]),
    )
    if qps:
        summary.update(
            completed_during_send_qps=sum(
                row["end"] <= begin + duration for row in successful
            )
            / duration,
            drain_s=max(0, end - (begin + duration)),
        )
    _write_json(directory / "summary.json", summary)
    _write_json(directory / "requests.json", rows)
    _write_json(directory / "responses.private.json", saved)
    print("E2E_METRIC " + json.dumps(summary, sort_keys=True), flush=True)
    return summary


class RocmMktCascadeE2ETest(unittest.TestCase):
    def test_service_rt_qps(self):
        mode = os.environ["MKT_E2E_MODE"]
        self.assertIn(mode, ("off", "legacy", "optimized"))
        model = _required_path("MKT_FIXTURE_MODEL_DIR", directory=True)
        requests_path = _required_path("MKT_FIXTURE_REQUESTS_JSON")
        env_path = _required_path("MKT_FIXTURE_ENV_JSON")
        samples = json.loads(requests_path.read_text())
        env = json.loads(env_path.read_text())
        self.assertTrue(set(env) <= CONFIG_KEYS)
        self.assertEqual(env["QUANTIZATION"], "FP8_PER_CHANNEL_COMPRESSED")
        self.assertEqual(env["CONCURRENCY_LIMIT"], "8")
        self.assertEqual(env["KV_CACHE_MEM_MB"], "32768")
        for sample in samples:
            config = sample["generate_config"]
            self.assertEqual(config["variable_num_beams"], [500, 1000])
            self.assertEqual(config["max_new_tokens"], 2)
        output = Path(os.environ["MKT_E2E_OUTPUT_DIR"]).absolute()
        self.assertFalse(output.exists(), "use a new result directory")
        output.mkdir(parents=True, mode=0o700)
        env.update(
            {
                "ENABLE_DECODE_CASCADE_ATTENTION": "0" if mode == "off" else "1",
                "RTP_ROCM_CASCADE_FP32": "1" if mode == "optimized" else "0",
                "SEQ_SIZE_PER_BLOCK": "16",
                "KERNEL_SEQ_SIZE_PER_BLOCK": "16",
                "FP8_KV_CACHE": "0",
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8",
                "RAYON_NUM_THREADS": "8",
                "FRONTEND_PRE_STOP_DRAIN_SECONDS": "0",
                "BACKEND_POST_FRONTEND_DRAIN_SECONDS": "0",
                "SHUTDOWN_TIMEOUT": "10",
            }
        )
        server_cpus = {int(x) for x in os.environ["MKT_E2E_SERVER_CPUS"].split(",")}
        client_cpus = {int(x) for x in os.environ["MKT_E2E_CLIENT_CPUS"].split(",")}
        original_cpus = os.sched_getaffinity(0)
        self.assertTrue(server_cpus <= original_cpus and client_cpus <= original_cpus)
        self.assertFalse(server_cpus & client_cpus)
        rates = [
            float(x) for x in os.getenv("MKT_E2E_QPS", "30,40,45,50,55").split(",") if x
        ]
        duration = float(os.getenv("MKT_E2E_DURATION", "20"))
        count = int(os.getenv("MKT_E2E_CLOSED_COUNT", "320"))
        self.assertTrue(
            duration > 0 and count > 0 and all(0 < rate <= 256 for rate in rates)
        )
        source_root = Path(__file__).resolve().parents[2]
        source_files = [
            source_root / name
            for name in (
                "models_py/triton_kernels/attention/beam_search_cascade.py",
                "models_py/triton_kernels/attention/beam_search_cascade_native.py",
                "models_py/triton_kernels/attention/beam_search_cascade_tail.py",
                "models_py/modules/factory/attention/rocm_impl/beam_search_cascade.py",
                "test/smoke/rocm_mkt_cascade_e2e_test.py",
            )
        ]
        provenance = dict(
            mode=mode,
            effective_environment=env,
            request_file_sha256=_sha_file(requests_path),
            model_config_sha256=_sha_file(model / "config.json"),
            source_sha256={
                str(path.relative_to(source_root)): _sha_file(path)
                for path in source_files
            },
            server_cpus=sorted(server_cpus),
            client_cpus=sorted(client_cpus),
            physical_gpu_visibility=os.getenv("HIP_VISIBLE_DEVICES"),
            graph_enabled=True,
            reuse_cache=False,
            response_sequence_count=1000,
            timing_boundary="client request construction through full HTTP response bytes; success RT only",
            arrival_pattern="fixed interval; success QPS includes drain",
        )
        _write_json(output / "provenance.json", provenance)
        umask = os.umask(0o077)
        summaries = []
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "TEST_UNDECLARED_OUTPUTS_DIR": str(output),
                    "MAGA_SERVER_WORK_DIR": str(output),
                },
            ), _private_manager_logs():
                _configure_local_service_proxy_environment()
                # Disable diagnostic hooks in the benchmark process and its children.
                for key in list(os.environ):
                    if key.startswith("RTP_HOT_HOOK"):
                        os.environ.pop(key)
                os.sched_setaffinity(0, server_cpus)
                manager = _PrivateServerManager(
                    env_args=env,
                    device_ids=[],
                    smoke_args_str=os.getenv("SMOKE_ARGS", ""),
                )
                try:
                    self.assertTrue(
                        manager.start_server(
                            model_path=str(model),
                            model_type="qwen_3",
                            tokenizer_path=str(model),
                            timeout=600,
                        ),
                        "service startup failed; inspect private logs",
                    )
                    os.sched_setaffinity(0, client_cpus)

                    async def run():
                        for name, concurrency, n, aux in (
                            ("warmup", 8, 64, False),
                            ("accuracy", 1, len(samples) * 2, True),
                            ("c1", 1, 80, False),
                            ("c2", 2, count, False),
                            ("c4", 4, count, False),
                            ("c8", 8, count * 2, False),
                        ):
                            result = await _phase(
                                manager.port,
                                samples,
                                output,
                                name,
                                concurrency=concurrency,
                                count=n,
                                aux=aux,
                            )
                            summaries.append(result)
                            self.assertEqual(
                                result["fail"], 0, "closed-loop request failed: " + name
                            )
                        for rate in rates:
                            result = await _phase(
                                manager.port,
                                samples,
                                output,
                                "qps" + str(rate),
                                qps=rate,
                                duration=duration,
                            )
                            summaries.append(result)
                            self.assertTrue(
                                set(result["errors"]) <= {"admission"},
                                "unexpected service errors under load",
                            )

                    asyncio.run(run())
                finally:
                    manager.stop_server()
        finally:
            os.sched_setaffinity(0, original_cpus)
            os.umask(umask)
            _write_json(output / "summary.json", dict(mode=mode, phases=summaries))
        self.assertEqual(_sha_file(requests_path), provenance["request_file_sha256"])
        self.assertEqual(
            {
                str(path.relative_to(source_root)): _sha_file(path)
                for path in source_files
            },
            provenance["source_sha256"],
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
