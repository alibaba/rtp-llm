"""Performance framework behavior and failure regressions."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.modes import ModeExecutionContext, PrefillAdapter
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.prefill.runner import PrefillRunner
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.storage import RunStore
from rtp_llm.test.perf_test.tests.support.fixtures import (
    fixture_config,
    fixture_engine,
    fixture_execution,
    fixture_model,
    fixture_workload,
)


def prefill_external_run_through_session(plan, session, run_root):
    """Drive the prefill runner factory exactly like the unified executor."""
    (run_root / "results").mkdir(parents=True, exist_ok=True)
    (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    adapter = PrefillAdapter(
        lambda context: PrefillRunner(
            context.session.client,
            prefill_external_Tokenizer(),
            commit_tail_tokens=prefill_external_COMMIT_TAIL_TOKENS,
            max_batch_tokens=plan.resolved_config["engine"]["max_batch_tokens"],
            monotonic=iter([0.0, 0.2]).__next__,
        )
    )
    return adapter(
        ModeExecutionContext(plan=plan, session=session, store=RunStore(run_root))
    )


class prefill_runner_Tokenizer:

    def encode(self, text):
        return text.split()


class prefill_runner_Client:

    def __init__(self, responses, batch_size=1):
        self.responses = {}
        seed_index = measure_index = 0
        for response in responses:
            if response["input_len"] <= 12:
                key = ("seed", None, seed_index)
                seed_index += 1
            else:
                key = (
                    "measure",
                    measure_index // batch_size,
                    measure_index % batch_size,
                )
                measure_index += 1
            self.responses[key] = response
        self.requests = []
        self.scheduler_calls = []

    def configure_scheduler(self, batch_size, mode="prefill"):
        assert mode == "prefill"
        self.scheduler_calls.append(batch_size)

    def request(self, **request):
        self.requests.append(request)
        if request["phase"] == "warmup":
            return self.responses["measure", 0, request["request_index"]]
        return self.responses[
            request["phase"], request["run_index"], request["request_index"]
        ]


def prefill_runner_response(
    input_len, reuse_len, latency=10.0, success=True, output_len=1
):
    return {
        "success": success,
        "input_len": input_len,
        "output_len": output_len,
        "reuse_len": reuse_len,
        "latency_ms": latency,
    }


def prefill_runner_case(cache_len=8, batch_size=1, measure_runs=2):
    return {
        "case_id": "prefill-case",
        "input_len": 16,
        "cache_len": cache_len,
        "compute_len": 16 - cache_len,
        "batch_size": batch_size,
        "measure_runs": measure_runs,
        "seed": 7,
        "prefix_policy": "shared_seed" if batch_size == 1 else "independent",
    }


def prefill_runner_run(client, case=None, **kwargs):
    """Run a scripted case with the common tokenizer and admission limits."""
    return PrefillRunner(
        client,
        prefill_runner_Tokenizer(),
        commit_tail_tokens=4,
        max_batch_tokens=4096,
        **kwargs,
    ).run_case(case or prefill_runner_case())


def prefill_runner_grouped_case():
    return dict(
        prefill_runner_case(batch_size=3, measure_runs=1),
        input_len=24,
        cache_len=8,
        compute_len=16,
        prefix_policy="independent",
        measurement_contract="concurrent_grouped_batch_v1",
        request_groups=[
            {"count": 1, "input_len": 16, "cache_len": 8},
            {"count": 2, "input_len": 24, "cache_len": 0},
        ],
    )


class SeedBatchClient:
    """Require complete concurrent scheduler groups without a GPU server."""

    def __init__(self, cache_lengths, *, failed_member=None, raised_member=None):
        self.cache_lengths = cache_lengths
        self.failed_member = failed_member
        self.raised_member = raised_member
        self.batches = []
        self.completed_seeds = set()
        self.lock = threading.Lock()

    def configure_scheduler(self, batch_size, mode="prefill"):
        assert mode == "prefill"
        if self.batches:
            assert self.batches[-1]["completed"] == len(self.batches[-1]["requests"])
        self.batches.append(
            {
                "barrier": threading.Barrier(batch_size, timeout=2),
                "requests": [],
                "completed": 0,
            }
        )

    def request(self, **request):
        with self.lock:
            batch = self.batches[-1]
            batch["requests"].append(request)
        try:
            batch["barrier"].wait()
            member = request["request_index"]
            seed = request["phase"] == "seed"
            if not seed:
                assert self.completed_seeds == {
                    index for (index, length) in enumerate(self.cache_lengths) if length
                }
            if seed and member == self.raised_member:
                raise RuntimeError("seed transport failed")
            return prefill_runner_response(
                len(request["input_ids"]),
                0 if seed else self.cache_lengths[member],
                success=not (seed and member == self.failed_member),
            )
        finally:
            with self.lock:
                batch["completed"] += 1
                if request["phase"] == "seed":
                    self.completed_seeds.add(request["request_index"])


def prefill_profile_setup(root):
    config = fixture_config(
        engine=fixture_engine(max_seq_len=4097, concurrency_limit=1),
        workload=fixture_workload(compute_steps=[256], cache_lengths=[0]),
    )
    identity = dict(prefill_resume_CASES[0], measurement_contract="concurrent_batch_v1")
    plan = build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(config),
        cases=[identity],
        result_directory=str(root),
    )
    initialize_run_directory(
        root, source_config=config, plan=plan, run_id="run", code_version="rev"
    )
    store = RunStore(root)
    store.write_checkpoint("prefill_progress", {"formal": True})
    store.write_prefill_case(
        plan.cases[0].case_id, {"case_id": plan.cases[0].case_id, "status": "ok"}
    )
    store.write_final_result({"status": "completed"})
    store.write_summary({"successful_cases": 1})
    return plan


def prefill_profile_formal_bytes(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
        and not {"profiles", "traces"}.intersection(path.relative_to(root).parts)
        and (path.name != ".writer.lock")
    }


def prefill_resume_config(extra_engine_args=None):
    return fixture_config(
        engine=fixture_engine(
            max_seq_len=4097, concurrency_limit=1, args=extra_engine_args or {}
        ),
        workload=fixture_workload(compute_steps=[256], cache_lengths=[0]),
    )


prefill_resume_CASES = [
    {
        "input_len": 512,
        "cache_len": 0,
        "compute_len": 512,
        "batch_size": 1,
        "measure_runs": 1,
        "seed": 1,
        "prefix_policy": "shared_seed",
    },
    {
        "input_len": 1024,
        "cache_len": 512,
        "compute_len": 512,
        "batch_size": 1,
        "measure_runs": 1,
        "seed": 1,
        "prefix_policy": "shared_seed",
    },
]


def prefill_resume_plan(root, action="run", config=None):
    return build_execution_plan(
        mode="prefill",
        action=action,
        resolved=resolve_config(config or prefill_resume_config()),
        cases=[
            dict(case, measurement_contract="concurrent_batch_v1")
            for case in prefill_resume_CASES
        ],
        result_directory=str(root),
    )


prefill_external_TOPOLOGY = {"tp": 2, "dp": 1, "pp": 1, "ep": 2}
prefill_external_COMMIT_TAIL_TOKENS = 8
prefill_external_CASE_INPUT_LEN = 24
prefill_external_CASE_CACHE_LEN = 16


def prefill_external_prefill_source(result_root):
    return fixture_config(
        model=fixture_model(
            label="external-prefill",
            tokenizer="/m",
            topology=dict(prefill_external_TOPOLOGY),
        ),
        engine=fixture_engine(max_seq_len=64, max_batch_tokens=64, concurrency_limit=1),
        workload=fixture_workload(
            transport="http_text",
            max_input_len=24,
            cache_alignment=8,
            commit_tail_tokens=prefill_external_COMMIT_TAIL_TOKENS,
            compute_steps=[8],
            cache_lengths=[prefill_external_CASE_CACHE_LEN],
            measure_runs=1,
            seed=7,
        ),
        execution=fixture_execution(devices=[0, 1]),
        output={"result_root": str(result_root)},
    )


def prefill_external_write_prefill_config(root):
    path = root / "prefill.json"
    path.write_text(
        json.dumps(prefill_external_prefill_source(root / "results")), encoding="utf-8"
    )
    return path


prefill_external_Tokenizer = prefill_runner_Tokenizer


def prefill_external_aux(input_len, reuse_len, latency_ms):
    return {
        "input_len": input_len,
        "output_len": 1,
        "reuse_len": reuse_len,
        "first_token_cost_time": latency_ms,
    }


class prefill_external_FakeModelServer:
    """Loopback model server that replays scripted aux_info payloads."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        outer = self

        class _Handler(BaseHTTPRequestHandler):

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(
                    {"path": self.path, "headers": dict(self.headers), "body": body}
                )
                if self.path == "/update_scheduler_info":
                    response = {"status": "ok"}
                else:
                    response = {"aux_info": outer._responses.pop(0)}
                payload = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def endpoint(self):
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
