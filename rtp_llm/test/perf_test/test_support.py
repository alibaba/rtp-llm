"""Shared performance fixtures; no test cases live here."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.storage import RunStore


def prefill_profile_setup(root):
    config = {
        "schema_version": 1,
        "model": fixture_model(),
        "engine": fixture_engine(max_seq_len=4097, concurrency_limit=1),
        "workload": fixture_workload(compute_steps=[256], cache_lengths=[0]),
        "execution": fixture_execution(),
        "output": {},
        "regression": {},
        "publish": {},
    }
    identity = {
        "input_len": 512,
        "cache_len": 0,
        "compute_len": 512,
        "batch_size": 1,
        "measure_runs": 1,
        "seed": 1,
        "prefix_policy": "shared_seed",
    }
    identity["measurement_contract"] = "concurrent_batch_v1"
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
        and "profiles" not in path.parts
        and (path.name != ".writer.lock")
    }


def prefill_resume_config(extra_engine_args=None):
    return {
        "schema_version": 1,
        "model": fixture_model(),
        "engine": fixture_engine(
            max_seq_len=4097, concurrency_limit=1, args=extra_engine_args or {}
        ),
        "workload": fixture_workload(compute_steps=[256], cache_lengths=[0]),
        "execution": fixture_execution(),
        "output": {},
        "regression": {},
        "publish": {},
    }


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


prefill_external_TOPOLOGY = {"tp": 1, "dp": 2, "pp": 1, "ep": 1}
prefill_external_COMMIT_TAIL_TOKENS = 8
prefill_external_CASE_INPUT_LEN = 24
prefill_external_CASE_CACHE_LEN = 16


def prefill_external_prefill_source(result_root):
    return {
        "schema_version": 1,
        "model": fixture_model(
            label="external-prefill",
            tokenizer="/m",
            topology=dict(prefill_external_TOPOLOGY),
        ),
        "engine": fixture_engine(
            max_seq_len=64, max_batch_tokens=64, concurrency_limit=1
        ),
        "workload": fixture_workload(
            max_input_len=24,
            cache_alignment=8,
            commit_tail_tokens=prefill_external_COMMIT_TAIL_TOKENS,
            compute_steps=[8],
            cache_lengths=[prefill_external_CASE_CACHE_LEN],
            measure_runs=1,
            seed=7,
        ),
        "execution": fixture_execution(devices=[0, 1]),
        "output": {"result_root": str(result_root)},
        "regression": {},
        "publish": {},
    }


def prefill_external_write_prefill_config(root):
    path = root / "prefill.json"
    path.write_text(
        json.dumps(prefill_external_prefill_source(root / "results")), encoding="utf-8"
    )
    return path


class prefill_external_Tokenizer:

    def encode(self, text):
        return text.split()


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


def config_resolution_base_config():
    return {
        "schema_version": 1,
        "model": fixture_model(
            label="test-model", type="registered_model", checkpoint="/models/test"
        ),
        "engine": fixture_engine(args={"act_type": "BF16", "reuse_cache": 1}),
        "workload": fixture_workload(),
        "execution": {"backend": "local"},
        "output": {},
        "regression": {},
        "publish": {},
    }


def fixture_execution(**overrides):
    """Fresh minimal execution fixture with case-specific overrides."""
    return {**{"backend": "local", "devices": [0]}, **overrides}


def fixture_workload(**overrides):
    """Fresh minimal workload fixture with case-specific overrides."""
    return {
        **{
            "kind": "cache_grid",
            "max_input_len": 2048,
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "compute_steps": [128],
            "batch_sizes": [1],
        },
        **overrides,
    }


def fixture_model(**overrides):
    """Fresh minimal model fixture with case-specific overrides."""
    return {**{"label": "test", "type": "registered", "checkpoint": "/m"}, **overrides}


def fixture_engine(**overrides):
    """Fresh minimal engine fixture with case-specific overrides."""
    return {
        **{"max_seq_len": 4096, "max_batch_tokens": 8192, "concurrency_limit": 8},
        **overrides,
    }
