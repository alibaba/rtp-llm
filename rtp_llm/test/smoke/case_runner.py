import concurrent.futures
import json
import logging
import os
import shutil
import signal
import threading
import time
import traceback

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class _TensorEncoder(json.JSONEncoder):
    def default(self, o):
        if _HAS_TORCH and isinstance(o, torch.Tensor):
            return o.tolist()
        return super().default(o)


from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from smoke.cache_status_comparer import CacheStatusComparer
from smoke.classifier_comparer import ClassifierComparer
from smoke.common_def import QueryStatus, SmokeException, Tracer
from smoke.dash_grpc_comparer import DASH_ENDPOINT, DashGrpcComparer
from smoke.dash_sc_grpc_comparer import (
    DASH_SC_GRPC_ENDPOINT,
    DashScGrpcComparer,
)
from smoke.embedding_comparer import EmbeddingComparer
from smoke.gpu_diagnostics import (
    ExceptionType,
    ProcessFailureType,
    classify_exception,
    classify_process_exit,
    dump_gpu_state,
    scan_process_log,
    snapshot_dmesg,
)
from smoke.normal_comparer import NormalComparer
from smoke.openai_comparer import OpenaiComparer
from smoke.remote_kvcm_server import RemoteKVCMServer
from smoke.reranker_comparer import RerankerComparer
from smoke.similarity_comparer import SimilarityComparer
from smoke.task_info import TaskInfo, TaskStates
from smoke.tau2_bench_comparer import Tau2BenchComparer
from smoke.worker_status_comparer import WorkerStatusComparer

from rtp_llm.test.utils.coredump_util import summarize_and_cleanup_coredumps
from rtp_llm.test.utils.maga_server_manager import MagaServerManager
from rtp_llm.utils.util import str_to_bool


def _iterate_modidfy_qr(origin: Dict[str, Any], new: Dict[str, Any]):
    assert isinstance(origin, dict) and isinstance(
        new, dict
    ), f"type_error: origin:{type(origin)} {origin} \n new:{type(new) }{new}"
    for key in list(origin.keys()):
        if key not in new:
            continue
        if isinstance(origin[key], dict):
            assert isinstance(new, dict), f"type_error, new:{type(new)} {new[key]}"
            _iterate_modidfy_qr(origin[key], new[key])
        else:
            origin[key] = new[key]


class CaseRunner(object):
    def __init__(
        self,
        task_info: TaskInfo,
        env_args: List[str],
        gpu_card: str,
        smoke_args: Union[str, Dict[str, str]] = "",
        batch_infer: bool = False,
        kvcm_config: Optional[Dict[str, str]] = None,
        sleep_time_qr: int = 0,
        kill_remote: bool = False,
        concurrency_test: bool = False,
        parallel_qr: int = 1,
    ):
        self.task_info = task_info
        self.env_args = env_args
        self.gpu_card = gpu_card
        if isinstance(smoke_args, dict):
            self.smoke_args = smoke_args
            if "main" in smoke_args:
                self.smoke_args_str = smoke_args["main"]
            elif smoke_args:
                self.smoke_args_str = list(smoke_args.values())[0]
            else:
                self.smoke_args_str = ""
        else:
            self.smoke_args = {}
            self.smoke_args_str = smoke_args if smoke_args else ""
        self.batch_infer = batch_infer
        self.kvcm_config = kvcm_config or {}
        self.sleep_time_qr = sleep_time_qr
        self.kill_remote = kill_remote
        self.concurrency_test = concurrency_test
        # ``parallel_qr`` dispatches up to N fixture queries concurrently via a
        # ThreadPoolExecutor; defaults to 1 (sequential) to preserve existing
        # smoke behavior. >1 requires the server's --concurrency_limit to be at
        # least that value or requests will queue server-side and we waste the
        # parallelism.
        self.parallel_qr = max(1, int(parallel_qr or 1))
        self.remote_kvcm_server: Optional[RemoteKVCMServer] = None

    @staticmethod
    def _extract_bool_arg(args_str: str, arg_name: str, default: bool = False) -> bool:
        """Extract a boolean argument value from a smoke_args string (e.g. '--enable_remote_cache true')."""
        if not args_str:
            return default
        tokens = args_str.split()
        for i, token in enumerate(tokens):
            if token == arg_name and i + 1 < len(tokens):
                return str_to_bool(tokens[i + 1])
        return default

    @staticmethod
    def _extract_int_arg(args_str: str, arg_name: str, default: int) -> int:
        """Extract an integer argument from a smoke argument string."""
        if not args_str:
            return default
        tokens = args_str.split()
        for i, token in enumerate(tokens):
            if token == arg_name and i + 1 < len(tokens):
                try:
                    return int(tokens[i + 1])
                except ValueError:
                    return default
        return default

    def run(self):
        # Subclasses override `_run_impl`, not `run`, so that coredump cleanup
        # always fires even when a runner subclass returns early (PD/DP/Vit/
        # FrontApp separation runners previously skipped cleanup entirely).
        try:
            return self._run_impl()
        finally:
            summarize_and_cleanup_coredumps(
                os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "")
            )

    def _run_impl(self):
        self._dmesg_baseline = snapshot_dmesg()
        keepalive_before_curl = self._keepalive_enabled()
        keepalive_after_curl = self._keepalive_enabled(after_curl=True)
        env_dict = self.create_env_from_args(self.env_args)
        enable_remote_cache = self._extract_bool_arg(
            self.smoke_args_str, "--enable_remote_cache"
        )
        if enable_remote_cache:
            self.remote_kvcm_server = self._start_remote_kvcm_server()
            assert self.remote_kvcm_server is not None, "remote kvcm shoule not be None"
            env_dict["RECO_SERVER_ADDRESS"] = self.remote_kvcm_server.address()
        task_states = TaskStates()
        logging.info(f"smoke_args_str: {self.smoke_args_str}")
        server_manager = self.start_server(
            env_dict,
            task_states,
            self.task_info,
            smoke_args_str=self.smoke_args_str,
        )
        if server_manager is None:
            task_states.ret = False
            return task_states
        if keepalive_before_curl:
            return self._keep_server_alive(server_manager, enable_remote_cache)
        task_states = self.curl_server(server_manager)
        if task_states.ret != True:
            server_manager.stop_server()
            if enable_remote_cache and self.remote_kvcm_server is not None:
                self.remote_kvcm_server.stop_server()
                self.remote_kvcm_server.copy_logs()
            return task_states
        if keepalive_after_curl:
            return self._keep_server_alive(server_manager, enable_remote_cache)
        server_manager.stop_server()
        if enable_remote_cache and self.remote_kvcm_server is not None:
            self.remote_kvcm_server.stop_server()
            self.remote_kvcm_server.copy_logs()
        return task_states

    def _keep_server_alive(
        self, server_manager: MagaServerManager, enable_remote_cache: bool
    ) -> TaskStates:
        return self._keep_servers_alive(
            {"main": server_manager}, enable_remote_cache=enable_remote_cache
        )

    @staticmethod
    def _keepalive_enabled(after_curl: bool = False) -> bool:
        before = str_to_bool(
            os.environ.get("SMOKE_KEEP_SERVER_ALIVE", "False")
        )
        after = str_to_bool(
            os.environ.get("SMOKE_KEEP_SERVER_ALIVE_AFTER_CURL", "False")
        )
        if before and after:
            raise ValueError(
                "SMOKE_KEEP_SERVER_ALIVE and "
                "SMOKE_KEEP_SERVER_ALIVE_AFTER_CURL are mutually exclusive"
            )
        return after if after_curl else before

    def _keep_servers_alive(
        self,
        servers: Dict[str, MagaServerManager],
        enable_remote_cache: bool = False,
    ) -> TaskStates:
        """Publish live endpoints and wait for an explicit stop request."""
        if not servers:
            raise ValueError("keepalive requires at least one server")

        output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())
        live_info_path = os.environ.get(
            "SMOKE_LIVE_INFO", os.path.join(output_dir, "smoke_live_info.json")
        )
        stop_file = os.environ.get(
            "SMOKE_STOP_FILE", os.path.join(output_dir, "smoke_stop")
        )
        stop_event = threading.Event()
        shutdown_errors = []

        def signal_handler(signum, _frame):
            logging.info("keep-alive smoke received signal %s", signum)
            stop_event.set()

        old_handlers = {}
        tmp_live_info = f"{live_info_path}.tmp.{os.getpid()}"
        try:
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGTERM, signal.SIGINT):
                    old_handlers[signum] = signal.getsignal(signum)
                    signal.signal(signum, signal_handler)

            servers_info = {
                role: {
                    "port": manager.port,
                    "server_pid": manager.server_pid,
                    "log_file": manager.log_file_path,
                }
                for role, manager in servers.items()
            }
            live_info = {
                "servers": servers_info,
                "stop_file": stop_file,
                "task_info": self.task_info.taskinfo_rel_path,
                "smoke_args": (
                    self.smoke_args
                    if isinstance(self.smoke_args, dict) and self.smoke_args
                    else self.smoke_args_str
                ),
            }
            if len(servers) == 1:
                only = next(iter(servers.values()))
                live_info.update(
                    {
                        "port": only.port,
                        "server_pid": only.server_pid,
                        "log_file": only.log_file_path,
                    }
                )

            live_info_dir = os.path.dirname(os.path.abspath(live_info_path))
            os.makedirs(live_info_dir, exist_ok=True)
            with open(tmp_live_info, "w", encoding="utf-8") as output:
                json.dump(live_info, output, indent=2, sort_keys=True)
                output.flush()
                os.fsync(output.fileno())
            os.replace(tmp_live_info, live_info_path)
            logging.info(
                "SMOKE_KEEP_SERVER_ALIVE active; live info: %s", live_info
            )

            while not stop_event.is_set():
                if os.path.exists(stop_file):
                    logging.info("keep-alive smoke stop file found: %s", stop_file)
                    break
                for role, manager in servers.items():
                    pid = manager.server_pid
                    if pid is None or not os.path.exists(f"/proc/{pid}"):
                        raise RuntimeError(
                            f"{role} server pid {pid} disappeared during keepalive"
                        )
                stop_event.wait(1)
        finally:
            try:
                if os.path.exists(tmp_live_info):
                    os.unlink(tmp_live_info)
            except Exception as exc:  # noqa: BLE001
                logging.warning("keepalive temp-file cleanup failed: %s", exc)
            for signum, handler in old_handlers.items():
                try:
                    signal.signal(signum, handler)
                except Exception as exc:  # noqa: BLE001
                    logging.warning(
                        "keepalive signal-handler restore failed for signal=%s: %s",
                        signum,
                        exc,
                    )
            for role, manager in reversed(list(servers.items())):
                try:
                    logging.info("stopping keepalive server role=%s", role)
                    manager.stop_server()
                except Exception as exc:  # noqa: BLE001
                    logging.warning(
                        "keepalive stop_server failed for role=%s: %s", role, exc
                    )
                    shutdown_errors.append(f"role={role}: {exc}")
            if (
                enable_remote_cache
                and getattr(self, "remote_kvcm_server", None) is not None
            ):
                try:
                    self.remote_kvcm_server.stop_server()
                except Exception as exc:  # noqa: BLE001
                    logging.warning("keepalive remote KVCM stop failed: %s", exc)
                    shutdown_errors.append(f"remote_kvcm: {exc}")
                try:
                    self.remote_kvcm_server.copy_logs()
                except Exception as exc:  # noqa: BLE001
                    logging.warning("keepalive remote KVCM log copy failed: %s", exc)

        if shutdown_errors:
            raise RuntimeError(
                "keepalive server shutdown failed: " + "; ".join(shutdown_errors)
            )

        return TaskStates()

    def _start_remote_kvcm_server(self) -> Optional[RemoteKVCMServer]:
        server_path = os.path.join(
            os.environ["TEST_SRCDIR"],
            os.environ["TEST_WORKSPACE"],
            "external/remote_kv_cache_manager_server",
        )
        kvcm_src_logs_path = os.path.join(os.environ["TEST_SRCDIR"], "rtp_llm/logs")
        bazel_outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())
        kvcm_dst_logs_path = os.path.join(bazel_outputs_dir, "kvcm_logs")
        remote_kvcm_server = RemoteKVCMServer(
            server_path, self.kvcm_config, kvcm_src_logs_path, kvcm_dst_logs_path
        )
        if remote_kvcm_server.start_server():
            return remote_kvcm_server
        logging.error("start remote_kvcm_server")
        return None

    def curl_server(self, server_manager: MagaServerManager) -> TaskStates:
        if self.concurrency_test:
            task_states = TaskStates()
            # Override hardcoded 5-request concurrency via env (used by DSv4
            # cuda-graph + concurrent decode stress tests). Defaults to 5 to
            # preserve existing smoke behavior.
            n_requests = int(os.environ.get("DSV4_STRESS_REQUESTS", "5"))
            n_workers = max(10, n_requests)

            # DSv4 stress with DIVERSE prompts: each thread sends a different
            # query (round-robin from a built-in list). Verifies per-request
            # KV slot routing — if cross-request stash leakage exists, a
            # request asking "list planets" might receive a "list colors"-
            # style answer (or a mix) because the captured-graph decode
            # would gather the wrong slot's KV state. Each prompt is paired
            # with substrings whose presence in the response proves the right
            # request got the right answer.
            stress_diverse = os.environ.get("DSV4_STRESS_DIVERSE", "0") == "1"
            if stress_diverse:
                task_states = self._run_diverse_stress(
                    server_manager, n_requests, n_workers
                )
                return task_states

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers
            ) as executor:
                futures = [
                    executor.submit(
                        self._curl_server_impl, server_manager, self.task_info
                    )
                    for _ in range(n_requests)
                ]
                results = []
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
                if results[0].ret == False:
                    task_states = results[0]
                else:
                    for result in results:
                        if str(result) != str(results[0]):
                            task_states = result
        else:
            task_states = self._curl_server_impl(server_manager, self.task_info)
        return task_states

    # Diverse-prompt stress mode for DSv4 cuda-graph + concurrent decode
    # validation. Each prompt has substrings whose presence proves the right
    # request received the right answer (cross-request KV leakage would make
    # a "list planets" request return e.g. color words).
    _DIVERSE_STRESS_PROMPTS: List[Tuple[str, List[str]]] = [
        (
            "List the planets of the solar system in order from the sun, "
            "one per line.",
            ["Mercury", "Neptune"],
        ),
        (
            "List the seven days of the week in order, one per line.",
            ["Monday", "Sunday"],
        ),
        (
            "List the four seasons of the year, one per line.",
            ["Spring", "Winter"],
        ),
        (
            "List the primary additive colors of light (red, green, blue), "
            "one per line.",
            ["red", "green", "blue"],
        ),
        (
            "Count from one to five in English words, one per line.",
            ["one", "five"],
        ),
        (
            "Name the chemical symbols of hydrogen, oxygen, and carbon, "
            "one per line.",
            ["H", "O", "C"],
        ),
        (
            "List the months of summer in the Northern Hemisphere, "
            "one per line.",
            ["June", "August"],
        ),
        (
            "List the five English vowel letters, one per line.",
            ["a", "e", "i"],
        ),
    ]

    def _run_diverse_stress(
        self,
        server_manager: MagaServerManager,
        n_requests: int,
        n_workers: int,
    ) -> "TaskStates":
        """Send n_requests parallel HTTP requests with DIFFERENT prompts
        (round-robin from _DIVERSE_STRESS_PROMPTS), then verify each
        response contains the substrings expected for ITS prompt.

        Cross-request KV stash leakage in cuda-graph + bsz>1 decode would
        cause a "List planets" request to receive a "List colors"-style
        answer (or empty / mixed text); the per-prompt substring check
        flags that immediately.
        """
        import requests as _requests

        prompts = self._DIVERSE_STRESS_PROMPTS
        url = f"http://0.0.0.0:{int(server_manager.port)}/v1/chat/completions"

        def _one_request(idx: int) -> Tuple[int, str, List[str], str, str]:
            prompt, expected_substrs = prompts[idx % len(prompts)]
            query = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64,
                "temperature": 0.0,
                "top_p": 0,
                "top_k": 1,
            }
            # Server enforces --concurrency_limit. With many parallel client
            # threads we'll hit ConcurrencyException → HTTP 500 quickly. Retry
            # with exponential backoff so the stress test exercises actual
            # batched decode rather than just rejecting requests.
            max_retries = 8
            backoff = 0.5
            last_err = ""
            for attempt in range(max_retries):
                try:
                    resp = _requests.post(url, json=query, timeout=180)
                    if resp.status_code == 200:
                        data = resp.json()
                        content = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        return (idx, prompt, expected_substrs, content, "")
                    body = resp.text[:200]
                    if resp.status_code == 500 and "Concurrency limit" in body:
                        last_err = body
                        time.sleep(backoff)
                        backoff = min(backoff * 1.5, 4.0)
                        continue
                    return (
                        idx,
                        prompt,
                        expected_substrs,
                        "",
                        f"HTTP {resp.status_code}: {body}",
                    )
                except Exception as e:
                    last_err = f"exception: {e}"
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, 4.0)
            return (
                idx,
                prompt,
                expected_substrs,
                "",
                f"retries exhausted: {last_err}",
            )

        logging.info(
            "[DSv4 STRESS] sending %d concurrent requests with %d distinct "
            "prompts to %s",
            n_requests,
            len(prompts),
            url,
        )
        results: List[Tuple[int, str, List[str], str, str]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_one_request, i) for i in range(n_requests)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda r: r[0])
        passed = 0
        failed_details: List[str] = []
        for idx, prompt, expected, content, err in results:
            if err:
                failed_details.append(f"[{idx}] prompt={prompt!r}: ERROR {err}")
                continue
            # Case-insensitive substring check: prompts ask for things like
            # "one ... five" but models commonly capitalize them ("One",
            # "Five") when starting a list — semantic content is correct
            # so don't flag that as cross-request KV leakage.
            content_lc = content.lower()
            missing = [s for s in expected if s.lower() not in content_lc]
            if missing:
                failed_details.append(
                    f"[{idx}] prompt={prompt!r}: response missing "
                    f"{missing!r}; got {content!r}"
                )
            else:
                passed += 1

        task_states = TaskStates()
        task_states.total_count = n_requests
        if passed == n_requests:
            logging.info(
                "[DSv4 STRESS] PASS: all %d requests received the correct "
                "per-prompt response",
                n_requests,
            )
        else:
            task_states.ret = False
            logging.error(
                "[DSv4 STRESS] FAIL: %d/%d requests passed; %d had wrong "
                "or missing content (cross-request KV leakage suspected)",
                passed,
                n_requests,
                len(failed_details),
            )
            for line in failed_details[:20]:
                logging.error("[DSv4 STRESS]   %s", line)
            if len(failed_details) > 20:
                logging.error(
                    "[DSv4 STRESS]   ... and %d more",
                    len(failed_details) - 20,
                )
            task_states.query_status.append(
                (
                    QueryStatus.COMPARE_FAILED,
                    f"diverse stress failed: {passed}/{n_requests} passed",
                    Tracer(),
                )
            )
        return task_states

    @staticmethod
    def _resolve_endpoint(q_r: Dict[str, Any], task_endpoint: Optional[str]) -> str:
        """Per-query endpoint resolution.

        Queries that carry `prompt_batch` must hit `/batch_infer` so the engine
        atomically enqueues the whole batch via BatchGenerateCall. Hitting the
        default `/` endpoint splits them into independent FIFOScheduler streams,
        which is non-deterministic for beam-search numerics.

        `/batch_infer` is non-streaming only — `prompt_batch` queries with
        `yield_generator: true` are rejected here so test data stays consistent
        with what the endpoint can actually serve.
        """
        explicit = q_r.get("endpoint")
        if explicit:
            return explicit
        query = q_r.get("query", {})
        if "prompt_batch" in query:
            if query.get("yield_generator"):
                raise SmokeException(
                    QueryStatus.VALID_FAILED,
                    "prompt_batch queries must be non-streaming "
                    "(set yield_generator=false); /batch_infer does not stream",
                )
            return "/batch_infer"
        return task_endpoint or "/"

    @staticmethod
    def _get_comparer_cls(q_r: Dict[str, Any], request_endpoint: str) -> Type:
        if request_endpoint == DASH_ENDPOINT:
            return DashGrpcComparer
        if request_endpoint == DASH_SC_GRPC_ENDPOINT:
            return DashScGrpcComparer
        if q_r.get("tau2_bench", False):
            return Tau2BenchComparer
        if "messages" in q_r["query"]:
            return OpenaiComparer
        elif request_endpoint in [
            "/v1/embeddings",
            "/v1/embeddings/dense",
            "/v1/embeddings/sparse",
            "/v1/embeddings/colbert",
        ]:
            return EmbeddingComparer
        elif request_endpoint.startswith("/rtp_llm/worker_status"):
            return WorkerStatusComparer
        elif request_endpoint.startswith("/rtp_llm/cache_status"):
            return CacheStatusComparer
        elif request_endpoint == "/v1/embeddings/similarity":
            return SimilarityComparer
        elif request_endpoint == "/v1/classifier":
            return ClassifierComparer
        elif request_endpoint == "/v1/reranker":
            return RerankerComparer
        elif q_r.get("mainse_module", None) == True:
            if q_r.get("use_decode_arpc", None) == True:
                from smoke.mainse.mainse_decode_arpc_comparer import (
                    MainseDecodeArpcComparer,
                )

                return MainseDecodeArpcComparer
            elif q_r.get("use_emb_arpc", None) == True:
                from smoke.mainse.mainse_embedding_arpc_comparer import (
                    MainseEmbeddingArpcComparer,
                )

                return MainseEmbeddingArpcComparer
            else:
                from smoke.mainse.mainse_comparer import MainseComparer

                return MainseComparer
        return NormalComparer

    def _run_stability_repeat(
        self,
        server_manager: MagaServerManager,
        task_info: TaskInfo,
        task_states: TaskStates,
    ) -> None:
        repeat_count = int(os.environ.get("STABILITY_REPEAT", "0"))
        if repeat_count <= 0 or task_states.ret == False:
            return

        qr_array = task_info.query_result
        task_endpoint = task_info.endpoint
        num_queries = len(qr_array)
        logging.info(
            f"[STABILITY_TEST] Starting {repeat_count} repeat iterations for {num_queries} queries"
        )

        per_query_pass: Dict[int, int] = defaultdict(int)
        per_query_fail: Dict[int, int] = defaultdict(int)
        per_query_responses: Dict[int, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        for iter_idx in range(repeat_count):
            for q_idx, q_r in enumerate(qr_array):
                request_endpoint = self._resolve_endpoint(q_r, task_endpoint)
                comparer_cls = self._get_comparer_cls(q_r, request_endpoint)
                try:
                    comparer_cls(
                        server_manager,
                        request_endpoint,
                        q_r,
                        Tracer(),
                        self.batch_infer,
                    ).run()
                    per_query_pass[q_idx] += 1
                    logging.info(
                        f"[STABILITY_TEST iter={iter_idx+1}/{repeat_count} query={q_idx}] PASS"
                    )
                except Exception as e:
                    exc_type = classify_exception(e)
                    if exc_type != ExceptionType.NOT_GPU_ERROR:
                        output_dir = os.environ.get(
                            "TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd()
                        )
                        dump_gpu_state(
                            exc=e,
                            failure_context=f"stability repeat ({exc_type.value})",
                            log_path=os.path.join(
                                output_dir, "gpu_state_stability.log"
                            ),
                            dmesg_baseline=getattr(self, "_dmesg_baseline", 0),
                        )
                    per_query_fail[q_idx] += 1
                    err_msg = str(e)
                    if "actual.response" in err_msg:
                        start = err_msg.find("actual.response = [")
                        if start != -1:
                            resp = err_msg[start + len("actual.response = [") :]
                            resp = resp.rstrip("]").rstrip()
                            per_query_responses[q_idx][resp] += 1
                    logging.warning(
                        f"[STABILITY_TEST iter={iter_idx+1}/{repeat_count} query={q_idx}] FAIL: {e}"
                    )

        total_checks = repeat_count * num_queries
        total_pass = sum(per_query_pass.values())
        total_fail = sum(per_query_fail.values())
        pass_rate = total_pass / total_checks * 100 if total_checks > 0 else 0

        logging.info(
            f"[STABILITY_SUMMARY] Total: {repeat_count} iterations x {num_queries} queries = {total_checks} checks"
        )
        logging.info(
            f"[STABILITY_SUMMARY] Pass: {total_pass}, Fail: {total_fail} (rate: {pass_rate:.1f}%)"
        )
        for q_idx in range(num_queries):
            p = per_query_pass.get(q_idx, 0)
            f = per_query_fail.get(q_idx, 0)
            line = f"[STABILITY_SUMMARY] query={q_idx}: {p}/{p+f} pass"
            if per_query_responses[q_idx]:
                line += f", unexpected_responses: {dict(per_query_responses[q_idx])}"
            logging.info(line)

        if total_fail > 0:
            task_states.ret = False
            task_states.query_status.append(
                (
                    QueryStatus.OTHERS,
                    f"Stability test: {total_fail}/{total_checks} failures in {repeat_count} iterations",
                    Tracer(),
                )
            )

    def _curl_server_impl(
        self, server_manager: MagaServerManager, task_info: TaskInfo
    ) -> TaskStates:
        task_states = TaskStates()
        qr_array = task_info.query_result
        task_endpoint = task_info.endpoint
        task_states.total_count = len(qr_array)
        comparer_cls = None

        # Pre-populate per-query metadata, request_endpoint, comparer_cls so the
        # dispatch worker only does the heavy I/O. We resolve comparer_cls here
        # too (so the outer scope final-assignment for SAVE_RESPONSE still gets
        # a value) — picking the last query's class is fine because all queries
        # in one fixture share an endpoint / comparer in practice.
        prepared = []
        for q_idx, q_r in enumerate(qr_array):
            q_r["_taskinfo_rel_path"] = task_info.taskinfo_rel_path
            q_r["_query_idx"] = q_idx
            # DashGrpcComparer needs to load a tokenizer client-side; the HTTP
            # comparers never see these fields so this is a no-op for them.
            q_r["_model_path"] = task_info.tokenizer_path or task_info.model_path
            q_r["_model_type"] = task_info.model_type
            if task_info.grammar_constraint_only:
                q_r["grammar_constraint_only"] = True
            request_endpoint = self._resolve_endpoint(q_r, task_endpoint)
            comparer_cls = self._get_comparer_cls(q_r, request_endpoint)
            prepared.append((q_idx, q_r, request_endpoint, comparer_cls))

        def _run_one(item):
            q_idx, q_r, request_endpoint, cls = item
            tracer = Tracer()
            try:
                cls(
                    server_manager, request_endpoint, q_r, tracer, self.batch_infer
                ).run()
                return (q_idx, QueryStatus.OK, "", tracer, None, "")
            except SmokeException as e:
                return (q_idx, e.error_status, e.message, tracer, None, "")
            except Exception as e:
                return (
                    q_idx,
                    QueryStatus.OTHERS,
                    str(e),
                    tracer,
                    e,
                    traceback.format_exc(),
                )

        results: list = [None] * len(prepared)
        if self.parallel_qr > 1 and len(prepared) > 1:
            # Parallel dispatch — sleep_time_qr is intentionally ignored here.
            # Server's --concurrency_limit must be >= parallel_qr or requests
            # will queue server-side and we get no real speedup.
            logging.info(
                "[CaseRunner] dispatching %d queries in parallel (workers=%d)",
                len(prepared),
                self.parallel_qr,
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.parallel_qr
            ) as executor:
                futures = {
                    executor.submit(_run_one, item): item[0] for item in prepared
                }
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    results[res[0]] = res
        else:
            for item in prepared:
                results[item[0]] = _run_one(item)
                if self.sleep_time_qr > 0:
                    time.sleep(self.sleep_time_qr)
                if self.kill_remote and self.remote_kvcm_server is not None:
                    self.remote_kvcm_server.stop_server()
                    logging.info("manually stop remote_kvcm_server")

        # Replay results in fixture order to preserve ``query_status`` indexing
        # (required by SAVE_RESPONSE golden-update logic + post-test logging).
        for q_idx, status, msg, tracer, exc, tb in results:
            if status != QueryStatus.OK:
                task_states.ret = False
            if exc is not None:
                exc_type = classify_exception(exc)
                if exc_type != ExceptionType.NOT_GPU_ERROR:
                    output_dir = os.environ.get(
                        "TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd()
                    )
                    dump_gpu_state(
                        exc=exc,
                        failure_context=f"query exception ({exc_type.value})",
                        log_path=os.path.join(output_dir, "gpu_state_query_error.log"),
                        dmesg_baseline=getattr(self, "_dmesg_baseline", 0),
                    )
                logging.error("query %d failed: %s\n%s", q_idx, msg, tb)
            task_states.query_status.append((status, msg, tracer))

        self._run_stability_repeat(server_manager, task_info, task_states)

        if (
            os.environ.get("SAVE_RESPONSE", "False") == "True"
            and comparer_cls != EmbeddingComparer
        ):
            with open(task_info.taskinfo_rel_path, "r") as f:
                try:
                    import json5

                    origin_json = json5.load(f)
                except ImportError:
                    origin_json = json.load(f)
                origin_qrs = origin_json["query_result"]
            for index, origin_qr in enumerate(origin_qrs):
                status = task_states.query_status[index][0]
                # Update golden for OK and COMPARE_FAILED (actual already in qr_array when SAVE_RESPONSE)
                if status != QueryStatus.OK and status != QueryStatus.COMPARE_FAILED:
                    continue
                now_result = qr_array[index]["result"]
                _iterate_modidfy_qr(origin_qr, now_result)
                if "response_batch" in now_result:
                    for idx, res in enumerate(now_result["response_batch"]):
                        _iterate_modidfy_qr(
                            origin_qr["result"]["response_batch"][idx], res
                        )
                else:
                    _iterate_modidfy_qr(origin_qr["result"], now_result)

            out_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())
            rewrite_path = os.path.join(
                out_dir, "smoke_actual", os.path.basename(task_info.taskinfo_rel_path)
            )
            os.makedirs(os.path.dirname(rewrite_path), exist_ok=True)
            with open(rewrite_path, "w") as f:
                json.dump(
                    origin_json,
                    indent=4,
                    separators=(",", ": "),
                    ensure_ascii=False,
                    fp=f,
                    cls=_TensorEncoder,
                )

        return task_states

    def start_server(
        self,
        env_dict: Dict[str, str],
        task_states: TaskStates,
        task_info: TaskInfo,
        port: Optional[str] = None,
        role_name: str = "main",
        smoke_args_str: Optional[str] = None,
        server_manager_callback: Optional[Callable[[MagaServerManager], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Optional[MagaServerManager]:
        # If smoke_args_str is not provided, try to get it from self.smoke_args dict based on role_name
        if smoke_args_str is None:
            if self.smoke_args and isinstance(self.smoke_args, dict):
                # Get smoke_args for this role, fallback to empty string if not found
                smoke_args_str = self.smoke_args.get(role_name, "")
            else:
                # Use the string value (for list env_args case)
                smoke_args_str = self.smoke_args_str
        server_manager = MagaServerManager(
            env_args=env_dict,
            port=port,
            role_name=role_name,
            smoke_args_str=smoke_args_str,
        )
        if server_manager_callback is not None:
            server_manager_callback(server_manager)
        if cancel_event is not None and cancel_event.is_set():
            task_states.ret = False
            task_states.err_msg = (
                f"{role_name} server start cancelled because another server failed"
            )
            server_manager.stop_server()
            return None
        tokenizer_path = (
            task_info.tokenizer_path
            if task_info.tokenizer_path is not None
            else task_info.model_path
        )

        ret = server_manager.start_server(
            task_info.model_path,
            task_info.model_type,
            tokenizer_path,
            task_info.lora_infos,
            task_info.ptuning_path,
            True,
            3600,
        )
        if task_info.update_lora_infos != None:
            for update_lora_info in task_info.update_lora_infos:
                exp_update_status, exp_update_response = (
                    update_lora_info.update_response[0],
                    update_lora_info.update_response[1],
                )
                update_status, update_response = server_manager.visit(
                    update_lora_info.update_lora_action, 1, "/update"
                )
                if (
                    exp_update_status != update_status
                    and update_response != exp_update_response
                ):
                    task_states.ret = False
                    task_states.err_msg = f"failed to update lora, real response is {update_response}, exp response is {exp_update_response}"
                    return None
        if ret is False:
            task_states.ret = False
            failure_type, failure_desc = classify_process_exit(server_manager.exit_code)
            task_states.err_msg = (
                f"start server failed: {failure_type.value} — {failure_desc}"
            )

            log_errors = scan_process_log(server_manager.log_file_path, max_lines=30)
            if log_errors:
                task_states.err_msg += "\n[process.log errors]\n" + "\n".join(
                    log_errors
                )

            output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())
            dump_gpu_state(
                exc=None,
                failure_context=f"server startup failed: {failure_desc}",
                log_path=os.path.join(output_dir, "gpu_state_server_failed.log"),
                server_pid=server_manager.server_pid,
                server_proc_status=getattr(server_manager, "server_proc_status", None),
                dmesg_baseline=getattr(self, "_dmesg_baseline", 0),
            )
            return None
        return server_manager

    def create_env_from_args(self, env_list: List[str]) -> Dict[str, str]:
        env_dict: Dict[str, str] = {}
        for env_str in env_list:
            k, v = env_str.split("=", 1)
            v = self._expand_env_value(v)
            if k == "MEMORY_CACHE_DISK_PATHS":
                self._prepare_memory_cache_disk_paths(v)
            env_dict.update({k: v})
            logging.info(f"env dict update {k}:{v}")
        return env_dict

    def _expand_env_value(self, value: str) -> str:
        output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())
        tmp_dir = os.environ.get("TEST_TMPDIR")
        if "__TEST_TMPDIR__" in value and tmp_dir is None:
            raise RuntimeError("TEST_TMPDIR is not set; cannot expand __TEST_TMPDIR__")
        tmp_dir = tmp_dir or output_dir
        return value.replace("__TEST_UNDECLARED_OUTPUTS_DIR__", output_dir).replace(
            "__TEST_TMPDIR__", tmp_dir
        )

    def _prepare_memory_cache_disk_paths(self, paths: str) -> None:
        safe_roots = [
            os.environ.get("TEST_TMPDIR"),
            os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"),
        ]
        safe_roots = [os.path.abspath(root) for root in safe_roots if root]
        for path in paths.split(","):
            path = path.strip()
            if not path:
                continue
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                if not any(
                    os.path.commonpath([abs_path, root]) == root for root in safe_roots
                ):
                    raise RuntimeError(
                        f"refuse to clean MEMORY_CACHE_DISK_PATHS outside test dirs: {abs_path}"
                    )
                shutil.rmtree(abs_path)
                logging.info("cleaned memory cache disk path: %s", abs_path)
            os.makedirs(abs_path, exist_ok=True)
            logging.info("prepared memory cache disk path: %s", abs_path)

    def start_servers_parallel(
        self, server_configs: List[Dict[str, Any]]
    ) -> Tuple[List[Any], List[Any]]:
        """
        并行启动多个服务器

        Args:
            server_configs: 服务器配置列表，每个配置包含:
                - env_dict: 环境变量字典
                - task_info: 任务信息
                - port: 端口号
                - role_name: 角色名称
                - smoke_args_str: smoke参数字符串(可选)

        Returns:
            Tuple[List[server_managers], List[task_states]]
        """

        failure_event = threading.Event()
        server_managers_by_role: Dict[str, MagaServerManager] = {}
        server_managers_lock = threading.Lock()

        def register_server_manager(
            role_name: str, server_manager: MagaServerManager
        ) -> None:
            with server_managers_lock:
                server_managers_by_role[role_name] = server_manager
            if failure_event.is_set():
                logging.warning(
                    "Stopping %s server immediately because parallel startup already failed",
                    role_name,
                )
                server_manager.stop_server()

        def stop_registered_servers(failed_role_name: str) -> None:
            with server_managers_lock:
                registered_managers = list(server_managers_by_role.items())
            for role_name, server_manager in registered_managers:
                logging.info(
                    "Stopping %s server after %s startup failure",
                    role_name,
                    failed_role_name,
                )
                server_manager.stop_server()

        def start_single_server(config):
            env_dict = config["env_dict"]
            task_info = config["task_info"]
            port = config["port"]
            role_name = config["role_name"]
            # If smoke_args_str is provided in config, use it; otherwise let start_server choose from dict
            smoke_args_str = config.get("smoke_args_str")

            task_states = TaskStates()
            if failure_event.is_set():
                task_states.ret = False
                task_states.err_msg = (
                    f"{role_name} server start cancelled because another server failed"
                )
                return None, task_states
            server_manager = self.start_server(
                env_dict,
                task_states,
                task_info,
                port=port,
                role_name=role_name,
                smoke_args_str=smoke_args_str,
                server_manager_callback=lambda manager: register_server_manager(
                    role_name, manager
                ),
                cancel_event=failure_event,
            )
            if task_states.ret != True:
                if not failure_event.is_set():
                    logging.error(
                        "Parallel server %s failed during startup, stopping siblings",
                        role_name,
                    )
                    failure_event.set()
                stop_registered_servers(role_name)
            return server_manager, task_states

        # 并行启动所有服务器
        server_managers = []
        task_states_list = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(server_configs)
        ) as executor:
            # 提交所有任务
            future_to_config = {
                executor.submit(start_single_server, config): config
                for config in server_configs
            }

            # 收集结果
            results = {}
            for future in concurrent.futures.as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    server_manager, task_states = future.result()
                    results[config["role_name"]] = (server_manager, task_states)
                except concurrent.futures.CancelledError:
                    task_states = TaskStates()
                    task_states.ret = False
                    task_states.err_msg = f"{config['role_name']} server start cancelled because another server failed"
                    results[config["role_name"]] = (None, task_states)
                except Exception as e:
                    task_states = TaskStates()
                    task_states.ret = False
                    task_states.err_msg = (
                        f"Failed to start server {config['role_name']}: {str(e)}"
                    )
                    results[config["role_name"]] = (None, task_states)

                if results[config["role_name"]][1].ret != True:
                    if not failure_event.is_set():
                        logging.error(
                            "Parallel server %s startup failed, cancelling siblings",
                            config["role_name"],
                        )
                        failure_event.set()
                    for pending_future in future_to_config:
                        if pending_future is not future:
                            pending_future.cancel()
                    stop_registered_servers(config["role_name"])

            # 按照原始顺序返回结果
            for config in server_configs:
                role_name = config["role_name"]
                server_managers.append(results[role_name][0])
                task_states_list.append(results[role_name][1])

        return server_managers, task_states_list
