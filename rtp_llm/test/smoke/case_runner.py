import concurrent.futures
import json
import logging
import os
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
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from smoke.cache_status_comparer import CacheStatusComparer
from smoke.classifier_comparer import ClassifierComparer
from smoke.common_def import QueryStatus, SmokeException, Tracer
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

PD_WRITEBACK_FAILURE_LOG_PATTERNS = [
    "read_transfer_not_done",
    "P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE",
]


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
        concurrency_request_count: int = 5,
        concurrency_workers: int = 10,
        stability_repeat: int = 0,
        assert_no_log_patterns: Optional[List[str]] = None,
        paired_baseline_envs: Optional[Union[List[str], Dict[str, List[str]]]] = None,
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
        self.concurrency_request_count = max(1, concurrency_request_count)
        self.concurrency_workers = max(1, concurrency_workers)
        self.stability_repeat = max(0, stability_repeat)
        self.assert_no_log_patterns = assert_no_log_patterns or []
        self.paired_baseline_envs = paired_baseline_envs or []

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

    def run(self):
        self._dmesg_baseline = snapshot_dmesg()
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
        try:
            server_manager = self.start_server(
                env_dict,
                task_states,
                self.task_info,
                smoke_args_str=self.smoke_args_str,
            )
            if server_manager is None:
                task_states.ret = False
                return task_states
            task_states = self.curl_server(server_manager)
            assert server_manager is not None, "server manager should not be None"
            server_manager.stop_server()
            if enable_remote_cache and self.remote_kvcm_server is not None:
                self.remote_kvcm_server.stop_server()
                self.remote_kvcm_server.copy_logs()
            self.assert_no_log_patterns_absent(task_states)
            return task_states
        finally:
            summarize_and_cleanup_coredumps(
                os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "")
            )

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

    def curl_server(
        self,
        server_manager: MagaServerManager,
        task_info: Optional[TaskInfo] = None,
    ) -> TaskStates:
        task_info = task_info or self.task_info
        if self.concurrency_test:
            task_states = TaskStates()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.concurrency_workers
            ) as executor:
                futures = [
                    executor.submit(self._curl_server_impl, server_manager, task_info)
                    for _ in range(self.concurrency_request_count)
                ]
                results = []
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
                if results[0].ret == False:
                    task_states = results[0]
                else:
                    for result in results:
                        if str(result) != str(str(results[0])):
                            task_states = result
        else:
            task_states = self._curl_server_impl(server_manager, task_info)
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
        if self.stability_repeat > 0:
            repeat_count = self.stability_repeat
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

    @staticmethod
    def _paired_expected_result(actual_result: Any) -> Dict[str, Any]:
        def pick_token_fields(response: Any) -> Dict[str, Any]:
            output_ids = getattr(response, "output_ids", None)
            if output_ids is None:
                raise SmokeException(
                    QueryStatus.COMPARE_FAILED,
                    "paired token diff requires generate_config.return_output_ids=true",
                )
            return {
                "response": getattr(response, "response"),
                "output_ids": output_ids,
            }

        if hasattr(actual_result, "response_batch"):
            return {
                "response_batch": [
                    pick_token_fields(response)
                    for response in actual_result.response_batch
                ]
            }
        return pick_token_fields(actual_result)

    def collect_actual_results(
        self, server_manager: MagaServerManager, task_info: TaskInfo
    ) -> Tuple[TaskStates, List[Dict[str, Any]]]:
        task_states = TaskStates(total_count=len(task_info.query_result))
        actual_results: List[Dict[str, Any]] = []
        for q_idx, q_r in enumerate(task_info.query_result):
            q_r["_taskinfo_rel_path"] = task_info.taskinfo_rel_path
            q_r["_query_idx"] = q_idx
            tracer = Tracer()
            request_endpoint = self._resolve_endpoint(q_r, task_info.endpoint)
            try:
                comparer_cls = self._get_comparer_cls(q_r, request_endpoint)
                comparer = comparer_cls(
                    server_manager,
                    request_endpoint,
                    q_r,
                    tracer,
                    self.batch_infer,
                )
                query_info = comparer.format_query(q_r["query"])
                tracer.query = query_info
                request_info = query_info.model_dump(exclude_defaults=True)
                comparer.maybe_set_concurrency(query_info)
                ret, res = server_manager.visit(
                    request_info,
                    int(os.environ.get("VISIT_RETRY_TIME", 4)),
                    request_endpoint,
                )
                if not ret:
                    raise SmokeException(
                        QueryStatus.VISIT_FAILED,
                        f"curl_error, query:[{request_info}], actual_res:[{res}]",
                    )
                actual_json = comparer.curl_response_to_json(query_info, res)
                actual_result = comparer.format_result(actual_json)
                tracer.actual_result = actual_result
                expected_result = self._paired_expected_result(actual_result)
                actual_results.append(expected_result)
                task_states.query_status.append((QueryStatus.OK, "", tracer))
            except SmokeException as e:
                task_states.ret = False
                task_states.query_status.append((e.error_status, e.message, tracer))
                break
            except Exception as e:
                exc_type = classify_exception(e)
                if exc_type != ExceptionType.NOT_GPU_ERROR:
                    output_dir = os.environ.get(
                        "TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd()
                    )
                    dump_gpu_state(
                        exc=e,
                        failure_context=f"paired baseline query exception ({exc_type.value})",
                        log_path=os.path.join(
                            output_dir, "gpu_state_paired_baseline.log"
                        ),
                        dmesg_baseline=getattr(self, "_dmesg_baseline", 0),
                    )
                task_states.ret = False
                logging.error("%s", traceback.format_exc())
                task_states.query_status.append((QueryStatus.OTHERS, str(e), tracer))
                break
            if self.sleep_time_qr > 0:
                time.sleep(self.sleep_time_qr)
        return task_states, actual_results

    @staticmethod
    def task_info_with_paired_results(
        task_info: TaskInfo, paired_results: List[Dict[str, Any]]
    ) -> TaskInfo:
        if len(task_info.query_result) != len(paired_results):
            raise SmokeException(
                QueryStatus.COMPARE_FAILED,
                "paired baseline result count mismatch: "
                f"task={len(task_info.query_result)}, baseline={len(paired_results)}",
            )
        paired_task_info = task_info.model_copy(deep=True)
        for query_result, result in zip(paired_task_info.query_result, paired_results):
            query_result["result"] = result
        return paired_task_info

    def assert_no_log_patterns_absent(self, task_states: TaskStates) -> None:
        if not self.assert_no_log_patterns:
            return

        output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "")
        if not output_dir or not os.path.isdir(output_dir):
            logging.warning(
                "skip log pattern assertion: TEST_UNDECLARED_OUTPUTS_DIR is unavailable"
            )
            return

        matches = []
        for root, _, files in os.walk(output_dir):
            for file_name in files:
                log_path = os.path.join(root, file_name)
                if not os.path.isfile(log_path):
                    continue
                try:
                    with open(log_path, "r", errors="ignore") as log_file:
                        for line_no, line in enumerate(log_file, 1):
                            for pattern in self.assert_no_log_patterns:
                                if pattern and pattern in line:
                                    matches.append(
                                        (pattern, log_path, line_no, line.strip()[:500])
                                    )
                                    break
                            if len(matches) >= 20:
                                break
                except OSError:
                    continue
                if len(matches) >= 20:
                    break
            if len(matches) >= 20:
                break

        if not matches:
            return

        failure_lines = [
            f"{pattern} at {path}:{line_no}: {line}"
            for pattern, path, line_no, line in matches
        ]
        task_states.ret = False
        task_states.query_status.append(
            (
                QueryStatus.OTHERS,
                "Smoke log assertion matched forbidden writeback failure pattern:\n"
                + "\n".join(failure_lines),
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
        for q_idx, q_r in enumerate(qr_array):
            q_r["_taskinfo_rel_path"] = task_info.taskinfo_rel_path
            q_r["_query_idx"] = q_idx
            tracer = Tracer()
            request_endpoint = self._resolve_endpoint(q_r, task_endpoint)
            try:
                comparer_cls = self._get_comparer_cls(q_r, request_endpoint)
                comparer_cls(
                    server_manager, request_endpoint, q_r, tracer, self.batch_infer
                ).run()
                task_states.query_status.append((QueryStatus.OK, f"", tracer))
            except SmokeException as e:
                task_states.ret = False
                task_states.query_status.append((e.error_status, e.message, tracer))
            except Exception as e:
                exc_type = classify_exception(e)
                if exc_type != ExceptionType.NOT_GPU_ERROR:
                    output_dir = os.environ.get(
                        "TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd()
                    )
                    dump_gpu_state(
                        exc=e,
                        failure_context=f"query exception ({exc_type.value})",
                        log_path=os.path.join(output_dir, "gpu_state_query_error.log"),
                        dmesg_baseline=getattr(self, "_dmesg_baseline", 0),
                    )
                task_states.ret = False
                logging.error("%s", traceback.format_exc())
                task_states.query_status.append((QueryStatus.OTHERS, str(e), tracer))
            if self.sleep_time_qr > 0:
                time.sleep(self.sleep_time_qr)
            if (
                self.kill_remote
                and getattr(self, "remote_kvcm_server", None) is not None
            ):
                self.remote_kvcm_server.stop_server()
                logging.info("manually stop remote_kvcm_server")

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
            env_dict.update({k: v})
            logging.info(f"env dict update {k}:{v}")
        return env_dict

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

        def start_single_server(config):
            env_dict = config["env_dict"]
            task_info = config["task_info"]
            port = config["port"]
            role_name = config["role_name"]
            # If smoke_args_str is provided in config, use it; otherwise let start_server choose from dict
            smoke_args_str = config.get("smoke_args_str")

            task_states = TaskStates()
            server_manager = self.start_server(
                env_dict,
                task_states,
                task_info,
                port=port,
                role_name=role_name,
                smoke_args_str=smoke_args_str,
            )
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
                except Exception as e:
                    task_states = TaskStates()
                    task_states.ret = False
                    task_states.err_msg = (
                        f"Failed to start server {config['role_name']}: {str(e)}"
                    )
                    results[config["role_name"]] = (None, task_states)

            # 按照原始顺序返回结果
            for config in server_configs:
                role_name = config["role_name"]
                server_managers.append(results[role_name][0])
                task_states_list.append(results[role_name][1])

        return server_managers, task_states_list
