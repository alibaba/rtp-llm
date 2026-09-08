import json
import logging
import os
import time
from typing import Any, Dict, List, Union

import requests
from pydantic import BaseModel
from smoke.common_def import QueryStatus, SmokeException, Tracer
from smoke.utils import no_compare, save_response

from rtp_llm.test.utils.maga_server_manager import MagaServerManager


def _actual_to_json_serializable(obj: Any) -> Any:
    """Convert actual result to JSON-serializable form (for dumping to artifact)."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return _actual_to_json_serializable(obj.model_dump())
    if isinstance(obj, dict):
        return {k: _actual_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_actual_to_json_serializable(x) for x in obj]
    if hasattr(obj, "tolist"):  # numpy/torch array
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


_COLD_CACHE_AUX_FIELDS = (
    "reuse_len",
    "local_reuse_len",
    "remote_reuse_len",
    "memory_reuse_len",
    "prefill_total_reuse_len",
    "prefill_local_reuse_len",
    "prefill_remote_reuse_len",
    "prefill_memory_reuse_len",
    "decode_total_reuse_len",
    "decode_local_reuse_len",
    "decode_remote_reuse_len",
    "decode_memory_reuse_len",
)


def _asserts_cold_cache(qr_info: Dict[str, Any]) -> bool:
    """True when this query's golden encodes a COLD prefix cache.

    Such a query cannot be retried: the first attempt's prefill publishes its
    blocks into the reuse cache even when the request later fails (e.g. the
    decode-side cache-store load times out), so re-sending the same query
    legitimately reports non-zero reuse and the cold-start golden can never
    match again. Only meaningful when the case actually runs with reuse on --
    plenty of non-reuse goldens carry ``reuse_len: 0`` simply because reuse is
    off, and those stay freely retryable.
    """
    if not qr_info.get("_reuse_cache_enabled"):
        return False
    aux = (qr_info.get("result") or {}).get("aux_info") or {}
    if not isinstance(aux, dict):
        return False
    present = [f for f in _COLD_CACHE_AUX_FIELDS if f in aux]
    return bool(present) and all(aux[f] == 0 for f in present)


class BaseComparer(object):
    def __init__(
        self,
        server_manager: MagaServerManager,
        request_endpoint: str,
        q_r: Dict[str, Any],
        tracer: Tracer,
        use_batch_scheduler: bool
    ):
        self.server_manager = server_manager
        self.request_endpoint = request_endpoint
        # for special usage
        self.qr_info = q_r
        self.tracer = tracer
        self.use_batch_scheduler = use_batch_scheduler

    def format_query(self, query_json: Dict[str, Any]) -> BaseModel:
        raise NotImplementedError()

    def format_result(self, result_json: Dict[str, Any]) -> BaseModel:
        raise NotImplementedError()

    def curl_response_to_json(
        self, query_info: BaseModel, curl_response: Any
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    def compare_result(self, expect_result: Any, actual_result: Any) -> None:
        raise NotImplementedError()

    def get_concurrency_batch(self, query_info: BaseModel) -> int:
        return 1

    def _maybe_rewrite_expect_result(
        self,
        smoke_response: Union[List[BaseModel], BaseModel],
        expect_result: Union[List[BaseModel], BaseModel],
        query_info: BaseModel,
    ) -> None:
        pass

    def _dump_actual_to_artifact(self, actual_result: Any) -> None:
        """Dump formatted actual result to current dir (or TEST_UNDECLARED_OUTPUTS_DIR) for artifact."""
        taskinfo_rel_path = self.qr_info.get("_taskinfo_rel_path")
        query_idx = self.qr_info.get("_query_idx")
        if taskinfo_rel_path is None or query_idx is None:
            return
        out_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())
        # smoke_actual/data/model/.../file.json -> smoke_actual/data/model/.../file.query_0.json
        base = os.path.splitext(taskinfo_rel_path)[0]
        rel_dir = os.path.dirname(base)
        name = os.path.basename(base)
        dest_dir = os.path.join(out_dir, "smoke_actual", rel_dir)
        os.makedirs(dest_dir, exist_ok=True)
        out_path = os.path.join(dest_dir, f"{name}.query_{query_idx}.json")
        try:
            data = _actual_to_json_serializable(actual_result)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logging.info("Dumped actual to %s", out_path)
        except Exception as e:
            logging.warning("Failed to dump actual to %s: %s", out_path, e)

    def maybe_set_concurrency(self, query_info: BaseModel):
        if not self.use_batch_scheduler:
            return
        concurrecy_batch = self.get_concurrency_batch(query_info)
        if concurrecy_batch <= 1:
            return
        url = f"http://0.0.0.0:{int(self.server_manager.port)}/update_scheduler_info"
        max_retries = 30
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url, json={"batch_size": concurrecy_batch}, timeout=10
                )
                resp_json = response.json()
                if response.status_code == 200 and "error" not in resp_json and resp_json.get("status") == "ok":
                    return
                logging.warning(
                    f"update_scheduler_info attempt {attempt+1}/{max_retries} error: {resp_json}"
                )
            except Exception as e:
                logging.warning(
                    f"update_scheduler_info attempt {attempt+1}/{max_retries} failed: {e}"
                )
            if attempt < max_retries - 1:
                time.sleep(2.0)
        raise Exception(
            f"failed to set concurrency after {max_retries} retries, batch_size={concurrecy_batch}"
        )

    def run(self):
        query_info: BaseModel = self.format_query(self.qr_info["query"])
        self.tracer.query = query_info
        visit_retry_time = int(os.environ.get("VISIT_RETRY_TIME", 4))
        if visit_retry_time > 1 and not save_response() and _asserts_cold_cache(self.qr_info):
            # Retrying here can only ever produce a *warm* cache result, which
            # the cold-start golden rejects with a confusing "N differences in
            # aux_info.*reuse_len" diff, burying the real error (e.g. an HTTP 500
            # CACHE_STORE_LOAD_BUFFER_TIMEOUT) that caused the first attempt to
            # fail. Fail on the first attempt so the actual cause is reported.
            logging.info(
                "query %s golden asserts a cold prefix cache; disabling visit "
                "retry (a retry would hit the cache the failed attempt warmed)",
                self.qr_info.get("_query_idx"),
            )
            visit_retry_time = 1
        request_info = query_info.model_dump(exclude_defaults=True)
        self.maybe_set_concurrency(query_info)
        ret, res = self.server_manager.visit(
            request_info,
            visit_retry_time,
            self.request_endpoint,
            self.qr_info.get("expected_status_code", 200),
        )
        if not ret:
            raise SmokeException(
                QueryStatus.VISIT_FAILED,
                f"curl_error, query:[{request_info}], actual_res:[{res}]",
            )
        try:
            actual_result = self.curl_response_to_json(query_info, res)
        except:
            logging.warning(f"parse response failed: {res}")
            raise
        test_with_sleep = bool(int(os.environ.get("TEST_WITH_SLEEP", 0)))
        logging.info(f"test_with_sleep: {test_with_sleep}")
        if test_with_sleep:
            time.sleep(3600 * 100)

        actual_result = self.format_result(actual_result)
        self.tracer.actual_result = actual_result
        expect_result: BaseModel = self.format_result(self.qr_info["result"])
        self.tracer.expect_result = expect_result
        self._maybe_rewrite_expect_result(actual_result, expect_result, query_info)
        self._dump_actual_to_artifact(actual_result)
        if save_response():
            self.qr_info["result"] = actual_result.model_dump(exclude_defaults=True)
        if no_compare():
            return
        self.compare_result(expect_result, actual_result)
