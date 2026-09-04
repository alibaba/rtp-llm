"""Explicit prefix-cache performance grid runner.

The ordinary ``GridRunner`` varies only batch size and total input length.  This
runner adds a cache dimension without adding an engine/server argument: each
case first inserts a unique prefix into the normal prefix cache, then sends
three unique continuations sharing exactly that prefix.  The measured
``aux_info.reuse_len`` is recorded, so a requested cache length is never
silently treated as a hit.  End-to-end TTFT is measured around the HTTP call;
the server's auxiliary timing remains available as a separate diagnostic.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter


def _make_http_session() -> requests.Session:
    """Keep one localhost connection alive while forwards remain serial."""
    session = requests.Session()
    session.mount(
        "http://",
        HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0),
    )
    session.headers.update({"Connection": "keep-alive"})
    return session


def _encode(tokenizer: Any, text: str) -> List[int]:
    return list(tokenizer.encode(text))


class PrefixPromptFactory:
    """Build exact-length prompts and verify their shared token prefix."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        # BPE tokenizers generally preserve this family exactly.  We still
        # verify every generated prompt before sending it to the server.
        self._word = " hello"
        # The filler is a single stable token for this tokenizer.  Use a
        # one-pass candidate and retain its token ids instead of repeatedly
        # re-encoding exponentially growing strings.
        self._filler_ids = _encode(self.tokenizer, self._word)
        self._fast_path = len(self._filler_ids) == 1
        self._last_exact_key = None
        self._last_exact_ids = None
        self._last_exact_text = None

    def _store_exact(self, key, text, ids):
        self._last_exact_key = key
        self._last_exact_text = text
        self._last_exact_ids = ids
        return text, ids

    def _exact_text_and_ids(self, target_len: int, prefix: str):
        if target_len <= 0:
            raise ValueError(f"input length must be positive, got {target_len}")
        key = (int(target_len), prefix)
        if self._last_exact_key == key and self._last_exact_ids is not None:
            return self._last_exact_text, self._last_exact_ids

        # Fast path: with a one-token filler, construct the exact target in one
        # full encode.  Prefix/suffix boundaries are checked by the caller.
        if self._fast_path:
            prefix_ids = _encode(self.tokenizer, prefix)
            needed = target_len - len(prefix_ids)
            if needed >= 0:
                candidate = prefix + self._word * needed
                candidate_ids = _encode(self.tokenizer, candidate)
                if len(candidate_ids) == target_len:
                    return self._store_exact(key, candidate, candidate_ids)

        # Conservative fallback for tokenizers whose filler merges or expands.
        text = prefix
        repeats = 1
        while len(_encode(self.tokenizer, text + self._word * repeats)) < target_len:
            repeats *= 2
        lo, hi = 0, repeats
        while lo < hi:
            mid = (lo + hi) // 2
            if len(_encode(self.tokenizer, text + self._word * mid)) < target_len:
                lo = mid + 1
            else:
                hi = mid
        candidate = text + self._word * lo
        candidate_ids = _encode(self.tokenizer, candidate)
        if len(candidate_ids) == target_len:
            return self._store_exact(key, candidate, candidate_ids)

        lo, hi = 0, len(candidate)
        best = prefix
        best_ids = _encode(self.tokenizer, best)
        while lo <= hi:
            mid = (lo + hi) // 2
            cur = candidate[:mid]
            cur_ids = _encode(self.tokenizer, cur)
            n = len(cur_ids)
            if n <= target_len:
                if n == target_len:
                    return self._store_exact(key, cur, cur_ids)
                best, best_ids = cur, cur_ids
                lo = mid + 1
            else:
                hi = mid - 1
        if len(best_ids) != target_len:
            raise ValueError(
                f"cannot construct exact tokenizer length {target_len}; "
                f"got {len(best_ids)}"
            )
        return self._store_exact(key, best, best_ids)

    def _exact_text(self, target_len: int, prefix: str) -> str:
        text, _ = self._exact_text_and_ids(target_len, prefix)
        return text

    def make_case(self, case_id: int, total_len: int, cache_len: int) -> Tuple[str, str, int]:
        if cache_len < 0 or cache_len >= total_len:
            raise ValueError(
                f"cache_len must satisfy 0 <= cache_len < total_len, "
                f"got {cache_len}/{total_len}"
            )
        # Keep cases independent. A shared prefix makes the observed reuse
        # depend on which earlier case happened to populate the prefix tree.
        marker = f"cache_grid_case_{case_id}_prefix_"
        if cache_len == 0:
            target, target_ids = self._exact_text_and_ids(total_len, marker)
            return target, "", len(target_ids)

        prefix, prefix_ids = self._exact_text_and_ids(cache_len, marker)
        target, target_ids = self._exact_text_and_ids(total_len, prefix + " __suffix_")
        if len(target_ids) != total_len or target_ids[:cache_len] != prefix_ids:
            target, target_ids = self._exact_text_and_ids(total_len, prefix)
        if len(target_ids) != total_len or target_ids[:cache_len] != prefix_ids:
            raise ValueError(
                f"unable to preserve exact prefix for case={case_id}: "
                f"prefix={cache_len}, target={len(target_ids)}"
            )
        return target, prefix, len(target_ids)

    def make_seed(
        self, case_id: int, prefix: str, cache_len: int, commit_tail_tokens: int
    ) -> str:
        if cache_len <= 0:
            return ""
        seed_len = cache_len + commit_tail_tokens
        seed, seed_ids = self._exact_text_and_ids(
            seed_len, prefix + f" __seed_commit_{case_id}_"
        )
        prefix_ids = _encode(self.tokenizer, prefix)
        if len(prefix_ids) != cache_len or seed_ids[:cache_len] != prefix_ids:
            seed, seed_ids = self._exact_text_and_ids(seed_len, prefix)
        if len(seed_ids) != seed_len or seed_ids[:cache_len] != prefix_ids:
            raise ValueError(
                f"unable to build cache seed for case={case_id}: "
                f"prefix={cache_len}, seed={len(seed_ids)}"
            )
        return seed


def _post_prefill(
    port: int,
    prompt: str,
    timeout: int,
    request_id: str,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    body = {
        "prompt": prompt,
        "generate_config": {
            "max_new_tokens": 1,
            "min_new_tokens": 1,
            "force_sp_accept": True,
        },
    }
    started = time.perf_counter()
    try:
        response = (session or requests).post(
            f"http://127.0.0.1:{port}", json=body, timeout=timeout
        )
    except Exception as exc:
        return {
            "success": False,
            "error": repr(exc),
            "request_id": request_id,
            "client_wall_time_ms": (time.perf_counter() - started) * 1000.0,
        }
    client_wall_time_ms = (time.perf_counter() - started) * 1000.0
    if response.status_code != 200:
        return {
            "success": False,
            "error": f"HTTP {response.status_code}: {response.text[:500]}",
            "request_id": request_id,
            "client_wall_time_ms": client_wall_time_ms,
        }
    try:
        data = response.json()
    except Exception as exc:
        return {
            "success": False,
            "error": f"invalid JSON: {exc}",
            "request_id": request_id,
            "client_wall_time_ms": client_wall_time_ms,
        }
    aux = data.get("aux_info") or {}
    return {
        "success": True,
        "request_id": request_id,
        "input_len": int(aux.get("input_len", 0)),
        "output_len": int(aux.get("output_len", 0)),
        "reuse_len": int(aux.get("reuse_len", 0)),
        "prefill_time_ms": float(aux.get("first_token_cost_time", 0.0)),
        "total_time_ms": float(aux.get("cost_time", 0.0)),
        "wait_time_ms": float(aux.get("wait_time", 0.0)),
        "client_wall_time_ms": client_wall_time_ms,
        "ttft_ms": client_wall_time_ms,
        "ttft_source": "client_http_wall_max_new_tokens_1",
    }


class CacheGridRunner:
    """Run and checkpoint a total-seq × prefix-cache grid."""

    def __init__(
        self,
        port: int,
        tokenizer: Any,
        cases: Iterable[Dict[str, int]],
        result_dir: str,
        *,
        request_timeout: int = 7200,
        measure_runs: int = 3,
        checkpoint_every: int = 1,
        cache_commit_tail_tokens: int = 4096,
        fail_fast: bool = True,
    ):
        self.port = port
        self.factory = PrefixPromptFactory(tokenizer)
        self.cases = list(cases)
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if measure_runs <= 0:
            raise ValueError("measure_runs must be positive")
        if cache_commit_tail_tokens <= 0:
            raise ValueError("cache_commit_tail_tokens must be positive")
        self.request_timeout = request_timeout
        self.measure_runs = measure_runs
        self.checkpoint_every = max(1, checkpoint_every)
        self.cache_commit_tail_tokens = cache_commit_tail_tokens
        self.fail_fast = fail_fast
        self.result_path = self.result_dir / "cache_grid_results.json"
        self._results: Dict[str, Dict[str, Any]] = {}
        self._http_session = _make_http_session()
        self._checkpoint_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="checkpoint-writer"
        )
        self._checkpoint_future: Optional[Future] = None
        if self.result_path.exists():
            with self.result_path.open(encoding="utf-8") as f:
                payload = json.load(f)
            self._results = {
                str(x["case_key"]): x for x in payload.get("metrics", [])
            }

    @staticmethod
    def case_key(case: Dict[str, int]) -> str:
        return f"bs{case['batch_size']}_seq{case['input_len']}_cache{case['cache_len']}"

    def _write_checkpoint(self, payload: Dict[str, Any]) -> None:
        tmp = self.result_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self.result_path)

    def _save(self, *, complete: bool = False, asynchronous: bool = False) -> None:
        payload = {
            "schema_version": 1,
            "mode": "prefix_cache_grid",
            "complete": complete,
            "total_cases": len(self.cases),
            "completed_cases": sum(
                row.get("status") == "ok" for row in self._results.values()
            ),
            "metrics": list(self._results.values()),
        }
        if asynchronous:
            if self._checkpoint_future is not None:
                self._checkpoint_future.result()
            self._checkpoint_future = self._checkpoint_executor.submit(
                self._write_checkpoint, payload
            )
            return
        if self._checkpoint_future is not None:
            self._checkpoint_future.result()
            self._checkpoint_future = None
        self._write_checkpoint(payload)

    def _close_resources(self) -> None:
        if self._checkpoint_future is not None:
            self._checkpoint_future.result()
            self._checkpoint_future = None
        self._checkpoint_executor.shutdown(wait=True)
        self._http_session.close()

    def _prepare_case_payload(self, case: Dict[str, int]) -> Dict[str, Any]:
        """Build prompts for one case without issuing a model request."""
        key = self.case_key(case)
        total_len = int(case["input_len"])
        cache_len = int(case["cache_len"])
        batch_size = int(case.get("batch_size", 1))
        if batch_size != 1:
            raise ValueError("prefix cache grid currently requires batch_size=1")
        if cache_len and cache_len % self.cache_commit_tail_tokens:
            raise ValueError(
                f"cache_len must align to commit tail "
                f"{self.cache_commit_tail_tokens}, got {cache_len}"
            )
        if cache_len and cache_len + self.cache_commit_tail_tokens > total_len:
            raise ValueError(
                f"cache_len must leave at least {self.cache_commit_tail_tokens} "
                f"tokens for seed commit, got {cache_len}/{total_len}"
            )
        target, prefix, built_len = self.factory.make_case(
            int(case["case_id"]), total_len, cache_len
        )
        seed = (
            self.factory.make_seed(
                int(case["case_id"]),
                prefix,
                cache_len,
                self.cache_commit_tail_tokens,
            )
            if cache_len
            else ""
        )
        prefix_ids = _encode(self.factory.tokenizer, prefix) if cache_len else []
        run_targets: List[str] = []
        for run_idx in range(self.measure_runs):
            if cache_len:
                run_target, run_ids = self.factory._exact_text_and_ids(
                    total_len, prefix + f" __run_{run_idx}_"
                )
                if run_ids[:cache_len] != prefix_ids:
                    raise ValueError(f"run {run_idx} did not preserve cache prefix")
            else:
                run_target, _ = self.factory._exact_text_and_ids(
                    total_len, f"case_{case['case_id']}_cold_run_{run_idx}_"
                )
            run_targets.append(run_target)
        return {
            "key": key,
            "total_len": total_len,
            "cache_len": cache_len,
            "batch_size": batch_size,
            "built_len": built_len,
            "seed": seed,
            "run_targets": run_targets,
        }

    def run(self) -> List[Dict[str, Any]]:
        pending = [
            case
            for case in self.cases
            if self._results.get(self.case_key(case), {}).get("status") != "ok"
        ]
        logging.info(
            "cache grid: %d total cases, %d already complete, %d pending",
            len(self.cases),
            len(self.cases) - len(pending),
            len(pending),
        )
        if not pending:
            self._save(complete=True)
            self._close_resources()
            return list(self._results.values())

        try:
            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="query-prefetch"
            ) as prep:
                prepared: Future = prep.submit(self._prepare_case_payload, pending[0])
                for idx, case in enumerate(pending, 1):
                    key = self.case_key(case)
                    started = time.time()
                    try:
                        try:
                            payload = prepared.result()
                        finally:
                            if idx < len(pending):
                                prepared = prep.submit(
                                    self._prepare_case_payload, pending[idx]
                                )
                        total_len = payload["total_len"]
                        cache_len = payload["cache_len"]
                        batch_size = payload["batch_size"]
                        built_len = payload["built_len"]
                        seed_result: Dict[str, Any] = {}
                        if cache_len:
                            seed_result = _post_prefill(
                                self.port,
                                payload["seed"],
                                self.request_timeout,
                                f"{key}:seed",
                                self._http_session,
                            )
                            if not seed_result.get("success"):
                                raise RuntimeError(
                                    seed_result.get("error", "seed request failed")
                                )

                        runs = [
                            _post_prefill(
                                self.port,
                                run_target,
                                self.request_timeout,
                                f"{key}:run{run_idx}",
                                self._http_session,
                            )
                            for run_idx, run_target in enumerate(payload["run_targets"])
                        ]

                        successful = [r for r in runs if r.get("success")]
                        expected_reuse = cache_len
                        reuse_values = [
                            int(r.get("reuse_len", -1)) for r in successful
                        ]
                        reuse_exact = bool(
                            len(successful) == self.measure_runs
                            and all(x == expected_reuse for x in reuse_values)
                        )
                        shape_exact = bool(
                            len(successful) == self.measure_runs
                            and all(
                                int(r.get("input_len", -1)) == total_len
                                for r in successful
                            )
                            and all(
                                int(r.get("output_len", -1)) == 1 for r in successful
                            )
                        )
                        ttft_values = [
                            float(r["ttft_ms"])
                            for r in successful
                            if float(r.get("ttft_ms", 0.0)) > 0.0
                        ]
                        timing_valid = len(ttft_values) == self.measure_runs
                        metric = {
                            "case_key": key,
                            "case_id": int(case["case_id"]),
                            "batch_size": batch_size,
                            "input_len": total_len,
                            "input_len_built": built_len,
                            "cache_len_requested": cache_len,
                            "cache_len_observed": reuse_values,
                            "input_len_observed": [
                                int(r.get("input_len", 0)) for r in successful
                            ],
                            "expected_reuse_len": expected_reuse,
                            "success_runs": len(successful),
                            "measure_runs": self.measure_runs,
                            "cache_commit_tail_tokens": self.cache_commit_tail_tokens,
                            "seed": seed_result,
                            "runs": runs,
                            "reuse_exact": reuse_exact,
                            "shape_exact": shape_exact,
                            "timing_valid": timing_valid,
                            "ttft_ms": ttft_values,
                            "median_ttft_ms": (
                                statistics.median(ttft_values)
                                if timing_valid
                                else None
                            ),
                            "avg_ttft_ms": (
                                statistics.fmean(ttft_values)
                                if timing_valid
                                else None
                            ),
                            "elapsed_s": time.time() - started,
                            "status": (
                                "ok"
                                if reuse_exact and shape_exact and timing_valid
                                else "invalid_shape"
                                if len(successful) == self.measure_runs
                                and not shape_exact
                                else "invalid_timing"
                                if len(successful) == self.measure_runs
                                and not timing_valid
                                else "invalid_reuse"
                                if len(successful) == self.measure_runs
                                else "failed"
                            ),
                        }
                    except Exception as exc:
                        metric = {
                            "case_key": key,
                            "case_id": int(case["case_id"]),
                            "batch_size": int(case.get("batch_size", 1)),
                            "input_len": int(case["input_len"]),
                            "cache_len_requested": int(case["cache_len"]),
                            "status": "error",
                            "error": repr(exc),
                            "elapsed_s": time.time() - started,
                        }
                    self._results[key] = metric
                    if idx % self.checkpoint_every == 0:
                        self._save(asynchronous=True)
                    logging.info(
                        "[CACHE_GRID] %d/%d %s status=%s reuse=%s "
                        "query_prefetch=next",
                        idx,
                        len(pending),
                        key,
                        metric.get("status"),
                        metric.get("cache_len_observed", []),
                    )
                    if self.fail_fast and metric.get("status") != "ok":
                        self._save()
                        raise RuntimeError(
                            f"cache grid stopped at {key}: "
                            f"status={metric.get('status')}"
                        )
            complete = all(
                self._results.get(self.case_key(case), {}).get("status") == "ok"
                for case in self.cases
            )
            self._save(complete=complete)
            return list(self._results.values())
        finally:
            self._close_resources()
