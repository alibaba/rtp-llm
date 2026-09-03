"""Explicit prefix-cache performance grid runner.

The ordinary ``GridRunner`` varies only batch size and total input length.  This
runner adds a cache dimension without adding an engine/server argument: each
case first inserts a unique prefix into the normal prefix cache, then sends
three unique continuations sharing exactly that prefix.  The measured
``aux_info.reuse_len`` is recorded, so a requested cache length is never
silently treated as a hit.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests


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
        # Share one deterministic prefix namespace across cache-grid cases.
        # This lets the prefix tree reuse common tokens and prevents cumulative
        # per-case cache duplication on long-sequence sweeps.
        marker = "cache_grid_shared_prefix_"
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

    def make_seed(self, case_id: int, cache_len: int) -> str:
        if cache_len <= 0:
            return ""
        return self._exact_text(cache_len, f"case_{case_id}_prefix_")


def _post_prefill(
    port: int, prompt: str, timeout: int, request_id: str
) -> Dict[str, Any]:
    body = {
        "prompt": prompt,
        "generate_config": {
            "max_new_tokens": 1,
            "min_new_tokens": 1,
            "force_sp_accept": True,
        },
    }
    try:
        response = requests.post(
            f"http://127.0.0.1:{port}", json=body, timeout=timeout
        )
    except Exception as exc:
        return {"success": False, "error": repr(exc), "request_id": request_id}
    if response.status_code != 200:
        return {
            "success": False,
            "error": f"HTTP {response.status_code}: {response.text[:500]}",
            "request_id": request_id,
        }
    try:
        data = response.json()
    except Exception as exc:
        return {"success": False, "error": f"invalid JSON: {exc}", "request_id": request_id}
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
        self.request_timeout = request_timeout
        self.measure_runs = measure_runs
        self.checkpoint_every = max(1, checkpoint_every)
        self.result_path = self.result_dir / "cache_grid_results.json"
        self._results: Dict[str, Dict[str, Any]] = {}
        if self.result_path.exists():
            with self.result_path.open(encoding="utf-8") as f:
                payload = json.load(f)
            self._results = {
                str(x["case_key"]): x for x in payload.get("metrics", [])
            }

    @staticmethod
    def case_key(case: Dict[str, int]) -> str:
        return f"bs{case['batch_size']}_seq{case['input_len']}_cache{case['cache_len']}"

    def _save(self, *, complete: bool = False) -> None:
        payload = {
            "schema_version": 1,
            "mode": "prefix_cache_grid",
            "complete": complete,
            "total_cases": len(self.cases),
            "completed_cases": len(self._results),
            "metrics": list(self._results.values()),
        }
        tmp = self.result_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self.result_path)

    def run(self) -> List[Dict[str, Any]]:
        pending = [c for c in self.cases if self.case_key(c) not in self._results]
        logging.info(
            "cache grid: %d total cases, %d already complete, %d pending",
            len(self.cases),
            len(self._results),
            len(pending),
        )
        for idx, case in enumerate(pending, 1):
            key = self.case_key(case)
            total_len = int(case["input_len"])
            cache_len = int(case["cache_len"])
            batch_size = int(case.get("batch_size", 1))
            if batch_size != 1:
                raise ValueError("prefix cache grid currently requires batch_size=1")
            started = time.time()
            try:
                target, prefix, built_len = self.factory.make_case(
                    int(case["case_id"]), total_len, cache_len
                )
                seed_result: Dict[str, Any] = {}
                if cache_len:
                    seed_result = _post_prefill(
                        self.port,
                        prefix,
                        self.request_timeout,
                        f"{key}:seed",
                    )
                    if not seed_result.get("success"):
                        raise RuntimeError(seed_result.get("error", "seed request failed"))

                prefix_ids = _encode(self.factory.tokenizer, prefix) if cache_len else []
                runs: List[Dict[str, Any]] = []
                for run_idx in range(self.measure_runs):
                    # Vary only the continuation suffix; this prevents the
                    # measured target itself from becoming a larger cache hit.
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
                    result = _post_prefill(
                        self.port, run_target, self.request_timeout, f"{key}:run{run_idx}"
                    )
                    runs.append(result)

                successful = [r for r in runs if r.get("success")]
                expected_reuse = cache_len
                reuse_values = [int(r.get("reuse_len", -1)) for r in successful]
                reuse_exact = bool(
                    len(successful) == self.measure_runs
                    and all(x == expected_reuse for x in reuse_values)
                )
                metric = {
                    "case_key": key,
                    "case_id": int(case["case_id"]),
                    "batch_size": batch_size,
                    "input_len": total_len,
                    "cache_len_requested": cache_len,
                    "cache_len_observed": reuse_values,
                    "input_len_observed": [int(r.get("input_len", 0)) for r in successful],
                    "expected_reuse_len": expected_reuse,
                    "success_runs": len(successful),
                    "measure_runs": self.measure_runs,
                    "seed": seed_result,
                    "runs": runs,
                    "reuse_exact": reuse_exact,
                    "elapsed_s": time.time() - started,
                    "status": (
                        "ok"
                        if len(successful) == self.measure_runs and reuse_exact
                        else "invalid_reuse"
                        if len(successful) == self.measure_runs
                        else "failed"
                    ),
                }
            except Exception as exc:
                metric = {
                    "case_key": key,
                    "case_id": int(case["case_id"]),
                    "batch_size": batch_size,
                    "input_len": total_len,
                    "cache_len_requested": cache_len,
                    "status": "error",
                    "error": repr(exc),
                    "elapsed_s": time.time() - started,
                }
            self._results[key] = metric
            if idx % self.checkpoint_every == 0:
                self._save()
            logging.info(
                "[CACHE_GRID] %d/%d %s status=%s reuse=%s",
                idx,
                len(pending),
                key,
                metric.get("status"),
                metric.get("cache_len_observed", []),
            )
        self._save(complete=len(self._results) == len(self.cases))
        return list(self._results.values())
