"""Concurrent stress test for reproducing CUDA-graph correctness bugs under load.

This module is imported by ``case_runner._run_concurrent_stress`` after the
normal smoke queries pass.  It reuses the first prompt from the task JSON,
overrides ``max_new_tokens`` to produce longer outputs, and fires N concurrent
requests at the server.  Any difference in responses across identical requests
indicates non-determinism (e.g. CUDA graph corruption in SP target-verify or
draft-prefill paths).

Environment variables
---------------------
CONCURRENT_STRESS_ITERS : int (default 0)
    Number of stress iterations.  0 = skip.
CONCURRENT_STRESS_CONCURRENCY : int (default 4)
    Number of concurrent requests per iteration.
CONCURRENT_STRESS_MAX_NEW_TOKENS : int (default 1000)
    Override max_new_tokens in generate_config.
CONCURRENT_STRESS_FAIL_TEST : str "0"|"1" (default "1")
    If "1", the smoke test fails on non-determinism.
CONCURRENT_STRESS_DETECT_REPEAT : str "0"|"1" (default "1")
    If "1", check for pathological repetition in responses.
CONCURRENT_STRESS_REPEAT_WINDOW : int (default 20)
    Minimum number of consecutive repeated tokens to flag as repetition.
CONCURRENT_STRESS_PROMPT_REPEAT : int (default 1)
    Repeat the user content (text between ``<|im_start|>user\\n`` and
    ``<|im_end|>``) this many times to inflate input length.  Used to drive
    ~20k-token inputs for CUDA-graph stress reproduction without touching
    the golden task_info JSON.  1 = no inflation.
CONCURRENT_STRESS_MAX_UNIQUE : int (default 1)
    Maximum number of unique responses allowed per iteration before counting
    it as a non-determinism failure.  1 = strict (all must match).
    2 = tolerate BF16/NCCL-level non-determinism (1 outlier allowed).
"""

import concurrent.futures
import copy
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


def _detect_repetition(text: str, window: int = 20) -> Optional[str]:
    """Return the repeated fragment if *text* contains a pathological loop."""
    if len(text) < window * 2:
        return None
    # Check for repeated n-grams (character level) with window size
    for frag_len in range(2, min(200, len(text) // 3) + 1):
        for start in range(len(text) - frag_len * window):
            frag = text[start : start + frag_len]
            if not frag.strip():
                continue
            count = 0
            pos = start
            while pos + frag_len <= len(text) and text[pos : pos + frag_len] == frag:
                count += 1
                pos += frag_len
            if count >= window:
                return frag
    return None


def _send_one_request(
    url: str,
    query: Dict[str, Any],
    timeout: int = 300,
) -> Tuple[int, str, float]:
    """POST *query* to *url* and return (status_code, response_text, elapsed_s)."""
    t0 = time.monotonic()
    try:
        resp = requests.post(url, json=query, timeout=timeout)
        elapsed = time.monotonic() - t0
        return resp.status_code, resp.text, elapsed
    except Exception as e:
        elapsed = time.monotonic() - t0
        return -1, str(e), elapsed


def _extract_response_text(raw: str) -> str:
    """Extract the ``response`` field from the server JSON reply."""
    try:
        obj = json.loads(raw)
        return obj.get("response", "")
    except (json.JSONDecodeError, TypeError):
        return raw


def _inflate_prompt(prompt: str, repeat: int) -> str:
    """Multiply the user body text inside the chat template.

    Target tokens ~= original_tokens * ``repeat``.  Chat tags stay intact so
    the server still tokenizes and samples normally; only the user-visible
    article body balloons.  Used for the 20k-token stress run.
    """
    if repeat <= 1 or not prompt:
        return prompt
    user_tag = "<|im_start|>user\n"
    end_tag = "<|im_end|>"
    i = prompt.find(user_tag)
    if i < 0:
        return prompt * repeat
    j = prompt.find(end_tag, i + len(user_tag))
    if j < 0:
        return prompt
    body = prompt[i + len(user_tag) : j]
    return prompt[: i + len(user_tag)] + (body * repeat) + prompt[j:]


def run_stress(
    base_url: str,
    query_template: Dict[str, Any],
    endpoint: str = "/",
    iterations: int = 5,
    concurrency: int = 4,
    max_new_tokens: int = 1000,
    detect_repeat: bool = True,
    repeat_window: int = 20,
    prompt_repeat: int = 1,
    max_unique: int = 1,
    heterogeneous: bool = False,
    query_pool: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Run the concurrent stress test.

    Returns (passed, summary_dict).

    When ``query_pool`` is non-empty (more than one entry), heterogeneous mode
    serves each worker a real different query from the pool (round-robin by
    worker_idx) instead of mutating one prompt with a marker.  This is the
    closest local proxy for production traffic where simultaneous users never
    send identical prompts.
    """
    url = f"{base_url}{endpoint}"

    def _normalize(q: Dict[str, Any]) -> Dict[str, Any]:
        q = copy.deepcopy(q)
        gc = q.setdefault("generate_config", {})
        gc["max_new_tokens"] = max_new_tokens
        gc["temperature"] = 0
        gc["top_k"] = 1
        gc["top_p"] = 0
        q["yield_generator"] = False
        q.pop("top_p", None)
        q.pop("top_k", None)
        q.pop("temperature", None)
        if prompt_repeat > 1 and isinstance(q.get("prompt"), str):
            q["prompt"] = _inflate_prompt(q["prompt"], prompt_repeat)
        return q

    query = _normalize(query_template)
    pool_normalized: List[Dict[str, Any]] = (
        [_normalize(q) for q in query_pool] if query_pool else [query]
    )
    use_pool = heterogeneous and len(pool_normalized) > 1
    if use_pool:
        logger.info(
            "[CONCURRENT_STRESS] heterogeneous query pool: %d real queries, "
            "lengths=%s",
            len(pool_normalized),
            [len(q.get("prompt", "")) for q in pool_normalized],
        )

    summary: Dict[str, Any] = {
        "iterations": iterations,
        "concurrency": concurrency,
        "max_new_tokens": max_new_tokens,
        "total_requests": 0,
        "http_errors": 0,
        "non_determinism_events": 0,
        "repetition_events": 0,
        "details": [],
    }

    all_passed = True

    for it in range(iterations):
        logger.info(
            "[CONCURRENT_STRESS iter=%d/%d] Sending %d concurrent requests (max_new_tokens=%d)",
            it + 1,
            iterations,
            concurrency,
            max_new_tokens,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            if use_pool:
                # Heterogeneous (multi-query) mode: round-robin pick from the
                # real query pool — each worker gets a genuinely different
                # full prompt, no synthetic marker injection.
                futures = []
                for w in range(concurrency):
                    pick = pool_normalized[
                        (it * concurrency + w) % len(pool_normalized)
                    ]
                    futures.append(pool.submit(_send_one_request, url, pick))
            elif heterogeneous and isinstance(query.get("prompt"), str):
                # Heterogeneous (single-prompt) mode: every worker gets a unique
                # 1-token tail appended to its prompt so per-iter batches are
                # guaranteed to contain different sequences.  Note: this can
                # confuse the model with awkwardly placed markers; prefer the
                # multi-query pool mode above when possible.
                base_prompt = query["prompt"]
                tail_marker = "<|im_start|>assistant\n"
                if tail_marker in base_prompt:
                    head, _, tail = base_prompt.rpartition(tail_marker)
                else:
                    head, tail = base_prompt, ""
                futures = []
                for w in range(concurrency):
                    qcopy = copy.deepcopy(query)
                    if tail:
                        qcopy["prompt"] = (
                            head.rstrip() + f"\n[req {it}.{w}]\n" + tail_marker
                        )
                    else:
                        qcopy["prompt"] = base_prompt + f"\n[req {it}.{w}]\n"
                    futures.append(pool.submit(_send_one_request, url, qcopy))
            else:
                futures = [
                    pool.submit(_send_one_request, url, query)
                    for _ in range(concurrency)
                ]
            results = [f.result() for f in futures]

        summary["total_requests"] += concurrency

        statuses = [r[0] for r in results]
        raw_texts = [r[1] for r in results]
        elapsed_list = [r[2] for r in results]
        responses = [_extract_response_text(t) for t in raw_texts]

        http_ok = [s == 200 for s in statuses]
        if not all(http_ok):
            n_err = sum(1 for ok in http_ok if not ok)
            summary["http_errors"] += n_err
            logger.warning(
                "[CONCURRENT_STRESS iter=%d] %d/%d HTTP errors: %s",
                it + 1,
                n_err,
                concurrency,
                [(s, t[:200]) for s, t in zip(statuses, raw_texts) if s != 200],
            )
            all_passed = False
            continue

        # Check determinism: all responses should be identical (or within max_unique tolerance)
        # In heterogeneous mode every prompt is unique by construction, so the
        # determinism check is meaningless — skip it and rely on detect_repeat
        # for the "garbled / stuck-loop" signal.
        unique_responses = set(responses)
        if not heterogeneous and len(unique_responses) > max_unique:
            summary["non_determinism_events"] += 1
            lens = [len(r) for r in responses]
            detail = {
                "iter": it + 1,
                "type": "non_determinism",
                "num_unique": len(unique_responses),
                "response_lengths": lens,
                "elapsed": [round(e, 2) for e in elapsed_list],
            }
            # Log first difference
            ref = responses[0]
            for idx, resp in enumerate(responses[1:], 1):
                if resp != ref:
                    diff_pos = next(
                        (
                            i
                            for i in range(min(len(ref), len(resp)))
                            if ref[i] != resp[i]
                        ),
                        min(len(ref), len(resp)),
                    )
                    detail["first_diff_at"] = diff_pos
                    detail["ref_around_diff"] = ref[
                        max(0, diff_pos - 30) : diff_pos + 30
                    ]
                    detail["other_around_diff"] = resp[
                        max(0, diff_pos - 30) : diff_pos + 30
                    ]
                    break

            summary["details"].append(detail)
            logger.error(
                "[CONCURRENT_STRESS iter=%d] NON-DETERMINISM: %d unique responses among %d requests. "
                "Lengths: %s. Detail: %s",
                it + 1,
                len(unique_responses),
                concurrency,
                lens,
                json.dumps(detail, ensure_ascii=False)[:500],
            )
            for ridx, resp in enumerate(responses):
                logger.error(
                    "[CONCURRENT_STRESS iter=%d] FULL RESPONSE req=%d (len=%d):\n%s",
                    it + 1,
                    ridx,
                    len(resp),
                    resp,
                )
            all_passed = False

        # Check for repetition patterns
        halt_now = False
        if detect_repeat:
            halt_on_event = (
                os.environ.get("CONCURRENT_STRESS_HALT_ON_EVENT", "0") == "1"
            )
            for idx, resp in enumerate(responses):
                repeated_frag = _detect_repetition(resp, repeat_window)
                if repeated_frag is not None:
                    summary["repetition_events"] += 1
                    detail = {
                        "iter": it + 1,
                        "type": "repetition",
                        "request_idx": idx,
                        "response_length": len(resp),
                        "repeated_fragment": repeated_frag[:100],
                        "elapsed": round(elapsed_list[idx], 2),
                    }
                    summary["details"].append(detail)
                    logger.error(
                        "[CONCURRENT_STRESS iter=%d req=%d] REPETITION detected: "
                        "fragment=%r (len=%d) in response of length %d",
                        it + 1,
                        idx,
                        repeated_frag[:60],
                        len(repeated_frag),
                        len(resp),
                    )
                    logger.error(
                        "[CONCURRENT_STRESS iter=%d req=%d] FULL REPEATED RESPONSE:\n%s",
                        it + 1,
                        idx,
                        resp,
                    )
                    all_passed = False
                    if halt_on_event:
                        halt_now = True

        if all(http_ok) and len(unique_responses) <= max_unique:
            logger.info(
                "[CONCURRENT_STRESS iter=%d] PASS: %d identical responses, len=%d, elapsed=%s",
                it + 1,
                concurrency,
                len(responses[0]),
                [round(e, 2) for e in elapsed_list],
            )

        if halt_now:
            logger.error(
                "[CONCURRENT_STRESS] HALT_ON_EVENT=1 — stopping after first "
                "repetition event at iter=%d to preserve dump state",
                it + 1,
            )
            break

    return all_passed, summary


def maybe_run_from_env(
    server_manager: Any,
    task_info: Any,
) -> Tuple[Optional[bool], Optional[Dict[str, Any]]]:
    """Entry point called by ``case_runner._run_concurrent_stress``.

    Returns ``(None, None)`` when CONCURRENT_STRESS_ITERS is 0 (skip).
    """
    iterations = int(os.environ.get("CONCURRENT_STRESS_ITERS", "0"))
    if iterations <= 0:
        return None, None

    concurrency = int(os.environ.get("CONCURRENT_STRESS_CONCURRENCY", "4"))
    max_new_tokens = int(os.environ.get("CONCURRENT_STRESS_MAX_NEW_TOKENS", "1000"))
    detect_repeat = os.environ.get("CONCURRENT_STRESS_DETECT_REPEAT", "1") == "1"
    repeat_window = int(os.environ.get("CONCURRENT_STRESS_REPEAT_WINDOW", "20"))
    prompt_repeat = int(os.environ.get("CONCURRENT_STRESS_PROMPT_REPEAT", "1"))
    max_unique = int(os.environ.get("CONCURRENT_STRESS_MAX_UNIQUE", "1"))
    heterogeneous = os.environ.get("CONCURRENT_STRESS_HETEROGENEOUS", "0") == "1"

    port = int(server_manager._port)
    port_offset = 5 if int(server_manager._env_args.get("HTTP_API_TEST", 0)) else 0
    base_url = f"http://0.0.0.0:{port + port_offset}"

    endpoint = getattr(task_info, "endpoint", "/") or "/"
    qr_array = task_info.query_result
    if not qr_array:
        logger.warning("[CONCURRENT_STRESS] No queries in task_info, skipping")
        return None, None

    query_template = qr_array[0].get("query", qr_array[0])
    # Build the full query pool — used by heterogeneous mode to round-robin
    # over real different queries instead of mutating one prompt with a
    # synthetic marker.
    query_pool = [qr.get("query", qr) for qr in qr_array]

    logger.info(
        "[CONCURRENT_STRESS] Starting: iters=%d concurrency=%d max_new_tokens=%d "
        "pool_size=%d heterogeneous=%s",
        iterations,
        concurrency,
        max_new_tokens,
        len(query_pool),
        heterogeneous,
    )

    passed, summary = run_stress(
        base_url=base_url,
        query_template=query_template,
        endpoint=endpoint,
        iterations=iterations,
        concurrency=concurrency,
        max_new_tokens=max_new_tokens,
        detect_repeat=detect_repeat,
        repeat_window=repeat_window,
        prompt_repeat=prompt_repeat,
        max_unique=max_unique,
        heterogeneous=heterogeneous,
        query_pool=query_pool,
    )

    level = logging.INFO if passed else logging.ERROR
    logger.log(
        level,
        "[CONCURRENT_STRESS] Result: %s | %s",
        "PASS" if passed else "FAIL",
        json.dumps(
            {k: v for k, v in summary.items() if k != "details"},
            ensure_ascii=False,
        ),
    )

    return passed, summary


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Concurrent stress test for RTP-LLM")
    parser.add_argument("--port", type=int, required=True, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--endpoint", default="/", help="Request endpoint")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--repeat-window", type=int, default=20)
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=1,
        help="Repeat the user body N times to inflate input tokens",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text (or reads from --prompt-file)",
    )
    parser.add_argument(
        "--prompt-file", type=str, default=None, help="JSON file with query template"
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    if args.prompt_file:
        with open(args.prompt_file) as f:
            data = json.load(f)
        if "query_result" in data:
            query_template = data["query_result"][0]["query"]
        elif "query" in data:
            query_template = data["query"]
        else:
            query_template = data
    elif args.prompt:
        query_template = {
            "prompt": args.prompt,
            "generate_config": {},
        }
    else:
        query_template = {
            "prompt": "<|im_start|>user\n请写一篇关于人工智能未来发展的文章，至少500字。<|im_end|>\n<|im_start|>assistant\n",
            "generate_config": {},
        }

    passed, summary = run_stress(
        base_url=base_url,
        query_template=query_template,
        endpoint=args.endpoint,
        iterations=args.iterations,
        concurrency=args.concurrency,
        max_new_tokens=args.max_new_tokens,
        detect_repeat=True,
        repeat_window=args.repeat_window,
        prompt_repeat=args.prompt_repeat,
    )

    print("\n" + "=" * 60)
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not passed:
        exit(1)
