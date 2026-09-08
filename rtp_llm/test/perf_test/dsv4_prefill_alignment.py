"""One-token Prefill serving benchmark with identical, verified token inputs.

This measures request wall time, including transport and sampling. It does not
measure pure model-forward time, prove whole-model accuracy, or qualify PD/MTP.
No servers are started, stopped, or reconfigured by this client.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")


def load_corpus(path: Path) -> list[dict[str, Any]]:
    # JSONL is delimited by newline, not by Unicode separators inside prompts.
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not rows:
        raise ValueError("empty corpus")
    seen = set()
    for row in rows:
        if not isinstance(row.get("id"), str) or row["id"] in seen:
            raise ValueError("corpus IDs must be unique strings")
        seen.add(row["id"])
        ids = row.get("input_ids")
        if not isinstance(row.get("text"), str) or not row["text"]:
            raise ValueError("each case needs non-empty text")
        if not isinstance(ids, list) or not ids:
            raise ValueError("each case needs non-empty input_ids")
        if any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("invalid token ID")
    return rows


def freeze(args: argparse.Namespace) -> None:
    # Imported only by this command: collection/comparison are stdlib-only.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=False
    )
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for line in args.source.read_text().split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        text = row[args.text_key]
        if not isinstance(text, str) or not text:
            raise ValueError("source text must be a non-empty string")
        ids = tokenizer.encode(text, add_special_tokens=True)
        if not ids or len(ids) + 1 > args.max_seq_len:
            raise ValueError("input exceeds context budget; truncation is forbidden")
        cases.append({"id": str(len(cases)), "text": text, "input_ids": ids})
        if args.limit and len(cases) == args.limit:
            break
    if not cases:
        raise ValueError("empty source")
    corpus = args.output / "corpus.jsonl"
    with corpus.open("x") as out:
        for case in cases:
            out.write(json.dumps(case, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(args.output / "manifest.json", {
        "schema": 1, "source_sha256": sha256(args.source),
        "corpus_sha256": sha256(corpus), "cases": len(cases),
        "text_key": args.text_key, "max_seq_len": args.max_seq_len,
        "tokenizer": str(args.tokenizer.resolve()),
        "tokenizer_files": {p.name: sha256(p) for p in sorted(args.tokenizer.iterdir())
                            if p.is_file() and p.name in {
                                "tokenizer.json", "tokenizer_config.json",
                                "special_tokens_map.json", "config.json"}},
        "add_special_tokens": True,
        "input_tokens": sum(len(c["input_ids"]) for c in cases),
        "min_input_len": min(len(c["input_ids"]) for c in cases),
        "max_input_len": max(len(c["input_ids"]) for c in cases),
    })


def request_body(backend: str, case: dict, accuracy: bool) -> dict:
    if backend == "rtp":
        return {"prompt": case["text"], "generate_config": {
            "max_new_tokens": 1, "min_new_tokens": 1, "top_k": 1,
            "top_p": 1.0, "temperature": 0.0, "is_streaming": False,
            "return_output_ids": True, "return_input_ids": accuracy,
            "return_logits": accuracy, "aux_info": True,
        }}
    if backend == "sglang":
        return {"input_ids": case["input_ids"], "stream": False,
                "sampling_params": {"max_new_tokens": 1, "min_new_tokens": 1,
                                    "temperature": 0.0, "top_k": 1,
                                    "top_p": 1.0, "ignore_eos": True},
                "return_logprob": accuracy, "logprob_start_len": 0,
                "top_logprobs_num": 20 if accuracy else 0,
                "return_text_in_logprobs": False}
    raise ValueError("unknown backend")


def one_vector(value: Any, name: str) -> list:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(isinstance(x, list) for x in value):
        raise ValueError(f"invalid {name} shape")
    return value


def normalize(backend: str, raw: dict, case: dict, accuracy: bool) -> dict:
    expected = case["input_ids"]
    if backend == "rtp":
        aux = raw.get("aux_info", {})
        if raw.get("finished") is not True:
            raise ValueError("RTP request did not finish")
        input_len, output_len = aux.get("input_len"), aux.get("output_len")
        ids = one_vector(raw.get("output_ids"), "output_ids")
        if accuracy and one_vector(raw.get("input_ids"), "input_ids") != expected:
            raise ValueError("RTP tokenizer/input IDs differ from frozen corpus")
        reused = aux.get("reuse_len")
        if reused is None or reused != 0:
            raise ValueError("RTP reuse_len must be present and zero")
        if aux.get("pd_sep") is not False:
            raise ValueError("stage 1 requires local Prefill, pd_sep=false")
        diagnostic = {k: aux.get(k) for k in (
            "first_token_cost_time", "wait_time", "cost_time", "iter_count")}
    else:
        aux = raw.get("meta_info", {})
        input_len, output_len = aux.get("prompt_tokens"), aux.get("completion_tokens")
        ids = one_vector(raw.get("output_ids"), "output_ids")
        if aux.get("cached_tokens") != 0:
            raise ValueError("SGLang cached_tokens must be present and zero")
        if not aux.get("finish_reason"):
            raise ValueError("SGLang request did not finish")
        diagnostic = {"finish_reason": aux["finish_reason"]}
    if type(input_len) is not int or input_len != len(expected):
        raise ValueError("actual input length differs from frozen corpus")
    if type(output_len) is not int or output_len != 1 or len(ids) != 1:
        raise ValueError("request must produce exactly one token")
    if type(ids[0]) is not int or ids[0] < 0:
        raise ValueError("invalid output token ID")
    result = {"input_len": input_len, "output_len": 1, "output_id": ids[0],
              "diagnostic_only": diagnostic}
    if accuracy and backend == "rtp":
        logits = one_vector(raw.get("logits"), "logits")
        if not logits or any(type(x) not in (int, float) or not math.isfinite(x)
                             for x in logits):
            raise ValueError("missing or non-finite RTP logits")
        if ids[0] >= len(logits):
            raise ValueError("output token outside logits vocabulary")
        maximum = max(logits)
        log_z = maximum + math.log(sum(math.exp(x - maximum) for x in logits))
        result["top_logprobs"] = {
            str(i): logits[i] - log_z
            for i in sorted(range(len(logits)), key=logits.__getitem__, reverse=True)[:20]
        }
        result["logits_argmax"] = max(range(len(logits)), key=logits.__getitem__)
    elif accuracy:
        tops = aux.get("output_top_logprobs")
        if not isinstance(tops, list) or len(tops) != 1 or not tops[0]:
            raise ValueError("missing SGLang next-token log probabilities")
        pairs = tops[0]
        if any(len(p) < 2 or type(p[1]) is not int or not math.isfinite(p[0])
               for p in pairs):
            raise ValueError("invalid SGLang next-token log probabilities")
        result["top_logprobs"] = {str(p[1]): p[0] for p in pairs}
    return result


def collect_one(url: str, backend: str, case: dict, accuracy: bool, timeout: float) -> dict:
    start = time.perf_counter()
    result = {"id": case["id"], "ok": False}
    try:
        data = json.dumps(request_body(backend, case, accuracy), allow_nan=False).encode()
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        # Internal inference endpoints must not inherit workstation HTTP proxies.
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            request, timeout=timeout
        ) as response:
            raw_text = response.read().decode("utf-8")
        # Preserve even a malformed/non-finite server response as exact text.
        result["raw_response"] = raw_text
        raw = json.loads(raw_text)
        result.update(normalize(backend, raw, case, accuracy))
        result["ok"] = True
    except urllib.error.HTTPError as error:
        result["raw_response"] = error.read().decode("utf-8", errors="replace")
        result["error"] = f"HTTPError: {error}"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["request_wall_ms"] = (time.perf_counter() - start) * 1000
    return result


def collect(args: argparse.Namespace) -> None:
    cases = load_corpus(args.corpus)
    declaration = json.loads(args.service_manifest.read_text())
    required = {"tp": 4, "cp": 1, "ep": 1, "dp": 1, "mtp": False,
                "prefix_cache": False, "pd": False}
    if any(declaration.get(k) != v for k, v in required.items()):
        raise ValueError("service declaration must describe TP4/CP1/EP1/DP1, MTP/cache/PD off")
    if declaration.get("runtime_verified") is not True or not declaration.get("runtime_evidence"):
        raise ValueError("record actual runtime verification before sending benchmark requests")
    if not declaration.get("checkpoint_manifest_sha256"):
        raise ValueError("checkpoint identity is required")
    corpus_sha = sha256(args.corpus)
    accuracy = args.mode == "accuracy"
    accuracy_assumed = bool(getattr(args, "assume_accuracy_for_performance", False))
    if accuracy_assumed and accuracy:
        raise ValueError("accuracy assumption is only valid for performance exploration")
    if not accuracy and not accuracy_assumed:
        if args.accuracy_run is None:
            raise ValueError("performance requires a successful accuracy collection")
        contract = json.loads((args.accuracy_run / "summary.json").read_text())
        if (contract.get("mode") != "accuracy" or not contract.get("valid")
                or contract.get("backend") != args.backend
                or contract.get("corpus_sha256") != corpus_sha
                or contract.get("service_manifest_sha256") != sha256(args.service_manifest)):
            raise ValueError("accuracy collection does not match this exact service and corpus")
    if args.concurrency < 1 or args.rounds < 1 or args.warmup_rounds < 1:
        raise ValueError("positive concurrency, rounds and warmup_rounds required")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "backend": args.backend, "mode": args.mode,
                "corpus_sha256": corpus_sha, "service": declaration,
                "service_manifest_sha256": sha256(args.service_manifest),
                "client_sha256": sha256(Path(__file__)), "url": args.url,
                "concurrency": args.concurrency, "warmup_rounds": args.warmup_rounds,
                "rounds": args.rounds, "started_unix": time.time(),
                "accuracy_assumed": accuracy_assumed, "stage1_qualified": False}
    write_json(args.output / "manifest.json", manifest)
    rounds = []
    with (args.output / "responses.jsonl").open("x") as log:
        for round_id in range(-args.warmup_rounds, args.rounds):
            started_unix = time.time()
            server_log = getattr(args, "server_log", None)
            log_start = server_log.stat().st_size if server_log else None
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(collect_one, args.url, args.backend, case,
                                       accuracy, args.timeout) for case in cases]
                responses = [f.result() for f in futures]
            elapsed = time.perf_counter() - start
            ended_unix = time.time()
            compile_markers = None
            if server_log:
                with server_log.open("rb") as server_source:
                    server_source.seek(log_start)
                    fragment = server_source.read().decode("utf-8", errors="replace")
                compile_markers = len(re.findall(r"warnings? generated when compiling|ptxas (?:info|warning)|nvcc fatal|clang: error", fragment))
            for response in responses:
                log.write(json.dumps({"round": round_id, **response}, ensure_ascii=False,
                                     allow_nan=False) + "\n")
            log.flush()
            valid = all(r["ok"] for r in responses) and len(responses) == len(cases)
            if not accuracy and round_id >= 0 and compile_markers:
                valid = False
            result = {"round": round_id, "warmup": round_id < 0, "valid": valid,
                      "success": sum(r["ok"] for r in responses), "requests": len(cases),
                      "wall_seconds": elapsed, "started_unix": started_unix,
                      "finished_unix": ended_unix, "jit_compile_markers": compile_markers,
                      "input_tokens": sum(r.get("input_len", 0) for r in responses if r["ok"]),
                      "output_tokens": sum(r.get("output_len", 0) for r in responses if r["ok"])}
            # Failed/incomplete rounds never receive a scored throughput.
            result["input_tokens_per_request_wall_second"] = (
                result["input_tokens"] / elapsed if valid and not accuracy else None)
            rounds.append(result)
            write_json(args.output / f"round-{round_id}.json", result)
            if not valid:
                break
    measured = [r for r in rounds if not r["warmup"]]
    valid = len(measured) == args.rounds and all(r["valid"] for r in rounds)
    values = [r["input_tokens_per_request_wall_second"] for r in measured
              if r["input_tokens_per_request_wall_second"] is not None]
    write_json(args.output / "summary.json", {
        **manifest, "valid": valid, "whole_model_accuracy_pass": False,
        "pd_or_mtp_qualified": False, "scored_performance": valid and not accuracy,
        "metric_boundary": "one-token request wall time including client work, transport, scheduling and sampling",
        "median_input_tokens_per_request_wall_second": statistics.median(values) if valid and values else None,
        "sample_cv": statistics.stdev(values) / statistics.mean(values) if valid and len(values) > 1 else None,
        "round_results": rounds,
    })
    if not valid:
        raise RuntimeError("incomplete or invalid collection; evidence retained")


def compare(args: argparse.Namespace) -> None:
    summaries = [json.loads((p / "summary.json").read_text()) for p in (args.rtp, args.sglang)]
    a, b = summaries
    if [a["backend"], b["backend"]] != ["rtp", "sglang"]:
        raise ValueError("expected RTP and SGLang runs in that order")
    for key in ("mode", "corpus_sha256", "concurrency", "rounds", "client_sha256", "accuracy_assumed"):
        if a[key] != b[key]:
            raise ValueError(f"comparison mismatch: {key}")
    for key in ("checkpoint_manifest_sha256", "tp", "cp", "ep", "dp", "mtp", "prefix_cache", "pd"):
        if a["service"][key] != b["service"][key]:
            raise ValueError(f"service comparison mismatch: {key}")
    if not a["valid"] or not b["valid"]:
        raise ValueError("cannot compare invalid runs")
    out = {"mode": a["mode"], "corpus_sha256": a["corpus_sha256"],
           "whole_model_accuracy_pass": False, "pd_or_mtp_qualified": False,
           "accuracy_assumed": a["accuracy_assumed"], "stage1_qualified": False}
    if a["mode"] == "performance":
        av, bv = [s["median_input_tokens_per_request_wall_second"] for s in summaries]
        out.update({"rtp_input_token_s": av, "sglang_input_token_s": bv,
                    "rtp_over_sglang": av / bv, "sample_cv": [a["sample_cv"], b["sample_cv"]],
                    "metric_boundary": a["metric_boundary"],
                    "paired_run_order_verified": False})
    else:
        data = []
        for path in (args.rtp, args.sglang):
            with (path / "responses.jsonl").open(encoding="utf-8") as source:
                rows = [json.loads(line) for line in source if line.strip()]
            data.append({(r["round"], r["id"]): r for r in rows if r["round"] >= 0})
        if not data[0] or data[0].keys() != data[1].keys():
            raise ValueError("missing paired cases")
        mismatches, deltas = [], []
        for key, ar in data[0].items():
            br = data[1][key]
            if ar["output_id"] != br["output_id"]:
                mismatches.append({"round": key[0], "id": key[1],
                                   "rtp": ar["output_id"], "sglang": br["output_id"]})
            ap, bp = ar["top_logprobs"], br["top_logprobs"]
            deltas.extend(abs(ap[t] - bp[t]) for t in ap.keys() & bp.keys())
        out.update({"paired_cases": len(data[0]), "top1_matches": len(data[0]) - len(mismatches),
                    "top1_mismatches": mismatches, "common_topk_logprob_samples": len(deltas),
                    "common_topk_max_abs_logprob_delta": max(deltas) if deltas else None,
                    "numerical_thresholds_frozen": False})
    write_json(args.output, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--text-key", required=True)
    p.add_argument("--tokenizer", required=True, type=Path)
    p.add_argument("--max-seq-len", type=int, default=10240)
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--output", required=True, type=Path)
    p.set_defaults(func=freeze)
    p = sub.add_parser("collect")
    p.add_argument("--backend", choices=("rtp", "sglang"), required=True)
    p.add_argument("--mode", choices=("accuracy", "performance"), required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--service-manifest", type=Path, required=True)
    p.add_argument("--accuracy-run", type=Path)
    p.add_argument("--assume-accuracy-for-performance", action="store_true",
                   help="Explicit exploratory measurement; never qualifies stage 1 accuracy")
    p.add_argument("--server-log", type=Path,
                   help="Audit new compilation messages during each measured round")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup-rounds", type=int, default=1)
    p.add_argument("--timeout", type=float, default=900)
    p.set_defaults(func=collect)
    p = sub.add_parser("compare")
    p.add_argument("--rtp", type=Path, required=True)
    p.add_argument("--sglang", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
