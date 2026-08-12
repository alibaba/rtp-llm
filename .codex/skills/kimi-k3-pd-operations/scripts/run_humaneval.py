#!/usr/bin/env python3
"""Run five Kimi K3 Eagle3 HumanEval prompts through the P/D frontend."""

import argparse
import glob
import json
import time
import urllib.request

import torch
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://11.163.39.114:27188/")
    parser.add_argument(
        "--samples",
        default="/mnt/nas1/hf/MAL_test_codes/samples/humaneval_*.pt",
    )
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--tokenizer", default="/data3/kimi-k3")
    parser.add_argument(
        "--output-json",
        default="/data1/xinfei.sxf/k3-pd-logs/humaneval-cudagraph.json",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(args.samples))[: args.count]
    if len(paths) != args.count:
        raise RuntimeError(f"expected {args.count} samples, found {len(paths)}")

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )
    results = []
    total_accepted = 0
    total_proposed = 0
    for path in paths:
        dump = torch.load(path, map_location="cpu", weights_only=False)
        input_len = int(dump["input_len"])
        ids = dump["input_ids"][:input_len].tolist()
        prompt = tokenizer.decode(ids)
        roundtrip_ids = tokenizer.encode(prompt)
        if roundtrip_ids != ids:
            raise RuntimeError(f"{path}: token round-trip mismatch")
        payload = {
            "prompt": prompt,
            "generate_config": {
                "max_new_tokens": args.max_new_tokens,
                "top_k": 1,
                "top_p": 0,
                "temperature": 0.0,
                "ignore_eos": False,
                "skip_special_tokens": False,
                "return_input_ids": True,
                "return_output_ids": True,
                "can_use_pd_separation": True,
                "force_disable_sp_run": False,
            },
        }
        request = urllib.request.Request(
            args.endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.time()
        with opener.open(request, timeout=1800) as response:
            result = json.loads(response.read())
        aux = result["aux_info"]
        output_ids = result.get("output_ids", [[]])[0]
        output_len = int(aux["output_len"])
        iterations = int(aux["iter_count"])
        accepted = output_len - iterations
        proposed = max(0, (iterations - 1) * 3)
        total_accepted += accepted
        total_proposed += proposed
        max_run = 0
        current_run = 0
        previous = None
        for token_id in output_ids:
            current_run = current_run + 1 if token_id == previous else 1
            max_run = max(max_run, current_run)
            previous = token_id
        row = {
            "query": dump.get("record_id", path),
            "input_len": input_len,
            "roundtrip_equal": True,
            "elapsed_s": round(time.time() - started, 3),
            "output_len": output_len,
            "iter_count": iterations,
            "accepted": accepted,
            "proposed": proposed,
            "rate": accepted / proposed if proposed else 0,
            "last_token": output_ids[-1] if output_ids else None,
            "max_consecutive_identical_tokens": max_run,
            "response": result.get("response", ""),
            "output_ids": output_ids,
        }
        results.append(row)
        with open(args.output_json, "w") as output_file:
            json.dump(results, output_file, ensure_ascii=False, indent=2)
        summary = {k: v for k, v in row.items() if k not in ("response", "output_ids")}
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(
        json.dumps(
            {
                "total_accepted": total_accepted,
                "total_proposed": total_proposed,
                "weighted_rate": total_accepted / total_proposed,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
