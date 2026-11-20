"""Check real CP8/DP8 KV transfer and reuse on existing full40 endpoints.

Use the built SM90 fixture's ordered Python dependency paths for the local
checkpoint tokenizer. This client does not launch or stop either service.
Known-answer semantics and performance are separate checks.
"""

import argparse
import hashlib
import json
import math
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.deepseek_v41_sm90_remote_pd_perf import request, routes


def checked(raw, vocab_size, rank_route, gamma, output_limit, *, expected_pd=True):
    assert "error_code" not in raw and raw.get("finished") is True, raw
    assert isinstance(raw.get("response"), str)
    if expected_pd:
        assert raw["response"].strip()
    aux = raw["aux_info"]
    assert aux["pd_sep"] is expected_pd
    assert len(raw["input_ids"]) == len(raw["output_ids"]) == 1
    assert len(raw["input_ids"][0]) == aux["input_len"]
    tokens = raw["output_ids"][0]
    assert 0 < len(tokens) <= output_limit and aux["output_len"] == len(tokens)
    assert all(type(token) is int and 0 <= token < vocab_size for token in tokens)
    assert 0 <= aux["prefill_total_reuse_len"] < aux["input_len"]
    if expected_pd:
        assert any(row.get("role") == "DECODE" and row.get("ip") == rank_route["ip"]
                   and int(row.get("grpc_port", -1)) == rank_route["grpc_port"]
                   for row in aux.get("role_addrs", [])), "Wrong decode route"
    else:
        # PrefillRpcServer deliberately handles max_new_tokens=1 locally.
        # An echoed DECODE role_addr is not proof that D executed this call.
        assert output_limit == 1 and aux.get("speculative_draft_rounds", 0) == 0
        assert sum(aux.get("speculative_accepted_tokens_per_pos", [])) == 0
    if not gamma:
        assert aux.get("speculative_draft_rounds", 0) == 0
        assert len(raw["logits"]) == 1 and len(raw["logits"][0]) == 15
        assert all(math.isfinite(value) for value in raw["logits"][0])
        assert aux["cum_log_probs"] and all(math.isfinite(value) for value in aux["cum_log_probs"])
    elif output_limit > 1:
        rounds = aux["speculative_draft_rounds"]
        accepted = aux["speculative_accepted_tokens_per_pos"]
        verified = aux["speculative_verified_draft_tokens"]
        assert rounds > 0 and len(accepted) == gamma
        assert all(0 <= count <= rounds for count in accepted)
        assert 0 <= sum(accepted) <= verified <= rounds * gamma and verified > 0
    return raw


def same(reference, actual, gamma):
    # Preserve the upstream per-path exact checks. Fresh BF16 SWA and cached
    # MXFP8 SWA are different paths; never demand their cross-path equality.
    for name in ("input_ids", "output_ids", "response"):
        assert reference[name] == actual[name], name
    if not gamma:
        assert reference["logits"] == actual["logits"], "generation logits"
        assert reference["aux_info"]["cum_log_probs"] == actual["aux_info"]["cum_log_probs"], "generation scores"


def checked_sampling_logprobs(actual, rank_route, gamma, stochastic, reference=None):
    """Keep the upstream DSpark sampling oracle and its exact tolerances."""
    check = unittest.TestCase()
    check.assertNotIn("error_code", actual)
    aux = actual["aux_info"]
    check.assertIs(aux["pd_sep"], True)
    check.assertTrue(any(row.get("role") == "DECODE" and row.get("ip") == rank_route["ip"]
                         and int(row.get("grpc_port", -1)) == rank_route["grpc_port"]
                         for row in aux.get("role_addrs", [])), "Wrong decode route")
    rounds = aux["speculative_draft_rounds"]
    accepted = aux["speculative_accepted_tokens_per_pos"]
    verified = aux["speculative_verified_draft_tokens"]
    check.assertGreater(rounds, 0)
    check.assertEqual(len(accepted), gamma)
    check.assertTrue(all(0 <= count <= rounds for count in accepted))
    check.assertTrue(0 <= sum(accepted) <= verified <= rounds * gamma and verified > 0)
    choice = actual["choices"][0]
    content = choice["message"]["content"]
    records = choice["logprobs"]["content"]
    check.assertTrue(choice["finish_reason"])
    check.assertGreater(len(records), 1, "must exercise speculative decode")
    check.assertEqual(b"".join(bytes(record["bytes"]) for record in records).decode("utf-8"), content)
    for record in records:
        score = record["logprob"]
        check.assertTrue(math.isfinite(score))
        check.assertLessEqual(score, 1e-5)
        top = record["top_logprobs"]
        check.assertEqual(len(top), 1)
        check.assertTrue(math.isfinite(top[0]["logprob"]))
        check.assertLessEqual(score, top[0]["logprob"] + 1e-5)
        if not stochastic:
            check.assertAlmostEqual(score, 0.0, places=5)
        if top[0]["bytes"] == record["bytes"]:
            check.assertAlmostEqual(score, top[0]["logprob"], places=5)
    if stochastic:
        check.assertTrue(any(record["logprob"] < -1e-5 for record in records))
    # As upstream: draft proposals use global RNG. Only greedy is reproducible.
    if not stochastic and reference is not None:
        check.assertEqual(content, reference["message"]["content"])
        check.assertEqual(records, reference["logprobs"]["content"])
    return choice


def sampling_logprobs(endpoint, prefill_route, decode_route, gamma, output_dir, report, save):
    """Run the original greedy/top-p/full modes twice against one real D rank."""
    assert gamma == 5
    query = dict(
        messages=[dict(role="user", content="Invent a peculiar name for a blue dragon and describe its unusual hobbies.")],
        max_tokens=32, enable_thinking=False, logprobs=True, top_logprobs=1,
        extra_configs=dict(random_seed=42, reuse_cache=False, can_use_pd_separation=True,
                           force_sp_accept=False, role_addrs=[prefill_route, decode_route]),
    )
    report["sampling_logprobs"] = []
    for label, stochastic, temperature, top_p in (
        ("greedy", False, 0.0, 1.0), ("top_p", True, 1.5, 0.9), ("full", True, 0.8, 1.0),
    ):
        query.update(temperature=temperature, top_p=top_p)
        query["extra_configs"].update(top_k=0 if stochastic else 1, do_sample=stochastic)
        reference = None
        for repeat in range(2):
            actual = request(endpoint, "/v1/chat/completions", query, timeout=600)
            path = output_dir / f"sampling_{label}_{repeat}.json"
            path.write_text(json.dumps(dict(query=query, actual=actual), ensure_ascii=False, indent=2))
            choice = checked_sampling_logprobs(actual, decode_route, gamma, stochastic, reference)
            aux = actual["aux_info"]
            report["sampling_logprobs"].append(dict(
                label=label, repeat=repeat, raw_path=str(path), decode_route=decode_route,
                pd_sep=aux["pd_sep"], draft_rounds=aux["speculative_draft_rounds"],
                accepted_tokens=sum(aux["speculative_accepted_tokens_per_pos"]),
                accepted_tokens_per_pos=aux["speculative_accepted_tokens_per_pos"],
                verified_draft_tokens=aux["speculative_verified_draft_tokens"],
                logprob_records=len(choice["logprobs"]["content"]), finish_reason=choice["finish_reason"],
            ))
            save()
            reference = choice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-endpoint", required=True)
    parser.add_argument("--decode-endpoint", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gamma", type=int, choices=(0, 5), required=True)
    args = parser.parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text())
    config = config.get("text_config", config)
    assert config["num_hidden_layers"] == 40
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), trust_remote_code=True, local_files_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", topology="CP8/DP8 EP8", gamma=args.gamma,
                  requests=[], cache_contract="fresh/fill/bypass exact; repeated hits exact",
                  logits_contract="exact 15 selected logits and cumulative scores" if not args.gamma
                  else "upstream specUpdate does not define native generation logits/scores")
    destination = args.output_dir / "pd-contract-summary.json"

    def save():
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    def generate(label, prompt, reuse, rank, limit=24):
        config = dict(max_new_tokens=limit, top_k=1, top_p=1.0, do_sample=False,
                      random_seed=42, reuse_cache=reuse, force_sp_accept=False,
                      can_use_pd_separation=True, return_input_ids=True,
                      return_output_ids=True, aux_info=True,
                      role_addrs=[prefill[0], decode[rank]])
        if not args.gamma:
            config.update(return_cum_log_probs=True, return_logits=True,
                          select_tokens_id=[vocab_size * i // 16 for i in range(1, 16)])
        query = dict(prompt=prompt, generate_config=config)
        raw = request(args.prefill_endpoint, "/", query, timeout=600)
        (args.output_dir / (label + ".json")).write_text(json.dumps(dict(query=query, actual=raw), ensure_ascii=False, indent=2))
        expected_pd = limit > 1
        checked(raw, vocab_size, decode[rank], args.gamma, limit, expected_pd=expected_pd)
        report["requests"].append(dict(label=label, decode_rank=rank if expected_pd else None,
                                       execution="PD" if expected_pd else "fresh_prefill_only_reference",
                                       input_len=raw["aux_info"]["input_len"],
                                       prefill_reuse_len=raw["aux_info"]["prefill_total_reuse_len"],
                                       draft_rounds=raw["aux_info"].get("speculative_draft_rounds", 0),
                                       accepted_tokens=sum(raw["aux_info"].get("speculative_accepted_tokens_per_pos", [])),
                                       verified_draft_tokens=raw["aux_info"].get("speculative_verified_draft_tokens", 0),
                                       output_ids=raw["output_ids"]))
        save()
        return raw

    save()
    try:
        vocab_size = int(config["vocab_size"])
        prefill = routes(args.prefill_endpoint, "PREFILL", 1)
        decode = routes(args.decode_endpoint, "DECODE", 8)
        report.update(prefill_routes=prefill, decode_routes=decode)
        short = "The capital of France is"
        references = []
        for rank in range(8):
            first = generate(f"rank{rank}-short-fresh", short, False, rank)
            assert first["aux_info"]["prefill_total_reuse_len"] == 0
            same(first, generate(f"rank{rank}-short-repeat", short, False, rank), args.gamma)
            # Independent full-prefill references for the first three tokens.
            prefix, expected = list(first["input_ids"][0]), []
            for step in range(min(3, len(first["output_ids"][0]))):
                prompt = tokenizer.decode(prefix, skip_special_tokens=False)
                assert tokenizer.encode(prompt) == prefix, "Tokenizer round trip changed actual input"
                one = generate(f"rank{rank}-fresh-prefill-{step}", prompt, False, rank, 1)
                assert one["input_ids"][0] == prefix and len(one["output_ids"][0]) == 1
                token = one["output_ids"][0][0]
                expected.append(token)
                prefix.append(token)
                if token == tokenizer.eos_token_id:
                    break
            assert len(expected) >= 2 and first["output_ids"][0][:len(expected)] == expected
            references.append(first)
        for actual in references[1:]:
            same(references[0], actual, args.gamma)
        # CP8 canonical cache keys span 256 * 8 tokens. Exercise at least two
        # keys, then send the same cached context to every real decode rank.
        # A previous client may have populated this still-running P fixture.
        # Keep this run's cold-fill prompt unique while all repeats share it.
        cache_case_id = hashlib.sha256(str(args.output_dir).encode()).hexdigest()[:16]
        report["cache_case_id"] = cache_case_id
        lines = []
        while True:
            lines.append(f"Notebook entry {len(lines)}: the archive stores books, maps, and letters.\n")
            prompt = f"Notebook {cache_case_id} records a journey.\n" + "".join(lines)
            if len(tokenizer.encode(prompt)) >= 4096:
                break
        prompt += "\nIn a few words, the journey was"
        report["prefix_input_tokens"] = len(tokenizer.encode(prompt))
        baseline = generate("long-uncached", prompt, False, 0)
        cold = generate("long-cache-fill", prompt, True, 0)
        assert baseline["aux_info"]["prefill_total_reuse_len"] == cold["aux_info"]["prefill_total_reuse_len"] == 0
        same(baseline, cold, args.gamma)
        first_hit = None
        for rank in range(8):
            hit = generate(f"rank{rank}-cache-hit", prompt, True, rank)
            second = generate(f"rank{rank}-cache-hit-repeat", prompt, True, rank)
            assert hit["aux_info"]["prefill_total_reuse_len"] >= 2048
            assert hit["aux_info"]["prefill_total_reuse_len"] == second["aux_info"]["prefill_total_reuse_len"]
            same(hit, second, args.gamma)
            if first_hit is not None:
                same(first_hit, hit, args.gamma)
            first_hit = hit
            bypass = generate(f"rank{rank}-cache-bypass", prompt, False, rank)
            assert bypass["aux_info"]["prefill_total_reuse_len"] == 0
            same(baseline, bypass, args.gamma)
        if args.gamma:
            assert any(row["draft_rounds"] > 1 for row in report["requests"])
            assert sum(row["accepted_tokens"] for row in report["requests"]) > 0
            sampling_logprobs(args.prefill_endpoint, prefill[0], decode[0], args.gamma,
                              args.output_dir, report, save)
        report["status"] = "PASS"
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()
    print(json.dumps(dict(status=report["status"], requests=len(report["requests"]))))


if __name__ == "__main__":
    main()
