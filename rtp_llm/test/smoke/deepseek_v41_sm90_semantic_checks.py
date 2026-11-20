"""Short full40 SM90 semantic checks against an already running HTTP endpoint.

The caller owns model startup, source binding, GPU leases and P/D topology. This
module only sends requests, so the same checks work for CP4/DP4 and CP8/DP8.
"""

import argparse
import concurrent.futures
import json
import re
import urllib.error
import urllib.request
from pathlib import Path


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _payload(kind):
    config = dict(
        max_new_tokens=24 if kind == "native_france" else 64,
        min_new_tokens=0,
        top_k=1,
        top_p=1.0,
        do_sample=False,
        force_sp_accept=False,
        random_seed=42,
        reuse_cache=False,
        can_use_pd_separation=True,
        return_output_ids=True,
        aux_info=True,
        ignore_eos=False,
        thinking_mode=1,  # GenerateConfig.ThinkingMode.DISABLED
    )
    if kind == "native_france":
        return "/", dict(prompt="The capital of France is", generate_config=config)
    return "/v1/chat/completions", dict(
        messages=[dict(role="user", content="What is 17 + 25? Reply with only the number.")],
        max_tokens=64,
        temperature=0.0,
        top_p=1.0,
        stream=True,
        stream_options=dict(include_usage=True),
        debug_info=True,  # Obtain the renderer's EOS ID; never invent one.
        enable_thinking=False,
        extra_configs=config,
    )


def _check(kind, raw, vocab_size, expected_pd):
    _require("error_code" not in raw, str(raw))
    aux = raw.get("aux_info") or {}
    _require(aux.get("pd_sep") is expected_pd, "Expected actual P/D execution")
    limit = 24 if kind == "native_france" else 64
    if kind == "native_france":
        _require(raw.get("finished") is True, "Native generation did not finish")
        content = raw.get("response")
        ids = raw.get("output_ids")
        _require(isinstance(content, str), "Missing native text")
        _require(re.search(r"\bParis\b", content, re.IGNORECASE), "France's capital must be Paris")
        _require(
            not re.search(r"\b(?:not|isn't|is not)\s+Paris\b", content, re.IGNORECASE),
            "Native answer contradicts the requested fact",
        )
        # This is a short factual completion, not a general table/repetition
        # heuristic for arbitrary model output. Catch the historical loops.
        words = re.findall(r"\w+", content.casefold())
        for width in range(1, 9):
            _require(
                not any(
                    words[index:index + width]
                    == words[index + width:index + 2 * width]
                    == words[index + 2 * width:index + 3 * width]
                    for index in range(len(words) - 3 * width + 1)
                ),
                "Native factual completion repeats a phrase three times",
            )
        # Native text is a continuation prompt. Its fixed24-token budget is
        # not an EOS contract; natural termination is asserted by chat below.
        finish = "finished_within_budget"
    else:
        choices = raw.get("choices")
        _require(isinstance(choices, list) and len(choices) == 1, "Expected one chat choice")
        choice = choices[0]
        _require(choice.get("finish_reason") == "stop", "Chat must stop naturally, not reach its token limit")
        message = choice.get("message") or {}
        content = message.get("content")
        _require(isinstance(content, str), "Missing chat text")
        _require(not message.get("reasoning_content"), "Thinking must be disabled")
        answer = content.strip().strip("`* ").strip()
        _require(
            re.fullmatch(r"(?i)(?:(?:the answer is|answer:)\s*)?(?:17\s*\+\s*25\s*=\s*)?42[.!]?", answer),
            "17 + 25 must yield the short answer 42",
        )
        ids = (raw.get("extra_outputs") or {}).get("output_ids")
        audit = raw.get("stream_audit") or {}
        _require(audit.get("complete") is True, "Missing complete audited chat stream")
        eos = audit.get("eos_token_id")
        _require(type(eos) is int and 0 <= eos < vocab_size, "Missing valid renderer EOS ID")
        _require(isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list), "Missing complete streamed IDs")
        _require(ids[0] and ids[0][-1] == eos and eos not in ids[0][:-1], "Chat IDs must end at the actual EOS")
        _require((raw.get("usage") or {}).get("completion_tokens") == aux.get("output_len"), "Chat usage and final output length disagree")
        finish = choice["finish_reason"]
    _require(isinstance(ids, list) and len(ids) == 1, "Missing one complete output-ID row")
    tokens = ids[0]
    _require(isinstance(tokens, list) and tokens, "Empty output IDs")
    _require(all(type(token) is int and 0 <= token < vocab_size for token in tokens), "Invalid output IDs")
    _require(
        len(tokens) <= limit if kind == "native_france" else len(tokens) < limit,
        "Output exceeded its native budget or chat failed to stop before its limit",
    )
    _require(aux.get("output_len") == len(tokens), "Output length and IDs disagree")
    return dict(content=content, output_ids=tokens, finish=finish)


def _collect_chat_sse(body):
    """Collect this server's incremental OpenAI events, retaining raw evidence.

    OpenAI streaming ends at EOF here; the server does not emit a [DONE]
    sentinel. A terminal choice with final aux/usage is therefore mandatory.
    Per-chunk output IDs are incremental, while usage counts are cumulative.
    """
    chunks, data = [], []
    done = False
    for line in body.splitlines() + [""]:
        if not line:
            if not data:
                continue
            encoded = "\n".join(data)
            data = []
            _require(not done, "Data after stream terminator")
            if encoded in ("[DONE]", "[done]"):
                done = True
                continue
            chunk = json.loads(encoded)
            _require(isinstance(chunk, dict) and "error_code" not in chunk and "error" not in chunk, "Stream error event")
            chunks.append(chunk)
        elif line.startswith(":"):
            continue
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
        else:
            _require(line.startswith(("event:", "id:", "retry:")), "Invalid SSE framing")
    _require(chunks, "Empty chat stream")
    tokens, content, reasoning, seen = [], [], [], set()
    usage = aux = None
    eos = None
    finished = False
    for chunk in chunks:
        fingerprint = json.dumps(chunk, sort_keys=True)
        _require(fingerprint not in seen, "Duplicate chat stream event")
        seen.add(fingerprint)
        _require(not finished, "Unexpected event after terminal chat choice")
        debug = chunk.get("debug_info") or {}
        if isinstance(debug, dict) and debug.get("eos_token_id") is not None:
            declared = debug["eos_token_id"]
            _require(type(declared) is int and declared >= 0, "Invalid renderer EOS ID")
            _require(eos is None or eos == declared, "Renderer EOS ID changed midstream")
            eos = declared
        choices = chunk.get("choices")
        _require(isinstance(choices, list) and len(choices) == 1, "Expected one stream choice")
        choice = choices[0]
        _require(choice.get("index") == 0, "Wrong streamed choice index")
        delta = choice.get("delta") or {}
        _require(not delta.get("tool_calls") and not delta.get("function_call"), "Unexpected tool output")
        for key, target in (("content", content), ("reasoning_content", reasoning)):
            value = delta.get(key)
            _require(value is None or isinstance(value, str), "Invalid streamed text")
            if value:
                target.append(value)
        ids = (chunk.get("extra_outputs") or {}).get("output_ids")
        if ids is not None:
            _require(isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list) and ids[0], "Invalid streamed ID row")
            _require(all(type(token) is int and token >= 0 for token in ids[0]), "Invalid streamed token ID")
            tokens.extend(ids[0])
        current_usage = chunk.get("usage")
        if current_usage is not None:
            count = current_usage.get("completion_tokens")
            _require(type(count) is int and count == len(tokens), "Missing or duplicated token chunk: cumulative usage mismatch")
            usage = current_usage
        elif ids is not None:
            _require(False, "Token chunk lacks cumulative usage")
        if chunk.get("aux_info") is not None:
            aux = chunk["aux_info"]
        finish = choice.get("finish_reason")
        if finish is not None:
            _require(finish == "stop", "Chat must stop naturally")
            _require(current_usage is not None and chunk.get("aux_info") is not None, "Terminal choice lacks final aux/usage")
            _require(aux.get("output_len") == len(tokens), "Final aux and complete streamed IDs disagree")
            _require(eos is not None and tokens and tokens[-1] == eos and eos not in tokens[:-1], "Complete stream lacks a single terminal EOS")
            finished = True
    _require(finished, "Chat stream ended without terminal choice")
    return dict(
        choices=[dict(message=dict(content="".join(content), reasoning_content="".join(reasoning)), finish_reason="stop")],
        extra_outputs=dict(output_ids=[tokens]), aux_info=aux, usage=usage,
        stream_audit=dict(complete=True, eos_token_id=eos, raw_chunks=chunks, raw_sse=body),
    )


def run_semantic_checks(post, output_dir, vocab_size, *, expected_pd=True):
    """Run two known-answer tasks, three serial repeats and two concurrent pairs.

    ``post(path, payload)`` returns the server JSON. No generated response is
    promoted to a golden: every answer is independently checked before repeat
    equality, and every original payload/response is retained.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(status="RUNNING", kinds=["native_france", "chat_arithmetic"], serial_repeats=3, concurrency=2, requests=[])
    destination = output_dir / "semantic-summary.json"

    def request(kind, label):
        path, payload = _payload(kind)
        raw = post(path, payload)
        (output_dir / (label + ".json")).write_text(
            json.dumps(dict(path=path, query=payload, actual=raw), ensure_ascii=False, indent=2)
        )
        checked = _check(kind, raw, vocab_size, expected_pd)
        return dict(label=label, kind=kind, **checked)

    try:
        references = {}
        for kind in report["kinds"]:
            for repeat in range(3):
                result = request(kind, f"{kind}_serial_{repeat}")
                if repeat:
                    _require(
                        {key: result[key] for key in ("content", "output_ids", "finish")} == references[kind],
                        f"Greedy serial output changed for {kind}",
                    )
                else:
                    references[kind] = {key: result[key] for key in ("content", "output_ids", "finish")}
                report["requests"].append(result)
        for repeat in range(2):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(request, kind, f"{kind}_concurrent_{repeat}") for kind in report["kinds"]]
                for kind, future in zip(report["kinds"], futures):
                    result = future.result()
                    _require(
                        {key: result[key] for key in ("content", "output_ids", "finish")} == references[kind],
                        f"Greedy concurrent output changed for {kind}",
                    )
                    report["requests"].append(result)
        report["status"] = "PASS"
        return report
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2))


def http_post(endpoint, timeout=180):
    def post(path, payload):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(
            endpoint.rstrip("/") + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with opener.open(request, timeout=timeout) as response:
                body = response.read().decode()
        except urllib.error.HTTPError as error:
            return dict(error_code="HTTP_ERROR", status=error.code, raw_body=error.read().decode(errors="replace"))
        if path == "/v1/chat/completions" and payload.get("stream"):
            try:
                return _collect_chat_sse(body)
            except (AssertionError, ValueError, KeyError, TypeError, AttributeError) as error:
                return dict(error_code="INVALID_CHAT_STREAM", error=str(error), raw_sse=body)
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return dict(error_code="INVALID_JSON", raw_body=body)
    return post


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="Existing P HTTP endpoint; no service is launched")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads((Path(args.checkpoint) / "config.json").read_text())
    text = config.get("text_config", config)
    _require(text["num_hidden_layers"] == 40, "This semantic fixture requires full40 weights")
    result = run_semantic_checks(http_post(args.endpoint), args.output_dir, int(text["vocab_size"]))
    print(json.dumps(dict(status=result["status"], requests=len(result["requests"]))))


if __name__ == "__main__":
    main()
