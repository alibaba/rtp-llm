"""Real model HTTP E2E; a separate Java harness owns tree building/publication."""

import json
import os
import time
import urllib.error
import urllib.request

WORKER = os.environ.get("CSR_E2E_WORKER", "http://127.0.0.1:18765")
MASTER = os.environ.get("CSR_E2E_MASTER", "http://127.0.0.1:18770")
EOS = 151645


def http(base, path, body=None, raw=False):
    payload = body if raw else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(
        base + path,
        data=payload,
        headers={
            "Content-Type": "application/octet-stream" if raw else "application/json"
        },
    )
    try:
        response = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as exc:
        response = exc
    data = response.read()
    try:
        data = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        pass
    return response.status, data, dict(response.headers)


def generate(beams=1, schedule=None):
    return http(
        WORKER,
        "/",
        {
            "prompt": "请推荐商品：",
            "generate_config": {
                "max_new_tokens": 12,
                "num_beams": beams,
                "variable_num_beams": schedule or [],
                "top_k": 1,
                "return_output_ids": True,
                "is_streaming": False,
            },
        },
    )


def publish(version, sids):
    code, body, _ = http(
        MASTER,
        "/rtp_llm/constraint_tree/build",
        {
            "version": version,
            "model": "engine_service",
            "start_token_id": 1699,
            "end_token_id": EOS,
            "sids": sids,
        },
    )
    assert code == 200 and body["state"] == "ACCEPTED", (code, body)
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        _, state, _ = http(MASTER, "/rtp_llm/constraint_tree/status")
        if state["state"] == "READY" and state["active_version"] == version:
            assert (
                state["published_worker_count"] == state["target_worker_count"] == 1
            ), state
            print(json.dumps({"master_ready": state}), flush=True)
            return
        time.sleep(0.2)
    raise AssertionError(state)


def check_generation(sids, beams=1, repeats=3, schedule=None):
    allowed = [list(map(int, sid.split("_"))) for sid in sids]
    for _ in range(repeats):
        code, body, _ = generate(beams, schedule)
        print(
            json.dumps({"inference_http": code, "result": body}, ensure_ascii=False),
            flush=True,
        )
        assert code == 200 and body.get("finished") is True, body
        outputs = body["output_ids"]
        assert len(outputs) == (schedule[-1] if schedule else beams), outputs
        assert len({tuple(tokens) for tokens in outputs}) == len(outputs), outputs
        for tokens in outputs:
            # Engine responses can include EOS padding for shorter beams.
            assert EOS in tokens, ("missing allowed EOS", tokens)
            if EOS in tokens:
                i = tokens.index(EOS)
                assert all(x == EOS for x in tokens[i:]), tokens
                tokens = tokens[:i]
            assert tokens in allowed, (tokens, allowed)


def variable_beam_checks():
    """Real model + Java publication; no special-case production beam widths."""
    import itertools
    import math

    sids = [
        "_".join(map(str, tokens))
        for tokens in itertools.product(
            range(169967, 169971), *[range(216540, 216544)] * 4
        )
    ]
    publish(int(time.time() * 1000), sids)
    for schedule in ([2, 4, 1, 1, 3], [1, 1, 3, 2, 4], [3]):
        check_generation(sids, max(schedule), repeats=3, schedule=schedule)

    # Only four two-token paths: requesting five must fail without returning
    # masked tokens or duplicate padding. A subsequent valid request must work.
    sids = [
        f"{first}_{second}" for first in (169967, 169968) for second in (216540, 216541)
    ]
    publish(int(time.time() * 1000), sids)
    code, body, _ = generate(5, [2, 5])
    assert code >= 400, (code, body)
    check_generation(sids, 3, repeats=3, schedule=[2, 3])

    # Actual business widths, with fewer root candidates than the final width.
    sids = [
        f"{first}_{second}"
        for first in range(169967, 170479)
        for second in range(215830, 215838)
    ]
    publish(int(time.time() * 1000), sids)
    allowed = {tuple(map(int, sid.split("_"))) for sid in sids}
    for _ in range(3):
        code, body, _ = generate(3500, [512, 3500])
        assert code == 200 and body.get("finished"), (code, str(body)[:1000])
        outputs = body["output_ids"]
        assert len(outputs) == len({tuple(tokens) for tokens in outputs}) == 3500
        for tokens in outputs:
            assert EOS in tokens and all(
                t == EOS for t in tokens[tokens.index(EOS) :]
            ), tokens
            assert tuple(tokens[: tokens.index(EOS)]) in allowed, tokens
        for aux in body["aux_info"]:
            scores = aux["cum_log_probs"]
            assert all(
                math.isfinite(x)
                for x in (scores if isinstance(scores, list) else [scores])
            )
        print(
            json.dumps(
                {
                    "variable_beams": [512, 3500],
                    "allowed_unique_outputs": len(outputs),
                    "eos_verified": True,
                    "aux": body["aux_info"][0],
                }
            ),
            flush=True,
        )
    print(
        "PASS: real-model variable beam growth/shrink/one-step holds, candidate shortage, business 512->3500 + EOS",
        flush=True,
    )


def main():
    code, state, _ = http(WORKER, "/constraint_tree_status")
    assert code == 200 and state["version"] == 0, state
    code, body, _ = generate()
    print(
        json.dumps({"no_tree_rejection": [code, body]}, ensure_ascii=False), flush=True
    )
    assert code >= 400 and "constraint" in str(body).lower(), (code, body)

    v = int(time.time() * 1000)
    first = ["169967_216546", "169968_215835_215836", "169969"]
    second = ["169970_216547", "169971_215838_215839", "169972"]
    publish(v, first)
    check_generation(first)
    check_generation(first, beams=3)
    publish(v + 1, second)
    check_generation(second)
    check_generation(second, beams=3)
    code, old, headers = http(MASTER, "/rtp_llm/constraint_tree/artifact?slot=backup")
    assert code == 200 and isinstance(old, bytes), (code, old)
    assert next(
        value
        for key, value in headers.items()
        if key.lower() == "x-constraint-tree-version"
    ) == str(v)
    code, body, _ = http(WORKER, "/update_constraint_tree", old, raw=True)
    assert code == 409 and body["status"] == "stale_version", (code, body)
    check_generation(second)
    code, body, _ = http(WORKER, "/update_constraint_tree", b"invalid-csr", raw=True)
    assert code >= 400, (code, body)
    check_generation(second)
    code, body, _ = generate(beams=4)
    assert code >= 400, (code, body)
    print(
        "PASS: real model generation, Java Master HTTP build/push/reconcile, variable SID lengths, fixed beam=1/3, hot update, backup, stale/corrupt rejection, root-beam admission",
        flush=True,
    )


def extended_checks():
    import struct

    v = int(time.time() * 1000)
    sids = ["169973", "169973_216540", "169974_216541_215830", "169975_216542"]
    publish(v, sids)
    for beams in (1, 2, 3):
        check_generation(sids, beams=beams, repeats=5)

    # Header passes admission, but invalid CSR content fails in the background loader.
    code, artifact, _ = http(MASTER, "/rtp_llm/constraint_tree/artifact")
    assert code == 200 and isinstance(artifact, bytes)
    bad = bytearray(artifact)
    struct.pack_into("<Q", bad, 16, v + 1)
    header_size = struct.unpack_from("<I", bad, 12)[0]
    states = struct.unpack_from("<I", bad, 32)[0]
    struct.pack_into("<i", bad, header_size + 4 * (states + 1), -5)
    code, body, _ = http(WORKER, "/update_constraint_tree", bytes(bad), raw=True)
    assert code == 200 and body["status"] == "accepted", (code, body)
    for _ in range(100):
        _, state, _ = http(WORKER, "/constraint_tree_status")
        if state["status"] == "failed":
            break
        time.sleep(0.02)
    assert state["status"] == "failed" and state["version"] == v, state
    print(json.dumps({"background_load_failure_retained_old_tree": state}), flush=True)
    check_generation(sids, beams=3)

    # A live streaming request must pin the old snapshot while a new one activates.
    long_tokens = [169976] + [216540 + i % 5 for i in range(190)]
    old_sid = "_".join(map(str, long_tokens))
    publish(v + 2, [old_sid])
    req = urllib.request.Request(
        WORKER + "/",
        data=json.dumps(
            {
                "prompt": "请推荐商品：",
                "generate_config": {
                    "max_new_tokens": 220,
                    "num_beams": 1,
                    "top_k": 1,
                    "return_output_ids": True,
                    "is_streaming": True,
                },
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    new_sids = ["169977_216544", "169978", "169979_216545_215831"]
    tokens = []
    activated = False
    started = time.monotonic()
    activation_ms = None
    with urllib.request.urlopen(req, timeout=60) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data.lower() == b"[done]":
                break
            chunk = json.loads(data)
            assert "error" not in chunk, chunk
            tokens.extend(chunk["output_ids"][0])
            if not activated:
                assert tokens and not chunk["finished"], chunk
                code, body, _ = http(
                    MASTER,
                    "/rtp_llm/constraint_tree/build",
                    {
                        "version": v + 3,
                        "model": "engine_service",
                        "start_token_id": 1699,
                        "end_token_id": EOS,
                        "sids": new_sids,
                    },
                )
                assert code == 200, (code, body)
                for _ in range(200):
                    _, state, _ = http(WORKER, "/constraint_tree_status")
                    if state["version"] == v + 3:
                        activated = True
                        activation_ms = (time.monotonic() - started) * 1000
                        break
                    time.sleep(0.001)
                assert activated, state
                print(
                    json.dumps(
                        {
                            "new_version_active_after_old_stream_tokens": len(tokens),
                            "worker": state,
                        }
                    ),
                    flush=True,
                )
    assert activated and tokens == long_tokens + [EOS], (len(tokens), tokens)
    # Prove real overlap, not just consumption of a response buffered before the switch.
    generation_ms = chunk["aux_info"][0]["cost_time"]
    assert chunk["finished"] and activation_ms < generation_ms, (activation_ms, chunk)
    print(
        json.dumps(
            {"activation_ms": activation_ms, "old_request_generation_ms": generation_ms}
        ),
        flush=True,
    )
    check_generation(new_sids, beams=3)
    print(
        "PASS: prefix-overlapping SIDs, beams 1/2/3, asynchronous load failure retaining old tree, live stream snapshot pinning",
        flush=True,
    )
    print(
        json.dumps({"restart_expected_version": v + 3, "restart_sids": new_sids}),
        flush=True,
    )


if __name__ == "__main__":
    main()
    extended_checks()
    variable_beam_checks()
