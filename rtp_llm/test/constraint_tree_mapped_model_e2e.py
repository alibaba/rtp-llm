"""Opt-in test: replaces trees on a dedicated LOCAL Master and real model Worker.

Run with --check-current after restarting the Worker to verify Master repush.
Uses the existing test HTTP helper but the real Worker tokenizer mapping.
"""

import argparse
import json
import re
import time

from constraint_tree_model_e2e import MASTER, WORKER, http


def state(base, path):
    code, body, _ = http(base, path)
    assert code == 200, (code, body)
    return body


def ready(version):
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        master = state(MASTER, "/rtp_llm/constraint_tree/status")
        worker = state(WORKER, "/constraint_tree_status")
        if (master["state"] == "READY" and master["active_version"] == version
                and worker["status"] == "ready" and worker["version"] == version):
            assert master["published_worker_count"] == master["target_worker_count"] == 1
            assert master["mapping_fingerprint"] == worker["mapping_fingerprint"]
            assert master["content_sha256"] == worker["content_sha256"]
            return master
        time.sleep(0.2)
    raise AssertionError((master, worker))


def publish(version, sids):
    code, body, _ = http(MASTER, "/rtp_llm/constraint_tree/build", {
        "version": version, "model": "engine_service", "sids": sids,
    })
    assert code == 200 and body["state"] == "ACCEPTED", (code, body)
    return ready(version)


def check(sids, mapping):
    allowed = {tuple(mapping["tokens"][s] for s in re.findall(r"C[0-9]+", sid)) for sid in sids}
    eos = mapping["end_token_id"]
    for beams in (1, 3):
        code, body, _ = http(WORKER, "/", {
            "prompt": "请推荐商品：",
            "generate_config": {"num_beams": beams, "max_new_tokens": 4, "top_k": 1,
                                "return_output_ids": True, "is_streaming": False},
        })
        assert code == 200 and body["finished"], (code, body)
        assert len(body["output_ids"]) == beams
        for tokens in body["output_ids"]:
            assert eos in tokens, tokens
            end = tokens.index(eos)
            assert tuple(tokens[:end]) in allowed, tokens
            assert all(t == eos for t in tokens[end:]), tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-current", action="store_true")
    args = parser.parse_args()
    mapping = state(WORKER, "/constraint_tree_mapping")
    first = ["C0C1", "C2C3", "C4C5"]
    second = ["C6C7", "C8C9", "C10C11"]
    if args.check_current:
        version = state(MASTER, "/rtp_llm/constraint_tree/status")["active_version"]
        ready(version)
        check(second, mapping)
        print(json.dumps({"restart_repush": "PASS", "version": version}), flush=True)
        return
    version = max(int(time.time() * 1000), state(MASTER, "/rtp_llm/constraint_tree/status")["requested_version"] + 1)
    publish(version, first)
    check(first, mapping)
    code, artifact, _ = http(MASTER, "/rtp_llm/constraint_tree/artifact")
    assert code == 200 and isinstance(artifact, bytes)
    active = publish(version + 1, second)
    assert active["backup_version"] == version, active
    check(second, mapping)
    code, backup, _ = http(MASTER, "/rtp_llm/constraint_tree/artifact?slot=backup")
    assert code == 200 and backup == artifact
    code, body, _ = http(WORKER, "/update_constraint_tree", artifact, raw=True)
    assert code == 409 and body["status"] == "stale_version", (code, body)
    code, _, _ = http(WORKER, "/update_constraint_tree", b"invalid-csr", raw=True)
    assert code >= 400
    check(second, mapping)
    code, _, _ = http(MASTER, "/rtp_llm/constraint_tree/build", {
        "version": version + 2, "model": "engine_service", "sids": ["C999999999C0"],
    })
    assert code == 200
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = state(MASTER, "/rtp_llm/constraint_tree/status")
        if status["state"] == "FAILED":
            break
        time.sleep(0.1)
    assert status["state"] == "FAILED" and status["active_version"] == version + 1, status
    check(second, mapping)
    publish(version + 3, second)
    print(json.dumps({"mapped_hot_update_backup_stale_corrupt_failed_build": "PASS",
                      "restart_expected_version": version + 3}), flush=True)


if __name__ == "__main__":
    main()
