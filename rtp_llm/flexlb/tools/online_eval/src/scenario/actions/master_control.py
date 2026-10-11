"""Authenticated test interventions on an explicitly opted-in isolated Master."""

import json
import urllib.request


def affinity_request(ctx, deadline, payload=None):
    token = ctx.env.spec.master_env["FLEXLB_TEST_CONTROL_TOKEN"]
    req = urllib.request.Request(
        f"http://127.0.0.1:{ctx.env.master_http_port}/rtp_llm/test/cache_affinity",
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Flexlb-Test-Token": token},
    )
    with urllib.request.urlopen(req, timeout=min(3, deadline.remaining())) as response:
        return json.load(response)
