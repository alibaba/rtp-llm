"""JSON access to live Master processes owned by the runtime environment."""

import json
import urllib.request


def require_process(ctx, target):
    if target == "single":
        if getattr(ctx.env, "masters", {}):
            raise ValueError("single target cannot address a dual-master environment")
        proc = ctx.env.master
    else:
        if set(getattr(ctx.env, "master_specs", {})) != {"A", "B"}:
            raise ValueError(
                "A/B target requires an actual dual-standalone environment"
            )
        proc = ctx.env.masters.get(target)
    if proc is None or not proc.alive():
        raise RuntimeError(f"owned master {target} is not running")
    return proc


def master_json(ctx, target, path, deadline, post=False):
    require_process(ctx, target)
    port = (
        ctx.env.master_http_port
        if target == "single"
        else ctx.env.master_specs[target].http_port
    )
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=b"{}" if post else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(
        request, timeout=min(5, deadline.remaining())
    ) as response:
        data = json.load(response)
    if not isinstance(data, dict):
        raise ValueError("master response is not a JSON object")
    return data
