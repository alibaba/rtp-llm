"""Test-only audit of emitted CPU token prefixes. Never enable for serving traffic."""
import hashlib
import json
import os
import struct
from pathlib import Path

_TARGETS = {
    "//rtp_llm/test/smoke:next_moe_nvfp4_cudagraph_tp2_sm100",
    "//rtp_llm/test/smoke:next_moe_nvfp4_deepep_ll_cudagraph_dp2_sm100",
}
_FIXTURES = {
    "single": "685f6eea97f0975cdde542d9e05470db05007dfd1b9d93fc5a54ea8bd7f2ad38",
    "batch": "2aa28305acb8987c2d9e5bf41e5d079e21efb32585d46ddeab2b461ba86b8197",
}


def approve(kind, body):
    """Only a Bazel test can authorize an exact fixture; request fields cannot opt in."""
    root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "")
    target = os.environ.get("TEST_TARGET", "").removesuffix("_sample_diag")
    if (os.environ.get("RTP_NEXT_SMOKE_PREFIX") != "1"
            or target not in _TARGETS or not os.path.isabs(root)):
        return None
    # The server injects the request ID and transport headers into kwargs.
    # They are not hashed, retained or written. Reject every other extra field.
    payload = {k: v for k, v in body.items() if k not in ("__request_id__", "headers")}
    expected = "prompt" if kind == "single" else "prompt_batch"
    if set(payload) != {expected, "generate_config"}:
        return None
    prompt, config = payload[expected], payload["generate_config"]
    prompts = [prompt] if kind == "single" else prompt
    if (not isinstance(prompts, list) or len(prompts) != (1 if kind == "single" else 3)
            or any(not isinstance(p, str) or len(p) > 100 for p in prompts)
            or not isinstance(config, dict) or len(config) > 4
            or any(type(v) not in (int, float) for v in config.values())):
        return None
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True).encode("ascii")
    if hashlib.sha256(canonical).hexdigest() != _FIXTURES.get(kind):
        return None
    return Path(root) / ("next-prefix-v1-" + kind + ".json")


def authorize(path, request_id, slots):
    """Private worker rendezvous: only exact HTTP fixtures can enable native sampling."""
    if (path is None or os.environ.get("RTP_NEXT_SAMPLER_DIAGNOSTIC") != "1"
            or not os.environ.get("TEST_TARGET", "").endswith("_sample_diag")
            or type(request_id) != int or not 0 <= request_id < 2**63):
        return
    root = os.environ.get("TEST_TMPDIR", "")
    if not os.path.isabs(root):
        return
    for offset, slot in enumerate(slots):
        # This private temporary worker directory is not an artifact. Marker
        # names do not contain raw request IDs; their contents contain only a slot.
        tag = ((request_id + offset) * 11400714819323198485) % (2**64)
        try:
            with (Path(root) / ("next-smoke-auth-" + str(tag))).open("x") as f:
                f.write(str(slot))
        except OSError:
            return


def _row(t):
    # The frontend has already decoded this terminal CPU tensor. No device
    # transfer, model-side flags, logits, re-tokenization or hooks into sampling.
    if (t is None or str(t.device) != "cpu" or str(t.dtype) not in ("torch.int32", "torch.int64")
            or t.ndim != 2 or t.shape[0] != 1 or not 1 <= t.shape[1] <= 20):
        return None
    ids = t[0].tolist()
    if any(type(i) != int or not 0 <= i < 2**31 for i in ids):
        return None
    h = hashlib.sha256(b"rtp/next-emitted-prefix/v1\x00")
    rows = []
    for i in ids:
        h.update(struct.pack(">I", i))
        rows.append(h.hexdigest())
    return rows


def capture(path, responses):
    """Write terminal-prefix digests only, never fail or change the request."""
    if (not isinstance(path, Path)
            or os.environ.get("RTP_NEXT_SMOKE_PREFIX") != "1"
            or os.environ.get("TEST_TARGET", "").removesuffix("_sample_diag") not in _TARGETS):
        return
    try:
        rows = []
        for res in responses:
            seqs = res.generate_outputs.generate_outputs
            if len(seqs) != 1 or not seqs[0].finished:
                return
            row = _row(seqs[0].output_ids)
            if not row:
                return
            rows.append(row)
        if len(rows) != (1 if path.name.endswith("-single.json") else 3):
            return
        # Fail closed on retries/races: the analyzer rejects duplicate markers.
        with path.open("x") as stream:
            json.dump({"version": 1, "rows": rows}, stream)
    except FileExistsError:
        try:
            path.with_suffix(".duplicate").touch()
        except OSError:
            pass
    except (OSError, ValueError, TypeError, AttributeError, RuntimeError):
        # Missing/malformed artifacts are diagnosed separately from the golden.
        return
