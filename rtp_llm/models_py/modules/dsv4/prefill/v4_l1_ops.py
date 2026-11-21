"""Private input relay and digest-only operator witness for one fixed V4 CP forward."""

import hashlib
import json
import os
import re
import threading
from contextlib import ExitStack
from functools import partial
from pathlib import Path

import torch

from .v4_fixed_replay import IDS, TARGET


def check(ok, why):
    if not ok:
        raise ValueError("V4 paired layer fixture: " + why)


def fingerprint(x):
    """Logical bytes plus physical layout, bounded host/device temporaries."""
    x = x.detach()
    check(x.ndim >= 1 and x.shape[0] > 0, "empty operand")
    chunk = max(1, (4 << 20) // max(1, x[0].numel() * x.element_size()))
    h = hashlib.sha256()
    for t in x.split(chunk, dim=0):
        h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return {
        "shape": list(x.shape),
        "stride": list(x.stride()),
        "dtype": str(x.dtype),
        "sha": h.hexdigest(),
    }


def instrument(trace, layer, call):
    role = os.environ.get("RTP_V4_CAUSAL_ROLE")
    if role is None:
        return call
    nonce = os.environ.get("RTP_ARM_PAIR_NONCE", "")
    check(
        os.getenv("TEST_TARGET") == TARGET
        and role in ("oldfst", "newfst")
        and re.fullmatch("[0-9a-f]{32}", nonce),
        "scope/nonce",
    )
    check(layer.layer_id == 1 and layer.attn.compress_ratio == 0, "not L1 SWA")
    return partial(run_layer, trace, layer, call, role, nonce)


def relay(trace, h, role, nonce):
    from safetensors import safe_open
    from safetensors.torch import save_file

    rank = trace.rank
    check(
        rank in (0, 1)
        and h.dtype == torch.bfloat16
        and h.shape[0] in (27, 28)
        and tuple(h.shape[1:]) == (4, 4096),
        "input geometry",
    )
    rows = trace.rows.long()
    check(
        rows.numel() == len(trace.pos) and len(set(trace.pos)) == len(trace.pos),
        "duplicate rows",
    )
    positions = torch.full((len(h),), -1, dtype=torch.int64)
    positions[rows.cpu()] = torch.tensor(trace.pos, dtype=torch.int64)
    meta = {
        "role": "oldfst",
        "family": "v4",
        "nonce": nonce,
        "rank": str(rank),
        "layer": "1",
    }
    name = "v4_r%d.safetensors" % rank
    if role == "oldfst":
        private = trace.root / "arm_causal_private"
        private.mkdir(mode=0o700, exist_ok=True)
        path = private / name
        check(not path.exists(), "duplicate relay")
        # No hidden values belonging to fake/padded rows leave this process.
        cpu = torch.zeros(h.shape, dtype=h.dtype)
        cpu.index_copy_(0, rows.cpu(), h.index_select(0, rows).cpu())
        save_file({"hidden": cpu, "positions": positions}, str(path), metadata=meta)
    else:
        path = (
            Path(os.environ["TEST_SRCDIR"])
            / os.environ["TEST_WORKSPACE"]
            / "rtp_llm/test/smoke/arm_causal_inputs"
            / name
        )
        check(0 < path.stat().st_size < (2 << 20), "relay size")
        with safe_open(str(path), framework="pt", device="cpu") as f:
            check(
                set(f.keys()) == {"hidden", "positions"} and f.metadata() == meta,
                "relay identity",
            )
            cpu = f.get_tensor("hidden")
            check(
                torch.equal(f.get_tensor("positions"), positions),
                "different CP row layout",
            )
    check(
        cpu.shape == h.shape
        and cpu.dtype == h.dtype
        and not torch.count_nonzero(cpu[positions < 0]).item(),
        "relay geometry/padding",
    )
    y = cpu.to(h.device)
    return (
        y,
        fingerprint(cpu)["sha"],
        torch.equal(h.index_select(0, rows), y.index_select(0, rows)),
    )


def runtime_backend(lin):
    e = getattr(lin, "_executor", None)
    return e if e is not None else lin


def effective_weights(layer, gate, experts, independent_shared):
    a = layer.attn
    tensors = {
        "attn_norm": layer.attn_norm.weight,
        "ffn_norm": layer.ffn_norm.weight,
        "sink": a.attn_sink,
        "q_norm": a.q_norm,
        "kv_norm": a.kv_norm,
        "wo_a.w": a._wo_a_stk_w,
        "wo_a.sf": a._wo_a_stk_s,
        "gate.weight": gate._weight_bf16(),
        "gate.tid2eid": gate.tid2eid,
    }
    for tag in ("attn", "ffn"):
        unit = getattr(layer, tag + "_hc")
        for k in ("fn", "base", "scale"):
            tensors[tag + "_hc." + k] = getattr(unit, k)
    for tag in ("wq_a", "wq_b", "wkv", "wo_b"):
        backend = runtime_backend(getattr(a, tag))
        tensors[tag + ".w"] = backend.weight
        tensors[tag + ".sf"] = backend.weight_scales
    for k in ("_mega_l1_w", "_mega_l1_sf", "_mega_l2_w", "_mega_l2_sf"):
        tensors[k] = getattr(experts, k)
    if independent_shared is not None:
        fp = independent_shared._fast_path
        for layer_id, pair in enumerate((fp._w13_parts, fp._w2_parts), 1):
            check(pair is not None, "shared GEMM weights not prepared")
            tensors["_se_l%d_w" % layer_id], tensors["_se_l%d_sf" % layer_id] = pair
    else:
        for k in ("_se_l1_w", "_se_l1_sf", "_se_l2_w", "_se_l2_sf"):
            tensors[k] = getattr(experts, k)
    return {k: fingerprint(v) for k, v in tensors.items()}


def run_layer(trace, layer, call, role, nonce, h, ids, positions, cu, **kwargs):
    check(
        tuple(ids.reshape(-1)[trace.rows].tolist()) == tuple(IDS[i] for i in trace.pos),
        "request mismatch",
    )
    vin, sig, natural_equal = relay(trace, h, role, nonce)
    natural_sig = fingerprint(h)["sha"]
    consumed_sig = fingerprint(vin)["sha"]
    stages = {}
    repeated = {}
    in_repeat = False
    replay = {}
    a = layer.attn
    gate = layer.ffn.gate
    experts = layer.ffn.fused_moe.fused_experts
    check(layer.ffn.strategy_name == "mega_moe_se" and gate.hash, "MoE path")
    chunked = getattr(layer.ffn, "_runtime", None) or layer.ffn
    shared = chunked._moe._shared_executor
    weights = effective_weights(layer, gate, experts, shared)
    thread = threading.get_ident()
    from .v4_norm40 import NormProbe

    norm_probe = NormProbe(trace)

    def row(tag, t, global_rows=False):
        record = repeated if in_repeat else stages
        check(tag not in record, "duplicate " + tag)
        x = t.detach()
        if x.ndim >= 2 and x.shape[0] == 1:
            x = x.squeeze(0)
        expected = 55 if global_rows else len(h)
        check(x.shape[0] == expected, tag + " row layout")
        selected = x if global_rows else x.index_select(0, trace.rows)
        pos = list(range(55)) if global_rows else trace.pos
        b = selected.contiguous().view(torch.uint8).cpu().reshape(len(pos), -1).numpy()
        record[tag] = {
            "pos": pos,
            "shape": list(selected.shape[1:]),
            "dtype": str(x.dtype),
            "rows": [hashlib.sha256(v.tobytes()).hexdigest() for v in b],
        }

    def hc(fn, tag, pre):
        def observed(*args, **kw):
            if tag == "ffn.hc_pre" and threading.get_ident() == thread:
                v = norm_probe.hc_call(fn, args, kw)
            else:
                v = fn(*args, **kw)
            if threading.get_ident() == thread:
                for k, t in zip((".mix", ".post", ".comb"), v) if pre else (("", v),):
                    row(tag + k, t)
            return v

        return observed

    with ExitStack() as stack:

        def replace(obj, key, val):
            own = vars(obj)
            existed = key in own
            prev = own.get(key)
            object.__setattr__(obj, key, val)
            (
                stack.callback(object.__setattr__, obj, key, prev)
                if existed
                else stack.callback(object.__delattr__, obj, key)
            )

        def around(obj, key, before=None, after=None):
            fn = getattr(obj, key)

            def wrapped(*args, **kw):
                if threading.get_ident() != thread:
                    return fn(*args, **kw)
                if before:
                    before(args, kw)
                result = fn(*args, **kw)
                if after:
                    after(result, args, kw)
                return result

            replace(obj, key, wrapped)

        norm_probe.hooks(layer, replace)
        fast = layer._prefill_fast_hc_impls()
        replace(
            layer,
            "_prefill_fast_hc_impls_cached",
            tuple(
                hc(fn, k, pre)
                for fn, k, pre in zip(
                    fast,
                    ("attn.hc_pre", "ffn.hc_pre", "attn.hc_post", "ffn.hc_post"),
                    (True, True, False, False),
                )
            ),
        )
        for k, unit in (("attn", layer.attn_hc), ("ffn", layer.ffn_hc)):
            for method, pre in (("pre", True), ("post", False)):
                replace(
                    unit, method, hc(getattr(unit, method), k + ".hc_" + method, pre)
                )

        def ab(args, kw):
            x = args[0].reshape(len(h), 4096)
            row("attn.norm", x)
            check(not args[1].any_cont, "not fresh SWA")
            row("attn.rope", args[1].freqs_cis)
            q, k = (runtime_backend(getattr(a, n)) for n in ("wq_a", "wkv"))
            native = [getattr(l, "_deepgemm_linear", l) for l in (q, k)]
            pair = native[0].quantize_input(x)
            for name, dispatcher, backend in zip(("wq_a", "wkv"), (q, k), native):
                ind = backend.quantize_input(x)
                own = dispatcher(x)
                shared = backend.forward_quantized(*pair)
                replay[name] = {
                    "activation_equal": torch.equal(
                        pair[0].view(torch.uint8), ind[0].view(torch.uint8)
                    ),
                    "scales_equal": torch.equal(pair[1], ind[1]),
                    "independent": fingerprint(own)["sha"],
                    "shared": fingerprint(shared)["sha"],
                }

        around(
            a,
            "_prefill_compute_qkv",
            before=ab,
            after=lambda v, args, kw: (
                row("attn.qr_norm", v.qr),
                row("attn.kv_rope_gather", v.kv_full, True),
            ),
        )

        def proj(v, args, kw):
            for k in ("wq_a", "wkv", "wq_b"):
                if args[0] is getattr(a, k):
                    row("attn." + k, v)
                    if k in replay:
                        replay[k]["live"] = fingerprint(v.reshape(len(h), -1))["sha"]

        for name in ("_lin", "_lin_from_shared_quant"):
            around(a, name, after=proj)
        around(
            a,
            "_prefill_write_swa_fp8_paged",
            before=lambda args, kw: row("attn.swa_write", args[1], True),
        )

        def fresh(args, kw):
            row("attn.swa_read", args[0].kv_full, True)

        around(a, "_attn_fp8_swa_via_kv_full", before=fresh)

        def mla(args, kw):
            for s, k in (
                ("q", "q"),
                ("kv", "kv"),
                ("indices", "indices"),
                ("length", "topk_length"),
            ):
                row("attn.mla." + s, kw[k], k == "kv")

        around(a, "_flash_mla_sparse_fwd_chunked_projected", before=mla)
        around(
            a,
            "_prefill_output_proj_into",
            before=lambda args, kw: row("attn.mla.out", args[0]),
            after=lambda v, args, kw: row("attn.wo_b", kw["out"]),
        )
        around(
            a,
            "_wo_a_einsum_from_fp8",
            before=lambda args, kw: (
                row("attn.wo_a.fp8", args[0]),
                row("attn.wo_a.scale", args[1]),
            ),
            after=lambda v, args, kw: row("attn.wo_a", v),
        )

        def gate_input(args, kw):
            if not in_repeat:
                norm_probe.save("live.gate_input", args[0])
            row("ffn.norm", args[0])

        around(
            gate,
            "_bf16_scores",
            before=gate_input,
            after=lambda v, args, kw: row("ffn.router", v),
        )

        def packed(args, kw):
            n = args[1]
            check(n == len(h), "unexpected MoE chunk")
            b = experts._mega_buf
            for tag in ("x", "x_sf", "topk_idx", "topk_weights"):
                row("ffn.pack." + tag, getattr(b, tag)[:n])

        around(
            experts,
            "_launch",
            before=packed,
            after=lambda v, args, kw: row("ffn.mega_moe", args[0]),
        )
        if shared is not None:
            fp = shared._fast_path

            def shared_view(v, args, kw):
                x = args[1]
                ws = fp._workspace_views(fp._ensure_workspace(x), len(x))
                for tag, field in (
                    ("x", "x_fp8"),
                    ("x_sf", "x_scale"),
                    ("gate_up", "gate_up_bf16"),
                    ("hidden", "hidden_fp8"),
                    ("hidden_sf", "hidden_scale"),
                    ("out", "out_bf16"),
                ):
                    row("ffn.shared." + tag, getattr(ws, field))

            around(fp, "_run_prepared", after=shared_view)
        # One extra matched collective on both CP/EP ranks, only at L1 in
        # the single gated diagnostic request. Re-enter from the MoE module
        # (router+pack+shared+routed+combine), not mid-collective with stale
        # rendezvous/dispatch scratch. Keep the first output independent of
        # the persistent workspace reused by the second call.
        first_ffn = layer.ffn.forward

        def repeated_ffn(x, input_ids, *args, **kw):
            nonlocal in_repeat
            if threading.get_ident() != thread:
                return first_ffn(x, input_ids, *args, **kw)
            norm_probe.save("live.moe_input", x)
            xs, ids_copy = x.clone(), input_ids.clone()
            first = first_ffn(x, input_ids, *args, **kw).clone()
            row("ffn.combined", first)
            in_repeat = True
            try:
                second = first_ffn(xs, ids_copy, *args, **kw)
                row("ffn.combined", second)
            finally:
                in_repeat = False
            return first

        replace(layer.ffn, "forward", repeated_ffn)
        row("input", h)
        natural_rows = stages.pop("input")
        row("input", vin)
        out = call(vin, ids, positions, cu, **kwargs)
        row("output", out)
    public = {
        "norm40": norm_probe.finish(layer),
        "family": "v4",
        "role": role,
        "nonce": nonce,
        "rank": trace.rank,
        "relay_sha": sig,
        "natural_rows_equal_relay": natural_equal,
        "natural_valid_rows": natural_rows,
        "natural_full_input_sha": natural_sig,
        "consumed_full_input_sha": consumed_sig,
        "physical_input_rows": len(h),
        "valid_input_rows": len(trace.pos),
        "weights": weights,
        "ops": stages,
        "ab": replay,
        "repeat_moe": repeated,
        "moe_executor": "routed_plus_shared" if shared is not None else "fused_se",
    }
    path = trace.root / ("v4_op_r%d.json" % trace.rank)
    with path.open("x") as f:
        json.dump(public, f, separators=(",", ":"))
    return out
