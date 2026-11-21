"""Single dedicated ARM proposal fixture. No collectives and no pooled bytes in public reports."""

import hashlib
import json
import os
import re
from pathlib import Path

import torch

TARGET = "//rtp_llm/test/smoke:deepseek_v41_first_draft_sm100_arm"
PKEY = {"positions", "target_features"}
DKEY = {
    "query_ids",
    "query_positions",
    "committed_positions",
    "layer0",
    "layer1",
    "layer2",
}


def need(x, why):
    if not x:
        raise ValueError("V4.1 first-draft: " + why)


def scope():
    if os.getenv("TEST_TARGET") != TARGET:
        return None
    role, nonce = os.getenv("RTP_V41_DRAFT_ROLE"), os.getenv("RTP_ARM_PAIR_NONCE", "")
    root = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
    need(
        role == os.getenv("RTP_V41_EXPECTED_LOADER")
        and role in ("oldfst", "newfst")
        and re.fullmatch("[0-9a-f]{32}", nonce),
        "identity",
    )
    marker = root / "v41_cache_trace_stage"
    if not marker.is_file() or marker.read_text() != nonce + " hit\n":
        return None
    return root, role, nonce


def sha(x):
    x = x.detach()
    if x.ndim == 0:
        x = x.reshape(1)
    h = hashlib.sha256()
    if len(x):
        n = max(1, (4 << 20) // max(1, x[0].numel() * x.element_size()))
        for c in x.split(n):
            h.update(c.contiguous().view(torch.uint8).cpu().numpy())
    return h.hexdigest()


def meta(rank, stage, nonce, role):
    return dict(rank=str(rank), stage=stage, nonce=nonce, role=role, family="v41draft")


def validate(data, stage):
    need(set(data) == (PKEY if stage == "p_commit" else DKEY), "wrong fields")
    if stage == "p_commit":
        p = data["positions"]
        h = data["target_features"]
        need(
            p.dtype == torch.int64
            and p.ndim == 1
            and 0 < len(p) <= 271
            and len(p.unique()) == len(p)
            and torch.all((p >= 512) & (p < 783)).item(),
            "CP positions",
        )
        need(
            h.dtype == torch.bfloat16
            and h.ndim == 2
            and len(h) == len(p)
            and 5120 <= h.shape[1] <= 5120 * 16
            and h.shape[1] % 5120 == 0,
            "feature shape",
        )
    else:
        need(
            data["query_ids"].dtype == torch.int32 and data["query_ids"].shape == (3,),
            "query IDs",
        )
        for k, start, end in (
            ("committed_positions", 655, 783),
            ("query_positions", 783, 786),
        ):
            need(
                data[k].dtype == torch.int64
                and torch.equal(data[k], torch.arange(start, end)),
                k,
            )
        for i in range(3):
            x = data["layer" + str(i)]
            need(x.dtype == torch.uint8 and x.shape == (128, 528), "committed cache")


def transfer(ctx, stem, rank, stage, tensors):
    from safetensors import safe_open
    from safetensors.torch import save_file

    root, role, nonce = ctx
    schema = meta(rank, stage, nonce, "oldfst")
    natural = {k: x.detach().to("cpu").contiguous() for k, x in tensors.items()}
    validate(natural, stage)
    if role == "oldfst":
        folder = root / "arm_causal_private"
        folder.mkdir(mode=0o700, exist_ok=True)
        path = folder / (stem + ".safetensors")
        need(not path.exists(), "duplicate producer")
        save_file(natural, str(path), metadata=schema)
        return natural
    path = (
        Path(os.environ["TEST_SRCDIR"])
        / os.environ["TEST_WORKSPACE"]
        / "rtp_llm/test/smoke/arm_causal_inputs"
        / (stem + ".safetensors")
    )
    need(
        0 < path.stat().st_size < (32 << 20 if stage == "p_commit" else 256 << 10),
        "relay size",
    )
    with safe_open(str(path), framework="pt", device="cpu") as f:
        need(
            f.metadata() == schema and set(f.keys()) == set(natural), "relay provenance"
        )
        data = {k: f.get_tensor(k) for k in f.keys()}
    validate(data, stage)
    need(
        all(
            v.shape == natural[k].shape and v.dtype == natural[k].dtype
            for k, v in data.items()
        ),
        "peer tensor shape",
    )
    for k in (
        ("positions",)
        if stage == "p_commit"
        else ("query_positions", "committed_positions")
    ):
        need(torch.equal(natural[k], data[k]), "peer position layout")
    return data


def publish(ctx, stem, rank, stage, **fields):
    root, role, nonce = ctx
    with (root / (stem + ".json")).open("x") as f:
        json.dump(
            dict(meta(rank, stage, nonce, role), **fields), f, separators=(",", ":")
        )


def fixed_features(model, features, positions, req, ends, cp):
    ctx = scope()
    if ctx is None:
        return features
    if int(cp.prefix_length) != 512 or ends.flatten().tolist() != [783]:
        return features
    rank = int(cp.cp_rank)
    need(
        int(cp.cp_size) == 2
        and rank in (0, 1)
        and features.shape[1] == model._dspark_aux_feature_dim,
        "wrong commit",
    )
    m = req.reshape(-1) >= 0
    pos = positions[m].long()
    natural = features[m]
    stem = f"v41draft_p{rank}"
    donor = transfer(
        ctx, stem, rank, "p_commit", dict(positions=pos, target_features=natural)
    )
    used = torch.zeros_like(features)
    used[m] = donor["target_features"].to(used.device)
    publish(
        ctx,
        stem,
        rank,
        "p_commit",
        positions=pos.tolist(),
        natural=sha(natural),
        used=sha(used[m]),
    )
    return used


def remember(model, lid, x, pos, prefix, indices, length, pool):
    # Snapshot the attention input on the stream at its actual consumer: the
    # shared late-layer workspace may be reused before post-replay inspection.
    # No host transfer or synchronization is issued during graph capture.
    if (
        os.getenv("TEST_TARGET") != TARGET
        or not torch.cuda.is_current_stream_capturing()
        or prefix.numel() != 1
    ):
        return
    if not hasattr(model, "_v41draft_captures"):
        model._v41draft_captures = {}
    model._v41draft_captures[int(lid)] = (
        x.detach().clone(),
        pos,
        prefix,
        indices,
        length,
        pool,
    )


def remember_queries(model, ids):
    if (
        os.getenv("TEST_TARGET") == TARGET
        and ids.shape == (1, 3)
        and torch.cuda.is_current_stream_capturing()
    ):
        model._v41draft_queries = ids


def selected(pool, slots):
    blocks = torch.div(slots, int(pool.shape[1]), rounding_mode="floor")
    return blocks, slots % int(pool.shape[1])


def begin(model, inputs):
    ctx = scope()
    if ctx is None or (ctx[0] / "v41draft_d0.json").exists():
        return None
    from rtp_llm.models_py.modules.dsv41.fp8._kv_cache_utils import (
        require_pool_tokens_per_block,
    )
    from rtp_llm.models_py.modules.dsv41.kv_cache_utils import (
        as_tagged_attention_inputs,
        cached_swa_region,
    )

    ai = as_tagged_attention_inputs(inputs.attention_inputs)
    prefix = ai.prefix_lengths_device
    if prefix.flatten().tolist() != [783]:
        return None
    raw = inputs.input_ids.reshape(-1)
    need(
        len(raw) == 3
        and model._gen_num_per_cycle == 3
        and int(model._v4_args.window_size) == 128
        and len(model.v4.layers) == 3,
        "graph shape",
    )
    pos = torch.arange(655, 786, device=raw.device, dtype=torch.long)
    req = torch.zeros_like(pos)
    tab = model._swa_block_table(ai, 1)
    natural_ids = torch.full(
        (3,), int(model._dspark_noise_token_id), device=raw.device, dtype=torch.int32
    )
    natural_ids[0] = raw[0]
    local = {
        "query_ids": natural_ids,
        "query_positions": pos[-3:],
        "committed_positions": pos[:128],
    }
    refs = {}
    for i in range(3):
        with model._swa_cache_bound(i, tab) as (_, entries, pool):
            tpb = int(
                require_pool_tokens_per_block(
                    model.kv_cache, region=cached_swa_region(model, model.kv_cache, i)
                )
            )
            slots = model._global_pool_slots(tab, req, pos, entries, tpb)
            need(
                pos.numel() == slots.numel() == 131
                and len(torch.unique(slots)) == 131
                and int(slots.min()) >= 0
                and int(slots.max()) < pool.shape[0] * entries
                and pool.ndim == 3
                and pool.dtype == torch.uint8
                and pool.shape[1:] == (entries, 528),
                "incomplete/aliased slots",
            )
            s = selected(pool, slots[:128])
            before = pool[s]
            local[f"layer{i}"] = before
            refs[i] = (pool, slots, s, before)
    need(
        len({item[0].data_ptr() for item in refs.values()}) == 3, "aliased draft layers"
    )
    donor = transfer(ctx, "v41draft_d0", 0, "d_propose", local)
    need(
        int(donor["query_ids"][1])
        == int(donor["query_ids"][2])
        == int(model._dspark_noise_token_id),
        "noise id",
    )
    state = dict(
        ctx=ctx,
        model=model,
        refs=refs,
        raw=raw,
        backup=raw[0].clone(),
        data=donor,
        local=local,
    )
    try:
        raw[0].copy_(donor["query_ids"][0].to(raw.device))
        for i, (pool, _, s, _) in refs.items():
            pool[s] = donor[f"layer{i}"].to(pool.device)
    except BaseException:
        restore(state)
        raise
    return state


def restore(state):
    if state is None:
        return
    state["raw"][0].copy_(state["backup"])
    for pool, _, s, old in state["refs"].values():
        pool[s] = old


def finish(state, head):
    if state is None:
        return
    ctx, model, donor = state["ctx"], state["model"], state["data"]
    live = model._v41draft_captures
    need(
        torch.equal(model._v41draft_queries.reshape(3).cpu(), donor["query_ids"]),
        "graph did not consume fixed query IDs",
    )
    need(set(live) == {0, 1, 2}, "missing graph operand binding")
    layers = {}
    for i, (pool, slots, s, _) in state["refs"].items():
        x, pos, prefix, idx, n, actual = live[i]
        need(
            actual.data_ptr() == pool.data_ptr()
            and prefix.flatten().tolist() == [783]
            and pos.flatten().tolist() == [783, 784, 785],
            "wrong active graph instance",
        )
        ix = idx.reshape(3, -1)
        need(
            torch.all(n == 131).item()
            and n.numel() in (1, 3)
            and torch.equal(ix[:, :131].long(), slots.expand(3, -1)),
            "wrong causal read indices",
        )
        value = sha(actual[s])
        need(value == sha(donor[f"layer{i}"]), "cache changed before read")
        layers[str(i)] = {
            "natural": sha(state["local"][f"layer{i}"]),
            "used": value,
            "query_kv": sha(actual[selected(actual, slots[128:])]),
            "attn_input": sha(x),
        }
    need(head.shape[0] >= 3 and head.shape[-1] == 5120, "head geometry")
    publish(
        ctx,
        "v41draft_d0",
        0,
        "d_propose",
        query_ids=donor["query_ids"].tolist(),
        natural_ids=state["local"]["query_ids"].tolist(),
        layers=layers,
        graph_head=sha(head[:3]),
    )
    # One-shot same-process handoff to the production Markov/softmax sampler.
    (ctx[0] / ("v41draft_pending_" + str(os.getpid()))).write_text(
        str(int(donor["query_ids"][0])) + "\n"
    )


def head_operand(hidden, kernel, scales, tp_rank, tp_size):
    ctx = scope()
    if ctx is None:
        return
    root = ctx[0]
    if not (root / ("v41draft_pending_" + str(os.getpid()))).is_file():
        return
    d = json.loads((root / "v41draft_d0.json").read_text())
    need(
        int(tp_rank) == 0 and int(tp_size) == 1 and sha(hidden) == d["graph_head"],
        "unmatched native draft LM head",
    )
    d["native_head"] = {
        "kernel": sha(kernel),
        "dtype": str(kernel.dtype),
        "shape": list(kernel.shape),
        "scales": None if scales is None else sha(scales),
    }
    # Only written on the dedicated single D stream before scores are emitted.
    (root / "v41draft_d0.json").write_text(json.dumps(d, separators=(",", ":")))


def draft_weights(model):
    # Report semantic runtime slots, not W.* or checkpoint-name aliases. A
    # backend outside this inventory remains explicitly unqualified.
    d, missing, paths = {}, {}, {}

    def slot(k, t, required=True):
        if isinstance(t, torch.Tensor):
            d[k] = dict(
                sha=sha(t),
                dtype=str(t.dtype),
                shape=list(t.shape),
                stride=list(t.stride()),
            )
        elif required:
            missing[k] = "not bound"

    def fields(k, obj, names):
        for n in names:
            slot(k + "." + n, getattr(obj, n, None))

    def linear(k, op):
        op = getattr(op, "v41_runtime_linear", op)
        fields(k, op, ("weight", "weight_scales"))
        slot(k + ".bias", getattr(op, "bias", None), False)

    v = model.v4
    fields("embedding", v.embed, ("weight",))
    fields("head_norm", v.norm, ("weight",))
    if getattr(v.head_hc, "_last_ffn_ref", None) is not None:
        need(v.head_hc._last_ffn_ref() is v.layers[-1].ffn_hc, "head delayed-HC source")
    else:
        fields("head_hc", v.head_hc, ("fn", "base", "scale"))
    for i, b in enumerate(v.layers):
        p = f"layer{i}."
        for n in ("attn_norm", "ffn_norm"):
            fields(p + n, getattr(b, n, None), ("weight",))
        for n in ("attn_hc", "ffn_hc"):
            unit = getattr(b, n, None)
            for key in (
                "fn",
                "base",
                "scale",
                "pre_fn",
                "pre_base",
                "pre_scale",
                "post_fn",
                "post_base",
                "post_scale",
            ):
                slot(p + n + "." + key, getattr(unit, key, None), False)
            if not any(k.startswith(p + n + ".") for k in d):
                missing[p + n] = "unknown HC representation"
        a = b.attn
        for n in ("wq_a", "wq_b", "wkv", "wo_b"):
            linear(p + n, getattr(a, n, None))
        combo = getattr(a, "wq_a_wkv", None)
        if combo is not None:
            linear(p + "wq_a_wkv", combo)
        fields(
            p + "attn",
            a,
            ("q_norm", "kv_norm", "attn_sink", "_wo_a_stk_w", "_wo_a_stk_s"),
        )
        for j, g in enumerate(getattr(a, "_wo_a_groups_v41", ()) or ()):
            linear(p + f"wo_group{j}", g)
        gate = b.ffn.gate
        slot(p + "gate.bf16", gate._weight_bf16())
        for n in ("bias", "tid2eid", "bias_vl"):
            slot(p + "gate." + n, getattr(gate, n, None), False)
        experts = b.ffn.fused_moe.fused_experts
        paths[p + "experts"] = type(experts).__name__
        found = 0
        for n in (
            "_mega_l1_w",
            "_mega_l1_sf",
            "_mega_l2_w",
            "_mega_l2_sf",
            "_se_l1_w",
            "_se_l1_sf",
            "_se_l2_w",
            "_se_l2_sf",
            "_w1",
            "_w1_scale",
            "_w2",
            "_w2_scale",
        ):
            t = getattr(experts, n, None)
            if isinstance(t, torch.Tensor):
                found += 1
                slot(p + "experts." + n, t)
        if found < 4:
            missing[p + "experts"] = "unknown expert pack"
        shared = b.ffn.shared_experts
        if shared is not None:
            linear(p + "shared.w13", getattr(shared, "w13", None))
            linear(p + "shared.w2", getattr(shared, "w2", None))
    return dict(slots=d, missing=missing, paths=paths)


def snapshot_ep(model):
    if (
        os.getenv("TEST_TARGET") != TARGET
        or getattr(model, "_v41draft_manifest_ready", False)
        or getattr(model, "_commit_only_prefill", False)
        or int(model.tp_size) != 1
    ):
        return
    if (
        torch.cuda.is_current_stream_capturing()
        or getattr(model.v4, "embed", None) is None
        or getattr(model, "kv_cache", None) is None
    ):
        return
    role = os.environ["RTP_V41_DRAFT_ROLE"]
    nonce = os.environ["RTP_ARM_PAIR_NONCE"]
    need(
        role == os.environ["RTP_V41_EXPECTED_LOADER"]
        and re.fullmatch("[0-9a-f]{32}", nonce),
        "weight witness provenance",
    )
    e = model.v4.layers[0].ffn.fused_moe.fused_experts
    cfg = getattr(e, "cfg", None)
    if cfg is not None and hasattr(cfg, "ep_rank"):
        ep = int(cfg.ep_rank)
        need(cfg.ep_size == 2, "EP runtime size")
    else:
        ep = int(model._v4_args.ep_rank)
        need(int(model._v4_args.ep_size) == 2, "EP group size")
    need(ep in (0, 1), "EP rank")
    ctx = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]), role, nonce
    doc = draft_weights(model)
    path = ctx[0] / f"v41draft_ep{ep}.json"
    if path.exists():
        other = json.loads(path.read_text())
        need(
            other == dict(meta(ep, "d_weights", nonce, role), **doc),
            "conflicting draft model weights",
        )
    else:
        publish(ctx, f"v41draft_ep{ep}", ep, "d_weights", **doc)
    model._v41draft_manifest_ready = True
