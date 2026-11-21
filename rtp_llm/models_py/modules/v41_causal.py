"""Paired ARM test only: V4.1 layer 1, known 783-token request with exactly 512 reused."""

import hashlib
import json
import os
import re
import struct
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

TARGET = "//rtp_llm/test/smoke:deepseek_v41_cache_rounds_sm100_arm"
KEYS = {"positions", "incoming", "previous_mix", "swa_tail"}


def _native_pair(op):
    # The owner may have raw checkpoint scales as well as its executor's
    # different, packed scales. Never substitute a raw field for a missing
    # native operand (the comparator reports missing tensors as incomplete).
    op = getattr(op, "v41_runtime_linear", op)
    return getattr(op, "weight", None), getattr(op, "weight_scales", None)


def _lookup_rows(table, ids, cuda_lookup):
    pair = table._uva if cuda_lookup else (table.weight, table.scales)
    if pair is None or len(pair) != 2:
        raise ValueError("runtime Engram lookup has no bound tables")
    for field, tensor in zip(("weight", "scales"), pair):
        if tensor.element_size() != 1 or tensor.ndim != 2:
            raise ValueError("unexpected runtime Engram table layout")
        yield field, tensor.view(torch.uint8).index_select(
            0, ids.to(device=tensor.device)
        )


def _shared_pairs(module, fast=None):
    if fast is None:
        return (_native_pair(module.w13), _native_pair(module.w2))
    # _run_prepared has completed, so these are the GEMM operands it
    # actually used, even when a stale wrapper refers to different buffers.
    if fast._prepared_shared_experts is not module:
        raise ValueError("unbound V4.1 shared expert workspace")
    return fast._w13_parts, fast._w2_parts


def _digest(x):
    x = x.detach()
    sha = hashlib.sha256()
    # A routed expert pack can be multi-GiB. Never materialize a second pack.
    stride = max(1, x[0].numel() * x.element_size()) if x.ndim and len(x) else 1
    for part in (
        x.reshape(1).split(1) if x.ndim == 0 else x.split(max(1, (8 << 20) // stride))
    ):
        sha.update(part.contiguous().view(torch.uint8).cpu().numpy())
    return {
        "dtype": str(x.dtype),
        "shape": list(x.shape),
        "sha256": sha.hexdigest(),
    }


def _gate_code_segments(cubin):
    """Fingerprint device instructions/constants, not DWARF or Bazel paths."""
    if (
        not isinstance(cubin, bytes)
        or len(cubin) < 64
        or cubin[:6] != b"\x7fELF\x02\x01"
    ):
        return {}
    offset = struct.unpack_from("<Q", cubin, 40)[0]
    stride, count, names = struct.unpack_from("<HHH", cubin, 58)
    if stride < 64 or names >= count or offset + stride * count > len(cubin):
        return {}

    def section(i):
        return struct.unpack_from("<IIQQQQ", cubin, offset + stride * i)

    _, _, _, _, start, size = section(names)
    strings = cubin[start : start + size]
    result = {}
    for i in range(count):
        name_offset, kind, _, _, start, size = section(i)
        end = strings.find(b"\0", name_offset)
        if end < name_offset or not size or kind == 8 or start + size > len(cubin):
            continue
        name = strings[name_offset:end]
        if name.startswith((b".text.", b".nv.constant", b".nv.merc.text")):
            result[name.decode("ascii")] = hashlib.sha256(
                cubin[start : start + size]
            ).hexdigest()
    return result


def _gate_witness(fn, a, kw, valid, record):
    """At the actual gate call: observe its launch and shadow the same pre-write values."""
    import triton

    from rtp_llm.models_py.modules.dsv41 import _engram_inject_triton as native
    from rtp_llm.models_py.modules.dsv41.engram import _engram_inject_supported

    h, kv, q, k, eps, mask = a
    if h.shape[0] < 2 or valid.dtype != torch.bool or valid.shape != h.shape[:1]:
        record["engram_gate"] = {"incomplete": "unexpected fixture row selection"}
        return fn(*a, **kw)

    def digest(x):
        return _digest(x[valid])["sha256"]

    def outcome(value):
        live = (
            value[valid]
            .detach()
            .contiguous()
            .view(torch.uint8)
            .flatten(1)
            .cpu()
            .numpy()
        )
        info["outcomes"][stage] = hashlib.sha256(live).hexdigest()
        info["outcome_rows"][stage] = [hashlib.sha256(row).hexdigest() for row in live]

    def abi(x):
        return dict(
            dtype=str(x.dtype),
            shape=list(x.shape),
            stride=list(x.stride()),
            offset=x.storage_offset(),
            alignment=x.data_ptr() % 256,
        )

    info = dict(
        eps=float(eps).hex(),
        torch=str(torch.__version__),
        triton=str(triton.__version__),
        capability=list(torch.cuda.get_device_capability(h.device)),
        inference=torch.is_inference_mode_enabled(),
        grad=torch.is_grad_enabled(),
        tf32=torch.backends.cuda.matmul.allow_tf32,
        abi={},
        operands={},
        launches={},
        outcomes={},
        outcome_rows={},
        out_is_hidden=kw.get("out") is h,
    )
    out_arg = kw.get("out")
    info["out_abi"] = None if out_arg is None else abi(out_arg)
    info["entry_strided_operands"] = []
    info["pre_fix_path"] = (
        "triton"
        if (
            _engram_inject_supported(h, q, k, mask)
            and kv.dtype == torch.bfloat16
            and kv.device == h.device
            and kv.shape == h.shape[:-2] + (5 * 5120,)
            and kv.is_contiguous()
        )
        else "torch"
    )
    for key, value in zip(("h", "kv", "q", "k", "mask"), (h, kv, q, k, mask)):
        if value is None:
            info["abi"][key] = None
            info["operands"][key] = None
            continue
        if not value.is_contiguous():
            info["entry_strided_operands"].append(key)
        info["abi"][key] = abi(value)
        info["operands"][key] = (
            digest(value) if key in ("h", "kv", "mask") else _digest(value)["sha256"]
        )
    real_kernel = native.engram_inject_kernel
    stage = "actual"

    class Witness:
        def __getitem__(self, grid):
            launch = real_kernel[grid]

            def run(*values, **params):
                from triton.runtime import driver

                launch_device = driver.active.get_current_device()
                launch_on_tensor_stream = (
                    launch_device == h.device.index
                    and driver.active.get_current_stream(launch_device)
                    == torch.cuda.current_stream(h.device).cuda_stream
                )
                compiled = launch(*values, **params)
                cubin = getattr(compiled, "asm", {}).get("cubin")
                info["launches"][stage] = dict(
                    grid=list(grid),
                    params=params,
                    device_matches_tensor=launch_device == h.device.index,
                    tensor_stream=launch_on_tensor_stream,
                    code_segments=_gate_code_segments(cubin),
                    binary_sha256=(
                        hashlib.sha256(cubin).hexdigest()
                        if isinstance(cubin, bytes)
                        else None
                    ),
                    aliases_hidden=values[0].data_ptr() == values[-1].data_ptr(),
                    operands=[abi(v) for v in values],
                )
                return compiled

            return run

    # The production call can overwrite h. Snapshot first; do not hand the
    # snapshot to production, as that could change dispatch and aliasing.
    saved = h.clone(memory_format=torch.contiguous_format)
    with patch.object(native, "engram_inject_kernel", Witness()):
        out = fn(*a, **kw)
        outcome(out)
        stage = "dense_native"
        operands = (saved, kv.contiguous(), q.contiguous(), k.contiguous(), eps)
        outcome(fn(*operands, None if mask is None else mask.contiguous()))
        stage = "torch_ref"
        live_mask = torch.ones_like(valid) if mask is None else mask
        strided_mask = torch.stack((live_mask, live_mask), -1)[:, 0]
        assert not strided_mask.is_contiguous()
        outcome(fn(*operands, strided_mask))
    info["actual_path"] = "triton" if "actual" in info["launches"] else "torch"
    info["complete"] = (
        "dense_native" in info["launches"] and "torch_ref" not in info["launches"]
    )
    record["engram_gate"] = info
    return out


def _schema(data, meta, nonce, rank):
    if meta != dict(
        nonce=nonce, role="oldfst", family="v41", rank=str(rank), stage="layer1.hit512"
    ):
        raise ValueError("wrong V4.1 fixture provenance")
    if set(data) != KEYS:
        raise ValueError("wrong V4.1 fixture fields")
    pos = data["positions"]
    if pos.ndim != 1 or pos.dtype != torch.int64 or not 1 <= len(pos) <= 271:
        raise ValueError("invalid V4.1 fixture CP extent")
    ids = pos.tolist()
    if len(set(ids)) != len(ids) or not all(512 <= p < 783 for p in ids):
        raise ValueError("invalid V4.1 fixture logical positions")
    for key, shape, dtype in (
        ("incoming", (len(ids), 4, 5120), torch.bfloat16),
        ("previous_mix", (len(ids), 4), torch.float32),
        ("swa_tail", (127, 512), torch.bfloat16),
    ):
        if tuple(data[key].shape) != shape or data[key].dtype != dtype:
            raise ValueError("invalid V4.1 fixture tensor " + key)


def _read(rank, nonce, positions, *, old_root=None):
    from safetensors import safe_open

    path = (
        (old_root / "arm_causal_private" / f"v41_r{rank}.safetensors")
        if old_root is not None
        else (
            Path(os.environ["TEST_SRCDIR"])
            / os.environ["TEST_WORKSPACE"]
            / (f"rtp_llm/test/smoke/arm_causal_inputs/v41_r{rank}.safetensors")
        )
    )
    if not path.is_file() or not 0 < path.stat().st_size < 12 << 20:
        raise ValueError("missing or oversized V4.1 private fixture")
    with safe_open(str(path), framework="pt", device="cpu") as f:
        if set(f.keys()) != KEYS:
            raise ValueError("unexpected V4.1 private fixture fields")
        data = {k: f.get_tensor(k) for k in KEYS}
        _schema(data, f.metadata(), nonce, rank)
    if not torch.equal(data["positions"], positions):
        raise ValueError("paired V4.1 CP rank owns different rows")
    return data


# "replay" retains the original both-Old artifact name.
CELLS = {
    "replay_nn": (False, False),
    "replay_on": (True, False),
    "replay_no": (False, True),
    "replay": (True, True),
}


def _slot_geometry(meta, entries):
    # Absolute physical page numbers vary between workers/requests. Preserve
    # the logical row -> (page equality class, ring offset) correspondence.
    slots = meta.cache_slot_mapping.detach().flatten()[:127].cpu().tolist()
    if len(slots) != 127 or not 0 < entries <= 65536:
        raise ValueError("invalid V4.1 SWA physical slot geometry")
    blocks, canonical = {}, []
    for position, slot in enumerate(slots, 385):
        if slot == -1:
            canonical.append([-1, -1])
            continue
        block, offset = divmod(slot, entries)
        if slot < 0 or offset != position % entries:
            raise ValueError("V4.1 SWA slot does not address its logical ring offset")
        canonical.append([blocks.setdefault(block, len(blocks)), offset])
    return dict(entries=entries, logical_start=385, slots=canonical)


def observe(trace, call, layer, cp, h, *args, **kwargs):
    role, nonce = os.environ["RTP_V41_CAUSAL_ROLE"], os.environ["RTP_ARM_PAIR_NONCE"]
    rank = int(cp.cp_rank)
    if (
        role not in ("oldfst", "newfst")
        or not re.fullmatch("[0-9a-f]{32}", nonce)
        or os.getenv("RTP_V41_EXPECTED_LOADER") != role
        or os.getenv("TEST_TARGET") != TARGET
        or trace.phase != "hit"
        or trace.prefix != 512
        or int(cp.cp_size) != 2
        or rank not in (0, 1)
        or layer.attn.layer_id != 1
        or layer.attn.compress_ratio != 0
        or h.dtype != torch.bfloat16
        or h.shape[1:] != (4, 5120)
    ):
        raise ValueError("unexpected V4.1 causal fixture scope")
    valid = cp.local_is_real.reshape(-1).bool()
    positions = cp.global_positions.reshape(-1)[valid].long().cpu()
    marker = trace.root / "v41_causal_replay_id"
    replay = marker.is_file()
    mode = "natural"
    if replay:
        pair = marker.read_text().split()
        if len(pair) != 2 or pair[0] != trace.run or pair[1] not in CELLS:
            raise ValueError("mismatched V4.1 repeat request")
        mode = pair[1]
    use_mix, use_swa = CELLS.get(mode, (False, False))
    incoming = (
        _read(rank, nonce, positions, old_root=trace.root if role == "oldfst" else None)
        if replay
        else {}
    )
    relay = {"positions": positions}
    record = dict(
        schema=1,
        family="v41",
        nonce=nonce,
        role=role,
        mode=mode,
        rank=rank,
        positions=positions.tolist(),
        events=[],
        request_id=trace.run,
        cp=dict(
            size=2,
            rank=rank,
            local_rows=len(valid),
            live_rows=valid.nonzero().flatten().tolist(),
        ),
        weights={},
        skipped=[],
    )

    def event(name, value, *, layout="local"):
        # Do not hash physical padding / invalid Q rows. All exported observations are digests.
        if layout == "local":
            if value.size(0) == len(valid):
                value = value[valid]
            elif value.size(0) != len(positions):
                raise ValueError("unmappable V4.1 rank rows: " + name)
        summary = _digest(value)
        if layout == "local":
            rows = (
                value.detach()
                .contiguous()
                .view(torch.uint8)
                .reshape(len(positions), -1)
                .cpu()
                .numpy()
            )
            summary["rows"] = [hashlib.sha256(row).hexdigest() for row in rows]
        record["events"].append([name, summary])

    def boundary(name, value):
        event(name + ".observed", value)
        if not replay:
            if role == "oldfst":
                relay[name] = value[valid].detach().to("cpu", copy=True).contiguous()
            return value
        # Only valid request rows are serialized. Canonicalize padding on BOTH
        # replay arms, without mutating the ordinary request or the source tensor.
        out = torch.zeros_like(value)
        live = value[valid].detach().contiguous()
        if name == "incoming" or use_mix:
            live = incoming[name].to(value.device)
        out[valid] = live
        event(name + ".used", out)
        return out

    def weight(name, value):
        if isinstance(value, torch.Tensor):
            record["weights"][name] = _digest(value)
        else:
            record["skipped"].append("weight:" + name)

    # Fingerprint actual device-bound native tensors, not checkpoint aliases.
    for key, owner in (
        ("engram_wkv", layer.engram.wkv),
        ("wq_b", layer.attn.wq_b),
        ("wo_b", layer.attn.wo_b),
    ):
        for field, value in zip(("weight", "weight_scales"), _native_pair(owner)):
            weight(key + "." + field, value)
    for label, obj, names in (
        ("engram", layer.engram, ("q_weight", "k_weight")),
        ("attn_hc", layer.attn_hc, ("fn", "base", "scale")),
        ("ffn_hc", layer.ffn_hc, ("fn", "base", "scale")),
        ("attn_norm", layer.attn_norm, ("weight",)),
        ("ffn_norm", layer.ffn_norm, ("weight",)),
        (
            "attention",
            layer.attn,
            ("q_norm", "kv_norm", "attn_sink", "_wo_a_stk_w", "_wo_a_stk_s"),
        ),
    ):
        for field in names:
            weight(label + "." + field, getattr(obj, field, None))
    moe_gate = layer.ffn.gate
    weight("gate.bf16", moe_gate._weight_bf16())
    for field in ("bias", "bias_vl", "tid2eid"):
        value = getattr(moe_gate, field, None)
        if value is not None:
            weight("gate." + field, value)
    table = layer.engram.embed_tokens
    # Only rows named by the matched request (never read or relay the whole table).
    ids = layer.engram_hashes[valid].detach().long().cpu().reshape(-1)
    ids = torch.unique(ids[ids >= 0], sorted=True)
    if (
        len(ids) > 271 * 24
        or table.weight.element_size() != 1
        or table.scales.element_size() != 1
    ):
        raise ValueError("unexpected Engram fixture lookup")
    weight("engram.addresses", ids)
    for field, rows in _lookup_rows(table, ids, layer.engram_hashes.is_cuda):
        weight("engram.selected." + field, rows)
    experts = layer.ffn.fused_moe.fused_experts
    for field in ("_mega_l1_w", "_mega_l1_sf", "_mega_l2_w", "_mega_l2_sf"):
        weight("experts." + field, getattr(experts, field, None))
    if layer.ffn.fused_moe.includes_shared_expert:
        for field in ("_se_l1_w", "_se_l1_sf", "_se_l2_w", "_se_l2_sf"):
            weight("experts." + field, getattr(experts, field, None))

    event("input_ids", args[0])
    event("hash_ids", layer.engram_hashes)
    event("token_mask", layer.engram_token_mask)
    previous = layer.attn_hc._previous_ref()
    h = boundary("incoming", h)
    mix = boundary("previous_mix", previous.pre_mix_out)
    with ExitStack() as stack:
        if replay:
            stack.enter_context(patch.object(previous, "pre_mix_out", mix))

        def wrap(obj, attr, intercept):
            fn = getattr(obj, attr)

            def wrapped(*a, **k):
                return intercept(fn, a, k)

            stack.enter_context(patch.object(obj, attr, wrapped))

        if h.is_cuda and hasattr(layer.engram, "q_weight"):
            from rtp_llm.models_py.modules.dsv41 import engram as gate_module

            def observe_engram_gate(fn, a, k):
                if (
                    len(a) == 6
                    and a[2] is layer.engram.q_weight
                    and a[3] is layer.engram.k_weight
                ):
                    return _gate_witness(fn, a, k, valid, record)
                return fn(*a, **k)

            wrap(gate_module, "gated_engram_residual", observe_engram_gate)

        shared = layer.ffn.shared_experts
        if not layer.ffn.fused_moe.includes_shared_expert and shared is not None:

            def shared_used(module, out, fast=None):
                for key, pair in zip(("w13", "w2"), _shared_pairs(module, fast)):
                    for field, value in zip(("weight", "weight_scales"), pair):
                        weight("shared." + key + "." + field, value)
                event(
                    "shared." + ("prepared" if fast is not None else "module") + ".out",
                    out,
                )

            stack.callback(
                shared.register_forward_hook(lambda m, _a, o: shared_used(m, o)).remove
            )
            runtime = getattr(layer.ffn, "_runtime", layer.ffn)
            executor = runtime._moe._shared_executor
            fast = getattr(executor, "_fast_path", None)
            if fast is not None:

                def fast_shared(fn, a, k):
                    out = fn(*a, **k)
                    try:
                        shared_used(a[0], out, fast)
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        # Production catches exceptions here and would rerun a
                        # fallback. A failed observation must not select it.
                        record["skipped"].append("prepared shared observation")
                    return out

                wrap(fast, "_run_prepared", fast_shared)

        for key, module in (
            ("engram_wkv", layer.engram.wkv),
            ("engram", layer.engram),
            ("attn_norm", layer.attn_norm),
            ("attention", layer.attn),
            ("ffn_norm", layer.ffn_norm),
            ("moe", layer.ffn),
        ):
            stack.callback(
                module.register_forward_pre_hook(
                    lambda _m, a, n=key: event(n + ".in", a[0])
                ).remove
            )
            stack.callback(
                module.register_forward_hook(
                    lambda _m, _a, out, n=key: event(n + ".out", out)
                ).remove
            )

        def hc(before):
            def run(fn, a, k):
                y, post, comb = fn(*a, **k)
                for label, value in (("readout", y), ("post", post), ("comb", comb)):
                    event(before + "." + label, value)
                return y, post, comb

            return run

        wrap(layer.attn_hc, "pre", hc("attn_hc"))
        wrap(layer.ffn_hc, "pre", hc("ffn_hc"))

        def qkv(fn, a, k):
            common = a[1]
            event("rope", common.freqs_cis)
            value = fn(*a, **k)
            event("qr", value.qr)
            event("new_kv", value.kv_full, layout="global")
            return value

        wrap(layer.attn, "_prefill_compute_qkv", qkv)

        def pre_qkv(fn, a, k):
            result = fn(*a, **k)
            if result is not None:
                for field, value in zip(
                    ("weight", "weight_scales"), _native_pair(layer.attn.wq_a_wkv)
                ):
                    weight("fused_qkv." + field, value)
                event("fused_qkv.qr", result[0].squeeze(0))
                event("fused_qkv.kv", result[1].squeeze(0))
            else:
                for key in ("wq_a", "wkv"):
                    op = getattr(layer.attn, key)
                    for field, value in zip(
                        ("weight", "weight_scales"), _native_pair(op)
                    ):
                        weight(key + "." + field, value)
            return result

        wrap(layer.attn, "_try_fused_qr_kv", pre_qkv)

        def packed(fn, a, k):
            result = fn(*a, **k)
            if result is not None:
                event("gate.scores", result.scores)
            return result

        wrap(moe_gate, "prepare_gate_payload", packed)

        def expert_pack(fn, a, k):
            result = fn(*a, **k)
            y, weights, ids = result
            event("gate.packed_weights", weights)
            event("gate.packed_ids", ids)
            event("routed.out", y)
            return result

        wrap(experts, "forward_gate_pack", expert_pack)

        def swa(fn, a, k):
            common = a[1]
            meta = common.swa_meta
            if (
                common.batch_size != 1
                or not common.cp_on
                or not common.any_cont
                or layer.attn._host_prefill_prefixes(common) != [512]
                or layer.attn._host_prefill_lengths(common) != [271]
                or meta.cache_gather_lens.reshape(-1).tolist() != [127]
            ):
                raise ValueError("unexpected V4.1 live SWA prefix geometry")
            out = fn(*a, **k)
            prefix = out[0, :127]
            if prefix.shape != (127, 512) or prefix.dtype != torch.bfloat16:
                raise ValueError("invalid V4.1 decoded SWA operand")
            event("swa_tail.observed", prefix, layout="global")
            record["swa_geometry"] = _slot_geometry(
                meta, int(layer.attn._swa_entries_per_block())
            )
            if not replay and role == "oldfst":
                relay["swa_tail"] = prefix.detach().to("cpu", copy=True).contiguous()
            if use_swa:
                prefix.copy_(incoming["swa_tail"])
            event("swa_tail.used", prefix, layout="global")
            return out

        wrap(layer.attn, "_swa_prefill_concat", swa)

        def query(fn, a, k):
            value = fn(*a, **k)
            event("q_rope", value)
            return value

        wrap(layer.attn, "_project_prefill_q", query)

        def flash_inputs(fn, a, k):
            event("flash.indices", k["indices"])
            event("flash.lens", k["topk_length"])
            return fn(*a, **k)

        wrap(layer.attn, "_prefill_sparse_attention", flash_inputs)

        def proj(fn, a, k):
            event("flash.out", a[0])
            result = fn(*a, **k)
            event("o_proj", k["out"])
            return result

        wrap(layer.attn, "_prefill_output_proj_into", proj)

        def mega(fn, a, k):
            result = fn(*a, **k)
            if result is not None:
                for label, t in zip(("residual", "norm", "post", "comb"), result):
                    event("mega_mhc." + label, t)
            return result

        wrap(layer, "_try_mega_mhc", mega)
        out = call(h, *args, **kwargs)
    event("outgoing", out)
    event("next_mix", layer.ffn_hc.pre_mix_out)
    if (
        not layer.ffn.fused_moe.includes_shared_expert
        and "shared.w13.weight" not in record["weights"]
    ):
        record["skipped"].append("shared executor operands")
    with (trace.root / f"v41_causal_{mode}_r{rank}.json").open("x") as f:
        json.dump(record, f, separators=(",", ":"))
    if not replay and role == "oldfst":
        from safetensors.torch import save_file

        provenance = dict(
            nonce=nonce, role=role, family="v41", rank=str(rank), stage="layer1.hit512"
        )
        _schema(relay, provenance, nonce, rank)
        private = trace.root / "arm_causal_private"
        private.mkdir(mode=0o700, exist_ok=True)
        output = private / f"v41_r{rank}.safetensors"
        if output.exists():
            raise ValueError("duplicate V4.1 private fixture")
        save_file(relay, str(output), metadata=provenance)
    return out
