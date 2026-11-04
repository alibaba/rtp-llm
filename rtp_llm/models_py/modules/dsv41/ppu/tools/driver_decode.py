#!/usr/bin/env python
"""Offline single-rank V4.1 PPU prefill->decode chain driver (L3).

Seeds the cache with one real prefill (L2's ``run_prefill`` through
``PPUModelCore``), wires the L3 decode orchestration onto the core
(``wire_decode``), then runs a fixed continuation of N decode steps — each
step samples argmax and feeds it back, so the Engram n-gram window spans the
step boundary exactly like production decode. Optionally validates the
§7.5 fixture contract (export -> load -> continue vs uninterrupted).

Device notes (single-rank bring-up, plan 7.5 — no P service needed):

* weights load with ``--ep-rank/--ep-size`` slicing (rank-0 shard of an EP8
  layout by default on real checkpoints); the LOCAL MoE path then computes
  only this rank's experts — routing to the other 7/8 of the experts
  contributes zero. This is a DEVICE-PATH smoke mode, recorded loudly as
  ``moe_mode`` in the summary, not a numerical claim;
* the Engram table loads as this ENP rank's owned rows only
  (``engram_coverage`` records it);
* positions/cu_seqlens/request_ids are passed EXPLICITLY on the device —
  L2's ``run_prefill`` still has CPU-default tensor creation on its default
  paths (reported; the explicit path avoids it).

Examples::

    # tiny synthetic model, CPU, fixture check included
    python -m rtp_llm.models_py.modules.dsv41.ppu.tools.driver_decode \
        --synthetic --tokens 33 --steps 8 --device cpu

    # real checkpoint on the D machine (inside the PPU container)
    python -m rtp_llm.models_py.modules.dsv41.ppu.tools.driver_decode \
        --config-path <ckpt>/config.json --checkpoint-path <ckpt> \
        --tokens 128 --steps 8 --device cuda --ep-rank 0 --ep-size 8 \
        --output <runs>/d_decode_chain.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config-path", help="path to config.json (real model)")
    source.add_argument(
        "--synthetic", action="store_true", help="tiny synthetic config/checkpoint"
    )
    parser.add_argument("--checkpoint-path", help="checkpoint directory")
    parser.add_argument("--tokens", type=int, default=33, help="prefill token count")
    parser.add_argument(
        "--requests", type=int, default=1, help="requests in the prefill batch"
    )
    parser.add_argument(
        "--steps", type=int, default=8, help="decode continuation steps"
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--ep-rank", type=int, default=0)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--enp-rank", type=int, default=None)
    parser.add_argument("--enp-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fixture-steps",
        type=int,
        default=0,
        help="also run the export/load/continue fixture check for N steps",
    )
    parser.add_argument(
        "--token-map-path",
        help="cache file for the tokenizer-derived engram_token_map "
        "(real checkpoints; derived once, reused across runs)",
    )
    parser.add_argument(
        "--mhc-torch",
        action="store_true",
        help="force the authoritative torch mHC path (workaround: the D "
        "container's Triton rejects the mix-stats kernel's int64 arg cast; "
        "the kernel fix compiles on P but not on D)",
    )
    parser.add_argument("--output", help="write the JSON summary to this path")
    return parser.parse_args(argv)


def _derive_token_map(checkpoint: str):
    """Tokenizer-derived engram token map (reference normalization)."""
    from transformers import AutoTokenizer

    from rtp_llm.models_py.modules.dsv41.ppu.engram import compressed_vocab_map

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    lookup, size = compressed_vocab_map(tokenizer)
    return lookup, size


def _load_config(args):
    if args.config_path:
        with open(args.config_path) as handle:
            raw = json.load(handle)
        text = {**raw, **raw.get("text_config", {})}
        text.pop("text_config", None)
        expected = text.get("engram_compressed_vocab_size")
        if "engram_token_map" not in text:
            # The checkpoint config does not carry the tokenizer-derived map;
            # derive it (cached next to the token-map path when given).
            cache_path = (
                Path(args.token_map_path)
                if args.token_map_path
                else Path(args.config_path).parent / "engram_token_map.json"
            )
            if cache_path.is_file():
                text["engram_token_map"] = json.loads(cache_path.read_text())
            else:
                lookup, size = _derive_token_map(args.checkpoint_path)
                if expected is not None and size != int(expected):
                    raise SystemExit(
                        f"derived compressed vocab {size} != config "
                        f"{expected}; refusing to run with a wrong token map"
                    )
                text["engram_token_map"] = lookup
                try:
                    cache_path.write_text(json.dumps(lookup))
                    print(f"[driver_decode] token map cached at {cache_path}")
                except OSError:
                    pass
        return text
    from rtp_llm.models_py.modules.dsv41.ppu.weight import synthetic_config

    return synthetic_config()


class _SoloEnpmGroup:
    """Single-process stand-in for the ENP group (single-rank smoke).

    Simulates an ``enp_size``-rank ENP group in which every OTHER rank has an
    empty query: ``all_gather_ids`` places this rank's (tail-padded) chunk at
    its fixed rank slot with invalid ``-1`` ids elsewhere, and the sum-combine
    degenerates to the local owner-side contribution. Rows outside this
    shard's owned range contribute zeros -- the documented single-rank
    coverage limit (``engram_coverage`` in the summary).
    """

    def __init__(self, rank: int, size: int):
        self.rank = int(rank)
        self.size = int(size)

    def all_max_count(self, count, device) -> int:
        return int(count)

    def all_gather_ids(self, chunk):
        import torch

        length = int(chunk.numel())
        if self.size == 1:
            return chunk
        gathered = torch.full(
            (self.size * length,), -1, dtype=chunk.dtype, device=chunk.device
        )
        gathered[self.rank * length : (self.rank + 1) * length] = chunk
        return gathered

    def all_reduce_sum(self, tensor):
        return tensor


class _DriverOps:
    """Ops wrapper for the offline decode drivers (fix-forward device fixes).

    ``sparse_attention`` replicates ``ppu/attention.sparse_attention`` with
    ONE fix: the per-HEAD ``attn_sink`` is passed UNSLICED. The L2-owned loop
    slices the sink by the query-row chunk (``attn_sink[start:end]``): a
    decode step's single row truncates the ``[h_q]`` sink to ``[1]``
    (flash_mla rejects: "attn_sink must have shape (h_q)") and a multi-chunk
    prefill silently drops all sink entries beyond the first chunk. Single
    chunk prefills with ``rows >= n_heads`` pass only because the row slice
    clamps back to the full sink. Handed to L2 (ownership log); remove this
    override when the attention.py fix lands.
    """

    def __init__(self, ops, device=None):
        self._ops = ops
        self._device = device

    def __getattr__(self, name):
        return getattr(self._ops, name)

    def sparse_attention(
        self, q, selected_blocks, *, cache, layer_id, context=None, **kwargs
    ):
        from rtp_llm.models_py.modules.dsv41.ppu import attention as _attention

        if "sm_scale" not in kwargs:
            scale = getattr(self._ops, "_sm_scale", None)
            if scale is not None:
                kwargs["sm_scale"] = scale
        if "window" not in kwargs:
            kwargs["window"] = getattr(self._ops, "_window", _attention.SWA_WINDOW)
        attn_sink = kwargs.pop("attn_sink", None)
        sm_scale = kwargs.pop("sm_scale", _attention.SM_SCALE)
        window = kwargs.pop("window", _attention.SWA_WINDOW)
        ratio = kwargs.pop("ratio", None)
        import torch

        kv, indices, topk_length = _attention.build_sparse_workspace(
            selected_blocks,
            cache=cache,
            layer_id=layer_id,
            context=context,
            window=window,
            ratio=ratio,
        )
        outputs = []
        rows = q.shape[0]
        chunk = (
            int(_attention.Q_CHUNK)
            if _attention.Q_CHUNK and _attention.Q_CHUNK > 0
            else rows or 1
        )
        for start in range(0, rows, chunk):
            end = min(start + chunk, rows)
            output, _, _ = _attention.flash_mla_sparse_fwd_ppu(
                q[start:end].contiguous(),
                kv,
                indices[start:end],
                sm_scale,
                _attention.HEAD_DIM,
                attn_sink=attn_sink,
                topk_length=topk_length[start:end],
            )
            outputs.append(output)
        if not outputs:
            return q.new_empty((0,) + tuple(q.shape[1:]))
        return torch.cat(outputs, dim=0)


class _DeviceOps(_DriverOps):
    """Single-rank device wrapper: Engram lookups stay on the host shard.

    The Engram shards are HOST tensors; the lookup runs on CPU rows and its
    result moves to the compute device (the DP8 driver handles the
    distributed ENP group instead).
    """

    def __init__(self, ops, device):
        super().__init__(ops, device)

    def engram_lookup(self, rows, *, layer_id, context=None):
        rows_cpu = rows.detach().to("cpu")
        out = self._ops.engram_lookup(rows_cpu, layer_id=layer_id, context=context)
        return out.to(self._device)


def _memory_summary(device):
    if device.type != "cuda":
        return {}
    try:
        import torch

        return {
            "max_allocated_mb": int(torch.cuda.max_memory_allocated() / 2**20),
            "allocated_mb": int(torch.cuda.memory_allocated() / 2**20),
        }
    except Exception as error:  # noqa: BLE001 - memory stats are best effort
        return {"error": f"{type(error).__name__}: {error}"}


def main(argv=None) -> int:
    args = _parse_args(argv)
    import torch

    from rtp_llm.models_py.modules.dsv41.ppu.decode import wire_decode
    from rtp_llm.models_py.modules.dsv41.ppu.kv_io import (
        compare_against_uninterrupted,
        load_fixture_state,
    )
    from rtp_llm.models_py.modules.dsv41.ppu.model import PPUModelCore
    from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
    from rtp_llm.models_py.modules.dsv41.ppu.weight import (
        V41Geometry,
        load_ppu_weights,
        write_synthetic_checkpoint,
    )

    if not args.config_path and not args.synthetic:
        args.synthetic = True
    if args.config_path and not args.checkpoint_path:
        raise SystemExit("--config-path requires --checkpoint-path")
    config = _load_config(args)
    device = torch.device(args.device)

    checkpoint = args.checkpoint_path
    synthetic_dir = None
    if checkpoint is None:
        synthetic_dir = Path(tempfile.mkdtemp(prefix="dsv41_ppu_synth_"))
        checkpoint = write_synthetic_checkpoint(config, synthetic_dir, seed=args.seed)

    geometry = V41Geometry.from_config(config)
    # Single-rank ops. PPUMoE's invariant (ep_size * num_local == routed)
    # forces ep_size 1 with the FULL expert count, but a real checkpoint
    # loads only this EP rank's shard; after construction we pin the instance
    # to the LOCAL reference path over exactly the bound experts. This is the
    # loudly-marked single-rank smoke mode (see moe_mode in the summary).
    ep_size_used = max(args.ep_size, 1)
    enp_size = args.enp_size if args.enp_size is not None else ep_size_used
    enp_rank = args.enp_rank if args.enp_rank is not None else args.ep_rank
    # ENP>1 without a process group: the solo shim keeps the shared lookup
    # protocol running over exactly this rank's bound shard.
    enpm_group = _SoloEnpmGroup(enp_rank, enp_size) if enp_size > 1 else None
    ops = PPUOps(
        ep_rank=args.ep_rank,
        ep_size=ep_size_used,
        num_routed_experts=geometry.n_routed_experts,
        num_local_experts=geometry.num_local_experts(ep_size_used),
        hidden_size=geometry.hidden_size,
        inter_dim=geometry.moe_inter_dim,
        swiglu_limit=geometry.swiglu_limit,
        mhc_config=geometry.mhc_config(),
        sm_scale=geometry.sm_scale(),
        window=geometry.window,
        enpm_group=enpm_group,
    )
    if args.ep_size > 1:
        # Route everything through the LOCAL reference path (no dispatch):
        # only the loaded shard's experts compute; other routed ids
        # contribute zero. Recorded as moe_mode below -- never silently.
        # (PPUOps.moe is the contract op method; moe_engine is the PPUMoE.)
        ops.moe_engine.ep_size = 1
    bundle = load_ppu_weights(
        config,
        checkpoint,
        ep_rank=args.ep_rank,
        ep_size=max(args.ep_size, 1),
        enp_rank=args.enp_rank,
        enp_size=args.enp_size,
        device=device,
    )
    bundle.bind(ops)
    if args.mhc_torch:
        # Workaround (fix-forward): the D container's Triton rejects the
        # mix-stats kernel's int64 scalar cast -- the wave2 _mhc_triton fix
        # compiles on P but not on D. mhc.py keeps the torch path as the
        # authority; force it through the public bind_mhc_config seam until
        # the kernel compiles on D. Recorded as mhc_backend in the summary.
        import dataclasses

        ops.bind_mhc_config(
            dataclasses.replace(geometry.mhc_config(), use_triton=False)
        )
    if device.type == "cuda":
        ops = _DeviceOps(ops, device)
    core = PPUModelCore(config, bundle, ops, device=device)

    generator = torch.Generator().manual_seed(args.seed)
    per_request = args.tokens // max(args.requests, 1)
    if per_request < 1:
        raise SystemExit("--tokens must be >= --requests")
    input_ids = torch.randint(
        0,
        geometry.vocab_size,
        (args.requests * per_request,),
        dtype=torch.int64,
        generator=generator,
    )
    cu = torch.arange(
        0, args.requests * per_request + 1, per_request, dtype=torch.int64
    )
    request_ids = torch.repeat_interleave(
        torch.arange(args.requests, dtype=torch.int64, device=device),
        torch.full((args.requests,), per_request, dtype=torch.int64, device=device),
    )
    # Positions are PER-REQUEST absolute positions: every request's first
    # token sits at position 0 (run_prefill validates exactly this).
    positions = torch.arange(per_request, dtype=torch.int64).repeat(args.requests)

    started = time.perf_counter()
    prefill = core.prefill(
        input_ids.to(device),
        positions=positions.to(device),
        cu_seqlens=cu.to(device),
        request_ids=request_ids,
        apply_lm_head=True,
    )
    prefill_seconds = time.perf_counter() - started
    if not bool(torch.isfinite(prefill.logits).all()):
        raise SystemExit("prefill logits contain non-finite values")

    orchestrator = wire_decode(core)
    histories = {}
    seed_tokens = {}
    for rid in range(args.requests):
        history = input_ids[rid * per_request : (rid + 1) * per_request].tolist()
        orchestrator.add_request(rid, history)
        histories[rid] = history
        seed_tokens[rid] = int(prefill.logits[int(cu[rid + 1]) - 1].argmax())

    pending = dict(seed_tokens)
    steps_report = []
    for step in range(args.steps):
        started = time.perf_counter()
        result = orchestrator.step(dict(pending))
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        finite = bool(torch.isfinite(result.hidden).all())
        if result.logits is not None and not bool(torch.isfinite(result.logits).all()):
            finite = False
        if not finite:
            steps_report.append({"step": step, "error": "non-finite values"})
            break
        row = 0
        for request_id in result.row_request_ids:
            token = int(result.logits[row].argmax())
            top_value = float(result.logits[row].max())
            steps_report.append(
                {
                    "step": step,
                    "request_id": request_id,
                    "position": result.row_positions[row],
                    "input_token": pending[request_id],
                    "argmax_token": token,
                    "top_logit": top_value,
                    "hidden_finite": True,
                    "step_seconds": round(elapsed, 4),
                }
            )
            histories[request_id].append(pending[request_id])
            pending[request_id] = token
            row += 1

    fixture_report = None
    if args.fixture_steps > 0:
        fixture_dir = Path(tempfile.mkdtemp(prefix="dsv41_decode_fixture_"))
        try:
            orchestrator.cache.export_fixture(
                fixture_dir, meta={"source_cp": 1, "driver": "driver_decode"}
            )
            restored = load_fixture_state(
                fixture_dir,
                ops=ops,
                bundle=bundle,
                request_histories={
                    rid: state.history for rid, state in orchestrator.requests.items()
                },
                device=device,
            )
            fixture_report = compare_against_uninterrupted(
                orchestrator,
                restored,
                steps=args.fixture_steps,
                seed_tokens=dict(pending),
            )
            fixture_report = {
                k: (v if not isinstance(v, float) else round(v, 6))
                for k, v in fixture_report.items()
            }
        finally:
            import shutil

            shutil.rmtree(fixture_dir, ignore_errors=True)

    states = {
        rid: {
            "seq_len": state.seq_len,
            "steps_done": state.steps_done,
        }
        for rid, state in orchestrator.requests.items()
    }
    summary = {
        "device": str(device),
        "tokens": int(input_ids.numel()),
        "requests": args.requests,
        "steps_requested": args.steps,
        "steps_completed": len({s["step"] for s in steps_report}),
        "layers": geometry.num_layers,
        "kv_source_layer_ids": list(geometry.kv_source_layer_ids),
        "engram_layer_ids": list(geometry.engram_layer_ids),
        "ep_rank": args.ep_rank,
        "ep_size": args.ep_size,
        "enp_size": args.enp_size if args.enp_size is not None else args.ep_size,
        "moe_mode": (
            f"local-reference: experts [{args.ep_rank * geometry.num_local_experts(max(args.ep_size, 1))}:"
            f"{(args.ep_rank + 1) * geometry.num_local_experts(max(args.ep_size, 1))}) "
            f"of {geometry.n_routed_experts}; routed ids outside the range "
            "contribute zero (single-rank device smoke, NOT a numerical claim)"
        ),
        "engram_coverage": (
            f"enp rank {args.enp_rank if args.enp_rank is not None else args.ep_rank}/"
            f"{args.enp_size if args.enp_size is not None else args.ep_size} owned rows "
            "only; lookups outside the shard miss"
        ),
        "mhc_backend": (
            "torch (forced via --mhc-torch; D Triton rejects the mix-stats "
            "kernel's int64 arg cast)"
            if args.mhc_torch
            else "auto"
        ),
        "prefill_seconds": round(prefill_seconds, 3),
        "prefill_logits_finite": True,
        "steps": steps_report,
        "request_states": states,
        "cache_regions": {
            key: {str(rid): count for rid, count in per_request.items()}
            for key, per_request in core.cache.valid_regions().items()
        },
        "fixture_check": fixture_report,
        "memory": _memory_summary(device),
        "checkpoint": str(checkpoint),
    }
    ok = bool(steps_report) and all("error" not in s for s in steps_report)
    if fixture_report is not None:
        ok = ok and bool(fixture_report.get("match"))
    summary["ok"] = ok
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    if synthetic_dir:
        import shutil

        shutil.rmtree(synthetic_dir, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
