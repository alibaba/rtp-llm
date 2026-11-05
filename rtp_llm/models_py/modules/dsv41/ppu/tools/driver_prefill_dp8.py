"""PPU890 DP8/EP8/ENP8 native Prefill functional qualification.

Launch with torch.distributed.run --standalone --nproc-per-node=8 -m
rtp_llm.models_py.modules.dsv41.ppu.tools.driver_prefill_dp8. Each rank owns
complete independent requests; there is no CP token split or KV all-gather.
This exercises NativeModelRunner, including fake streams, with logical KV
pools. Native physical-page publication is covered by test_prefill_dp8;
HTTP/CacheStore/RDMA acceptance requires a separate service run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS


def request_lengths(scenario, rank, tokens):
    counts = {
        "all": (1,) * 8,
        "one-busy": (0, 0, 0, 0, 1, 0, 0, 0),
        "skew": (2, 0, 1, 0, 3, 0, 0, 1),
        "idle": (0,) * 8,
    }
    return [tokens + (rank + row) % 3 for row in range(counts[scenario][rank])]


def make_inputs(geometry, lengths, rank, device):
    import torch

    parts, windows = [], []
    for row, length in enumerate(lengths):
        tokens = (torch.arange(length) + rank * 997 + row * 17) % geometry.vocab_size
        parts.append(tokens)
        windows.extend(
            [
                [
                    int(tokens[pos - back]) if pos >= back else -1
                    for back in range(geometry.engram_max_ngram_size)
                ]
                for pos in range(length)
            ]
        )
    ids = torch.cat(parts) if parts else torch.zeros(1, dtype=torch.int64)
    primary = NS(
        is_prefill=True,
        input_lengths=torch.tensor(lengths or [1]),
        prefix_lengths=torch.zeros(max(1, len(lengths)), dtype=torch.int64),
    )
    return NS(
        input_ids=ids.to(device),
        attention_inputs={"swa_kv": primary},
        engram_token_windows=torch.tensor(windows, dtype=torch.int64, device=device),
        is_fake_stream=not lengths,
        warmup=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--token-map-cache")
    parser.add_argument("--tokens", type=int, default=33)
    parser.add_argument("--scenarios", default="all,one-busy,skew,idle")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    scenarios = args.scenarios.split(",")
    if args.tokens < 1 or any(
        s not in ("all", "one-busy", "skew", "idle") for s in scenarios
    ):
        parser.error("positive --tokens and scenarios all,one-busy,skew,idle required")
    if int(os.environ.get("WORLD_SIZE", "0")) != 8:
        parser.error("launch through torch.distributed.run with eight workers")

    import torch
    import torch.distributed as dist

    from rtp_llm.models_py.modules.dsv41.ppu.engram import DistributedEnpmGroup
    from rtp_llm.models_py.modules.dsv41.ppu.model import PPUModelCore
    from rtp_llm.models_py.modules.dsv41.ppu.native_model import NativeModelRunner
    from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
    from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_prefill_cp8 import (
        _load_config,
    )
    from rtp_llm.models_py.modules.dsv41.ppu.weight import V41Geometry, load_ppu_weights

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout))
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=args.timeout))
    config = [None]
    if rank == 0:
        config[0] = _load_config(
            NS(
                config_path=str(args.checkpoint_path / "config.json"),
                tokenizer_path=None,
                token_map_cache=args.token_map_cache,
            )
        )
    dist.broadcast_object_list(config, src=0, group=control)
    config = config[0]
    geo = V41Geometry.from_config(config)
    if args.tokens + 2 > geo.max_seq_len:
        raise ValueError("requested tokens exceed the checkpoint context limit")
    enp = DistributedEnpmGroup(dist.group.WORLD)
    ops = PPUOps(
        ep_rank=rank,
        ep_size=8,
        num_routed_experts=geo.n_routed_experts,
        num_local_experts=geo.num_local_experts(8),
        hidden_size=geo.hidden_size,
        inter_dim=geo.moe_inter_dim,
        swiglu_limit=geo.swiglu_limit,
        mhc_config=geo.mhc_config(),
        sm_scale=geo.sm_scale(),
        window=geo.window,
        enpm_group=enp,
        deepep_group=dist.group.WORLD,
    )
    bundle = load_ppu_weights(
        config,
        args.checkpoint_path,
        ep_rank=rank,
        ep_size=8,
        enp_rank=rank,
        enp_size=8,
        device=device,
    )
    bundle.bind(ops, enpm_group=enp)
    bundle.lm_head = None

    class RecordingCore(PPUModelCore):
        last_requests = None

        def prefill(self, *inputs, **kwargs):
            result = super().prefill(*inputs, **kwargs)
            self.last_requests = dict(kwargs["cache"]._request_tokens)
            return result

    core = RecordingCore(config, bundle, ops, device=device)
    runner = NativeModelRunner(core, dp_rank=rank, dp_size=8)
    runner.initialize(NS(kv_cache=None, is_decode_role=False))
    results = []
    with torch.inference_mode():
        for scenario in scenarios:
            lengths = request_lengths(scenario, rank, args.tokens)
            inputs = make_inputs(geo, lengths, rank, device)
            core.last_requests = None
            dist.barrier(group=control)
            start = time.monotonic()
            hidden = runner.forward(inputs)
            torch.cuda.synchronize()
            checks = {
                "finite": bool(torch.isfinite(hidden).all()),
                "local_rows": hidden.shape
                == (sum(lengths) if lengths else 1, geo.hidden_size),
                "local_cache": core.last_requests
                == (dict(enumerate(lengths)) if lengths else None),
                "idle_output": bool(lengths) or int(hidden.count_nonzero()) == 0,
            }
            results.append(
                {
                    "scenario": scenario,
                    "lengths": lengths,
                    "seconds": time.monotonic() - start,
                    "checks": checks,
                }
            )
            print(json.dumps({"rank": rank, **results[-1]}), flush=True)
            if not all(checks.values()):
                raise AssertionError(results[-1])
    stats = ops.moe_engine.execution_stats()
    record = {
        "rank": rank,
        "layers": geo.num_layers,
        "dp": 8,
        "cp": 1,
        "ep": 8,
        "enp": 8,
        "scenarios": results,
        "moe": stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_name(f"{args.output.stem}.rank{rank}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    records = [None] * 8
    dist.all_gather_object(records, record, group=control)
    if rank == 0:
        args.output.write_text(
            json.dumps({"ranks_reported": 8, "ranks": records, "ok": True}, indent=2)
            + "\n"
        )
    torch.cuda.synchronize()
    dist.barrier(group=control)
    ops.moe_engine.close()
    # DeepEP's IPC buffers must be freed before the process groups.
    from rtp_llm.models_py.modules.dsv41.ppu.moe import PPUMoE

    with PPUMoE._BUFFER_LOCK:
        PPUMoE._BUFFER_CACHE.clear()
    dist.barrier(group=control)
    dist.destroy_process_group(control)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
