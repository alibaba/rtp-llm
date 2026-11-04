#!/usr/bin/env python
"""Offline single-rank V4.1 PPU prefill driver.

Builds weights (synthetic checkpoint or a real checkpoint directory), a bound
``PPUOps`` and a ``PPUModelCore``, runs one prefill over ``--tokens`` random
tokens and prints/writes a next-token logits summary (finite check, argmax,
top-5). Runs entirely on CPU with ``--device cpu --synthetic``; on a PPU host
pass ``--device cuda`` and a real ``--checkpoint-path``.

Examples::

    # tiny synthetic model, CPU, no checkpoint needed
    python -m rtp_llm.models_py.modules.dsv41.ppu.tools.driver_prefill \
        --synthetic --tokens 33 --device cpu

    # real checkpoint on the P machine (inside the PPU container)
    python -m rtp_llm.models_py.modules.dsv41.ppu.tools.driver_prefill \
        --config-path /path/to/checkpoint/config.json \
        --checkpoint-path /path/to/checkpoint \
        --tokens 128 --device cuda --ep-rank 0 --ep-size 8 \
        --output /tmp/rtp-llm-v41-runs/prefill_smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config-path", help="path to config.json (real model)")
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="use a tiny synthetic config/checkpoint (CPU friendly)",
    )
    parser.add_argument(
        "--checkpoint-path",
        help="checkpoint directory (default: a synthetic one in a temp dir)",
    )
    parser.add_argument(
        "--tokenizer-path",
        help="tokenizer.json for the engram_token_map (default: <config dir>/tokenizer.json)",
    )
    parser.add_argument(
        "--token-map-cache",
        help="JSON cache path for the computed engram_token_map (recommended on real runs)",
    )
    parser.add_argument("--tokens", type=int, default=33, help="prefill token count")
    parser.add_argument(
        "--requests", type=int, default=1, help="number of requests in the batch"
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--ep-rank", type=int, default=0)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="keep the synthetic checkpoint dir and print its path",
    )
    parser.add_argument("--output", help="write the JSON summary to this path")
    return parser.parse_args(argv)


def _load_token_map(tokenizer_path: str, cache_path=None) -> list:
    """The tokenizer-derived Engram compressed-vocab map (``engram_token_map``).

    ``compressed_vocab_map`` (ppu/engram.py, the rtp-llm reference math)
    needs a HF fast tokenizer; the raw ``tokenizers.Tokenizer`` is wrapped in
    a minimal shim when transformers is unavailable. The result is cached at
    ``cache_path`` (plain JSON list) so repeated runs skip the ~129k-decode
    walk.
    """
    import json as _json
    import os as _os

    if cache_path and _os.path.exists(cache_path):
        with open(cache_path) as handle:
            return _json.load(handle)

    from rtp_llm.models_py.modules.dsv41.ppu.engram import compressed_vocab_map

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            _os.path.dirname(_os.path.abspath(tokenizer_path)), trust_remote_code=False
        )
    except Exception:
        from tokenizers import Tokenizer

        raw = Tokenizer.from_file(tokenizer_path)

        class _Shim:
            backend_tokenizer = raw

            def __len__(self):
                return raw.get_vocab_size(with_added_tokens=True)

        tokenizer = _Shim()
    lookup, _ = compressed_vocab_map(tokenizer)
    if cache_path:
        _os.makedirs(_os.path.dirname(_os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w") as handle:
            _json.dump(list(int(x) for x in lookup), handle)
    return list(int(x) for x in lookup)


def _load_config(args):
    if args.config_path:
        with open(args.config_path) as handle:
            raw = json.load(handle)
        text = {**raw, **raw.get("text_config", {})}
        text.pop("text_config", None)
        if "engram_token_map" not in text:
            # Real configs: the Engram hash needs the training-time
            # compressed-vocab map derived from the checkpoint tokenizer.
            tokenizer_path = args.tokenizer_path
            if tokenizer_path is None:
                tokenizer_path = str(Path(args.config_path).parent / "tokenizer.json")
            if not Path(tokenizer_path).exists():
                raise SystemExit(
                    f"engram_token_map missing from the config and no tokenizer "
                    f"at {tokenizer_path}; pass --tokenizer-path"
                )
            text["engram_token_map"] = _load_token_map(
                tokenizer_path, args.token_map_cache
            )
        return text
    from rtp_llm.models_py.modules.dsv41.ppu.weight import synthetic_config

    return synthetic_config()


def main(argv=None) -> int:
    args = _parse_args(argv)
    import torch

    from rtp_llm.models_py.modules.dsv41.ppu.model import PPUModelCore
    from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
    from rtp_llm.models_py.modules.dsv41.ppu.weight import (
        load_ppu_weights,
        write_synthetic_checkpoint,
    )

    if not args.config_path and not args.synthetic:
        args.synthetic = True
    config = _load_config(args)
    device = torch.device(args.device)

    checkpoint = args.checkpoint_path
    synthetic_dir = None
    if checkpoint is None:
        if args.config_path:
            raise SystemExit(
                "--config-path requires --checkpoint-path (no synthetic weights "
                "for a real config)"
            )
        synthetic_dir = Path(tempfile.mkdtemp(prefix="dsv41_ppu_synth_"))
        checkpoint = write_synthetic_checkpoint(config, synthetic_dir, seed=args.seed)

    from rtp_llm.models_py.modules.dsv41.ppu.weight import V41Geometry

    geometry = V41Geometry.from_config(config)
    ops = PPUOps(
        ep_rank=args.ep_rank,
        ep_size=args.ep_size,
        num_routed_experts=geometry.n_routed_experts,
        num_local_experts=geometry.num_local_experts(args.ep_size),
        hidden_size=geometry.hidden_size,
        inter_dim=geometry.moe_inter_dim,
        swiglu_limit=geometry.swiglu_limit,
        mhc_config=geometry.mhc_config(),
        sm_scale=geometry.sm_scale(),
        window=geometry.window,
    )
    bundle = load_ppu_weights(
        config,
        checkpoint,
        ep_rank=args.ep_rank,
        ep_size=args.ep_size,
        device=device,
    )
    bundle.bind(ops)
    core = PPUModelCore(config, bundle, ops, device=device)

    generator = torch.Generator().manual_seed(args.seed)
    per_request = args.tokens // args.requests
    if per_request < 1:
        raise SystemExit("--tokens must be >= --requests")
    input_ids = torch.randint(
        0,
        geometry.vocab_size,
        (args.requests * per_request,),
        dtype=torch.int64,
        generator=generator,
    )
    cu_seqlens = torch.arange(
        0, args.requests * per_request + 1, per_request, dtype=torch.int64
    )
    result = core.prefill(
        input_ids.to(device),
        cu_seqlens=cu_seqlens.to(device),
        apply_lm_head=True,
    )
    logits = result.logits
    finite = bool(torch.isfinite(logits).all())
    last = logits[-1]
    top_values, top_ids = last.topk(min(5, last.shape[-1]))
    summary = {
        "device": str(device),
        "tokens": int(input_ids.numel()),
        "requests": int(args.requests),
        "layers": geometry.num_layers,
        "hidden_size": geometry.hidden_size,
        "kv_source_layer_ids": list(geometry.kv_source_layer_ids),
        "engram_layer_ids": list(geometry.engram_layer_ids),
        "ep_rank": args.ep_rank,
        "ep_size": args.ep_size,
        "logits_finite": finite,
        "hidden_shape": list(result.hidden.shape),
        "hidden_finite": bool(torch.isfinite(result.hidden).all()),
        "argmax": int(last.argmax()),
        "top5_ids": [int(x) for x in top_ids],
        "top5_values": [float(x) for x in top_values],
        "cache_bytes": {
            tag: int(size) for tag, size in result.cache.bytes_summary().items()
        },
        "checkpoint": str(checkpoint),
    }
    if not finite:
        summary["error"] = "logits contain non-finite values"
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    if synthetic_dir and not args.keep_checkpoint:
        import shutil

        shutil.rmtree(synthetic_dir, ignore_errors=True)
    elif synthetic_dir:
        print(f"synthetic checkpoint kept at {synthetic_dir}", file=sys.stderr)
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
