"""Real 3FS 40-layer 680.56B Qwen 4.5-Flash token generation & PD-disaggregation verifier.

Modes:
  - --mode unified-ep4: 4 GPUs (EP=4), reads directly from 3FS read-only checkpoint,
    runs greedy generation on real Chinese/English prompts using exact Qwen2BPETokenizer.
  - --mode pd-ep4x2: 8 GPUs (Ranks 0..3 Prefill EP=4 -> real NCCL CacheStore 8-tag P2P ->
    Ranks 4..7 Decode EP=4), verifies PD-disaggregated token generation on real 3FS weights.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import build_qwen4_5_kv_cache_plan
from rtp_llm.models.qwen4_5.qwen4_5_weight import validate_qwen4_5_weight_map
from rtp_llm.models_py.model_desc.qwen4_5 import Qwen4_5ForCausalLM
from rtp_llm.models_py.model_desc.qwen4_5_runtime import (
    LocalKVCache,
    LocalMemoryCacheStoreWriter,
    LocalPyModelInputs,
    QWEN4_5_CACHE_TAGS,
    Qwen4_5RTPModel,
    build_block_tables_for_requests,
    make_tagged_attention_inputs,
)
from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import (
    SafetensorsShardReader,
    load_qwen4_5_from_checkpoint,
)
from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig
from rtp_llm.models_py.modules.qwen4_5.tokenizer import Qwen2BPETokenizer

DEFAULT_3FS_CKPT = "/mnt/hf3fs/3fs/models/Qwen4.5-Flash-1109-bf16-re-mxfp4-ple-fp8"

CHAT_PROMPTS: List[str] = [
    "中国的首都是哪里？请用一句话直接回答。",
    "计算 17 * 24 + 58 等于多少？请直接给出计算步骤和最终结果。",
    "Write a concise Python function to check if a string is a palindrome.",
]

BASE_PROMPTS: List[str] = [
    "中国的首都是",
    "The capital of China is",
    'def is_palindrome(s: str) -> bool:\n    """Check whether string s is a palindrome."""\n',
    "17 * 24 = ",
]

DEFAULT_PROMPTS: List[str] = BASE_PROMPTS


def _select_prompts_and_encode(tok: Qwen2BPETokenizer, prompt_style: str) -> Tuple[List[str], List[List[int]]]:
    if prompt_style == "base":
        prompts = list(BASE_PROMPTS)
        prompt_ids = [tok.encode(p) for p in prompts]
    elif prompt_style == "chat":
        prompts = list(CHAT_PROMPTS)
        prompt_ids = [tok.encode_chat(p, enable_thinking=False) for p in prompts]
    else:
        raise ValueError(f"Unsupported prompt_style: {prompt_style}")
    return prompts, prompt_ids


def _validate_real_headers(ckpt_dir: Path, cfg: Qwen4_5ArchConfig) -> Dict[str, Any]:
    t0 = time.perf_counter()
    with SafetensorsShardReader(ckpt_dir) as reader:
        shards = sorted({s for s in reader.weight_map.values()})
        actual_specs: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
        for sname in shards:
            hdr, _ = reader._get_header(sname)
            for k, meta in hdr.items():
                if k != "__metadata__":
                    actual_specs[k] = (meta["dtype"], tuple(int(d) for d in meta["shape"]))
    res = validate_qwen4_5_weight_map(cfg, actual_specs)
    res["scan_sec"] = round(time.perf_counter() - t0, 3)
    res["shard_count"] = len(shards)
    return res


@torch.no_grad()
def _run_unified_ep4(
    ckpt_dir: Path,
    *,
    prompt_style: str,
    max_new_tokens: int,
    num_workers: int,
    out_json: Optional[Path],
) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 4, f"Expected 4 ranks for unified-ep4, got {world}"

    cfg = Qwen4_5ArchConfig.from_dict(json.loads((ckpt_dir / "config.json").read_text()))
    tok = Qwen2BPETokenizer(ckpt_dir)

    if rank == 0:
        val = _validate_real_headers(ckpt_dir, cfg)
        print(f"[Header Check] {val}", flush=True)
    else:
        val = {}
    dist.barrier()

    t_load0 = time.perf_counter()
    model = Qwen4_5ForCausalLM(
        cfg,
        use_triton_mxfp4=True,
        compute_mode="bf16",
        dtype=torch.bfloat16,
        device=device,
        init_mxfp4_weights=False,
        ep_size=4,
        ep_rank=rank,
        ep_group=dist.group.WORLD,
    )
    load_qwen4_5_from_checkpoint(
        model, ckpt_dir, ple_mode="gpu_sharded", num_workers=num_workers, verbose=True
    )
    dist.barrier()
    load_sec = time.perf_counter() - t_load0

    mem_gib = torch.cuda.memory_allocated(device) / (1024 ** 3)
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f"[Rank {rank}] Weights resident HBM={mem_gib:.2f} GiB (peak={peak_gib:.2f} GiB)", flush=True)

    rtp = Qwen4_5RTPModel.from_causal_lm(model)
    plan = build_qwen4_5_kv_cache_plan(cfg)
    tpb = 16
    kv_pool = LocalKVCache(
        plan,
        tokens_per_block=tpb,
        num_blocks_per_tag={t: 64 for t in QWEN4_5_CACHE_TAGS},
        device=device,
        dtype=torch.bfloat16,
    )
    rtp.kv_cache = kv_pool

    prompts, prompt_ids = _select_prompts_and_encode(tok, prompt_style)
    in_lens = [len(p) for p in prompt_ids]
    max_lens = [ln + max_new_tokens for ln in in_lens]
    tables = build_block_tables_for_requests(kv_pool, max_lens)
    flat_ids = torch.tensor([t for seq in prompt_ids for t in seq], dtype=torch.int64, device=device)

    torch.cuda.synchronize(device)
    t_p0 = time.perf_counter()
    p_attn = make_tagged_attention_inputs(
        tables,
        is_prefill=True,
        input_lengths=in_lens,
        sequence_lengths=in_lens,
        prefix_lengths=[0] * len(prompt_ids),
        device=device,
    )
    p_out = rtp.forward(LocalPyModelInputs(input_ids=flat_ids, attention_inputs=p_attn))
    last_idx = []
    acc = 0
    for ln in in_lens:
        acc += ln
        last_idx.append(acc - 1)
    first_logits = rtp.compute_logits(
        p_out.hidden_states[torch.tensor(last_idx, dtype=torch.long, device=device)]
    )
    assert torch.isfinite(first_logits).all(), "Non-finite logits detected on Prefill!"
    first_toks = torch.argmax(first_logits, dim=-1).tolist()
    torch.cuda.synchronize(device)
    prefill_ms = (time.perf_counter() - t_p0) * 1000.0

    gen_tokens: List[List[int]] = [[int(t)] for t in first_toks]
    cur_lens = list(in_lens)
    cur_toks = list(first_toks)
    dec_times_ms: List[float] = []
    for _ in range(max_new_tokens - 1):
        torch.cuda.synchronize(device)
        td0 = time.perf_counter()
        d_attn = make_tagged_attention_inputs(
            tables,
            is_prefill=False,
            input_lengths=[1] * len(prompt_ids),
            sequence_lengths=cur_lens,
            prefix_lengths=[0] * len(prompt_ids),
            device=device,
        )
        d_out = rtp.forward(
            LocalPyModelInputs(
                input_ids=torch.tensor(cur_toks, dtype=torch.int64, device=device),
                attention_inputs=d_attn,
            )
        )
        logits = rtp.compute_logits(d_out.hidden_states)
        assert torch.isfinite(logits).all(), "Non-finite logits detected on Decode!"
        cur_toks = torch.argmax(logits, dim=-1).tolist()
        torch.cuda.synchronize(device)
        dec_times_ms.append((time.perf_counter() - td0) * 1000.0)
        for b in range(len(prompt_ids)):
            gen_tokens[b].append(int(cur_toks[b]))
            cur_lens[b] += 1

    dist.barrier()
    if rank == 0:
        results = []
        print("\n=== REAL 3FS UNIFIED EP=4 GENERATION ===", flush=True)
        print(
            f"load={load_sec:.2f}s | prefill(B={len(prompt_ids)}, lens={in_lens})={prefill_ms:.1f}ms | decode avg TPOT={sum(dec_times_ms)/max(1,len(dec_times_ms)):.2f}ms/step",
            flush=True,
        )
        for b, prompt_txt in enumerate(prompts):
            dec_text = tok.decode(gen_tokens[b], skip_special_tokens=True, stop_at_eos=True)
            print(f"[Prompt {b}] {prompt_txt}", flush=True)
            print(f"  tokens[:16]={gen_tokens[b][:16]} | text={dec_text!r}", flush=True)
            results.append(
                {
                    "prompt": prompt_txt,
                    "prompt_len": in_lens[b],
                    "generated_token_ids": gen_tokens[b],
                    "decoded_text": dec_text,
                }
            )
        report = {
            "mode": "unified_ep4_real_3fs",
            "prompt_style": prompt_style,
            "ckpt_path": str(ckpt_dir),
            "header_validation": val,
            "load_sec": round(load_sec, 2),
            "resident_hbm_gib": round(mem_gib, 2),
            "prefill_ms": round(prefill_ms, 2),
            "decode_avg_ms_per_step": round(sum(dec_times_ms) / max(1, len(dec_times_ms)), 2),
            "outputs": results,
        }
        if out_json is not None:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Saved report to {out_json}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


@torch.no_grad()
def _run_pd_ep4x2(
    ckpt_dir: Path,
    *,
    prompt_style: str,
    max_new_tokens: int,
    num_workers: int,
    out_json: Optional[Path],
    unified_ref_json: Optional[Path],
) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 8, f"Expected 8 ranks (P EP=4 GPUs 0..3 + D EP=4 GPUs 4..7), got {world}"

    p_ep_group = dist.new_group([0, 1, 2, 3])
    d_ep_group = dist.new_group([4, 5, 6, 7])
    is_prefill_role = rank < 4
    ep_rank = rank if is_prefill_role else (rank - 4)
    ep_group = p_ep_group if is_prefill_role else d_ep_group

    cfg = Qwen4_5ArchConfig.from_dict(json.loads((ckpt_dir / "config.json").read_text()))
    tok = Qwen2BPETokenizer(ckpt_dir)

    t_load0 = time.perf_counter()
    model = Qwen4_5ForCausalLM(
        cfg,
        use_triton_mxfp4=True,
        compute_mode="bf16",
        dtype=torch.bfloat16,
        device=device,
        init_mxfp4_weights=False,
        ep_size=4,
        ep_rank=ep_rank,
        ep_group=ep_group,
    )
    load_qwen4_5_from_checkpoint(
        model, ckpt_dir, ple_mode="gpu_sharded", num_workers=num_workers, verbose=(rank in (0, 4))
    )
    dist.barrier()
    load_sec = time.perf_counter() - t_load0

    rtp = Qwen4_5RTPModel.from_causal_lm(model)
    plan = build_qwen4_5_kv_cache_plan(cfg)
    tpb = 16
    kv_pool = LocalKVCache(
        plan,
        tokens_per_block=tpb,
        num_blocks_per_tag={t: 64 for t in QWEN4_5_CACHE_TAGS},
        device=device,
        dtype=torch.bfloat16,
    )
    rtp.kv_cache = kv_pool

    prompts, prompt_ids = _select_prompts_and_encode(tok, prompt_style)
    in_lens = [len(p) for p in prompt_ids]
    max_lens = [ln + max_new_tokens for ln in in_lens]

    if is_prefill_role:
        p_tables = build_block_tables_for_requests(kv_pool, in_lens)
        writer = LocalMemoryCacheStoreWriter(record_payload=True)
        flat_ids = torch.tensor([t for seq in prompt_ids for t in seq], dtype=torch.int64, device=device)

        torch.cuda.synchronize(device)
        t_p0 = time.perf_counter()
        p_attn = make_tagged_attention_inputs(
            p_tables,
            is_prefill=True,
            input_lengths=in_lens,
            sequence_lengths=in_lens,
            prefix_lengths=[0] * len(prompt_ids),
            device=device,
            cache_store_writer=writer,
        )
        p_out = rtp.forward(LocalPyModelInputs(input_ids=flat_ids, attention_inputs=p_attn))
        last_idx = []
        acc = 0
        for ln in in_lens:
            acc += ln
            last_idx.append(acc - 1)
        first_logits = rtp.compute_logits(
            p_out.hidden_states[torch.tensor(last_idx, dtype=torch.long, device=device)]
        )
        assert torch.isfinite(first_logits).all(), "Non-finite logits on P Prefill!"
        first_toks = torch.argmax(first_logits, dim=-1).tolist()
        torch.cuda.synchronize(device)
        prefill_ms = (time.perf_counter() - t_p0) * 1000.0

        # Real peer-to-peer CacheStore transfer from Prefill rank `ep_rank` -> Decode rank `ep_rank + 4`
        peer = ep_rank + 4
        sorted_items = sorted(writer.published.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        t_tx0 = time.perf_counter()
        meta = [
            {
                "first_toks": first_toks,
                "p_tables": p_tables,
                "items": [(lid, tag, list(blks)) for (lid, tag), (blks, _) in sorted_items],
                "prefill_ms": prefill_ms,
            }
        ]
        dist.send_object_list(meta, dst=peer, device=device)
        tx_bytes = 0
        for (_, _), (_, payload) in sorted_items:
            assert payload is not None
            tx_bytes += int(payload.numel() * payload.element_size())
            dist.send(payload.contiguous(), dst=peer)
        torch.cuda.synchronize(device)
        tx_ms = (time.perf_counter() - t_tx0) * 1000.0
        pub_events = len(writer.events)
    else:
        peer = ep_rank
        meta_recv: List[Any] = [None]
        dist.recv_object_list(meta_recv, src=peer, device=device)
        m0 = meta_recv[0]
        first_toks = m0["first_toks"]
        p_tables = m0["p_tables"]
        items = m0["items"]
        prefill_ms = float(m0["prefill_ms"])

        t_rx0 = time.perf_counter()
        d_tables = build_block_tables_for_requests(kv_pool, in_lens)
        id_map_by_tag: Dict[str, Dict[int, int]] = {}
        for tag, s_rows in p_tables.items():
            mp: Dict[int, int] = {}
            for sr, dr in zip(s_rows, d_tables[tag]):
                for sb, db in zip(sr, dr):
                    mp[int(sb)] = int(db)
            id_map_by_tag[tag] = mp

        tx_bytes = 0
        for lid, tag, src_blks in items:
            dst_cache = kv_pool.get_layer_cache(lid, tag)
            buf = torch.empty(
                (len(src_blks), *dst_cache.kv_cache_base.shape[1:]),
                dtype=dst_cache.kv_cache_base.dtype,
                device=device,
            )
            tx_bytes += int(buf.numel() * buf.element_size())
            dist.recv(buf, src=peer)
            id_map = id_map_by_tag[tag]
            for i, sb in enumerate(src_blks):
                if sb in id_map:
                    dst_cache.kv_cache_base[id_map[sb]].copy_(buf[i])
        torch.cuda.synchronize(device)
        tx_ms = (time.perf_counter() - t_rx0) * 1000.0

        d_tables = build_block_tables_for_requests(kv_pool, max_lens, existing_tables=d_tables)
        gen_tokens = [[int(t)] for t in first_toks]
        cur_lens = list(in_lens)
        cur_toks = list(first_toks)
        dec_times_ms: List[float] = []
        for _ in range(max_new_tokens - 1):
            torch.cuda.synchronize(device)
            td0 = time.perf_counter()
            d_attn = make_tagged_attention_inputs(
                d_tables,
                is_prefill=False,
                input_lengths=[1] * len(prompt_ids),
                sequence_lengths=cur_lens,
                prefix_lengths=[0] * len(prompt_ids),
                device=device,
            )
            d_out = rtp.forward(
                LocalPyModelInputs(
                    input_ids=torch.tensor(cur_toks, dtype=torch.int64, device=device),
                    attention_inputs=d_attn,
                )
            )
            logits = rtp.compute_logits(d_out.hidden_states)
            assert torch.isfinite(logits).all(), "Non-finite logits on D Decode!"
            cur_toks = torch.argmax(logits, dim=-1).tolist()
            torch.cuda.synchronize(device)
            dec_times_ms.append((time.perf_counter() - td0) * 1000.0)
            for b in range(len(prompt_ids)):
                gen_tokens[b].append(int(cur_toks[b]))
                cur_lens[b] += 1

    dist.barrier()
    # Gather final Decode report onto rank 0
    if rank == 4:
        dist.send_object_list(
            [
                {
                    "gen_tokens": gen_tokens,
                    "rx_ms": tx_ms,
                    "dec_avg_ms": sum(dec_times_ms) / max(1, len(dec_times_ms)),
                    "tx_mib": tx_bytes / (1024 ** 2),
                    "prefill_ms": prefill_ms,
                    "num_items": len(items),
                }
            ],
            dst=0,
            device=device,
        )
    elif rank == 0:
        d_res_box: List[Any] = [None]
        dist.recv_object_list(d_res_box, src=4, device=device)
        d_res = d_res_box[0]
        pd_gen = d_res["gen_tokens"]

        ref_tokens = None
        if unified_ref_json is not None and unified_ref_json.exists():
            ref_obj = json.loads(unified_ref_json.read_text(encoding="utf-8"))
            ref_tokens = [item["generated_token_ids"] for item in ref_obj["outputs"]]

        print("\n=== REAL 3FS TRUE PD-DISAGGREGATED EP=4 (GPU0..3) -> NCCL -> EP=4 (GPU4..7) ===", flush=True)
        print(
            f"load={load_sec:.2f}s | prefill={d_res['prefill_ms']:.1f}ms | nccl_transfer({d_res['num_items']} published regions, {d_res['tx_mib']:.2f} MiB)={d_res['rx_ms']:.1f}ms | decode TPOT={d_res['dec_avg_ms']:.2f}ms/step | pub_events={pub_events}",
            flush=True,
        )
        outputs = []
        total_match = 0
        total_tok = len(prompts) * max_new_tokens
        for b, prompt_txt in enumerate(prompts):
            dec_text = tok.decode(pd_gen[b], skip_special_tokens=True, stop_at_eos=True)
            match_str = ""
            if ref_tokens is not None:
                m = sum(int(x == y) for x, y in zip(pd_gen[b], ref_tokens[b]))
                total_match += m
                match_str = f" | match_vs_unified={m}/{max_new_tokens}"
            print(f"[PD Prompt {b}] {prompt_txt}", flush=True)
            print(f"  tokens[:16]={pd_gen[b][:16]}{match_str} | text={dec_text!r}", flush=True)
            outputs.append(
                {
                    "prompt": prompt_txt,
                    "prompt_len": in_lens[b],
                    "generated_token_ids": pd_gen[b],
                    "decoded_text": dec_text,
                }
            )
        if ref_tokens is not None:
            print(f"[PD vs Unified EP=4] exact_token_match={total_match}/{total_tok}", flush=True)

        report = {
            "mode": "pd_ep4x2_real_3fs",
            "prompt_style": prompt_style,
            "ckpt_path": str(ckpt_dir),
            "load_sec": round(load_sec, 2),
            "prefill_ms": round(d_res["prefill_ms"], 2),
            "nccl_transfer_ms": round(d_res["rx_ms"], 2),
            "published_regions_per_rank": d_res["num_items"],
            "transfer_mib_per_rank": round(d_res["tx_mib"], 2),
            "decode_avg_ms_per_step": round(d_res["dec_avg_ms"], 2),
            "exact_match_vs_unified": f"{total_match}/{total_tok}" if ref_tokens is not None else "N/A",
            "outputs": outputs,
        }
        if out_json is not None:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Saved PD report to {out_json}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify real 3FS Qwen 4.5-Flash token generation")
    parser.add_argument("--ckpt-path", type=str, default=DEFAULT_3FS_CKPT)
    parser.add_argument("--mode", choices=["unified-ep4", "pd-ep4x2"], required=True)
    parser.add_argument("--prompt-style", choices=["base", "chat"], default="base")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--out-json", type=str, default="")
    parser.add_argument("--unified-ref-json", type=str, default="")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_path)
    out_json = Path(args.out_json) if args.out_json else None
    ref_json = Path(args.unified_ref_json) if args.unified_ref_json else None

    if args.mode == "unified-ep4":
        _run_unified_ep4(
            ckpt_dir,
            prompt_style=args.prompt_style,
            max_new_tokens=args.max_new_tokens,
            num_workers=args.num_workers,
            out_json=out_json,
        )
    else:
        _run_pd_ep4x2(
            ckpt_dir,
            prompt_style=args.prompt_style,
            max_new_tokens=args.max_new_tokens,
            num_workers=args.num_workers,
            out_json=out_json,
            unified_ref_json=ref_json,
        )


if __name__ == "__main__":
    main()
