"""Lightweight Accuracy & PD-Consistency Benchmark for Qwen 4.5-Flash.

Two execution modes:
  1. Default lightweight built-in mode (zero external files, ~10-20s on B300):
     Runs all 4 accuracy gates on an 8-layer 1109-isomorphic safetensors checkpoint:
       - Gate 1: Checkpoint 1109 Schema & Reload Parity
       - Gate 2: Triton/MXFP4 vs Pure-PyTorch FP32 Reference Logits Parity
       - Gate 3: PD-Disaggregated (GPU0->GPU1) vs Unified Autoregressive Token Generation
       - Gate 4: Prefix-Reuse + EOS-Boundary PLE PD Consistency
  2. Real checkpoint immediate serving & golden verification mode (--ckpt-path):
     Loads a real HuggingFace 1109 checkpoint (or 4-layer slice), executes real
     PD-disaggregated Prefill -> CacheStore -> Decode generation, prints generated
     tokens/text, and optionally validates against a HF golden JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import build_qwen4_5_kv_cache_plan
from rtp_llm.models_py.model_desc.qwen4_5 import Qwen4_5ForCausalLM
from rtp_llm.models_py.model_desc.qwen4_5_runtime import (
    LocalKVCache,
    LocalMemoryCacheStoreWriter,
    LocalPyModelInputs,
    PDDisaggregatedSession,
    QWEN4_5_CACHE_TAGS,
    Qwen4_5RTPModel,
    SimpleByteTokenizer,
    build_block_tables_for_requests,
    make_tagged_attention_inputs,
    transfer_published_blocks_to_decode,
)
from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import (
    load_qwen4_5_from_checkpoint,
    save_qwen4_5_safetensors_checkpoint,
)
from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig
from rtp_llm.models_py.modules.qwen4_5.test.test_e2e_qwen4_5_model import (
    _stabilize_router_margins,
    make_mini_qwen4_5_config,
)


def _compute_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    """Compare logits/tensors a (test) against b (reference)."""
    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()
    cos = float(F.cosine_similarity(af.reshape(1, -1), bf.reshape(1, -1)).item())

    # Symmetric KL divergence KL(softmax(b) || softmax(a)) over last vocab axis
    p_ref = F.softmax(bf.reshape(-1, bf.shape[-1]), dim=-1).clamp_min(1e-8)
    log_p_ref = torch.log(p_ref)
    log_p_test = F.log_softmax(af.reshape(-1, af.shape[-1]), dim=-1)
    kl = float((p_ref * (log_p_ref - log_p_test)).sum(dim=-1).mean().item())

    top1_a = torch.argmax(af, dim=-1)
    top1_b = torch.argmax(bf, dim=-1)
    top1_agree = float((top1_a == top1_b).float().mean().item())
    return {
        "max_abs": round(float(diff.max().item()), 6),
        "mean_abs": round(float(diff.mean().item()), 6),
        "cosine": round(cos, 8),
        "kl_div": round(kl, 8),
        "top1_agreement": round(top1_agree, 6),
    }


def _run_unified_baseline(
    model: Qwen4_5ForCausalLM,
    prompts: List[List[int]],
    max_new_tokens: int,
    *,
    enable_yoco_decode: bool = True,
) -> tuple[List[List[int]], List[torch.Tensor]]:
    device = model.model.embed_tokens.weight.device
    all_gen: List[List[int]] = [[] for _ in prompts]
    step_lists: List[List[torch.Tensor]] = [[] for _ in range(max_new_tokens)]
    for b, p_toks in enumerate(prompts):
        st = model.create_runtime_state(
            batch_size=1,
            max_seq_len=len(p_toks) + max_new_tokens + 4,
            device=device,
            enable_yoco_decode=enable_yoco_decode,
        )
        cur = torch.tensor([p_toks], dtype=torch.int64, device=device)
        for s in range(max_new_tokens):
            lg = model(
                cur,
                state=st,
                logits_to_keep=1,
                enable_yoco_decode=enable_yoco_decode,
            )[:, 0, :]
            step_lists[s].append(lg.detach().cpu())
            nxt = int(torch.argmax(lg, dim=-1).item())
            all_gen[b].append(nxt)
            cur = torch.tensor([[nxt]], dtype=torch.int64, device=device)
    return all_gen, [torch.cat(step_lists[s], dim=0) for s in range(max_new_tokens)]


@torch.no_grad()
def run_lightweight_suite() -> Dict[str, Any]:
    """Run all 4 lightweight accuracy & PD consistency gates on an 8-layer 1109 schema."""
    t_start = time.perf_counter()
    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1") if torch.cuda.device_count() >= 2 else dev0
    cfg = make_mini_qwen4_5_config()
    tok = SimpleByteTokenizer(
        cfg.vocab_size, bos_id=cfg.bos_token_id, eos_id=cfg.eos_token_id
    )

    p_causal = Qwen4_5ForCausalLM(cfg, dtype=torch.bfloat16, device=dev0)
    p_causal.init_synthetic_weights(seed=202611, ple_mode="gpu_full")
    _stabilize_router_margins(p_causal)
    p_causal.prepare_weights()

    gates: Dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Gate 1: Checkpoint 1109 Schema & Exact Reload
        ckpt_report = save_qwen4_5_safetensors_checkpoint(p_causal, tmpdir)
        d_causal = Qwen4_5ForCausalLM(cfg, dtype=torch.bfloat16, device=dev1)
        load_qwen4_5_from_checkpoint(d_causal, tmpdir, ple_mode="gpu_full")

        g_inp = torch.Generator(device=dev0).manual_seed(202611)
        test_inp = torch.randint(10, cfg.vocab_size, (2, 20), device=dev0, generator=g_inp)
        st0 = p_causal.create_runtime_state(2, 32)
        logits_src = p_causal(test_inp, state=st0)

        # Reload back onto dev0 to compare numerical identity
        chk_causal = Qwen4_5ForCausalLM(cfg, dtype=torch.bfloat16, device=dev0)
        load_qwen4_5_from_checkpoint(chk_causal, tmpdir, ple_mode="gpu_full")
        st1 = chk_causal.create_runtime_state(2, 32)
        logits_reload = chk_causal(test_inp, state=st1)
        m1 = _compute_metrics(logits_reload, logits_src)
        g1_pass = (
            ckpt_report["missing"] == []
            and ckpt_report["unexpected"] == []
            and m1["max_abs"] == 0.0
        )
        gates["Gate_1_Checkpoint_1109_Schema_Reload"] = {
            "pass": g1_pass,
            "num_tensors": ckpt_report["num_matched"],
            "file_bytes": ckpt_report["total_file_bytes"],
            **m1,
        }

        # Gate 2: Triton/MXFP4 vs Pure-PyTorch FP32 Reference Prefill + Direct Layer-1 LatentMoE MXFP4 Check
        st_tri = p_causal.create_runtime_state(2, 32)
        st_ref = p_causal.create_runtime_state(2, 32)
        l_tri = p_causal(test_inp, state=st_tri, use_reference=False)
        l_ref = p_causal(test_inp, state=st_ref, use_reference=True)
        m2 = _compute_metrics(l_tri, l_ref)

        moe_in = (
            torch.randn(
                40,
                cfg.hidden_size,
                dtype=torch.bfloat16,
                device=dev0,
                generator=g_inp,
            )
            * 0.1
        )
        moe_tri = p_causal.model.layers[1].mlp(moe_in, use_triton_mxfp4=True)
        moe_ref = p_causal.model.layers[1].mlp(moe_in, use_triton_mxfp4=False)
        moe_cos = float(
            F.cosine_similarity(
                moe_tri.float().reshape(1, -1), moe_ref.float().reshape(1, -1)
            ).item()
        )
        moe_max_abs = float((moe_tri.float() - moe_ref.float()).abs().max().item())

        g2_pass = (
            m2["cosine"] >= 0.999
            and m2["kl_div"] < 1e-3
            and m2["max_abs"] < 0.15
            and moe_cos >= 0.9999
            and moe_max_abs < 0.01
        )
        gates["Gate_2_Triton_MXFP4_vs_PureTorch_Ref"] = {
            "pass": g2_pass,
            **m2,
            "moe_triton_vs_dequant_cos": round(moe_cos, 8),
            "moe_triton_vs_dequant_max_abs": round(moe_max_abs, 6),
        }

        # Gate 3: PD-Disaggregated (GPU0 -> GPU1) vs Unified Generation (Ragged + Cross-Page)
        p_rtp = Qwen4_5RTPModel.from_causal_lm(p_causal)
        d_rtp = Qwen4_5RTPModel.from_causal_lm(d_causal)
        eos = cfg.eos_token_id
        prompts = [
            tok.encode("Qwen 4.5 PD serving baseline gate!") + [eos, 29, 31],
            tok.encode("Multi-page ragged prompt crossing 16 tokens!") + [eos],
            tok.encode("B=3 decode accuracy"),
        ]
        max_new = 16
        ref_toks, ref_lgs = _run_unified_baseline(
            p_causal, prompts, max_new, enable_yoco_decode=True
        )
        sess_y = PDDisaggregatedSession(
            p_rtp, d_rtp, tokens_per_block=16, max_blocks_per_tag=128
        )
        pd_toks, pd_lgs = sess_y.generate(
            prompts, max_new_tokens=max_new, enable_yoco_decode=True
        )
        sess_ny = PDDisaggregatedSession(
            p_rtp,
            d_rtp,
            tokens_per_block=16,
            max_blocks_per_tag=128,
            enable_yoco_decode=False,
        )
        ny_toks, ny_lgs = sess_ny.generate(
            prompts, max_new_tokens=max_new, enable_yoco_decode=False
        )

        m3 = _compute_metrics(torch.stack(pd_lgs, dim=0), torch.stack(ref_lgs, dim=0))
        yoco_diff = float(
            (torch.stack(pd_lgs).float() - torch.stack(ny_lgs).float())
            .abs()
            .max()
            .item()
        )
        exact_toks = sum(
            int(pd_toks[b][s] == ref_toks[b][s])
            for b in range(len(prompts))
            for s in range(max_new)
        )
        total_toks = len(prompts) * max_new
        g3_pass = (
            exact_toks == total_toks
            and pd_toks == ny_toks
            and m3["cosine"] >= 0.9995
            and m3["kl_div"] < 1e-4
            and yoco_diff == 0.0
        )
        gates["Gate_3_PD_Disagg_vs_Unified_Generation"] = {
            "pass": g3_pass,
            "pd_devices": f"{dev0}->{dev1}",
            "batch_size": len(prompts),
            "steps_per_req": max_new,
            "exact_token_match": f"{exact_toks}/{total_toks}",
            "yoco_on_vs_off_max_abs": round(yoco_diff, 6),
            "cache_store_events": len(sess_y.writer.events),
            **m3,
        }

        # Gate 4: Prefix Reuse + EOS-Boundary PLE PD Consistency
        prefix = tok.encode("System prompt prefix len >= 16 bytes")[:16]
        suffix = [eos] + tok.encode(" user query!")
        full_p = prefix + suffix
        pfx_ref_toks, pfx_ref_lgs = _run_unified_baseline(
            p_causal, [full_p], max_new_tokens=8
        )

        plan = build_qwen4_5_kv_cache_plan(cfg)
        p_kv = LocalKVCache(
            plan,
            tokens_per_block=16,
            num_blocks_per_tag={t: 64 for t in QWEN4_5_CACHE_TAGS},
            device=dev0,
            dtype=torch.bfloat16,
        )
        d_kv = LocalKVCache(
            plan,
            tokens_per_block=16,
            num_blocks_per_tag={t: 64 for t in QWEN4_5_CACHE_TAGS},
            device=dev1,
            dtype=torch.bfloat16,
        )
        p_rtp.kv_cache = p_kv
        d_rtp.kv_cache = d_kv

        tbl_p = build_block_tables_for_requests(p_kv, [len(prefix)])
        _ = p_rtp.forward(
            LocalPyModelInputs(
                input_ids=torch.tensor(prefix, dtype=torch.int64, device=dev0),
                attention_inputs=make_tagged_attention_inputs(
                    tbl_p,
                    is_prefill=True,
                    input_lengths=[len(prefix)],
                    sequence_lengths=[len(prefix)],
                    prefix_lengths=[0],
                    device=dev0,
                ),
            )
        )
        tbl_p = build_block_tables_for_requests(
            p_kv, [len(full_p)], existing_tables=tbl_p
        )
        writer4 = LocalMemoryCacheStoreWriter(record_payload=True)
        out4 = p_rtp.forward(
            LocalPyModelInputs(
                input_ids=torch.tensor(suffix, dtype=torch.int64, device=dev0),
                attention_inputs=make_tagged_attention_inputs(
                    tbl_p,
                    is_prefill=True,
                    input_lengths=[len(suffix)],
                    sequence_lengths=[len(full_p)],
                    prefix_lengths=[len(prefix)],
                    device=dev0,
                    cache_store_writer=writer4,
                ),
            )
        )
        lg0 = p_rtp.compute_logits(out4.hidden_states[-1:])
        cur_tok = int(torch.argmax(lg0, dim=-1).item())
        pfx_pd_toks = [cur_tok]
        pfx_pd_lgs = [lg0.cpu()]
        tbl_d = transfer_published_blocks_to_decode(
            writer4.published, tbl_p, d_kv, [len(full_p)]
        )
        tbl_d = build_block_tables_for_requests(
            d_kv, [len(full_p) + 8], existing_tables=tbl_d
        )
        cur_len = len(full_p)
        for _ in range(7):
            d_out = d_rtp.forward(
                LocalPyModelInputs(
                    input_ids=torch.tensor([cur_tok], dtype=torch.int64, device=dev1),
                    attention_inputs=make_tagged_attention_inputs(
                        tbl_d,
                        is_prefill=False,
                        input_lengths=[1],
                        sequence_lengths=[cur_len],
                        prefix_lengths=[0],
                        device=dev1,
                    ),
                )
            )
            lg = d_rtp.compute_logits(d_out.hidden_states)
            cur_tok = int(torch.argmax(lg, dim=-1).item())
            pfx_pd_toks.append(cur_tok)
            pfx_pd_lgs.append(lg.cpu())
            cur_len += 1

        m4 = _compute_metrics(torch.stack(pfx_pd_lgs), torch.stack(pfx_ref_lgs))
        g4_pass = (
            pfx_pd_toks == pfx_ref_toks[0]
            and m4["cosine"] >= 0.999
            and m4["kl_div"] < 1e-3
        )
        gates["Gate_4_Prefix_Reuse_EOS_PLE_PD"] = {
            "pass": g4_pass,
            "exact_token_match": f"{sum(a == b for a, b in zip(pfx_pd_toks, pfx_ref_toks[0]))}/8",
            **m4,
        }

    elapsed = time.perf_counter() - t_start
    all_passed = all(g["pass"] for g in gates.values())
    return {
        "mode": "lightweight_builtin_8layer_1109_schema",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "gpu_name": torch.cuda.get_device_name(dev0),
        "elapsed_sec": round(elapsed, 2),
        "overall_pass": all_passed,
        "gates": gates,
    }


def _load_tokenizer(ckpt_path: Path, cfg: Qwen4_5ArchConfig) -> Any:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            str(ckpt_path), trust_remote_code=True, local_files_only=True
        )
    except Exception:
        return SimpleByteTokenizer(
            cfg.vocab_size, bos_id=cfg.bos_token_id, eos_id=cfg.eos_token_id
        )


@torch.no_grad()
def run_real_ckpt_serving_check(
    ckpt_path: Path,
    prompt: str,
    max_new_tokens: int,
    hf_golden_json: Optional[Path],
) -> Dict[str, Any]:
    """Load a real checkpoint, run immediate PD Prefill->Decode token generation, check golden."""
    t0 = time.perf_counter()
    raw_cfg = json.loads((ckpt_path / "config.json").read_text(encoding="utf-8"))
    cfg = Qwen4_5ArchConfig.from_dict(raw_cfg)

    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1") if torch.cuda.device_count() >= 2 else dev0

    p_model = Qwen4_5ForCausalLM(cfg, dtype=torch.bfloat16, device=dev0)
    load_qwen4_5_from_checkpoint(p_model, ckpt_path, ple_mode="cpu_uva")
    if dev1 != dev0:
        d_model = Qwen4_5ForCausalLM(cfg, dtype=torch.bfloat16, device=dev1)
        load_qwen4_5_from_checkpoint(d_model, ckpt_path, ple_mode="cpu_uva")
    else:
        d_model = p_model

    tok = _load_tokenizer(ckpt_path, cfg)
    enc = tok.encode(prompt)
    prompt_ids: List[int] = enc if isinstance(enc, list) else list(enc)

    p_rtp = Qwen4_5RTPModel.from_causal_lm(p_model)
    d_rtp = Qwen4_5RTPModel.from_causal_lm(d_model)
    sess = PDDisaggregatedSession(p_rtp, d_rtp, tokens_per_block=64, max_blocks_per_tag=256)
    gen_toks, pd_lgs = sess.generate([prompt_ids], max_new_tokens=max_new_tokens)
    decoded_text = tok.decode(gen_toks[0])

    result: Dict[str, Any] = {
        "mode": "real_checkpoint_pd_serving",
        "ckpt_path": str(ckpt_path),
        "pd_devices": f"{dev0}->{dev1}",
        "prompt": prompt,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": gen_toks[0],
        "generated_text": decoded_text,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "overall_pass": all(torch.isfinite(x).all().item() for x in pd_lgs),
    }

    if hf_golden_json is not None:
        golden = json.loads(hf_golden_json.read_text(encoding="utf-8"))
        # Optional golden schema: {"first_token_logits": [...], "expected_tokens": [...]}
        if "first_token_logits" in golden:
            g_lgs = torch.tensor(golden["first_token_logits"], dtype=torch.float32).reshape(1, -1)
            result["golden_first_step_metrics"] = _compute_metrics(pd_lgs[0].float(), g_lgs)
            result["overall_pass"] = result["overall_pass"] and (
                result["golden_first_step_metrics"]["cosine"] >= 0.99
            )
        if "expected_tokens" in golden:
            exp = golden["expected_tokens"][: len(gen_toks[0])]
            result["golden_token_match"] = f"{sum(a == b for a, b in zip(gen_toks[0], exp))}/{len(exp)}"
            result["overall_pass"] = result["overall_pass"] and (gen_toks[0] == exp)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen 4.5-Flash Lightweight Accuracy & PD Benchmark")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Optional real HF checkpoint directory")
    parser.add_argument("--prompt", type=str, default="你好，请简要介绍一下混合线性注意力模型。", help="Prompt for --ckpt-path mode")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--hf-golden-json", type=str, default=None)
    parser.add_argument(
        "--out-json",
        type=str,
        default="/data0/luohaocheng.lhc/artifacts/qwen45-flash-20261123/accuracy_benchmark_report.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen 4.5-Flash accuracy benchmark")

    if args.ckpt_path:
        report = run_real_ckpt_serving_check(
            Path(args.ckpt_path),
            args.prompt,
            args.max_new_tokens,
            Path(args.hf_golden_json) if args.hf_golden_json else None,
        )
    else:
        report = run_lightweight_suite()

    print("=== Qwen 4.5-Flash Lightweight Accuracy Benchmark Report ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out_json and os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR") is None:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved report to {out_p}")

    if not report["overall_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
