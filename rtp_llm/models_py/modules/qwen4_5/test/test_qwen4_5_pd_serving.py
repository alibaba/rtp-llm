"""PD-disaggregated serving, paged CacheStore publication, and real 2-process NCCL tests."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import (
    FULL_ATTN_KV_TAG,
    PLE_STATE_TAG,
    build_qwen4_5_kv_cache_plan,
)
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
    prepare_qwen4_5_cache_store_writers,
    publish_qwen4_5_layer_caches,
    transfer_published_blocks_to_decode,
    validate_qwen4_5_pd_cache_inputs,
)
from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import (
    load_qwen4_5_from_checkpoint,
    save_qwen4_5_safetensors_checkpoint,
)
from rtp_llm.models_py.modules.qwen4_5.test.test_e2e_qwen4_5_model import (
    _stabilize_router_margins,
    make_mini_qwen4_5_config,
)


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item()
    )


def _run_unified_baseline(
    model: Qwen4_5ForCausalLM,
    prompts: list[list[int]],
    max_new_tokens: int,
    *,
    enable_yoco_decode: bool = True,
) -> tuple[list[list[int]], list[torch.Tensor]]:
    """Run independent per-request full-sequence autoregressive generation as exact reference."""
    device = model.model.embed_tokens.weight.device
    all_gen: list[list[int]] = [[] for _ in prompts]
    per_step_logits: list[list[torch.Tensor]] = [[] for _ in range(max_new_tokens)]
    for b, p_toks in enumerate(prompts):
        max_len = len(p_toks) + max_new_tokens + 4
        st = model.create_runtime_state(
            batch_size=1,
            max_seq_len=max_len,
            device=device,
            enable_yoco_decode=enable_yoco_decode,
        )
        cur_in = torch.tensor([p_toks], dtype=torch.int64, device=device)
        for step in range(max_new_tokens):
            logits = model(
                cur_in,
                state=st,
                logits_to_keep=1,
                enable_yoco_decode=enable_yoco_decode,
            )[:, 0, :]
            per_step_logits[step].append(logits.detach().cpu())
            nxt = int(torch.argmax(logits, dim=-1).item())
            all_gen[b].append(nxt)
            cur_in = torch.tensor([[nxt]], dtype=torch.int64, device=device)

    step_tensors = [torch.cat(per_step_logits[s], dim=0) for s in range(max_new_tokens)]
    return all_gen, step_tensors


def _run_unified_rtp_baseline(
    model: Qwen4_5ForCausalLM,
    prompts: list[list[int]],
    max_new_tokens: int,
    *,
    tokens_per_block: int = 16,
) -> tuple[list[list[int]], list[torch.Tensor]]:
    """Run batch-matched unified Qwen4_5RTPModel reference (identical GEMM batch shape)."""
    device = model.model.embed_tokens.weight.device
    rtp = Qwen4_5RTPModel.from_causal_lm(model)
    plan = build_qwen4_5_kv_cache_plan(model.config)
    kv_pool = LocalKVCache(
        plan,
        tokens_per_block=tokens_per_block,
        num_blocks_per_tag={t: 128 for t in QWEN4_5_CACHE_TAGS},
        device=device,
        dtype=torch.bfloat16,
    )
    rtp.kv_cache = kv_pool
    in_lens = [len(seq) for seq in prompts]
    tables = build_block_tables_for_requests(
        kv_pool, [ln + max_new_tokens for ln in in_lens]
    )
    flat_ids = torch.tensor(
        [t for seq in prompts for t in seq], dtype=torch.int64, device=device
    )
    p_attn = make_tagged_attention_inputs(
        tables,
        is_prefill=True,
        input_lengths=in_lens,
        sequence_lengths=in_lens,
        prefix_lengths=[0] * len(prompts),
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
    first_toks = torch.argmax(first_logits, dim=-1).tolist()
    gen_tokens = [[int(t)] for t in first_toks]
    step_logits: list[torch.Tensor] = [first_logits.detach().cpu()]
    cur_lens = list(in_lens)
    cur_toks = list(first_toks)
    for _ in range(max_new_tokens - 1):
        d_attn = make_tagged_attention_inputs(
            tables,
            is_prefill=False,
            input_lengths=[1] * len(prompts),
            sequence_lengths=cur_lens,
            prefix_lengths=[0] * len(prompts),
            device=device,
        )
        d_out = rtp.forward(
            LocalPyModelInputs(
                input_ids=torch.tensor(cur_toks, dtype=torch.int64, device=device),
                attention_inputs=d_attn,
            )
        )
        logits = rtp.compute_logits(d_out.hidden_states)
        step_logits.append(logits.detach().cpu())
        cur_toks = torch.argmax(logits, dim=-1).tolist()
        for b in range(len(prompts)):
            gen_tokens[b].append(int(cur_toks[b]))
            cur_lens[b] += 1
    return gen_tokens, step_logits


class TestQwen4_5PDServing(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required for Qwen 4.5-Flash PD tests")
        cls.cfg = make_mini_qwen4_5_config()
        cls.dev0 = torch.device("cuda:0")
        cls.dev1 = torch.device("cuda:1") if torch.cuda.device_count() >= 2 else cls.dev0

    @torch.no_grad()
    def test_01_safetensors_checkpoint_roundtrip_1109_schema(self) -> None:
        src_model = Qwen4_5ForCausalLM(self.cfg, dtype=torch.bfloat16, device=self.dev0)
        src_model.init_synthetic_weights(seed=1109, ple_mode="gpu_full")
        _stabilize_router_margins(src_model)
        src_model.prepare_weights()

        with tempfile.TemporaryDirectory() as tmpdir:
            report = save_qwen4_5_safetensors_checkpoint(src_model, tmpdir)
            self.assertEqual(report["missing"], [])
            self.assertEqual(report["unexpected"], [])
            self.assertEqual(report["num_matched"], report["num_expected"])

            dst_model = Qwen4_5ForCausalLM(self.cfg, dtype=torch.bfloat16, device=self.dev0)
            load_qwen4_5_from_checkpoint(dst_model, tmpdir, ple_mode="gpu_full")

        inp = torch.randint(10, self.cfg.vocab_size, (2, 18), device=self.dev0)
        st_a = src_model.create_runtime_state(2, 32)
        st_b = dst_model.create_runtime_state(2, 32)
        out_a = src_model(inp, state=st_a)
        out_b = dst_model(inp, state=st_b)
        max_abs = float((out_a.float() - out_b.float()).abs().max().item())
        print(
            f"[Checkpoint Roundtrip 1109] matched={report['num_matched']} "
            f"reload_max_abs={max_abs:.6f}"
        )
        self.assertEqual(max_abs, 0.0)

    def test_02_pd_cache_store_publication_contract(self) -> None:
        events: list[tuple[str, str, torch.Tensor]] = []
        writer = SimpleNamespace(
            write=lambda cs_in, kv: events.append(
                (cs_in.tag, kv.tag, kv.kv_cache_base.clone())
            )
        )

        def make_inp(tag: str, *, store: bool = True, **overrides):
            obj = SimpleNamespace(
                is_prefill=True,
                is_cuda_graph=False,
                is_target_verify=False,
                cache_store_inputs=SimpleNamespace(tag=tag) if store else None,
                cache_store_writer=writer,
            )
            obj.__dict__.update(overrides)
            return obj

        tags = sorted(QWEN4_5_CACHE_TAGS)
        caches = [SimpleNamespace(tag=t, kv_cache_base=torch.zeros(4)) for t in tags]
        kv_mock = SimpleNamespace(
            get_layer_cache_groups=lambda layer: caches[:4] if layer == 0 else caches[4:],
            get_layer_cache=lambda layer, tag: next(c for c in caches if c.tag == tag),
        )
        inputs = {t: make_inp(t) for t in tags}

        broken = dict(inputs)
        broken[tags[0]] = make_inp(tags[0], store=False)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            prepare_qwen4_5_cache_store_writers(broken, kv_mock, layer_count=2)
        self.assertEqual(events, [])

        for ov, msg in (
            ({"cache_store_writer": None}, "writer is missing"),
            ({"is_prefill": False}, "prefill inputs"),
            ({"is_cuda_graph": True}, "eager prefill"),
            ({"is_target_verify": True}, "target-verify"),
        ):
            with self.assertRaisesRegex(RuntimeError, msg):
                validate_qwen4_5_pd_cache_inputs((make_inp(FULL_ATTN_KV_TAG, **ov),))
        self.assertEqual(events, [])

        missing_one = dict(inputs)
        del missing_one[PLE_STATE_TAG]
        with self.assertRaisesRegex(RuntimeError, "missing"):
            prepare_qwen4_5_cache_store_writers(missing_one, kv_mock, layer_count=2)

        writers = prepare_qwen4_5_cache_store_writers(
            inputs,
            kv_mock,
            layer_count=2,
            write_op_factory=lambda val, cache: (val.cache_store_writer, val.cache_store_inputs),
        )
        self.assertEqual(set(writers), set(tags))
        for c in caches[:4]:
            c.kv_cache_base.fill_(7.0)
        publish_qwen4_5_layer_caches(kv_mock, 0, writers)
        self.assertEqual(len(events), 4)
        self.assertTrue(all(ev[0] == ev[1] and torch.all(ev[2] == 7.0) for ev in events))

    @torch.no_grad()
    def test_03_ragged_batch_pd_disagg_token_generation(self) -> None:
        p_causal = Qwen4_5ForCausalLM(self.cfg, dtype=torch.bfloat16, device=self.dev0)
        p_causal.init_synthetic_weights(seed=2027, ple_mode="gpu_full")
        _stabilize_router_margins(p_causal)
        p_causal.prepare_weights()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_qwen4_5_safetensors_checkpoint(p_causal, tmpdir)
            d_causal = Qwen4_5ForCausalLM(self.cfg, dtype=torch.bfloat16, device=self.dev1)
            load_qwen4_5_from_checkpoint(d_causal, tmpdir, ple_mode="gpu_full")

        p_rtp = Qwen4_5RTPModel.from_causal_lm(p_causal)
        d_rtp = Qwen4_5RTPModel.from_causal_lm(d_causal)
        tok = SimpleByteTokenizer(
            self.cfg.vocab_size,
            bos_id=self.cfg.bos_token_id,
            eos_id=self.cfg.eos_token_id,
        )

        # Ragged prompts crossing tokens_per_block=16 boundaries AND containing EOS tokens
        eos = self.cfg.eos_token_id
        prompts = [
            tok.encode("Hello Qwen 4.5 PD!") + [eos, 45, 77, 88],
            tok.encode("Second request across 2 KV pages now!") + [eos],
            tok.encode("Ragged short B=3") + [33, 99],
        ]
        max_new = 16

        ref_tokens, ref_logits = _run_unified_baseline(
            p_causal, prompts, max_new, enable_yoco_decode=True
        )

        session_yoco = PDDisaggregatedSession(
            p_rtp, d_rtp, tokens_per_block=16, max_blocks_per_tag=128
        )
        pd_tokens, pd_logits = session_yoco.generate(
            prompts, max_new_tokens=max_new, enable_yoco_decode=True
        )

        session_noyoco = PDDisaggregatedSession(
            p_rtp,
            d_rtp,
            tokens_per_block=16,
            max_blocks_per_tag=128,
            enable_yoco_decode=False,
        )
        ny_tokens, ny_logits = session_noyoco.generate(
            prompts, max_new_tokens=max_new, enable_yoco_decode=False
        )

        total = len(prompts) * max_new
        matched = sum(
            int(pd_tokens[b][s] == ref_tokens[b][s])
            for b in range(len(prompts))
            for s in range(max_new)
        )
        min_cos = min(_cos_sim(pd_logits[s], ref_logits[s]) for s in range(max_new))
        max_abs = max(
            float((pd_logits[s].float() - ref_logits[s].float()).abs().max().item())
            for s in range(max_new)
        )
        yoco_diff = max(
            float((pd_logits[s].float() - ny_logits[s].float()).abs().max().item())
            for s in range(max_new)
        )
        print(
            f"[PD Disagg {self.dev0}->{self.dev1} Ragged B={len(prompts)}x{max_new}] "
            f"tokens={matched}/{total}, min_cos={min_cos:.6f}, max_abs={max_abs:.6f}, "
            f"YoCo_ON_vs_OFF={yoco_diff:.6f} | published_events={len(session_yoco.writer.events)}"
        )
        self.assertEqual(matched, total)
        self.assertEqual(pd_tokens, ny_tokens)
        self.assertGreater(min_cos, 0.9995)
        self.assertLess(max_abs, 0.08)
        self.assertEqual(yoco_diff, 0.0)

    @torch.no_grad()
    def test_04_prefix_cache_reuse_pd_equivalence(self) -> None:
        causal = Qwen4_5ForCausalLM(self.cfg, dtype=torch.bfloat16, device=self.dev0)
        causal.init_synthetic_weights(seed=3031, ple_mode="gpu_full")
        _stabilize_router_margins(causal)
        causal.prepare_weights()

        tok = SimpleByteTokenizer(
            self.cfg.vocab_size,
            bos_id=self.cfg.bos_token_id,
            eos_id=self.cfg.eos_token_id,
        )
        prefix = tok.encode("System prompt shared prefix over 16")[:16]
        suffix = tok.encode(" user query!")
        full_prompt = prefix + suffix

        ref_tokens, ref_logits = _run_unified_baseline(
            causal, [full_prompt], max_new_tokens=8
        )

        rtp = Qwen4_5RTPModel.from_causal_lm(causal)
        plan = build_qwen4_5_kv_cache_plan(self.cfg)
        p_kv = LocalKVCache(
            plan,
            tokens_per_block=16,
            num_blocks_per_tag={t: 64 for t in QWEN4_5_CACHE_TAGS},
            device=self.dev0,
            dtype=torch.bfloat16,
        )
        d_kv = LocalKVCache(
            plan,
            tokens_per_block=16,
            num_blocks_per_tag={t: 64 for t in QWEN4_5_CACHE_TAGS},
            device=self.dev1,
            dtype=torch.bfloat16,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            save_qwen4_5_safetensors_checkpoint(causal, tmpdir)
            d_causal = Qwen4_5ForCausalLM(self.cfg, dtype=torch.bfloat16, device=self.dev1)
            load_qwen4_5_from_checkpoint(d_causal, tmpdir, ple_mode="gpu_full")
        d_rtp = Qwen4_5RTPModel.from_causal_lm(d_causal, kv_cache=d_kv)
        rtp.kv_cache = p_kv

        # Step A: Prefill prefix of length 16
        tbl_p = build_block_tables_for_requests(p_kv, [len(prefix)])
        attn_pfx = make_tagged_attention_inputs(
            tbl_p,
            is_prefill=True,
            input_lengths=[len(prefix)],
            sequence_lengths=[len(prefix)],
            prefix_lengths=[0],
            device=self.dev0,
        )
        _ = rtp.forward(
            LocalPyModelInputs(
                input_ids=torch.tensor(prefix, dtype=torch.int64, device=self.dev0),
                attention_inputs=attn_pfx,
            )
        )

        # Step B: Extend suffix reusing prefix page 0 -> publish -> transfer to D
        tbl_p = build_block_tables_for_requests(
            p_kv, [len(full_prompt)], existing_tables=tbl_p
        )
        writer = LocalMemoryCacheStoreWriter(record_payload=True)
        attn_suf = make_tagged_attention_inputs(
            tbl_p,
            is_prefill=True,
            input_lengths=[len(suffix)],
            sequence_lengths=[len(full_prompt)],
            prefix_lengths=[len(prefix)],
            device=self.dev0,
            cache_store_writer=writer,
        )
        out_suf = rtp.forward(
            LocalPyModelInputs(
                input_ids=torch.tensor(suffix, dtype=torch.int64, device=self.dev0),
                attention_inputs=attn_suf,
            )
        )
        logits0 = rtp.compute_logits(out_suf.hidden_states[-1:])
        tok0 = int(torch.argmax(logits0, dim=-1).item())
        pd_tokens = [tok0]
        pd_logits = [logits0.cpu()]

        tbl_d = transfer_published_blocks_to_decode(
            writer.published, tbl_p, d_kv, [len(full_prompt)]
        )
        tbl_d = build_block_tables_for_requests(
            d_kv, [len(full_prompt) + 8], existing_tables=tbl_d
        )
        cur_len = len(full_prompt)
        cur_tok = tok0
        for _ in range(7):
            d_attn = make_tagged_attention_inputs(
                tbl_d,
                is_prefill=False,
                input_lengths=[1],
                sequence_lengths=[cur_len],
                prefix_lengths=[0],
                device=self.dev1,
            )
            d_out = d_rtp.forward(
                LocalPyModelInputs(
                    input_ids=torch.tensor([cur_tok], dtype=torch.int64, device=self.dev1),
                    attention_inputs=d_attn,
                )
            )
            lg = d_rtp.compute_logits(d_out.hidden_states)
            cur_tok = int(torch.argmax(lg, dim=-1).item())
            pd_tokens.append(cur_tok)
            pd_logits.append(lg.cpu())
            cur_len += 1

        min_cos = min(_cos_sim(pd_logits[s], ref_logits[s]) for s in range(8))
        print(
            f"[Prefix Reuse + PD Transfer] tokens={pd_tokens} == ref={ref_tokens[0]} | "
            f"min_cos={min_cos:.6f}"
        )
        self.assertEqual(pd_tokens, ref_tokens[0])
        self.assertGreater(min_cos, 0.999)

    @torch.no_grad()
    def test_05_decode_rolling_opaque_pool_and_fp8_kv_pd_equivalence(self) -> None:
        """Verify Decode fixed rolling OPAQUE_STATE pool + single tail-block handoff + FP8 KV."""
        p_causal = Qwen4_5ForCausalLM(
            self.cfg, dtype=torch.bfloat16, kv_dtype="fp8_e4m3", device=self.dev0
        )
        p_causal.init_synthetic_weights(seed=4049, ple_mode="gpu_full")
        _stabilize_router_margins(p_causal)
        p_causal.prepare_weights()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_qwen4_5_safetensors_checkpoint(p_causal, tmpdir)
            d_causal = Qwen4_5ForCausalLM(
                self.cfg, dtype=torch.bfloat16, kv_dtype="fp8_e4m3", device=self.dev1
            )
            load_qwen4_5_from_checkpoint(d_causal, tmpdir, ple_mode="gpu_full")

        p_rtp = Qwen4_5RTPModel.from_causal_lm(p_causal)
        d_rtp = Qwen4_5RTPModel.from_causal_lm(d_causal)
        tok = SimpleByteTokenizer(
            self.cfg.vocab_size,
            bos_id=self.cfg.bos_token_id,
            eos_id=self.cfg.eos_token_id,
        )

        # Prompt 0 has length == 32 (exact 2*16 block boundary); prompt 1 has length == 29 (crosses 32 during decode).
        tpb = 16
        prompts = [
            tok.encode("Exact boundary length 32 prompt!")[:32],
            tok.encode("Rolling state decode test!"),
        ]
        while len(prompts[0]) < 32:
            prompts[0].append(42)
        in_lens = [len(p) for p in prompts]
        max_new = 20

        bf16_ref_tokens, bf16_ref_logits = _run_unified_baseline(
            p_causal, prompts, max_new, enable_yoco_decode=True
        )

        # Unified FP8 KV FULL-group reference (non-rolling) to verify rolling eviction & single-tail transfer exactness.
        plan_fp8_full = build_qwen4_5_kv_cache_plan(
            self.cfg, kv_dtype="fp8_e4m3", enable_yoco_decode=True
        )
        ref_kv_fp8 = LocalKVCache(
            plan_fp8_full,
            tokens_per_block=tpb,
            num_blocks_per_tag={t: 64 for t in QWEN4_5_CACHE_TAGS},
            device=self.dev0,
            dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
        )
        p_rtp.kv_cache = ref_kv_fp8
        ref_tables = build_block_tables_for_requests(
            ref_kv_fp8, [ln + max_new for ln in in_lens]
        )
        ref_flat_ids = torch.tensor(
            [t for seq in prompts for t in seq], dtype=torch.int64, device=self.dev0
        )
        ref_p_out = p_rtp.forward(
            LocalPyModelInputs(
                input_ids=ref_flat_ids,
                attention_inputs=make_tagged_attention_inputs(
                    ref_tables,
                    is_prefill=True,
                    input_lengths=in_lens,
                    sequence_lengths=in_lens,
                    prefix_lengths=[0] * len(prompts),
                    device=self.dev0,
                ),
            ),
            enable_yoco_decode=True,
        )
        ref_last = [in_lens[0] - 1, in_lens[0] + in_lens[1] - 1]
        ref_first_lg = p_rtp.compute_logits(
            ref_p_out.hidden_states[torch.tensor(ref_last, dtype=torch.long, device=self.dev0)]
        )
        ref_cur_toks = torch.argmax(ref_first_lg, dim=-1).tolist()
        fp8_ref_tokens = [[int(t)] for t in ref_cur_toks]
        fp8_ref_logits = [ref_first_lg.detach().cpu()]
        ref_cur_lens = list(in_lens)
        for _ in range(max_new - 1):
            ref_d_out = p_rtp.forward(
                LocalPyModelInputs(
                    input_ids=torch.tensor(ref_cur_toks, dtype=torch.int64, device=self.dev0),
                    attention_inputs=make_tagged_attention_inputs(
                        ref_tables,
                        is_prefill=False,
                        input_lengths=[1] * len(prompts),
                        sequence_lengths=ref_cur_lens,
                        prefix_lengths=[0] * len(prompts),
                        device=self.dev0,
                    ),
                ),
                enable_yoco_decode=True,
            )
            lg = p_rtp.compute_logits(ref_d_out.hidden_states)
            fp8_ref_logits.append(lg.detach().cpu())
            ref_cur_toks = torch.argmax(lg, dim=-1).tolist()
            for b in range(len(prompts)):
                fp8_ref_tokens[b].append(int(ref_cur_toks[b]))
                ref_cur_lens[b] += 1

        rolling_pool_blocks = 8  # Smaller than total logical pages across both reqs to force rolling reuse

        def _run_pd_rolling(enable_yoco: bool) -> tuple[list[list[int]], list[torch.Tensor]]:
            plan_fp8 = build_qwen4_5_kv_cache_plan(
                self.cfg, kv_dtype="fp8_e4m3", enable_yoco_decode=enable_yoco
            )
            # Create Prefill first so active tags exclude unused QSA tags (e.g., indexer_kv).
            p_kv = LocalKVCache(
                plan_fp8,
                tokens_per_block=tpb,
                num_blocks_per_tag={t: 64 for t in QWEN4_5_CACHE_TAGS},
                device=self.dev0,
                dtype=torch.bfloat16,
                kv_dtype=torch.float8_e4m3fn,
            )
            active_tags = p_kv.group_tags()
            state_tags = [t for t in active_tags if t != FULL_ATTN_KV_TAG]
            d_num_blocks = {FULL_ATTN_KV_TAG: 64}
            for t in state_tags:
                d_num_blocks[t] = rolling_pool_blocks

            d_kv = LocalKVCache(
                plan_fp8,
                tokens_per_block=tpb,
                num_blocks_per_tag=d_num_blocks,
                device=self.dev1,
                dtype=torch.bfloat16,
                kv_dtype=torch.float8_e4m3fn,
            )
            p_rtp.kv_cache = p_kv
            d_rtp.kv_cache = d_kv

            p_tables = build_block_tables_for_requests(p_kv, in_lens)
            flat_ids = torch.tensor(
                [t for seq in prompts for t in seq], dtype=torch.int64, device=self.dev0
            )
            writer = LocalMemoryCacheStoreWriter(record_payload=True)
            p_attn = make_tagged_attention_inputs(
                p_tables,
                is_prefill=True,
                input_lengths=in_lens,
                sequence_lengths=in_lens,
                prefix_lengths=[0] * len(prompts),
                device=self.dev0,
                cache_store_writer=writer,
            )
            p_out = p_rtp.forward(
                LocalPyModelInputs(input_ids=flat_ids, attention_inputs=p_attn),
                enable_yoco_decode=enable_yoco,
            )
            last_idx = [in_lens[0] - 1, in_lens[0] + in_lens[1] - 1]
            first_logits = p_rtp.compute_logits(
                p_out.hidden_states[torch.tensor(last_idx, dtype=torch.long, device=self.dev0)]
            )
            first_toks = torch.argmax(first_logits, dim=-1).tolist()

            # Build Decode block tables: full FULL_ATTN_KV_TAG + 2-slot rolling OPAQUE_STATE pools (blocks 1..rolling_pool_blocks-1)
            max_seq_total = max(in_lens) + max_new + 2
            max_pages = (max_seq_total + tpb - 1) // tpb
            d_tables: dict[str, list[list[int]]] = {
                t: [[0] * max_pages for _ in prompts] for t in active_tags
            }
            full_next = 1
            for b, ln in enumerate(in_lens):
                req_pages = (ln + max_new + tpb - 1) // tpb
                p_pages = (ln + tpb - 1) // tpb
                for pg in range(req_pages):
                    d_tables[FULL_ATTN_KV_TAG][b][pg] = full_next
                    if pg < p_pages:
                        src_blk = p_tables[FULL_ATTN_KV_TAG][b][pg]
                        for layer_id in range(p_kv.layer_count()):
                            if FULL_ATTN_KV_TAG not in p_kv.layer_caches[layer_id]:
                                continue
                            dst_c = d_kv.get_layer_cache(layer_id, FULL_ATTN_KV_TAG)
                            dst_c.kv_cache_base[full_next].copy_(
                                writer.published[(layer_id, FULL_ATTN_KV_TAG)][1][
                                    writer.published[(layer_id, FULL_ATTN_KV_TAG)][0].index(src_blk)
                                ].to(self.dev1)
                            )
                            p_c = p_kv.get_layer_cache(layer_id, FULL_ATTN_KV_TAG)
                            if dst_c.kv_scale_base is not None and p_c.kv_scale_base is not None:
                                dst_c.kv_scale_base[full_next].copy_(
                                    p_c.kv_scale_base[src_blk].to(self.dev1)
                                )
                    full_next += 1

            # Each rolling tag keeps an independent ring allocator of slots 1..rolling_pool_blocks-1,
            # loading ONLY the single tail block (k_tail = (ln - 1) // tpb) from Prefill CacheStore.
            free_slots = {t: list(range(1, rolling_pool_blocks)) for t in state_tags}
            active_prev_slot = {t: [0] * len(prompts) for t in state_tags}
            active_cur_slot = {t: [0] * len(prompts) for t in state_tags}
            for t in state_tags:
                for b, ln in enumerate(in_lens):
                    k_tail = (ln - 1) // tpb
                    src_blk = p_tables[t][b][k_tail]
                    slot = free_slots[t].pop(0)
                    d_tables[t][b][k_tail] = slot
                    active_cur_slot[t][b] = slot
                    for layer_id in range(p_kv.layer_count()):
                        if t not in p_kv.layer_caches[layer_id]:
                            continue
                        dst_c = d_kv.get_layer_cache(layer_id, t)
                        pub_blks, pub_payload = writer.published[(layer_id, t)]
                        dst_c.kv_cache_base[slot].copy_(
                            pub_payload[pub_blks.index(src_blk)].to(self.dev1)
                        )

            gen_tokens = [[int(t)] for t in first_toks]
            step_logits = [first_logits.detach().cpu()]
            cur_lens = list(in_lens)
            cur_toks = list(first_toks)
            for _ in range(max_new - 1):
                # Mirror FIFOScheduler incrKVBlock + LINEAR retainedDecodeBlocks ([prev_blk, cur_blk]) eviction
                for b, cur_len in enumerate(cur_lens):
                    write_pg = cur_len // tpb
                    for t in state_tags:
                        if d_tables[t][b][write_pg] == 0:
                            old_evicted = active_prev_slot[t][b]
                            if old_evicted != 0:
                                evict_pg = write_pg - 2
                                if evict_pg >= 0:
                                    d_tables[t][b][evict_pg] = 0
                                free_slots[t].append(old_evicted)
                            new_slot = free_slots[t].pop(0)
                            d_tables[t][b][write_pg] = new_slot
                            active_prev_slot[t][b] = active_cur_slot[t][b]
                            active_cur_slot[t][b] = new_slot

                d_attn = make_tagged_attention_inputs(
                    d_tables,
                    is_prefill=False,
                    input_lengths=[1] * len(prompts),
                    sequence_lengths=cur_lens,
                    prefix_lengths=[0] * len(prompts),
                    device=self.dev1,
                )
                d_out = d_rtp.forward(
                    LocalPyModelInputs(
                        input_ids=torch.tensor(cur_toks, dtype=torch.int64, device=self.dev1),
                        attention_inputs=d_attn,
                    ),
                    enable_yoco_decode=enable_yoco,
                )
                lg = d_rtp.compute_logits(d_out.hidden_states)
                step_logits.append(lg.detach().cpu())
                cur_toks = torch.argmax(lg, dim=-1).tolist()
                for b in range(len(prompts)):
                    gen_tokens[b].append(int(cur_toks[b]))
                    cur_lens[b] += 1
            return gen_tokens, step_logits

        pd_tokens, pd_logits = _run_pd_rolling(True)
        ny_tokens, ny_logits = _run_pd_rolling(False)
        min_cos_fp8_ref = min(
            _cos_sim(pd_logits[s], fp8_ref_logits[s]) for s in range(max_new)
        )
        prefill_cos_vs_bf16 = _cos_sim(pd_logits[0], bf16_ref_logits[0])
        matched_fp8 = sum(
            int(pd_tokens[b][s] == fp8_ref_tokens[b][s])
            for b in range(len(prompts))
            for s in range(max_new)
        )
        matched_bf16 = sum(
            int(pd_tokens[b][s] == bf16_ref_tokens[b][s])
            for b in range(len(prompts))
            for s in range(max_new)
        )
        yoco_diff = max(
            float((pd_logits[s].float() - ny_logits[s].float()).abs().max().item())
            for s in range(max_new)
        )
        print(
            f"[PD Rolling OPAQUE_STATE + FP8 KV B=2x{max_new}] "
            f"vs_fp8_ref={matched_fp8}/{len(prompts)*max_new} (min_cos={min_cos_fp8_ref:.6f}), "
            f"vs_bf16_ref={matched_bf16}/{len(prompts)*max_new} (p0_cos={prefill_cos_vs_bf16:.6f}), "
            f"YoCo_ON_vs_OFF={yoco_diff:.6f}"
        )
        self.assertGreater(prefill_cos_vs_bf16, 0.999)
        self.assertGreater(min_cos_fp8_ref, 0.9999)
        self.assertEqual(pd_tokens, fp8_ref_tokens)
        self.assertEqual(pd_tokens, ny_tokens)
        self.assertEqual(yoco_diff, 0.0)

    @torch.no_grad()
    def test_06_asymmetric_p_tp4_to_d_tp2_pd_disagg_end_to_end(self) -> None:
        """End-to-end P(sp4_tp4_ep4) -> C++ TpCacheTransferMode reassembly -> D(dp2_tpsp2_ep4) PD serving."""
        import socket
        import torch.multiprocessing as mp
        from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import (
            FFN_OUTPUT_SCONV_TAG,
            GDN_CONSUMER_Q_CONV_TAG,
            GDN_CONV_STATE_TAG,
            GDN_SSM_STATE_TAG,
            build_qwen4_5_kv_cache_spec_descs,
        )
        from rtp_llm.ops import RoleType, TpCacheTransferMode

        def _free_port() -> int:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                return int(s.getsockname()[1])

        tok = SimpleByteTokenizer(
            self.cfg.vocab_size,
            bos_id=self.cfg.bos_token_id,
            eos_id=self.cfg.eos_token_id,
        )
        prompts = [
            tok.encode("Qwen4.5 P_tp=4 to D_tp=2 PD check!"),
            tok.encode("Ragged asymmetric TP handoff seq2!"),
        ]
        in_lens = [len(p) for p in prompts]
        max_new = 6
        tpb = 16
        seed = 5454

        # 1. Verify C++ descriptor transfer contracts for P(tp4) -> D(tp2)
        mcfg = SimpleNamespace(
            qwen4_5_arch_config=self.cfg,
            attn_config=SimpleNamespace(tokens_per_block=tpb, kernel_tokens_per_block=tpb),
            gen_num_per_cycle=0,
        )
        p_descs = build_qwen4_5_kv_cache_spec_descs(
            mcfg, self.cfg, parallel_mode="sp4_tp4_ep4", kv_dtype="fp8_e4m3", enable_yoco_decode=True, role_type=RoleType.PREFILL
        )
        d_descs = build_qwen4_5_kv_cache_spec_descs(
            mcfg, self.cfg, parallel_mode="dp2_tpsp2_ep4", kv_dtype="fp8_e4m3", enable_yoco_decode=True, role_type=RoleType.DECODE
        )
        d_desc_map: dict[tuple[int, str], object] = {
            (lid, str(d.tag)): d for lid, layer_list in enumerate(d_descs) for d in layer_list
        }
        p_desc_map: dict[tuple[int, str], object] = {
            (lid, str(d.tag)): d for lid, layer_list in enumerate(p_descs) for d in layer_list
        }
        self.assertEqual(d_desc_map[(0, GDN_SSM_STATE_TAG)].tp_transfer_mode, TpCacheTransferMode.SHARDED_STATE)
        self.assertEqual(d_desc_map[(0, GDN_CONV_STATE_TAG)].tp_transfer_mode, TpCacheTransferMode.SHARDED_STATE)
        self.assertEqual(d_desc_map[(5, GDN_CONSUMER_Q_CONV_TAG)].tp_transfer_mode, TpCacheTransferMode.SHARDED_STATE)
        self.assertEqual(d_desc_map[(0, FFN_OUTPUT_SCONV_TAG)].tp_transfer_mode, TpCacheTransferMode.REPLICATED)
        self.assertEqual(d_desc_map[(3, FULL_ATTN_KV_TAG)].tp_transfer_mode, TpCacheTransferMode.MHA_HEADS)

        # 2. Single-rank FP8 unified reference sequence (teacher-forced targets & logits)
        ref_causal = Qwen4_5ForCausalLM(
            self.cfg, dtype=torch.bfloat16, kv_dtype="fp8_e4m3", device=self.dev0
        )
        ref_causal.init_synthetic_weights(seed=seed, ple_mode="gpu_full")
        _stabilize_router_margins(ref_causal)
        ref_causal.prepare_weights()
        ref_rtp = Qwen4_5RTPModel.from_causal_lm(ref_causal)
        ref_plan = build_qwen4_5_kv_cache_plan(self.cfg, kv_dtype="fp8_e4m3", enable_yoco_decode=True)
        ref_kv = LocalKVCache(
            ref_plan,
            tokens_per_block=tpb,
            num_blocks_per_tag={t: 64 for t in QWEN4_5_CACHE_TAGS},
            device=self.dev0,
            dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
        )
        ref_rtp.kv_cache = ref_kv
        ref_tables = build_block_tables_for_requests(ref_kv, [ln + max_new for ln in in_lens])
        flat_ids = torch.tensor([t for seq in prompts for t in seq], dtype=torch.int64, device=self.dev0)
        ref_p = ref_rtp.forward(
            LocalPyModelInputs(
                input_ids=flat_ids,
                attention_inputs=make_tagged_attention_inputs(
                    ref_tables,
                    is_prefill=True,
                    input_lengths=in_lens,
                    sequence_lengths=in_lens,
                    prefix_lengths=[0, 0],
                    device=self.dev0,
                ),
            ),
            enable_yoco_decode=True,
        )
        last_idx = [in_lens[0] - 1, in_lens[0] + in_lens[1] - 1]
        ref_lg0 = ref_rtp.compute_logits(ref_p.hidden_states[torch.tensor(last_idx, dtype=torch.long, device=self.dev0)])
        ref_cur = torch.argmax(ref_lg0, dim=-1).tolist()
        ref_step_inputs: list[list[int]] = [list(ref_cur)]
        ref_tokens = [[int(t)] for t in ref_cur]
        ref_logits = [ref_lg0.detach().cpu()]
        cur_lens = list(in_lens)
        for _ in range(max_new - 1):
            d_out = ref_rtp.forward(
                LocalPyModelInputs(
                    input_ids=torch.tensor(ref_cur, dtype=torch.int64, device=self.dev0),
                    attention_inputs=make_tagged_attention_inputs(
                        ref_tables,
                        is_prefill=False,
                        input_lengths=[1, 1],
                        sequence_lengths=cur_lens,
                        prefix_lengths=[0, 0],
                        device=self.dev0,
                    ),
                ),
                enable_yoco_decode=True,
            )
            lg = ref_rtp.compute_logits(d_out.hidden_states)
            ref_logits.append(lg.detach().cpu())
            ref_cur = torch.argmax(lg, dim=-1).tolist()
            ref_step_inputs.append(list(ref_cur))
            for b in range(2):
                ref_tokens[b].append(int(ref_cur[b]))
                cur_lens[b] += 1
        del ref_causal, ref_rtp, ref_kv
        torch.cuda.empty_cache()

        ctx = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmpdir:
            for mode_name, yoco_flag in (("yoco", True), ("noyoco", False)):
                p_descs_m = build_qwen4_5_kv_cache_spec_descs(
                    mcfg, self.cfg, parallel_mode="sp4_tp4_ep4", kv_dtype="fp8_e4m3", enable_yoco_decode=yoco_flag, role_type=RoleType.PREFILL
                )
                d_descs_m = build_qwen4_5_kv_cache_spec_descs(
                    mcfg, self.cfg, parallel_mode="dp2_tpsp2_ep4", kv_dtype="fp8_e4m3", enable_yoco_decode=yoco_flag, role_type=RoleType.DECODE
                )
                p_map_m = {(lid, str(d.tag)): d for lid, layer_list in enumerate(p_descs_m) for d in layer_list}
                d_map_m = {(lid, str(d.tag)): d for lid, layer_list in enumerate(d_descs_m) for d in layer_list}

                # 3. Run P_tp=4 (sp4_tp4_ep4) Prefill and publish per-rank CacheStore buffers
                p_port = _free_port()
                err_q = ctx.Queue()
                p_procs = [
                    ctx.Process(
                        target=_p4_d2_gloo_worker,
                        args=(r, "prefill", self.cfg, prompts, ref_step_inputs, max_new, tpb, seed, yoco_flag, p_port, tmpdir, err_q),
                    )
                    for r in range(4)
                ]
                for pr in p_procs:
                    pr.start()
                for pr in p_procs:
                    pr.join(timeout=120)
                while not err_q.empty():
                    raise RuntimeError(f"Prefill({mode_name}) worker failed: {err_q.get()}")
                self.assertTrue(all(pr.exitcode == 0 for pr in p_procs))

                p_rank_data = [torch.load(os.path.join(tmpdir, f"prefill_r{r}.pt"), map_location="cpu") for r in range(4)]

                # 4. Reassemble P_tp=4 -> D_tp=2 according to C++ TpCacheTransferMode + segments
                def _reassemble_sharded_state(desc, p_desc, b0: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
                    self.assertEqual(desc.tp_transfer_mode, TpCacheTransferMode.SHARDED_STATE)
                    qe, ke, ve = int(desc.tp_state_q_elems), int(desc.tp_state_k_elems), int(desc.tp_state_v_elems)
                    if qe == 0 and ke == 0 and ve == 0:
                        return torch.cat([b0, b1], dim=-1)
                    pq, pk, pv = int(p_desc.tp_state_q_elems), int(p_desc.tp_state_k_elems), int(p_desc.tp_state_v_elems)
                    self.assertEqual(pq * 2, qe)
                    self.assertEqual(pk * 2, ke)
                    self.assertEqual(pv * 2, ve)
                    tot_b = b0.shape[1]
                    qb = tot_b * pq // (pq + pk + pv)
                    kb = tot_b * pk // (pq + pk + pv)
                    vb = tot_b - qb - kb
                    return torch.cat(
                        [
                            b0[:, :qb],
                            b1[:, :qb],
                            b0[:, qb : qb + kb],
                            b1[:, qb : qb + kb],
                            b0[:, qb + kb : qb + kb + vb],
                            b1[:, qb + kb : qb + kb + vb],
                        ],
                        dim=-1,
                    )

                for d_tp_rank in range(2):
                    p0_d = p_rank_data[d_tp_rank * 2]
                    p1_d = p_rank_data[d_tp_rank * 2 + 1]
                    reassembled_payload: dict[tuple[int, str], torch.Tensor] = {}
                    reassembled_scales: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
                    for (lid, tag), (blk_ids, ten0) in p0_d["published"].items():
                        ten1 = p1_d["published"][(lid, tag)][1]
                        if tag == "linear":
                            cnv_tag = GDN_CONV_STATE_TAG if (lid, GDN_CONV_STATE_TAG) in p_map_m else GDN_CONSUMER_Q_CONV_TAG
                            p_ssm = p_map_m[(lid, GDN_SSM_STATE_TAG)]
                            p_cnv = p_map_m[(lid, cnv_tag)]
                            d_ssm = d_map_m[(lid, GDN_SSM_STATE_TAG)]
                            d_cnv = d_map_m[(lid, cnv_tag)]
                            p_ssm_bytes = int(p_ssm.entry_elems) * int(p_ssm.explicit_entry_count) * 4
                            p_cnv_bytes = int(p_cnv.entry_elems) * int(p_cnv.explicit_entry_count) * 2
                            b0 = ten0.contiguous().view(torch.uint8).reshape(len(blk_ids), -1)
                            b1 = ten1.contiguous().view(torch.uint8).reshape(len(blk_ids), -1)
                            merged_ssm = _reassemble_sharded_state(d_ssm, p_ssm, b0[:, :p_ssm_bytes], b1[:, :p_ssm_bytes])
                            merged_cnv = _reassemble_sharded_state(
                                d_cnv, p_cnv, b0[:, p_ssm_bytes : p_ssm_bytes + p_cnv_bytes], b1[:, p_ssm_bytes : p_ssm_bytes + p_cnv_bytes]
                            )
                            merged = torch.cat([merged_ssm, merged_cnv], dim=-1)
                            d_ssm_bytes = int(d_ssm.entry_elems) * int(d_ssm.explicit_entry_count) * 4
                            d_cnv_bytes = int(d_cnv.entry_elems) * int(d_cnv.explicit_entry_count) * 2
                            self.assertEqual(merged.shape[1], d_ssm_bytes + d_cnv_bytes)
                            full_bytes = ten0.shape[1] * 4
                            if merged.shape[1] < full_bytes:
                                merged = torch.cat(
                                    [merged, torch.zeros((len(blk_ids), full_bytes - merged.shape[1]), dtype=torch.uint8)],
                                    dim=-1,
                                )
                            reassembled_payload[(lid, tag)] = merged.contiguous().view(torch.float32)
                            continue

                        desc = d_map_m[(lid, tag)]
                        p_desc = p_map_m[(lid, tag)]
                        mode = desc.tp_transfer_mode
                        if mode == TpCacheTransferMode.SHARDED_STATE:
                            b0 = ten0.contiguous().view(torch.uint8).reshape(len(blk_ids), -1)
                            b1 = ten1.contiguous().view(torch.uint8).reshape(len(blk_ids), -1)
                            merged_b = _reassemble_sharded_state(desc, p_desc, b0, b1)
                            out_ten = merged_b.contiguous().view(ten0.dtype)
                            out_ten = out_ten.reshape(len(blk_ids), int(desc.entry_elems), int(desc.explicit_entry_count))
                            reassembled_payload[(lid, tag)] = out_ten
                        elif mode == TpCacheTransferMode.MHA_HEADS:
                            self.assertEqual(tuple(ten0.shape), tuple(ten1.shape))
                            reassembled_payload[(lid, tag)] = ten0.clone()
                        elif mode == TpCacheTransferMode.REPLICATED:
                            self.assertTrue(torch.equal(ten0, ten1), f"Replicated tag ({lid}, {tag}) mismatch between peers")
                            reassembled_payload[(lid, tag)] = ten0.clone()
                        else:
                            raise AssertionError(f"Unexpected transfer mode {mode} for ({lid}, {tag})")

                    for lid, (sc_blks, sc0) in p0_d["scales"].items():
                        reassembled_scales[(lid, FULL_ATTN_KV_TAG)] = (sc_blks, sc0.clone())

                    torch.save(
                        {
                            "tables": p0_d["tables"],
                            "published": {k: (p0_d["published"][k][0], v) for k, v in reassembled_payload.items()},
                            "scales": reassembled_scales,
                            "first_toks": p0_d["first_toks"],
                            "first_logits": p0_d["first_logits"],
                        },
                        os.path.join(tmpdir, f"decode_tp{d_tp_rank}.pt"),
                    )

                # 5. Run D_tp=2 (dp2_tpsp2_ep4) Decode (plus unified D_tp=2 parity reference inside rank)
                d_port = _free_port()
                d_procs = [
                    ctx.Process(
                        target=_p4_d2_gloo_worker,
                        args=(r, "decode", self.cfg, prompts, ref_step_inputs, max_new, tpb, seed, yoco_flag, d_port, tmpdir, err_q),
                    )
                    for r in range(4)
                ]
                for pr in d_procs:
                    pr.start()
                for pr in d_procs:
                    pr.join(timeout=120)
                while not err_q.empty():
                    raise RuntimeError(f"Decode({mode_name}) worker failed: {err_q.get()}")
                self.assertTrue(all(pr.exitcode == 0 for pr in d_procs))
                os.replace(os.path.join(tmpdir, "decode_out.pt"), os.path.join(tmpdir, f"decode_{mode_name}.pt"))

            res_y = torch.load(os.path.join(tmpdir, "decode_yoco.pt"), map_location="cpu")
            res_n = torch.load(os.path.join(tmpdir, "decode_noyoco.pt"), map_location="cpu")
            pd_tokens = res_y["tokens"]
            pd_logits = res_y["logits"]
            uni_tokens = res_y["uni_tokens"]
            uni_logits = res_y["uni_logits"]
            ny_tokens = res_n["tokens"]
            ny_logits = res_n["logits"]

            total = len(prompts) * max_new
            matched_uni = sum(int(pd_tokens[b][s] == uni_tokens[b][s]) for b in range(2) for s in range(max_new))
            min_cos_tp1 = min(_cos_sim(pd_logits[s], ref_logits[s]) for s in range(max_new))
            min_cos_uni = min(_cos_sim(pd_logits[s], uni_logits[s]) for s in range(max_new))
            yoco_diff = max(float((pd_logits[s].float() - ny_logits[s].float()).abs().max().item()) for s in range(max_new))
            print(
                f"[PD P_tp=4 -> D_tp=2 Asymmetric FP8 KV B=2x{max_new}] "
                f"vs_D_tp2_uni={matched_uni}/{total} (min_cos={min_cos_uni:.6f}), "
                f"vs_TP1_ref_min_cos={min_cos_tp1:.6f}, YoCo_ON_vs_OFF={yoco_diff:.6f}"
            )
            self.assertGreater(min_cos_tp1, 0.999)
            self.assertGreater(min_cos_uni, 0.999)
            self.assertEqual(pd_tokens, uni_tokens)
            self.assertEqual(pd_tokens, ny_tokens)
            self.assertEqual(yoco_diff, 0.0)


@torch.no_grad()
def _p4_d2_gloo_worker(
    rank: int,
    role: str,
    cfg,
    prompts: list[list[int]],
    ref_step_inputs: list[list[int]],
    max_new: int,
    tpb: int,
    seed: int,
    enable_yoco: bool,
    port: int,
    tmpdir: str,
    err_q,
) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=rank, world_size=4)
        try:
            dev = torch.device("cuda:0")
            torch.cuda.set_device(dev)
            if role == "prefill":
                pmode = "sp4_tp4_ep4"
                g0 = dist.new_group([0, 1, 2, 3])
                tp_r, dp_r = rank, 0
                kw = dict(
                    parallel_mode=pmode, tp_size=4, tp_rank=rank, tp_group=g0,
                    sp_size=4, sp_rank=rank, sp_group=g0, ep_size=4, ep_rank=rank, ep_group=g0,
                )
            else:
                pmode = "dp2_tpsp2_ep4"
                tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
                dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
                ep_group = dist.new_group([0, 1, 2, 3])
                dp_r, tp_r = rank // 2, rank % 2
                kw = dict(
                    parallel_mode=pmode, dp_size=2, dp_rank=dp_r, dp_group=dp_groups[tp_r],
                    tp_size=2, tp_rank=tp_r, tp_group=tp_groups[dp_r],
                    sp_size=2, sp_rank=tp_r, sp_group=tp_groups[dp_r], ep_size=4, ep_rank=rank, ep_group=ep_group,
                )

            m = Qwen4_5ForCausalLM(cfg, dtype=torch.bfloat16, kv_dtype="fp8_e4m3", device=dev, **kw)
            m.init_synthetic_weights(seed=seed, ple_mode="gpu_sharded")
            _stabilize_router_margins(m)
            m.prepare_weights()
            rtp = Qwen4_5RTPModel.from_causal_lm(m)
            plan = build_qwen4_5_kv_cache_plan(cfg, parallel_mode=pmode, kv_dtype="fp8_e4m3", enable_yoco_decode=enable_yoco)
            in_lens = [len(p) for p in prompts]

            if role == "prefill":
                p_kv = LocalKVCache(
                    plan, tokens_per_block=tpb, num_blocks_per_tag={t: 64 for t in QWEN4_5_CACHE_TAGS},
                    device=dev, dtype=torch.bfloat16, kv_dtype=torch.float8_e4m3fn,
                )
                rtp.kv_cache = p_kv
                p_tables = build_block_tables_for_requests(p_kv, in_lens)
                writer = LocalMemoryCacheStoreWriter(record_payload=True)
                flat_ids = torch.tensor([t for seq in prompts for t in seq], dtype=torch.int64, device=dev)
                p_out = rtp.forward(
                    LocalPyModelInputs(
                        input_ids=flat_ids,
                        attention_inputs=make_tagged_attention_inputs(
                            p_tables, is_prefill=True, input_lengths=in_lens, sequence_lengths=in_lens,
                            prefix_lengths=[0, 0], device=dev, cache_store_writer=writer,
                        ),
                    ),
                    enable_yoco_decode=enable_yoco,
                )
                last_idx = [in_lens[0] - 1, in_lens[0] + in_lens[1] - 1]
                first_lg = rtp.compute_logits(p_out.hidden_states[torch.tensor(last_idx, dtype=torch.long, device=dev)])
                scales = {}
                for lid in range(p_kv.layer_count()):
                    if FULL_ATTN_KV_TAG in p_kv.layer_caches[lid]:
                        c = p_kv.get_layer_cache(lid, FULL_ATTN_KV_TAG)
                        if c.kv_scale_base is not None:
                            blks = writer.published[(lid, FULL_ATTN_KV_TAG)][0]
                            idx_t = torch.tensor(blks, dtype=torch.long, device=dev)
                            scales[lid] = (blks, c.kv_scale_base.index_select(0, idx_t).detach().cpu())
                torch.save(
                    {
                        "tables": p_tables,
                        "published": {k: (blks, v.cpu()) for k, (blks, v) in writer.published.items()},
                        "scales": scales,
                        "first_toks": torch.argmax(first_lg, dim=-1).tolist(),
                        "first_logits": first_lg.detach().cpu(),
                    },
                    os.path.join(tmpdir, f"prefill_r{rank}.pt"),
                )
            else:
                bundle = torch.load(os.path.join(tmpdir, f"decode_tp{tp_r}.pt"), map_location="cpu")

                # 1) Unified D_tp=2 baseline on exact same teacher-forced step tokens
                u_kv = LocalKVCache(
                    plan, tokens_per_block=tpb, num_blocks_per_tag={t: 64 for t in QWEN4_5_CACHE_TAGS},
                    device=dev, dtype=torch.bfloat16, kv_dtype=torch.float8_e4m3fn,
                )
                rtp.kv_cache = u_kv
                u_tables = build_block_tables_for_requests(u_kv, [ln + max_new for ln in in_lens])
                flat_ids = torch.tensor([t for seq in prompts for t in seq], dtype=torch.int64, device=dev)
                u_p = rtp.forward(
                    LocalPyModelInputs(
                        input_ids=flat_ids,
                        attention_inputs=make_tagged_attention_inputs(
                            u_tables, is_prefill=True, input_lengths=in_lens, sequence_lengths=in_lens,
                            prefix_lengths=[0, 0], device=dev,
                        ),
                    ),
                    enable_yoco_decode=enable_yoco,
                )
                last_idx = [in_lens[0] - 1, in_lens[0] + in_lens[1] - 1]
                u_lg0 = rtp.compute_logits(u_p.hidden_states[torch.tensor(last_idx, dtype=torch.long, device=dev)])
                uni_tokens = [[int(t)] for t in torch.argmax(u_lg0, dim=-1).tolist()]
                uni_logits = [u_lg0.detach().cpu()]
                cur_lens = list(in_lens)
                for step_i in range(max_new - 1):
                    u_attn = make_tagged_attention_inputs(
                        u_tables, is_prefill=False, input_lengths=[1, 1], sequence_lengths=cur_lens,
                        prefix_lengths=[0, 0], device=dev,
                    )
                    u_out = rtp.forward(
                        LocalPyModelInputs(input_ids=torch.tensor(ref_step_inputs[step_i], dtype=torch.int64, device=dev), attention_inputs=u_attn),
                        enable_yoco_decode=enable_yoco,
                    )
                    ulg = rtp.compute_logits(u_out.hidden_states)
                    uni_logits.append(ulg.detach().cpu())
                    utoks = torch.argmax(ulg, dim=-1).tolist()
                    for b in range(2):
                        uni_tokens[b].append(int(utoks[b]))
                        cur_lens[b] += 1

                # 2) PD Disaggregated D_tp=2 decode from reassembled P_tp=4 CacheStore state
                d_kv = LocalKVCache(
                    plan, tokens_per_block=tpb, num_blocks_per_tag={t: 64 for t in QWEN4_5_CACHE_TAGS},
                    device=dev, dtype=torch.bfloat16, kv_dtype=torch.float8_e4m3fn,
                )
                rtp.kv_cache = d_kv
                d_tables = transfer_published_blocks_to_decode(
                    bundle["published"], bundle["tables"], d_kv, [ln + max_new for ln in in_lens]
                )
                for (lid, tag), (src_blks, sc_ten) in bundle["scales"].items():
                    dst_c = d_kv.get_layer_cache(lid, tag)
                    assert dst_c.kv_scale_base is not None
                    s_rows, d_rows = bundle["tables"][tag], d_tables[tag]
                    id_map = {int(sb): int(db) for sr, dr in zip(s_rows, d_rows) for sb, db in zip(sr, dr)}
                    sc_dev = sc_ten.to(dev)
                    for i_b, sb in enumerate(src_blks):
                        if sb in id_map:
                            dst_c.kv_scale_base[id_map[sb]].copy_(sc_dev[i_b])

                gen_tokens = [[int(t)] for t in bundle["first_toks"]]
                step_logits = [bundle["first_logits"]]
                cur_lens = list(in_lens)
                for step_i in range(max_new - 1):
                    d_attn = make_tagged_attention_inputs(
                        d_tables, is_prefill=False, input_lengths=[1, 1], sequence_lengths=cur_lens,
                        prefix_lengths=[0, 0], device=dev,
                    )
                    d_out = rtp.forward(
                        LocalPyModelInputs(input_ids=torch.tensor(ref_step_inputs[step_i], dtype=torch.int64, device=dev), attention_inputs=d_attn),
                        enable_yoco_decode=enable_yoco,
                    )
                    lg = rtp.compute_logits(d_out.hidden_states)
                    step_logits.append(lg.detach().cpu())
                    cur_toks = torch.argmax(lg, dim=-1).tolist()
                    for b in range(2):
                        gen_tokens[b].append(int(cur_toks[b]))
                        cur_lens[b] += 1

                if rank == 0:
                    torch.save(
                        {"tokens": gen_tokens, "logits": step_logits, "uni_tokens": uni_tokens, "uni_logits": uni_logits},
                        os.path.join(tmpdir, "decode_out.pt"),
                    )
            dist.barrier()
        finally:
            dist.destroy_process_group()
    except BaseException as exc:
        import traceback
        err_q.put((rank, f"{exc!r}\n{traceback.format_exc()}"))
        raise


@torch.no_grad()
def run_two_process_nccl_pd_test() -> None:
    """True 2-process PD disaggregation over NCCL: Rank 0 (Prefill, GPU0) -> Rank 1 (Decode, GPU1)."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, f"Expected 2 processes (P + D), got world_size={world_size}"

    cfg = make_mini_qwen4_5_config()
    tok = SimpleByteTokenizer(
        cfg.vocab_size, bos_id=cfg.bos_token_id, eos_id=cfg.eos_token_id
    )
    prompts = [
        tok.encode("Qwen 4.5 Flash PD serving test on B300!"),
        tok.encode("Prefill GPU0 -> NCCL -> Decode GPU1!"),
    ]
    in_lens = [len(p) for p in prompts]
    max_new = 16
    tpb = 16

    with tempfile.TemporaryDirectory() as tmpdir_raw:
        obj_list = [tmpdir_raw if rank == 0 else None]
        dist.broadcast_object_list(obj_list, src=0, device=device)
        ckpt_dir = Path(obj_list[0])  # type: ignore[arg-type]

        if rank == 0:
            src_m = Qwen4_5ForCausalLM(cfg, dtype=torch.bfloat16, device=device)
            src_m.init_synthetic_weights(seed=8888, ple_mode="gpu_full")
            _stabilize_router_margins(src_m)
            src_m.prepare_weights()
            save_qwen4_5_safetensors_checkpoint(src_m, ckpt_dir)
            dist.barrier()
            role_model = src_m
        else:
            dist.barrier()
            role_model = Qwen4_5ForCausalLM(cfg, dtype=torch.bfloat16, device=device)
            load_qwen4_5_from_checkpoint(role_model, ckpt_dir, ple_mode="gpu_full")

        rtp = Qwen4_5RTPModel.from_causal_lm(role_model)
        plan = build_qwen4_5_kv_cache_plan(cfg)
        kv_pool = LocalKVCache(
            plan,
            tokens_per_block=tpb,
            num_blocks_per_tag={t: 128 for t in QWEN4_5_CACHE_TAGS},
            device=device,
            dtype=torch.bfloat16,
        )
        rtp.kv_cache = kv_pool

        if rank == 0:
            causal_ref_tokens, causal_ref_logits = _run_unified_baseline(
                role_model, prompts, 1
            )
            ref_tokens, ref_logits = _run_unified_rtp_baseline(
                role_model, prompts, max_new, tokens_per_block=tpb
            )
            p_tables = build_block_tables_for_requests(kv_pool, in_lens)
            writer = LocalMemoryCacheStoreWriter(record_payload=True)
            flat_ids = torch.tensor(
                [t for seq in prompts for t in seq], dtype=torch.int64, device=device
            )
            p_attn = make_tagged_attention_inputs(
                p_tables,
                is_prefill=True,
                input_lengths=in_lens,
                sequence_lengths=in_lens,
                prefix_lengths=[0, 0],
                device=device,
                cache_store_writer=writer,
            )
            p_out = rtp.forward(
                LocalPyModelInputs(input_ids=flat_ids, attention_inputs=p_attn)
            )
            last_idx = torch.tensor(
                [in_lens[0] - 1, in_lens[0] + in_lens[1] - 1],
                dtype=torch.long,
                device=device,
            )
            first_logits = rtp.compute_logits(p_out.hidden_states[last_idx])
            first_toks = torch.argmax(first_logits, dim=-1).tolist()

            meta = [
                {
                    "first_toks": first_toks,
                    "p_tables": p_tables,
                    "items": [
                        (lid, tag, list(blks))
                        for (lid, tag), (blks, _) in writer.published.items()
                    ],
                }
            ]
            dist.broadcast_object_list(meta, src=0, device=device)
            for (lid, tag), (_, payload) in writer.published.items():
                assert payload is not None
                dist.send(payload.contiguous(), dst=1)

            d_result: list[object] = [None]
            dist.broadcast_object_list(d_result, src=1, device=device)
            res_dict = d_result[0]  # type: ignore[index]
            pd_tokens = res_dict["gen_tokens"]
            pd_logits = [first_logits.detach().cpu()] + res_dict["step_logits"]

            total = len(prompts) * max_new
            matched = sum(
                int(pd_tokens[b][s] == ref_tokens[b][s])
                for b in range(len(prompts))
                for s in range(max_new)
            )
            min_cos = min(_cos_sim(pd_logits[s], ref_logits[s]) for s in range(max_new))
            print(
                f"[2-Proc NCCL PD GPU0->GPU1] matched={matched}/{total}, min_cos={min_cos:.6f} | "
                f"req0_toks={pd_tokens[0][:8]} text={tok.decode(pd_tokens[0][:8])!r}",
                flush=True,
            )
            prefill_cos = _cos_sim(pd_logits[0], causal_ref_logits[0])
            assert prefill_cos > 0.9999, f"Prefill CausalLM parity too low: {prefill_cos}"
            assert pd_tokens[0][0] == causal_ref_tokens[0][0] and pd_tokens[1][0] == causal_ref_tokens[1][0]
            assert matched == total, f"Token mismatch: {pd_tokens} vs {ref_tokens}"
            assert min_cos > 0.9999, f"Cosine too low: {min_cos}"
            print(
                "PASSED: 2-process NCCL PD-disaggregated generation exact match!",
                flush=True,
            )
        else:
            meta: list[object] = [None]
            dist.broadcast_object_list(meta, src=0, device=device)
            m0 = meta[0]  # type: ignore[index]
            first_toks = m0["first_toks"]
            p_tables = m0["p_tables"]
            items = m0["items"]

            d_tables = build_block_tables_for_requests(kv_pool, in_lens)
            id_map_by_tag: dict[str, dict[int, int]] = {}
            for tag, s_rows in p_tables.items():
                mp: dict[int, int] = {}
                for sr, dr in zip(s_rows, d_tables[tag]):
                    for sb, db in zip(sr, dr):
                        mp[int(sb)] = int(db)
                id_map_by_tag[tag] = mp

            for lid, tag, src_blks in items:
                dst_cache = kv_pool.get_layer_cache(lid, tag)
                buf = torch.empty(
                    (len(src_blks), *dst_cache.kv_cache_base.shape[1:]),
                    dtype=dst_cache.kv_cache_base.dtype,
                    device=device,
                )
                dist.recv(buf, src=0)
                id_map = id_map_by_tag[tag]
                for i, sb in enumerate(src_blks):
                    if sb in id_map:
                        dst_cache.kv_cache_base[id_map[sb]].copy_(buf[i])

            d_tables = build_block_tables_for_requests(
                kv_pool, [ln + max_new for ln in in_lens], existing_tables=d_tables
            )
            gen_tokens = [[int(first_toks[b])] for b in range(len(prompts))]
            step_logits_cpu: list[torch.Tensor] = []
            cur_lens = list(in_lens)
            cur_toks = list(first_toks)
            for _ in range(max_new - 1):
                d_attn = make_tagged_attention_inputs(
                    d_tables,
                    is_prefill=False,
                    input_lengths=[1, 1],
                    sequence_lengths=cur_lens,
                    prefix_lengths=[0, 0],
                    device=device,
                )
                d_out = rtp.forward(
                    LocalPyModelInputs(
                        input_ids=torch.tensor(cur_toks, dtype=torch.int64, device=device),
                        attention_inputs=d_attn,
                    )
                )
                logits = rtp.compute_logits(d_out.hidden_states)
                step_logits_cpu.append(logits.detach().cpu())
                cur_toks = torch.argmax(logits, dim=-1).tolist()
                for b in range(len(prompts)):
                    gen_tokens[b].append(int(cur_toks[b]))
                    cur_lens[b] += 1

            dist.broadcast_object_list(
                [{"gen_tokens": gen_tokens, "step_logits": step_logits_cpu}],
                src=1,
                device=device,
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if "--nccl-pd-test" in sys.argv:
        run_two_process_nccl_pd_test()
    else:
        unittest.main(verbosity=2)
