# Omni Correctness Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone script (`verify_omni_correctness.py`) that compares rtp-llm's Qwen2.5-Omni pipeline output against HuggingFace's reference implementation stage by stage, reporting pass/fail with numerical details.

**Architecture:** Single Python script with sequential execution — load HF model, run and save intermediates, free GPU, load rtp-llm engines, run and save intermediates, compare all checkpoints. Uses greedy decoding on both paths.

**Tech Stack:** Python 3.10, PyTorch, transformers (>=4.52 with Qwen2.5-Omni), rtp-llm C++ engine, NumPy

**Spec:** `docs/superpowers/specs/2026-06-10-omni-correctness-verification-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `verify_omni_correctness.py` | Create | Main verification script with all logic |

This is a single-file script (not a library). All functions live in one file, following the pattern of `bench_omni_e2e.py` and `bench_omni_perf.py` which are also single-file standalone scripts in the repo root.

---

### Task 1: Script skeleton — CLI, data classes, comparison utilities

**Files:**
- Create: `verify_omni_correctness.py`

This task creates the foundation: argument parsing, result data classes, the comparison logic, and the report printer. No model loading yet.

- [ ] **Step 1: Create the script with imports, data classes, and arg parsing**

```python
"""Verify rtp-llm omni pipeline correctness against HuggingFace reference.

Runs the same prompt through both HF's Qwen2_5OmniForConditionalGeneration
(decomposed into thinker/talker/token2wav) and rtp-llm's C++ engine pipeline,
comparing intermediate outputs at each stage.

Usage:
    # On mateng04 (requires GPU + rtp-llm build + transformers>=4.52):
    python verify_omni_correctness.py \
        --ckpt /root/models/Qwen/Qwen2.5-Omni-7B \
        --prompt "Tell me a joke."
"""
import argparse
import gc
import inspect
import json
import logging
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify_omni")


@dataclass
class StageResult:
    """Intermediates from one stage."""
    output_ids: Optional[torch.Tensor] = None     # [1, seq_len] int32
    hidden_states: Optional[torch.Tensor] = None   # [seq_len, hidden_dim] bf16
    codec_tokens: Optional[torch.Tensor] = None    # [1, seq_len] int64
    waveform: Optional[torch.Tensor] = None         # [1, num_samples] float32


@dataclass
class CheckpointResult:
    name: str
    passed: bool
    details: str


def compare_tokens(name: str, hf: torch.Tensor, rtp: torch.Tensor) -> CheckpointResult:
    """Compare token ID tensors for exact match."""
    hf_flat = hf.flatten().cpu()
    rtp_flat = rtp.flatten().cpu()

    min_len = min(len(hf_flat), len(rtp_flat))
    if len(hf_flat) != len(rtp_flat):
        # Still compare the overlapping prefix
        match = torch.equal(hf_flat[:min_len], rtp_flat[:min_len])
        prefix = "prefix match" if match else "prefix mismatch"
        return CheckpointResult(
            name=name,
            passed=False,
            details=f"Length mismatch: HF={len(hf_flat)}, rtp={len(rtp_flat)}, {prefix}",
        )

    if torch.equal(hf_flat, rtp_flat):
        return CheckpointResult(
            name=name,
            passed=True,
            details=f"{len(hf_flat)} tokens, exact match",
        )

    mismatches = (hf_flat != rtp_flat).nonzero(as_tuple=True)[0]
    first_pos = mismatches[0].item()
    return CheckpointResult(
        name=name,
        passed=False,
        details=(
            f"{len(mismatches)}/{len(hf_flat)} tokens differ, "
            f"first at pos {first_pos}: HF={hf_flat[first_pos].item()} vs rtp={rtp_flat[first_pos].item()}"
        ),
    )


def compare_tensors(
    name: str,
    hf: torch.Tensor,
    rtp: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> CheckpointResult:
    """Compare floating-point tensors with allclose."""
    hf_f = hf.float().cpu()
    rtp_f = rtp.float().cpu()

    if hf_f.shape != rtp_f.shape:
        return CheckpointResult(
            name=name,
            passed=False,
            details=f"Shape mismatch: HF={list(hf_f.shape)}, rtp={list(rtp_f.shape)}",
        )

    diff = (hf_f - rtp_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    passed = torch.allclose(hf_f, rtp_f, rtol=rtol, atol=atol)

    return CheckpointResult(
        name=name,
        passed=passed,
        details=f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, rtol={rtol}, atol={atol}",
    )


def compare_waveform(
    name: str,
    hf: torch.Tensor,
    rtp: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> CheckpointResult:
    """Compare waveforms with allclose and SNR."""
    hf_f = hf.float().cpu().flatten()
    rtp_f = rtp.float().cpu().flatten()

    min_len = min(len(hf_f), len(rtp_f))
    if len(hf_f) != len(rtp_f):
        hf_f = hf_f[:min_len]
        rtp_f = rtp_f[:min_len]
        len_note = f" (truncated to {min_len}, HF={hf.numel()}, rtp={rtp.numel()})"
    else:
        len_note = ""

    diff = (hf_f - rtp_f).abs()
    max_diff = diff.max().item()

    signal_power = (hf_f ** 2).mean().item()
    noise_power = (diff ** 2).mean().item()
    if noise_power > 0 and signal_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float('inf')

    passed = torch.allclose(hf_f, rtp_f, rtol=rtol, atol=atol)

    return CheckpointResult(
        name=name,
        passed=passed,
        details=f"max_diff={max_diff:.2e}, SNR={snr_db:.1f}dB{len_note}",
    )


def print_report(prompt: str, results: List[CheckpointResult]):
    """Print formatted comparison report."""
    print()
    print("=" * 72)
    print("  Omni Correctness Verification")
    print("=" * 72)
    print(f"Prompt: \"{prompt}\"")
    print()
    print(f"{'Checkpoint':<35} {'Status':<8} {'Details'}")
    print("-" * 72)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"{r.name:<35} {status:<8} {r.details}")
    print("-" * 72)

    all_pass = all(r.passed for r in results)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 72)
    print()


def save_wav(waveform: torch.Tensor, path: str, sample_rate: int = 24000):
    """Save a waveform tensor as a WAV file."""
    audio = waveform.squeeze().detach().cpu().float().numpy()
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    with open(path, "wb") as f:
        num_samples = len(audio_int16)
        data_size = num_samples * 2
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2))
        f.write(struct.pack("<H", 2))
        f.write(struct.pack("<H", 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(audio_int16.tobytes())
    logger.info(f"Saved WAV: {path} ({num_samples / sample_rate:.2f}s)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify rtp-llm omni vs HuggingFace reference"
    )
    parser.add_argument("--ckpt", required=True, help="Model checkpoint path")
    parser.add_argument("--prompt", default="Tell me a short joke.")
    parser.add_argument("--speaker", default="Chelsie")
    parser.add_argument("--max-thinker-tokens", type=int, default=256)
    parser.add_argument("--max-talker-tokens", type=int, default=2048)
    parser.add_argument("--save-intermediates", action="store_true",
                        help="Save .pt files for debugging")
    parser.add_argument("--output-dir", default="verify_output",
                        help="Directory for output WAVs and .pt files")
    return parser.parse_args()
```

- [ ] **Step 2: Add the `main()` stub that calls everything in sequence**

Add at the bottom of the same file:

```python
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Phase 1: HF reference
    logger.info("=" * 72)
    logger.info("Phase 1: Running HuggingFace reference path")
    logger.info("=" * 72)
    hf_result = run_hf_path(args)

    # Free GPU before loading rtp-llm
    logger.info("Freeing HF model GPU memory...")
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 2: rtp-llm
    logger.info("=" * 72)
    logger.info("Phase 2: Running rtp-llm engine path")
    logger.info("=" * 72)
    rtp_result = run_rtp_path(args)

    # Phase 3: Compare
    logger.info("=" * 72)
    logger.info("Phase 3: Comparing outputs")
    logger.info("=" * 72)
    results = compare_all(hf_result, rtp_result, args)

    print_report(args.prompt, results)

    if args.save_intermediates:
        save_path = os.path.join(args.output_dir, "intermediates.pt")
        torch.save({"hf": hf_result, "rtp": rtp_result}, save_path)
        logger.info(f"Saved intermediates to {save_path}")

    all_pass = all(r.passed for r in results)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the file parses**

Run: `python3 -c "import ast; ast.parse(open('verify_omni_correctness.py').read()); print('OK')"`
Expected: `OK` (will fail at runtime since `run_hf_path` etc. are not defined yet — that's fine, syntax is what matters here)

- [ ] **Step 4: Commit**

```bash
git add verify_omni_correctness.py
git commit -m "feat(omni): add correctness verification script skeleton"
```

---

### Task 2: HF reference path — `run_hf_path()`

**Files:**
- Modify: `verify_omni_correctness.py`

This task implements the HuggingFace reference path, decomposing `Qwen2_5OmniForConditionalGeneration.generate()` into its three stages to extract intermediates.

- [ ] **Step 1: Implement `run_hf_path()`**

Insert this function before `main()` in `verify_omni_correctness.py`:

```python
def run_hf_path(args) -> StageResult:
    """Run HF reference, decomposing generate() to extract per-stage intermediates."""
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    logger.info("Loading HF model...")
    t0 = time.time()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    logger.info(f"HF model loaded in {time.time() - t0:.1f}s")

    processor = Qwen2_5OmniProcessor.from_pretrained(args.ckpt)

    # Prepare input
    messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=text_input, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(model.device)
    prompt_len = input_ids.shape[1]
    logger.info(f"Input token count: {prompt_len}")

    # Load speaker params
    speaker_params = model.speaker_map[args.speaker]

    # === Stage 1: Thinker ===
    logger.info("Running HF thinker...")
    t_start = time.time()
    thinker_result = model.thinker.generate(
        input_ids=input_ids,
        output_hidden_states=True,
        return_dict_in_generate=True,
        max_new_tokens=args.max_thinker_tokens,
    )
    t_thinker = time.time() - t_start

    thinker_output_ids = thinker_result.sequences[:, prompt_len:]
    num_thinker_tokens = thinker_output_ids.shape[1]
    logger.info(f"HF thinker: {num_thinker_tokens} tokens in {t_thinker:.2f}s")

    # Extract hidden states the same way HF generate() does internally:
    # hidden_states is a tuple of (num_steps,) each containing (num_layers+1,) tensors
    # layer 0 = input embeddings, layer -1 = last hidden layer output
    embeds_to_talker = thinker_result.hidden_states[0][0].clone().to(input_ids.device)
    processed_thinker_hidden = (
        (embeds_to_talker,) + thinker_result.hidden_states[0][1:],
    ) + thinker_result.hidden_states[1:]

    thinker_generate_ids = thinker_result.sequences[:, prompt_len:].to(input_ids.device)

    thinker_token_embeds = [
        step[0].to(input_ids.device) for step in processed_thinker_hidden
    ]
    thinker_hidden_states = [
        step[-1].to(input_ids.device) for step in processed_thinker_hidden
    ]

    # Build the combined representation: last_hidden + input_embed per step
    # This is what HF passes to the talker as inputs_embeds
    hf_combined_hs_list = [
        hs + embed for hs, embed in zip(thinker_hidden_states, thinker_token_embeds)
    ]

    # For checkpoint comparison: stack the per-generated-token hidden states
    # (skip prompt steps — only the generated portion)
    # thinker_hidden_states[0] covers the prompt, [1:] cover generated tokens
    hf_last_hidden_gen = torch.cat(
        [step.squeeze(0) for step in thinker_hidden_states[1:]], dim=0
    )  # [num_gen_tokens, hidden_dim]

    logger.info(f"HF thinker hidden states shape: {hf_last_hidden_gen.shape}")

    # === Stage 2: Talker ===
    logger.info("Running HF talker...")
    t_start = time.time()

    # Reconstruct talker inputs exactly as HF generate() does
    talker_text_bos_token = speaker_params["bos_token"]
    talker_input_text_ids = torch.cat(
        [
            input_ids,
            torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=input_ids.device),
            thinker_generate_ids[:, :1],
        ],
        dim=-1,
    )

    talker_input_ids = torch.cat(
        [
            torch.full_like(input_ids, fill_value=model.talker.codec_mask_token),
            torch.tensor([[model.talker.codec_pad_token]], dtype=torch.long, device=input_ids.device),
            torch.tensor([[model.talker.codec_bos_token]], dtype=torch.long, device=input_ids.device),
        ],
        dim=1,
    )

    thinker_embed_tokens = model.thinker.get_input_embeddings()

    thinker_reply_part = (
        torch.cat(thinker_hidden_states[1:], dim=1)
        + torch.cat(thinker_token_embeds[1:], dim=1)
    )
    talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
    talker_text_bos_t = torch.tensor(
        [[talker_text_bos_token]], dtype=torch.long, device=input_ids.device
    )
    talker_text_bos_embed = thinker_embed_tokens(talker_text_bos_t).to(input_ids.device)
    talker_inputs_embeds = torch.cat(
        [talker_inputs_embeds, talker_text_bos_embed, thinker_reply_part[:, :1, :]],
        dim=1,
    )

    eos_token = torch.tensor(
        [[model.talker.text_eos_token]], dtype=torch.long, device=input_ids.device
    )
    eos_embedding = thinker_embed_tokens(eos_token).to(input_ids.device)
    pad_token = torch.tensor(
        [[model.talker.text_pad_token]], dtype=torch.long, device=input_ids.device
    )
    pad_embedding = thinker_embed_tokens(pad_token).to(input_ids.device)
    thinker_reply_part = torch.cat(
        [thinker_reply_part[:, 1:, :], eos_embedding, pad_embedding], dim=1
    )

    talker_result = model.talker.generate(
        input_ids=talker_input_ids,
        input_text_ids=talker_input_text_ids,
        thinker_reply_part=thinker_reply_part,
        inputs_embeds=talker_inputs_embeds,
        suppress_tokens=[model.talker.codec_bos_token],
        do_sample=False,
        top_k=1,
        max_new_tokens=args.max_talker_tokens,
        eos_token_id=[8292, 8294],
    )
    codec_tokens = talker_result[:, talker_input_ids.shape[1]:-1]
    t_talker = time.time() - t_start
    logger.info(f"HF talker: {codec_tokens.shape[1]} codec tokens in {t_talker:.2f}s")

    # === Stage 3: Token2wav ===
    logger.info("Running HF token2wav...")
    t_start = time.time()
    if model.token2wav.dtype != torch.float:
        model.token2wav.float()
    wav = model.token2wav(
        codec_tokens.to(input_ids.device),
        conditioning=speaker_params["cond"].to(input_ids.device).float(),
        reference_mel=speaker_params["ref_mel"].to(input_ids.device).float(),
    )
    t_t2w = time.time() - t_start
    logger.info(f"HF token2wav: {wav.numel()/24000:.2f}s audio in {t_t2w:.2f}s")

    # Save HF wav
    save_wav(wav, os.path.join(args.output_dir, "hf_output.wav"))

    result = StageResult(
        output_ids=thinker_output_ids.cpu(),
        hidden_states=hf_last_hidden_gen.cpu(),
        codec_tokens=codec_tokens.cpu(),
        waveform=wav.cpu(),
    )

    # Free HF model
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

    return result
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('verify_omni_correctness.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add verify_omni_correctness.py
git commit -m "feat(omni): implement HF reference path for correctness verification"
```

---

### Task 3: rtp-llm engine path — `run_rtp_path()`

**Files:**
- Modify: `verify_omni_correctness.py`

Implements the rtp-llm path using the C++ engine for both thinker and talker, following patterns from `bench_omni_perf.py`.

- [ ] **Step 1: Add engine creation helper**

Insert these helper functions before `run_hf_path()`:

```python
def create_engine_config(kv_cache_mb=4096, start_port=-100):
    """Create rtp-llm engine config (no gRPC server)."""
    from rtp_llm.ops import (
        ParallelismConfig, RuntimeConfig, FMHAConfig, DeviceResourceConfig,
        MoeConfig, NcclCommConfig, PDSepConfig, ConcurrencyConfig,
        ProfilingDebugLoggingConfig, HWKernelConfig, ModelSpecificConfig,
        SpeculativeExecutionConfig, CacheStoreConfig, MiscellaneousConfig,
        ArpcConfig, GrpcConfig,
    )
    from rtp_llm.config.kv_cache_config import KVCacheConfig
    from rtp_llm.config.py_config_modules import ServerConfig, LoadConfig
    from rtp_llm.config.engine_config import EngineConfig

    server_config = ServerConfig()
    server_config.start_port = start_port

    kv_cache_config = KVCacheConfig()
    kv_cache_config.kv_cache_mem_mb = kv_cache_mb
    kv_cache_config.test_block_num = 0

    return EngineConfig(
        parallelism_config=ParallelismConfig(),
        runtime_config=RuntimeConfig(),
        nccl_comm_config=NcclCommConfig(),
        server_config=server_config,
        pd_sep_config=PDSepConfig(),
        concurrency_config=ConcurrencyConfig(),
        fmha_config=FMHAConfig(),
        kv_cache_config=kv_cache_config,
        profiling_debug_logging_config=ProfilingDebugLoggingConfig(),
        hw_kernel_config=HWKernelConfig(),
        device_resource_config=DeviceResourceConfig(),
        moe_config=MoeConfig(),
        model_specific_config=ModelSpecificConfig(),
        sp_config=SpeculativeExecutionConfig(),
        cache_store_config=CacheStoreConfig(),
        misc_config=MiscellaneousConfig(),
        arpc_config=ArpcConfig(),
        grpc_config=GrpcConfig(),
        load_config=LoadConfig(),
    )


def make_engine(model_cls, ckpt_path, engine_config, model_type,
                max_seq_len=4096, vit_config=None):
    """Load model and start C++ engine."""
    from rtp_llm.async_decoder_engine.engine_creator import create_engine

    config = model_cls._create_config(ckpt_path)
    config.ckpt_path = ckpt_path
    config.tokenizer_path = ckpt_path
    config.model_type = model_type
    config.max_seq_len = max_seq_len
    config.use_kvcache = True
    config.phy2log_path = ""
    config.init_precision_config(
        kv_cache_config=engine_config.kv_cache_config, act_type=None
    )

    kwargs = dict(
        model_config=config,
        parallelism_config=engine_config.parallelism_config,
        hw_kernel_config=engine_config.hw_kernel_config,
        kv_cache_config=engine_config.kv_cache_config,
        fmha_config=engine_config.fmha_config,
        moe_config=engine_config.moe_config,
        load_method=engine_config.load_config.load_method,
        max_generate_batch_size=engine_config.runtime_config.max_generate_batch_size,
        vit_config=vit_config,
        merge_lora=False,
        device_resource_config=engine_config.device_resource_config,
        force_cpu_load_weights=engine_config.load_config.force_cpu_load_weights,
    )
    sig = inspect.signature(model_cls.from_config)
    if 'load_python_model' in sig.parameters:
        kwargs['load_python_model'] = True
    if 'skip_python_model' in sig.parameters:
        kwargs['skip_python_model'] = False

    model = model_cls.from_config(**kwargs)
    engine = create_engine(
        model=model,
        engine_config=engine_config,
        alog_conf_path=engine_config.profiling_debug_logging_config.ft_alog_conf_path,
        world_info=None,
    )
    engine.start()
    return engine, model
```

- [ ] **Step 2: Implement `run_rtp_path()`**

Insert after `run_hf_path()`:

```python
def run_rtp_path(args) -> StageResult:
    """Run rtp-llm engine path, extracting per-stage intermediates."""
    from rtp_llm.omni.models.qwen2_5_omni.thinker import Qwen2_5OmniThinker
    from rtp_llm.omni.models.qwen2_5_omni.talker import Qwen2_5OmniTalker
    from rtp_llm.omni.models.qwen2_5_omni.token2wav_model import Token2WavModel
    from rtp_llm.config.py_config_modules import VitConfig
    from transformers import AutoTokenizer

    device = "cuda:0"

    # Tokenize prompt
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text)
    logger.info(f"rtp-llm input tokens: {len(prompt_ids)}")

    # === Stage 1: Thinker ===
    logger.info("Loading rtp-llm thinker engine...")
    t0 = time.time()
    vit_config = VitConfig()
    thinker_ec = create_engine_config(kv_cache_mb=512)
    thinker_engine, thinker_model = make_engine(
        Qwen2_5OmniThinker, args.ckpt, thinker_ec,
        "qwen2_5_omni_thinker", max_seq_len=2048, vit_config=vit_config,
    )
    logger.info(f"Thinker engine loaded in {time.time() - t0:.1f}s")

    input_ids = torch.tensor(prompt_ids, dtype=torch.int32)
    eos_token_id = tokenizer.eos_token_id or 151643

    logger.info("Running rtp-llm thinker (return_hidden_states=True)...")
    t_start = time.time()
    output_ids, hidden_states = thinker_engine.rtp_llm_op_.generate(
        input_ids,
        max_new_tokens=args.max_thinker_tokens,
        eos_token_id=eos_token_id,
        return_hidden_states=True,
    )
    t_thinker = time.time() - t_start

    num_thinker_tokens = output_ids.shape[1]
    logger.info(
        f"rtp-llm thinker: {num_thinker_tokens} tokens in {t_thinker:.2f}s, "
        f"hidden_states shape: {hidden_states.shape}"
    )

    # Free thinker engine before loading talker
    thinker_engine.stop()
    del thinker_engine, thinker_model
    gc.collect()
    torch.cuda.empty_cache()

    # === Stage 2: Talker ===
    logger.info("Loading rtp-llm talker engine...")
    t0 = time.time()
    talker_ec = create_engine_config(kv_cache_mb=256)
    talker_engine, talker_model = make_engine(
        Qwen2_5OmniTalker, args.ckpt, talker_ec,
        "qwen2_5_omni_talker", max_seq_len=2048,
    )
    logger.info(f"Talker engine loaded in {time.time() - t0:.1f}s")

    # Load speaker data
    spk_dict = torch.load(os.path.join(args.ckpt, "spk_dict.pt"), map_location=device)
    speaker_data = spk_dict[args.speaker]
    speaker_bos = speaker_data["bos_token"]
    cond = speaker_data["cond"].float().to(device)
    ref_mel = speaker_data["ref_mel"].float().to(device)

    # Set thinker hidden states on the Python model
    thinker_hs = hidden_states.to(dtype=torch.bfloat16, device=device)
    py_model = talker_model.py_model
    if py_model is not None and hasattr(py_model, 'set_thinker_hidden_states'):
        py_model.set_thinker_hidden_states(thinker_hs)

    logger.info("Running rtp-llm talker...")
    initial_tokens = torch.tensor([speaker_bos], dtype=torch.int32)
    t_start = time.time()
    codec_tokens, _ = talker_engine.rtp_llm_op_.generate(
        initial_tokens, args.max_talker_tokens, 8294
    )
    t_talker = time.time() - t_start
    logger.info(f"rtp-llm talker: {codec_tokens.shape[1]} codec tokens in {t_talker:.2f}s")

    if py_model is not None and hasattr(py_model, 'clear_thinker_hidden_states'):
        py_model.clear_thinker_hidden_states()

    # Filter special tokens for token2wav (same logic as omni_engine.py)
    mask = codec_tokens[0] < 8292
    codec_filtered = codec_tokens[0][mask].unsqueeze(0)
    logger.info(f"Codec tokens after filter: {codec_filtered.shape[1]}")

    # Free talker engine
    talker_engine.stop()
    del talker_engine, talker_model
    gc.collect()
    torch.cuda.empty_cache()

    # === Stage 3: Token2wav ===
    logger.info("Loading token2wav...")
    t0 = time.time()
    token2wav = Token2WavModel.from_pretrained(args.ckpt, device=device)
    logger.info(f"Token2wav loaded in {time.time() - t0:.1f}s")

    logger.info("Running rtp-llm token2wav...")
    t_start = time.time()
    with torch.no_grad():
        wav = token2wav(
            codec_filtered.to(device),
            conditioning=cond,
            reference_mel=ref_mel,
        )
    t_t2w = time.time() - t_start
    logger.info(f"rtp-llm token2wav: {wav.numel()/24000:.2f}s audio in {t_t2w:.2f}s")

    save_wav(wav, os.path.join(args.output_dir, "rtp_output.wav"))

    del token2wav
    gc.collect()
    torch.cuda.empty_cache()

    return StageResult(
        output_ids=output_ids.cpu(),
        hidden_states=hidden_states.cpu(),
        codec_tokens=codec_tokens.cpu(),
        waveform=wav.cpu(),
    )
```

- [ ] **Step 3: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('verify_omni_correctness.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add verify_omni_correctness.py
git commit -m "feat(omni): implement rtp-llm engine path for correctness verification"
```

---

### Task 4: Comparison orchestration — `compare_all()`

**Files:**
- Modify: `verify_omni_correctness.py`

Ties together the comparison utilities to compare all checkpoints.

- [ ] **Step 1: Implement `compare_all()`**

Insert after `run_rtp_path()`:

```python
def compare_all(
    hf: StageResult, rtp: StageResult, args
) -> List[CheckpointResult]:
    """Compare all checkpoints between HF and rtp-llm results."""
    results = []

    # Checkpoint 1a: Thinker output token IDs
    if hf.output_ids is not None and rtp.output_ids is not None:
        results.append(compare_tokens("Thinker output_ids", hf.output_ids, rtp.output_ids))
    else:
        results.append(CheckpointResult("Thinker output_ids", False, "Missing data"))

    # Checkpoint 1b: Thinker hidden states
    if hf.hidden_states is not None and rtp.hidden_states is not None:
        results.append(compare_tensors(
            "Thinker hidden_states",
            hf.hidden_states, rtp.hidden_states,
            rtol=1e-2, atol=1e-3,
        ))
    else:
        results.append(CheckpointResult("Thinker hidden_states", False, "Missing data"))

    # Checkpoint 2: Talker codec tokens
    if hf.codec_tokens is not None and rtp.codec_tokens is not None:
        results.append(compare_tokens("Talker codec_tokens", hf.codec_tokens, rtp.codec_tokens))
    else:
        results.append(CheckpointResult("Talker codec_tokens", False, "Missing data"))

    # Checkpoint 3: Waveform
    if hf.waveform is not None and rtp.waveform is not None:
        results.append(compare_waveform(
            "Waveform", hf.waveform, rtp.waveform,
            rtol=1e-4, atol=1e-5,
        ))
    else:
        results.append(CheckpointResult("Waveform", False, "Missing data"))

    # Save comparison artifacts
    if args.save_intermediates:
        out_dir = args.output_dir
        for name, hf_t, rtp_t in [
            ("thinker_output_ids", hf.output_ids, rtp.output_ids),
            ("thinker_hidden_states", hf.hidden_states, rtp.hidden_states),
            ("codec_tokens", hf.codec_tokens, rtp.codec_tokens),
            ("waveform", hf.waveform, rtp.waveform),
        ]:
            if hf_t is not None and rtp_t is not None:
                torch.save(
                    {"hf": hf_t.cpu(), "rtp": rtp_t.cpu()},
                    os.path.join(out_dir, f"{name}.pt"),
                )
        logger.info(f"Saved comparison .pt files to {out_dir}/")

    return results
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('verify_omni_correctness.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add verify_omni_correctness.py
git commit -m "feat(omni): add comparison orchestration for correctness verification"
```

---

### Task 5: Environment setup and remote test

**Files:**
- No new files — remote execution only

This task sets up the environment on mateng04 and runs the verification script.

- [ ] **Step 1: Install transformers with Qwen2.5-Omni support in rtp-llm env**

Run on mateng04:
```bash
ssh root@mateng04 "/root/miniconda3/envs/rtp-llm-py310/bin/pip install 'transformers>=4.52'"
```

Expected: Successfully installs transformers 4.52+ with Qwen2.5-Omni model support.

- [ ] **Step 2: Sync the script to remote**

```bash
scp verify_omni_correctness.py root@mateng04:/root/mateng/rtp-llm/
```

- [ ] **Step 3: Run the verification on mateng04**

```bash
ssh root@mateng04 "cd /root/mateng/rtp-llm && \
    CUDA_VISIBLE_DEVICES=5,6 /root/miniconda3/envs/rtp-llm-py310/bin/python \
    verify_omni_correctness.py \
    --ckpt /root/models/Qwen/Qwen2.5-Omni-7B \
    --prompt 'Tell me a short joke.' \
    --save-intermediates"
```

Expected output: The verification report table showing PASS/FAIL for each checkpoint.

- [ ] **Step 4: Analyze results**

If any checkpoint fails:
- Check `verify_output/*.pt` files to inspect the divergent tensors
- Listen to `verify_output/hf_output.wav` and `verify_output/rtp_output.wav`
- If thinker hidden states diverge, investigate whether rtp-llm `return_hidden_states` returns last-layer-only vs last-layer + embeddings

- [ ] **Step 5: Commit the final script**

```bash
git add verify_omni_correctness.py
git commit -m "feat(omni): complete correctness verification script with remote test"
```
