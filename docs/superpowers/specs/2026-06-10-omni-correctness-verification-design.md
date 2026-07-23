# Omni Pipeline Correctness Verification — Design Spec

## Goal

Verify that rtp-llm's Qwen2.5-Omni multi-stage pipeline produces numerically identical results to the HuggingFace reference implementation (`Qwen2_5OmniForConditionalGeneration`), stage by stage.

## Scope

- **All three stages**: thinker, talker, token2wav
- **Exact greedy match** for token IDs, `allclose` for floating-point tensors
- **Standalone script** (`verify_omni_correctness.py`) run manually on mateng04
- **Single-process**: both HF and rtp-llm engines loaded in the same Python process

## Architecture

```
                    Same prompt (greedy, temp=0)
                           │
              ┌────────────┴────────────┐
              │                         │
         HF Reference              rtp-llm Path
              │                         │
    ┌─────────┴─────────┐    ┌─────────┴─────────┐
    │  HF Thinker        │    │  rtp Thinker       │
    │  (model.thinker    │    │  (C++ engine,      │
    │   .generate())     │    │   return_hidden_   │
    └────────┬───────────┘    │   states=True)     │
             │                └────────┬───────────┘
             │                         │
    ═══ Checkpoint 1: thinker output_ids + hidden_states ═══
             │                         │
    ┌────────┴───────────┐    ┌────────┴───────────┐
    │  HF Talker         │    │  rtp Talker        │
    │  (model.talker     │    │  (C++ engine,      │
    │   .generate())     │    │   greedy decode)   │
    └────────┬───────────┘    └────────┬───────────┘
             │                         │
    ═══ Checkpoint 2: talker codec token IDs ═══
             │                         │
    ┌────────┴───────────┐    ┌────────┴───────────┐
    │  Token2Wav (shared │    │  Token2Wav (shared │
    │  PyTorch module)   │    │  PyTorch module)   │
    └────────┬───────────┘    └────────┬───────────┘
             │                         │
    ═══ Checkpoint 3: waveform ═══
```

## HF Reference Decomposition

The HF `Qwen2_5OmniForConditionalGeneration.generate()` bundles all stages. To extract per-stage intermediates, we decompose it following the same internal logic:

### Step 1: Thinker

```python
thinker_result = model.thinker.generate(
    input_ids=input_ids,
    output_hidden_states=True,
    return_dict_in_generate=True,
    max_new_tokens=max_thinker_tokens,
    **processor_kwargs,
)
# thinker_result.sequences: [1, prompt_len + gen_len]
# thinker_result.hidden_states: tuple of (num_steps,) each (num_layers+1,) tensors
```

### Step 2: Hidden State Processing (thinker → talker transform)

The HF model constructs talker input embeddings as:

```python
# Per-step: input_embed (layer 0) + last_hidden (layer -1)
thinker_token_embeds = [step[0] for step in thinker_result.hidden_states]
thinker_hidden_states = [step[-1] for step in thinker_result.hidden_states]
# Combined: hidden + embed for each step
talker_hs = [hs + embed for hs, embed in zip(thinker_hidden_states, thinker_token_embeds)]
```

This is a critical comparison point — the rtp-llm path gets hidden states from `rtp_llm_op_.generate(return_hidden_states=True)` which returns the last-layer hidden states only. The rtp-llm `Qwen2_5OmniTalkerModel.forward()` then adds `embed_tokens(codec_token) + thinker_hidden_states` and projects.

**Key difference to verify**: The rtp-llm thinker `return_hidden_states` returns last-layer hidden states. The HF path uses `last_hidden + input_embeddings`. We need to check whether the rtp-llm path matches this combined representation or if there's a discrepancy.

### Step 3: Talker

```python
# HF: model.talker.generate() with custom inputs_embeds
talker_result = model.talker.generate(
    input_ids=talker_input_ids,        # codec token track
    input_text_ids=talker_input_text_ids,  # text token track
    inputs_embeds=talker_inputs_embeds,     # combined hs+embed
    thinker_reply_part=thinker_reply_part,
    do_sample=False, top_k=1,          # force greedy for comparison
    max_new_tokens=max_talker_tokens,
    eos_token_id=[8292, 8294],
)
codec_tokens = talker_result[:, input_len:-1]
```

### Step 4: Token2wav

```python
wav = model.token2wav(
    codec_tokens,
    conditioning=speaker_params["cond"].float(),
    reference_mel=speaker_params["ref_mel"].float(),
)
```

## rtp-llm Path

### Thinker

Uses the rtp-llm C++ engine via HTTP API (streaming, `return_hidden_states=True`):

```python
# Option A: via HTTP API (requires running thinker server)
text, per_token_hs = call_thinker_streaming(prompt, thinker_url, max_tokens)
# per_token_hs: list of [hidden_size] vectors (last-layer hidden states)

# Option B: via direct engine call (loaded in-process)
output_ids, hidden_states = thinker_engine.rtp_llm_op_.generate(
    input_ids, max_new_tokens, eos_token_id, return_hidden_states=True
)
```

For the comparison script, we use Option B (direct engine call) to avoid HTTP serialization differences.

### Talker

```python
# Set thinker hidden states on the Python model
py_model.set_thinker_hidden_states(thinker_hs)

# Generate via C++ engine
codec_tokens, _ = talker_engine.rtp_llm_op_.generate(
    initial_tokens, max_talker_tokens, eos_token_id
)
```

### Token2wav

Same PyTorch module as HF — `Token2WavModel.from_pretrained()`.

## Comparison Tolerances

| Checkpoint | Metric | Tolerance |
|---|---|---|
| Thinker output_ids | Exact match | `torch.equal` |
| Thinker hidden_states | bf16 allclose | `rtol=1e-2, atol=1e-3` |
| Thinker→talker transform | bf16 allclose | `rtol=1e-2, atol=1e-3` |
| Talker codec_tokens | Exact match | `torch.equal` |
| Waveform | float32 allclose | `rtol=1e-4, atol=1e-5` |

## Script Interface

```bash
# Run on mateng04 with GPU
python verify_omni_correctness.py \
    --ckpt /root/models/Qwen/Qwen2.5-Omni-7B \
    --prompt "Tell me a joke." \
    --max-thinker-tokens 256 \
    --max-talker-tokens 2048 \
    --speaker Chelsie \
    --save-intermediates  # optional: save .pt files for debugging
```

## Script Structure

```
verify_omni_correctness.py
├── load_hf_model(ckpt)           → model, processor
├── load_rtp_engines(ckpt)        → thinker_engine, talker_engine
├── run_hf_decomposed(model, ...) → HFResult(output_ids, hidden_states,
│                                              talker_hs, codec_tokens, waveform)
├── run_rtp_decomposed(engines, ...)→ RTPResult(same fields)
├── compare_checkpoint(name, hf, rtp, tolerance) → CompareResult(pass/fail, details)
├── print_report(results)         → formatted table
└── main()                        → parse args, run both, compare, exit code
```

## Output Format

```
=== Omni Correctness Verification ===
Prompt: "Tell me a joke."
Checkpoint                    | Status | Details
──────────────────────────────┼────────┼─────────────────────────
Thinker output_ids            | PASS   | 47 tokens, exact match
Thinker hidden_states         | PASS   | max_diff=2.3e-4, mean=1.1e-5
Thinker→talker hs transform   | PASS   | max_diff=3.1e-4, mean=2.0e-5
Talker codec_tokens           | PASS   | 312 tokens, exact match
Waveform                      | PASS   | max_diff=1.2e-6, SNR=85.3dB
──────────────────────────────┼────────┼─────────────────────────
Overall: ALL PASS
```

Exit code 0 = all pass, 1 = any fail. Does not abort early — runs all checkpoints.

## Environment Requirements

- **Conda env**: rtp-llm-py310 with `transformers>=4.52` installed (for Qwen2.5-Omni HF model support)
- **GPU**: 2+ GPUs recommended (HF model uses device_map="auto", rtp-llm engine uses separate GPU)
- **Model checkpoint**: `/root/models/Qwen/Qwen2.5-Omni-7B`

## Known Risks

1. **Hidden state representation mismatch**: The HF path uses `last_hidden + input_embeds` while rtp-llm's `return_hidden_states` might return only last-layer hidden states. If so, we need to also extract input embeddings from the rtp-llm thinker to reconstruct the combined representation. The script should detect this by comparing the thinker hidden states against both `last_hidden` alone and `last_hidden + input_embeds`, and report which one matches.

2. **Talker input construction**: The HF talker has a complex input setup with dual tracks (codec IDs + text IDs), while rtp-llm's talker py_model does a simpler `embed + hs → proj`. The comparison script will expose whether these are equivalent. If they diverge, the script will print the first divergent token position and the logit distributions at that point.

3. **GPU memory**: Loading both HF model (~14GB) and rtp-llm engines (~7GB each) in one process requires ~28GB VRAM. Strategy: load HF model first with `device_map="auto"`, run and save intermediates, then `del model` and `torch.cuda.empty_cache()` before loading rtp-llm engines. This sequences GPU usage rather than requiring simultaneous allocation.

4. **Determinism**: Both paths must use greedy decoding. HF talker defaults to sampling — the script overrides with `do_sample=False, top_k=1`. The rtp-llm C++ engine already defaults to `top_k=1, do_sample=false` in `RtpLLMOp::generate()`.
