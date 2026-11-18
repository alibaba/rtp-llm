# ROCm ATOM full MTP draft-loop graph

Opt-in port of the full draft-loop graph from alibaba/rtp-llm PR #1697.
This is not the deferred-tail-wait experiment. The captured loop includes
both draft forwards, LM heads, TP gathers, greedy sampling and inter-step
metadata updates for `gen_num_per_cycle=3`.

## Installation and activation

Build RTP-LLM from the accompanying C++ changes. Install the pinned ATOM
`0.1.8+g9af739efc` wheel and existing Qwen compatibility/HC/TopK patches first,
then run:

```sh
python patches/atom/draft_loop/apply.py
python patches/atom/draft_loop/apply.py --verify-only
export RTP_LLM_MTP_DRAFT_LOOP_GRAPH=1
```

Set the variable only on the decode service; restart workers after changing it.
It defaults off. NVIDIA, other plugins, prefill,
fake/warmup requests and requests ineligible for CUDA graph keep the old path.
The hook checks shared device sequence-length storage and captured metadata.
Padded rows remain at zero; device lengths and all three token-major MRoPE
axes advance together between steps and are restored after each loop.
The recurrent input is the pre-collapse HC hidden tensor, not the LM-head input.
This branch's `computeLogits` already gathers TP logits: no second gather is added.

The installer checks the wheel version, exact source/patch/output hashes,
Python syntax and supports repeat application. It rejects unknown modifications.
The original wheel hash is unchanged. Apply base patches before this overlay.
Do not rerun the base patch verifier after overlays that change its tracked files.
