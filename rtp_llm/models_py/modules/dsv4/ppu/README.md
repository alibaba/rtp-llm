# DeepSeek V4 Flash on PPU

This package contains the PPU implementations of DSV4 attention, MoE, HC,
metadata, and prefill/decode orchestration. It follows the existing model
layers: checkpoint loading stays in `rtp_llm/models/deepseek_v4.py`,
Python model entry points in `models_py/model_desc/deepseek_v4_ppu`, reusable
operators in the existing `factory/*/ppu_impl` directories, and kernels/JIT
sources in `models_py/kernels/ppu`. Native bindings stay in `bindings/ppu`.

The standard `DeepSeekV4` and `DeepSeekV4Mtp` classes own loading and select
PPU implementations only for Flash on PPU. The factory keeps the normal
registry and prepares scheduler capacities before target and draft loading.

`PpuModuleProvider` binds prefill operators directly; `PpuDecodeProvider`
extends it with decode buffers and scheduling. The FP4 attention class owns
its inverse-RoPE projection, and `attention/ppu_wo_a.py` owns the grouped
output projection. Routing arithmetic and PPU weight binding live together
in `moe/gate.py`; the shared gate remains their common weight contract.
The private modules require an instance-bound PPU provider; they do not
carry a second CUDA provider or a capability registry. Prefill retains the
ordinary metadata factory and no fused MTP input; Decode supplies its
optional metadata and fusion implementations. The DeepEP MXFP4 dispatch,
expert compute, and combine adapter belongs to `moe/ppu_deepep_fp4.py`.

The MTP implementation inherits the shared constructor, forward path, hidden
handoff, and chunked fusion loop. Its PPU specialization supplies FP8
projections, restores sharded TP embeddings before normalization, and offers
an optional fused input operator. The shared embedding hook defaults to the
original embedding call; other platforms retain their behavior.

MegaMoE is supplied by the separately built [`deep-gemm-mega` fork](https://github.com/fxp19950422/DeepGEMM-for-sail/tree/c144b7ac0f3930feb4cb8e32dc10c74f28115bb7), pinned
with its SHA256 in the PPU dependency lock. It coexists with the official
`deep_gemm` package and owns its native transport libraries and LUT. RTP-LLM
does not patch or build that dependency during installation.

Tests live in `test/`; automatic PPU contracts and numerical tests retain
810E placement, while M890P-only tests remain explicitly manual. See
[OPTIONAL_FEATURES.md](OPTIONAL_FEATURES.md) for runtime feature gates.
