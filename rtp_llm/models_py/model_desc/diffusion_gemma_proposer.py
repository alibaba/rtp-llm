"""DiffusionGemma block-diffusion proposer (DSpark PROPOSE role).

One engine round = one full canvas (block_length*8 = 256 tokens by default):
the python forward_propose runs the entropy-bounded denoising loop over the
canvas (bidirectional decoder attention, read-only encoder KV), runs the EB
sampler exactly like the HF reference (temperature schedule 0.8 -> 0.4,
lowest-entropy acceptance under the mutual-information bound, full renoising,
adaptive stopping; the entropy and the per-step categorical draw are fused
into single-launch Triton kernels on CUDA — see the gate notes below, both
A/B-checked against their torch reference paths — and since opt r5 the
softcap/temperature transform is applied in-register by those kernels plus
a single softcap_argmax launch, while the accept mask runs as one
accept_mask launch), and returns the final
argmax canvas tokens through
``PyModelOutputs.diffusion_tokens``. The C++ target-verify forward of the
same round is the canvas commit (encoder causal forward writing the main KV
cache). All backbone weights are the main model's modules — nothing is loaded
twice and no separate draft KV exists: the canvas attention reads the main
cache through the proposer's reference to the main model.
"""

import logging
import math
import os
from typing import Any, Optional

import torch
from torch import nn

from rtp_llm.models.diffusion_gemma_registry import get_main_model
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs

logger = logging.getLogger(__name__)

# Contract #5-iv: single-launch fused entropy kernel (logsumexp + p*logp in
# one reduction) for the EB sampler's two Categorical.entropy() calls per
# step — the torch path burns 8+ full-vocab fp32 passes per call. The
# reference path stays available: DGM_FUSED_ENTROPY=0 forces it (A/B gate).
_FUSED_ENTROPY_ENV = os.environ.get("DGM_FUSED_ENTROPY", "1") != "0"
_fused_entropy_kernel = None
try:
    from rtp_llm.models_py.triton_kernels.common.categorical_entropy import (
        categorical_entropy,
    )

    _fused_entropy_kernel = categorical_entropy
except Exception as _entropy_error:  # CPU-only sandboxes without triton
    logger.info(
        "DiffusionGemma fused entropy kernel unavailable, torch path: %s",
        _entropy_error,
    )

# Contract opt-r4 lane B: single-launch fused sampling kernel (Gumbel-max
# argmax over logits + noise) for the EB sampler's per-step Categorical draw
# — the torch chain materializes a full [rows, V] fp32 softmax block (read +
# write) and then walks it again inside multinomial, three-plus full-vocab
# passes per step where the profiled softmax alone cost 73.1ms x 210 over one
# decode window. The kernel reads the logits once and emits token ids; it
# samples the SAME distribution (diffusion_gemma_proposer_test: same-seed
# determinism + empirical TV/chi-square agreement with the torch reference)
# through its own counter-based RNG stream, so token sequences differ from
# the torch path by design. The reference path stays available:
# DGM_FUSED_SAMPLING=0 forces it (A/B gate; also the state the recorded
# smoke goldens were produced under).
_FUSED_SAMPLING_ENV = os.environ.get("DGM_FUSED_SAMPLING", "1") != "0"
_fused_sample_kernel = None
try:
    from rtp_llm.models_py.triton_kernels.common.categorical_sample import (
        categorical_sample,
    )

    _fused_sample_kernel = categorical_sample
except Exception as _sampling_error:  # CPU-only sandboxes without triton
    logger.info(
        "DiffusionGemma fused sampling kernel unavailable, torch path: %s",
        _sampling_error,
    )

# opt r5 ⑥-clearing lane, item 2 (softcap inline): the per-step epilogue's
# full-vocab elementwise chain — fp32 cast + tanh(x/s) pass + *(s/t) pass
# (52.6 ms/window) plus torch.argmax (15.8 ms) — collapses into one
# softcap_argmax launch that reads the RAW bf16 logits once, materializes
# the transformed fp32 row for the self-conditioning consumer and emits the
# row argmax; the entropy/sampling kernels additionally read the raw logits
# with the transform applied in-register (contract #5-iii merged algebra
# tanh(x/s)*(s/t), <= 2 ulp vs the sequential chain). DGM_FUSED_SOFTCAP=0
# restores the materialized torch chain (A/B gate; the state the r75 goldens
# were recorded under modulo the RNG stream).
_FUSED_SOFTCAP_ENV = os.environ.get("DGM_FUSED_SOFTCAP", "1") != "0"


def _disable_softcap() -> None:
    """Degrade switch: after a kernel failure the fused softcap path stays
    off for the rest of the process (r4 fused-kernel contract)."""
    global _FUSED_SOFTCAP_ENV
    _FUSED_SOFTCAP_ENV = False


_softcap_argmax_kernel = None
try:
    from rtp_llm.models_py.triton_kernels.common.softcap_argmax import (
        softcap_argmax,
    )

    _softcap_argmax_kernel = softcap_argmax
except Exception as _softcap_error:  # CPU-only sandboxes without triton
    logger.info(
        "DiffusionGemma fused softcap_argmax kernel unavailable, torch path: %s",
        _softcap_error,
    )

# opt r9 ⑥ tail-clear lane, item 2 (softmax fold): with the fused softcap
# path active, the ONLY consumer of the materialized fp32 ``processed``
# tensor is the next step's self-conditioning
# ``softmax(processed).to(embed_dtype)`` matmul in forward_canvas — the
# ATen cunn_SoftMaxForward (42.2 ms / 122x per decode window) plus the
# full-vocab fp32->bf16 cast. ``emit_softmax`` computes the row softmax
# inside the softcap kernel (row max / exp-sum / normalize in fp32
# registers, three raw-logits reads) and hands forward_canvas bf16
# probabilities directly: processed, the ATen softmax and the cast pass
# all disappear. The entropy/sampling/accept kernels keep reading the RAW
# logits through logits_ctx (unchanged); their torch fallbacks reconstruct
# the reference chain from the ctx when the caller runs in probs mode
# (``_ctx_logits`` — bit-identical to the materialized torch chain).
# DGM_FUSED_PROBS=0 restores the processed+ATen-softmax path (A/B gate;
# DGM_PROBS_AB=1 prints the fused-vs-reference referee stats,
# [dgm-probs-ab]).
_FUSED_PROBS_ENV = os.environ.get("DGM_FUSED_PROBS", "1") != "0"
_PROBS_AB_CHECK = os.environ.get("DGM_PROBS_AB", "") not in ("", "0")


def _ctx_logits(logits: Optional[torch.Tensor], logits_ctx) -> torch.Tensor:
    """Materialize the reference softcap chain for the torch fallbacks.

    In probs mode the caller passes ``logits=None`` (nothing materialized);
    the reference the fallback must match is the exact torch sequence
    ``tanh(raw.to(fp32) / softcap) * out_scale`` reconstructed from the ctx
    (same ops in the same order as the proposer's torch chain — bit-equal).
    """
    if logits is not None:
        return logits
    if logits_ctx is None:
        raise ValueError("no logits and no logits_ctx to reconstruct from")
    raw, softcap, out_scale = logits_ctx
    return torch.tanh(raw.to(torch.float32) / softcap) * out_scale

# opt r5 ⑥-clearing lane, item 3 (accept-mask compact): the reference
# sort/cumsum/scatter chain on [B, canvas] entropy rows was 3540 radixSort
# invocations / 68.1 ms per decode window (six-plus tiny launches per step —
# launch-overhead bound). One accept_mask launch preserves the exact
# ascending-entropy budget semantics (the sort IS the acceptance rule, not
# mask compaction — see the kernel module docstring).
# DGM_FUSED_ACCEPT=0 forces the torch chain.
_FUSED_ACCEPT_ENV = os.environ.get("DGM_FUSED_ACCEPT", "1") != "0"
_fused_accept_kernel = None
try:
    from rtp_llm.models_py.triton_kernels.common.accept_mask import (
        accept_mask,
    )

    _fused_accept_kernel = accept_mask
except Exception as _accept_error:  # CPU-only sandboxes without triton
    logger.info(
        "DiffusionGemma fused accept_mask kernel unavailable, torch path: %s",
        _accept_error,
    )


def _categorical_entropy(logits: torch.Tensor, logits_ctx=None) -> torch.Tensor:
    """Categorical entropy over the last dim: fused Triton kernel on CUDA
    fp32, torch.distributions reference path otherwise. The fused kernel is
    A/B-checked against the reference (fp32 reduction-order noise, entropy
    diff < 1e-5 — see diffusion_gemma_proposer_test); any kernel failure
    degrades to the reference path with a logged warning.

    ``logits_ctx`` = (raw_logits, softcap, out_scale) (opt r5 softcap
    inline): when given, the kernel reads the RAW logits and computes the
    entropy of Categorical(tanh(x/s) * out_scale) in-register, skipping the
    materialized transform passes; ``logits`` (the materialized processed
    row) only serves the reference fallback. In r9 probs mode ``logits``
    is None and the reference fallback reconstructs the chain from the ctx
    (``_ctx_logits``)."""
    if logits_ctx is not None and (
        _FUSED_ENTROPY_ENV and _fused_entropy_kernel is not None
    ):
        raw, softcap, out_scale = logits_ctx
        if raw.is_cuda and raw.numel() > 0:
            try:
                return _fused_entropy_kernel(
                    raw, softcap=softcap, out_scale=out_scale
                )
            except Exception as error:
                logger.warning(
                    "fused entropy kernel failed (%s); torch reference path taken",
                    error,
                )
    if logits is None:
        # probs mode: nothing materialized — reconstruct the reference
        # chain from the ctx before the (fused-or-torch) logits paths.
        logits = _ctx_logits(None, logits_ctx)
    if (
        _FUSED_ENTROPY_ENV
        and _fused_entropy_kernel is not None
        and logits.is_cuda
        and logits.dtype == torch.float32
        and logits.numel() > 0
    ):
        try:
            return _fused_entropy_kernel(logits)
        except Exception as error:
            logger.warning(
                "fused entropy kernel failed (%s); torch reference path taken",
                error,
            )
    dist = torch.distributions.Categorical(logits=logits)
    return dist.entropy()


class EntropyBoundDenoiser:
    """Reference-faithful EB sampler (transformers generation_diffusion_gemma):

    - initialize: uniform random canvas
    - per step: temperature t = t_min + (t_max - t_min) * step / max_steps;
      logits /= t; denoiser_canvas = multinomial(softmax(logits));
      accept lowest-entropy tokens while cumsum(sorted) - sorted <= bound;
      renoise everything else; final canvas = argmax of last logits.
    - adaptive stopping: mean entropy < confidence_threshold and argmax
      canvas stable across stability_threshold consecutive steps.

    The multinomial(softmax(logits)) draw runs through the fused Gumbel-max
    kernel on CUDA fp32 (DGM_FUSED_SAMPLING gate, default on): identical
    sampling distribution, deterministic per-round RNG stream, one logits
    read instead of the torch chain's full softmax materialization plus a
    second full-width multinomial pass.
    """

    def __init__(
        self,
        canvas_length: int,
        vocab_size: int,
        max_denoising_steps: int,
        t_min: float,
        t_max: float,
        entropy_bound: float,
        confidence_threshold: float,
        stability_threshold: int,
        generator: Optional[torch.Generator] = None,
    ):
        self.canvas_length = canvas_length
        self.vocab_size = vocab_size
        self.max_denoising_steps = max_denoising_steps
        self.t_min = t_min
        self.t_max = t_max
        self.entropy_bound = entropy_bound
        self.confidence_threshold = confidence_threshold
        self.stability_threshold = stability_threshold
        self.generator = generator
        self._sample_draws = 0

    def temperature(self, step: int) -> float:
        # The reference schedules by steps REMAINING
        # (reversed(range(1, max_denoising_steps + 1))): the temperature
        # starts at t_max (flat/exploratory) and decays to t_min (sharp) as
        # the canvas converges. Our loop counts ELAPSED steps from 0, so the
        # remaining count is max_denoising_steps - step; mapping it through
        # the reference formula t = t_min + (t_max - t_min) * (remaining /
        # max_steps) reproduces the schedule exactly (first step t_max, last
        # step ~t_min). The inverted schedule (sharp first, flat later)
        # collapses the canvas by accepting wrong tokens at step 0.
        remaining = self.max_denoising_steps - step
        return self.t_min + (self.t_max - self.t_min) * (
            remaining / max(1, self.max_denoising_steps)
        )

    def initialize_canvas(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.randint(
            low=0,
            high=self.vocab_size,
            size=(batch, self.canvas_length),
            device=device,
            generator=self.generator,
        )

    def _accept_mask(self, logits: torch.Tensor, logits_ctx=None) -> torch.Tensor:
        token_entropy = _categorical_entropy(logits, logits_ctx)
        if (
            _FUSED_ACCEPT_ENV
            and _fused_accept_kernel is not None
            and token_entropy.is_cuda
            and token_entropy.dtype == torch.float32
            and token_entropy.numel() > 0
            and int(token_entropy.shape[-1]) <= 2048
        ):
            try:
                return _fused_accept_kernel(token_entropy, self.entropy_bound)
            except Exception as error:
                logger.warning(
                    "fused accept_mask kernel failed (%s); torch reference path taken",
                    error,
                )
        sorted_entropy, sorted_idx = torch.sort(token_entropy, dim=-1, descending=False)
        cumulative = torch.cumsum(sorted_entropy, dim=-1)
        sorted_mask = cumulative - sorted_entropy <= self.entropy_bound
        return torch.scatter(
            torch.zeros_like(sorted_mask), -1, sorted_idx, sorted_mask
        )

    def _fused_sample_seed(self) -> int:
        """Deterministic per-round, per-draw seed for the fused sampler.

        The torch path draws its randomness from the round generator's own
        stream; the fused kernel uses a counter-based hash stream instead,
        seeded from the same round generator's initial seed plus a per-draw
        counter — reproducible for a fixed round seed (golden
        reproducibility) and distinct across steps and rounds.
        """
        base = 0
        if self.generator is not None:
            base = int(self.generator.initial_seed())
        seed = (base + self._sample_draws * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        self._sample_draws += 1
        return seed

    def _sample(self, logits: torch.Tensor, logits_ctx=None) -> torch.Tensor:
        """One token per row of ``logits`` drawn from ``softmax(logits)``:
        fused Gumbel-max Triton kernel under the DGM_FUSED_SAMPLING gate on
        CUDA fp32 (fp32 logits in, int64 token ids out, one launch), the
        torch softmax+multinomial reference path otherwise. The two paths
        sample the same distribution but not the same tokens (independent
        RNG streams); the acceptance evidence is distributional — see
        diffusion_gemma_proposer_test. Any kernel failure degrades to the
        reference path with a logged warning.

        ``logits_ctx`` = (raw_logits, softcap, out_scale) (opt r5 softcap
        inline): when given, the kernel draws from
        ``softmax(tanh(raw/s) * out_scale)`` with the transform in-register,
        reading the raw bf16/fp32 logits instead of the materialized row.
        In r9 probs mode ``logits`` is None and the torch fallback
        reconstructs the reference chain from the ctx (``_ctx_logits``)."""
        if _FUSED_SAMPLING_ENV and _fused_sample_kernel is not None:
            if logits_ctx is not None:
                raw, softcap, out_scale = logits_ctx
                if raw.is_cuda and raw.numel() > 0:
                    try:
                        return _fused_sample_kernel(
                            raw,
                            self._fused_sample_seed(),
                            softcap=softcap,
                            out_scale=out_scale,
                        )
                    except Exception as error:
                        logger.warning(
                            "fused sampling kernel failed (%s); torch reference path taken",
                            error,
                        )
            elif (
                logits is not None
                and logits.is_cuda
                and logits.dtype == torch.float32
                and logits.numel() > 0
            ):
                try:
                    return _fused_sample_kernel(logits, self._fused_sample_seed())
                except Exception as error:
                    logger.warning(
                        "fused sampling kernel failed (%s); torch reference path taken",
                        error,
                    )
        if logits is None:
            # probs mode: nothing materialized — reconstruct the reference
            # chain from the ctx for this torch tail.
            logits = _ctx_logits(None, logits_ctx)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        flat = probs.reshape(-1, probs.shape[-1])
        return torch.multinomial(flat, 1, generator=self.generator).squeeze(-1)

    def step(
        self,
        current_canvas: torch.Tensor,
        logits: torch.Tensor,
        logits_ctx=None,
    ):
        """One accept/re noise step. Returns (new_canvas, accepted_mask)."""
        mask = self._accept_mask(logits, logits_ctx)
        denoiser = self._sample(logits, logits_ctx).view(current_canvas.shape)
        accepted = torch.where(mask, denoiser, current_canvas)
        random_canvas = self.initialize_canvas(current_canvas.shape[0], current_canvas.device)
        renoised = torch.where(mask, accepted, random_canvas)
        return renoised, mask

    def stop(
        self,
        argmax_canvas: torch.Tensor,
        logits: torch.Tensor,
        history: Optional[torch.Tensor],
        logits_ctx=None,
    ):
        entropy = _categorical_entropy(logits, logits_ctx)
        confident = entropy.mean(dim=-1) < self.confidence_threshold
        if history is None or self.stability_threshold == 0:
            stable = torch.ones_like(confident)
        else:
            stable = (history == argmax_canvas.unsqueeze(0)).all(dim=-1).all(dim=0)
        return stable & confident


class DiffusionGemmaProposer(nn.Module):
    """DSpark PROPOSE/COMMIT wrapper around the shared main model."""

    def __init__(self, config, parallelism_config, ckpt_path: str):
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.ckpt_path = ckpt_path
        self.layer_num = int(config.num_layers)
        self.hidden_size = int(config.hidden_size)
        self.canvas_length = int(config.dgm_canvas_length)
        self.vocab_size = int(config.vocab_size)
        self.max_denoising_steps = int(
            getattr(config, "dgm_max_denoising_steps", 48)
        )
        self.t_min = float(getattr(config, "dgm_t_min", 0.4))
        self.t_max = float(getattr(config, "dgm_t_max", 0.8))
        self.entropy_bound = float(getattr(config, "dgm_entropy_bound", 0.1))
        self.confidence_threshold = float(
            getattr(config, "dgm_confidence_threshold", 0.005)
        )
        self.stability_threshold = int(
            getattr(config, "dgm_stability_threshold", 1)
        )
        self.final_logit_softcapping = float(
            getattr(config, "dgm_final_logit_softcapping", 30.0)
        )
        self.kv_cache = None  # engine MTP module cache: unused; reads go to the main cache
        self._main_model = None
        self._resolved = False
        self._seed = 0

    # ------------------------------------------------------------------
    # Engine plumbing
    # ------------------------------------------------------------------

    def _get_main_model(self):
        if not self._resolved:
            model = get_main_model(self.ckpt_path, self.layer_num)
            if model is None:
                raise RuntimeError(
                    "DiffusionGemma proposer requires the main model to be "
                    f"registered first (ckpt={self.ckpt_path}, layers={self.layer_num})"
                )
            self._main_model = model
            self._resolved = True
        return self._main_model

    def initialize(self, init_resource) -> bool:
        # Intentionally ignore the engine-provided MTP module cache; canvas
        # attention reads the main model's cache through _get_main_model().
        return True

    def prepare_fmha_impl(self, inputs, is_cuda_graph: bool = False, *args, **kwargs):
        return None  # the proposer owns its attention ops via the main model

    def cuda_graph_input_hidden_size(self) -> int:
        return 0

    def _probs_ab(
        self,
        raw_logits: torch.Tensor,
        softcap: float,
        out_scale: float,
        probs: torch.Tensor,
    ) -> None:
        """Debug A/B (DGM_PROBS_AB=1): the fused in-kernel softmax against
        the exact torch reference ``softmax(tanh(raw/s)*o).to(dtype)`` —
        bf16 ulp-distance stats over the full [canvas, vocab] block (first
        call per process; the [dgm-probs-ab] lines are the r9 evidence)."""
        if getattr(self, "_probs_ab_logged", False):
            return
        self._probs_ab_logged = True
        try:
            ref_processed = torch.tanh(
                raw_logits.to(torch.float32) / softcap
            ) * out_scale
            ref = ref_processed.softmax(dim=-1, dtype=torch.float32).to(probs.dtype)
            ref = ref.view(probs.shape)
            exact = int((probs == ref).sum())
            abs_diff = float((probs.float() - ref.float()).abs().max())
            if probs.dtype in (torch.bfloat16, torch.float16):
                ulp = (probs.view(torch.int16) - ref.view(torch.int16)).abs()
                print(
                    f"[dgm-probs-ab] shape={tuple(probs.shape)} dtype={probs.dtype}: "
                    f"exact_frac={exact / probs.numel():.6f} "
                    f"max_ulp={int(ulp.max())} mean_ulp={float(ulp.float().mean()):.4f} "
                    f"max_abs_diff={abs_diff:.3e}",
                    flush=True,
                )
            else:
                print(
                    f"[dgm-probs-ab] shape={tuple(probs.shape)} dtype={probs.dtype}: "
                    f"exact_frac={exact / probs.numel():.6f} "
                    f"max_abs_diff={abs_diff:.3e}",
                    flush=True,
                )
        except Exception as error:
            print(f"[dgm-probs-ab] referee failed: {error}", flush=True)

    # ------------------------------------------------------------------
    # DSpark roles
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward_propose(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        main_model = self._get_main_model()
        device = main_model.embed_tokens.weight.device
        attention = inputs.attention_inputs
        if isinstance(attention, dict):
            primary = next(iter(attention.values()))
        else:
            primary = attention
        prefix_lengths = primary.prefix_lengths
        if prefix_lengths is None or prefix_lengths.numel() == 0:
            # warmup / fake streams: return well-shaped empties.
            tokens = int(inputs.input_ids.numel())
            width = self.canvas_length
            batch = max((tokens + width - 1) // width, 1)
            hidden = torch.zeros(
                (batch * width, self.hidden_size),
                dtype=main_model.embed_tokens.weight.dtype,
                device=device,
            )
            out = PyModelOutputs(hidden)
            out.diffusion_tokens = torch.zeros(
                (batch, width), dtype=torch.int32, device=device
            )
            return out

        batch = int(prefix_lengths.numel())
        prefix = prefix_lengths.to(device=device, dtype=torch.long)
        # Positions per request: [L, L + canvas) — matches the C++ dspark
        # expandDSparkPositionIds with sample_from_anchor=True (gamma rows).
        flat_positions = (
            prefix.unsqueeze(1)
            + torch.arange(self.canvas_length, device=device, dtype=torch.long).unsqueeze(0)
        ).reshape(-1)

        generator = torch.Generator(device=device)
        generator.manual_seed(
            int(self._seed) * 1000003 + int(prefix.sum().item()) % 1000003
        )
        self._seed += 1

        denoiser = EntropyBoundDenoiser(
            self.canvas_length,
            self.vocab_size,
            self.max_denoising_steps,
            self.t_min,
            self.t_max,
            self.entropy_bound,
            self.confidence_threshold,
            self.stability_threshold,
            generator=generator,
        )
        canvas = denoiser.initialize_canvas(batch, device)
        self_conditioning_logits = None
        # opt r9 item 2: probs mode — the fused softcap kernel's bf16 softmax
        # output (fed to forward_canvas directly; None on the torch path).
        self_conditioning_probs = None
        argmax_canvas = canvas
        history = None
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        embed_weight = main_model.embed_tokens.weight
        softcap = self.final_logit_softcapping
        last_hidden = None

        for step in range(self.max_denoising_steps):
            hidden = main_model.forward_canvas(
                inputs,
                canvas,
                flat_positions,
                self_conditioning_logits,
                self_conditioning_probs=self_conditioning_probs,
            )
            last_hidden = hidden
            self._last_step = step
            raw_logits = torch.nn.functional.linear(
                hidden.to(embed_weight.dtype), embed_weight
            )
            # Reference: t = t_min + (t_max - t_min) * step / max_steps
            # (transformers TemperatureScheduleByDenoisingStep).
            temperature = denoiser.temperature(step)
            out_scale = softcap / temperature
            logits_ctx = None
            # opt r9 item 2: in probs mode nothing is materialized — the
            # softmax over the transformed row is computed inside the
            # softcap kernel and handed to forward_canvas directly; the
            # entropy/sampling/accept kernels keep reading the RAW logits
            # through logits_ctx (their torch fallbacks reconstruct the
            # reference chain via _ctx_logits when logits is None).
            processed = None
            probs = None
            if (
                _FUSED_SOFTCAP_ENV
                and _softcap_argmax_kernel is not None
                and raw_logits.is_cuda
                and raw_logits.dtype
                in (torch.bfloat16, torch.float16, torch.float32)
                and raw_logits.numel() > 0
            ):
                # opt r5 item 2 (softcap inline): one launch replaces the
                # fp32 cast + tanh + scale passes and torch.argmax — the
                # raw bf16 logits are read once, the fp32 transformed row
                # (self-conditioning consumer) is materialized and the
                # argmax falls out of the same registers. logits_ctx hands
                # (raw, softcap, out_scale) to the entropy/sampling kernels
                # so they apply the #5-iii merged algebra in-register
                # instead of reading the materialized tensor.
                try:
                    if (
                        _FUSED_PROBS_ENV
                        and embed_weight.dtype
                        in (torch.bfloat16, torch.float16, torch.float32)
                    ):
                        probs, new_argmax = _softcap_argmax_kernel(
                            raw_logits,
                            softcap,
                            out_scale,
                            emit_softmax=True,
                            probs_dtype=embed_weight.dtype,
                        )
                        probs = probs.view(batch, self.canvas_length, -1)
                    else:
                        processed, new_argmax = _softcap_argmax_kernel(
                            raw_logits, softcap, out_scale
                        )
                        processed = processed.view(batch, self.canvas_length, -1)
                    new_argmax = new_argmax.view(batch, self.canvas_length)
                    logits_ctx = (
                        raw_logits.view(batch, self.canvas_length, -1),
                        softcap,
                        out_scale,
                    )
                    if _PROBS_AB_CHECK and probs is not None:
                        self._probs_ab(raw_logits, softcap, out_scale, probs)
                except Exception as error:
                    logger.warning(
                        "fused softcap_argmax kernel failed (%s); torch path "
                        "for this process",
                        error,
                    )
                    _disable_softcap()
                    probs = None
            if logits_ctx is None:
                # Contract #5-iii: the softcap chain ((x/s -> tanh) * s) / t
                # is merged algebraically into tanh(x/s) * (s/t) — one
                # full-vocab elementwise pass less. Values agree with the
                # sequential chain to the last fp32 ulp (two roundings
                # either way).
                logits = raw_logits.to(torch.float32)
                logits = torch.tanh(logits / softcap)
                logits = logits.view(batch, self.canvas_length, -1)
                processed = logits * out_scale
                new_argmax = torch.argmax(processed, dim=-1)
            if bool(os.environ.get("DGM_FI_DIAG", "")) and step == 0:
                # Reuse bring-up fork: with the C++ reuse fix the matched
                # blocks are shared and real (run-44 read probe), yet the
                # canvas still degenerates to all-EOS. Decide between
                # "canvas hidden ~constant (uniform attention)" and
                # "healthy hidden, degenerate logits/sampler" with one line.
                hidden_std = (
                    hidden.reshape(batch, self.canvas_length, -1)
                    .std(dim=1)
                    .mean()
                    .item()
                )
                diag_processed = (
                    processed
                    if processed is not None
                    else _ctx_logits(None, logits_ctx)
                )
                top5 = (
                    diag_processed[0, :3].topk(3, dim=-1).indices.tolist()
                )
                entropy0 = torch.distributions.Categorical(
                    logits=diag_processed
                ).entropy()[0, :3].tolist()
                print(
                    f"[dgm-fi] step0: prefix={prefix.tolist()} "
                    f"hidden_row_std={hidden_std:.4f} "
                    f"logits[0,:3]top3={top5} entropy[0,:3]="
                    f"[{entropy0[0]:.3f}, {entropy0[1]:.3f}, {entropy0[2]:.3f}]",
                    flush=True,
                )
            if history is None:
                history = torch.full(
                    (max(1, self.stability_threshold), batch, self.canvas_length),
                    -1,
                    dtype=new_argmax.dtype,
                    device=new_argmax.device,
                )
            if finished.any():
                # Reference semantics: freeze rows that already stopped
                # denoising (their canvas/argmax/logits no longer update).
                new_argmax = torch.where(finished[:, None], argmax_canvas, new_argmax)
                frozen_canvas = canvas
            else:
                frozen_canvas = canvas
            # processed is None in probs mode: the entropy/sampling/accept
            # kernels read raw through logits_ctx and their torch fallbacks
            # reconstruct the reference chain (bit-equal to this tensor).
            finished = finished | denoiser.stop(
                new_argmax, processed, history, logits_ctx=logits_ctx
            )
            history = torch.cat([history[1:], new_argmax.unsqueeze(0)], dim=0)
            renoised, _mask = denoiser.step(
                frozen_canvas, processed, logits_ctx=logits_ctx
            )
            renoised = torch.where(finished[:, None], frozen_canvas, renoised)
            argmax_canvas = new_argmax
            canvas = renoised
            # Contract #5-iii: keep the self-conditioning signal fp32 end to
            # end (probs mode: bf16 probabilities straight from the kernel —
            # the same values forward_canvas's reference produced via
            # softmax(processed).to(embed dtype), bf16-ulp class).
            self_conditioning_logits = processed
            self_conditioning_probs = probs
            if bool(finished.all().item()):
                break

        final_tokens = argmax_canvas
        if bool(os.environ.get("DGM_FI_DIAG", "")):
            # Reuse bring-up (run 40/41): cache-hit queries degenerate to a
            # 1-token response while cold queries are healthy; the anchor
            # logit is correct, so this dump pinpoints whether the propose
            # canvas itself is garbage or the C++ verify side rejects it.
            print(
                f"[dgm-fi] propose: steps_used={self._last_step} "
                f"finished={finished.tolist()} prefix={prefix.tolist()} "
                f"canvas[:12]={final_tokens.reshape(-1)[:12].tolist()} "
                f"canvas_min={int(final_tokens.min().item())} "
                f"canvas_max={int(final_tokens.max().item())} "
                f"hidden_absmax={float(last_hidden.abs().max().item()):.3f}",
                flush=True,
            )
        # hidden rows must match the C++ query block (B * canvas_length); the
        # diffusion executor consumes only diffusion_tokens, so reuse the last
        # denoising hidden instead of a third forward.
        out = PyModelOutputs(
            last_hidden.reshape(batch * self.canvas_length, self.hidden_size)
        )
        out.diffusion_tokens = final_tokens.to(torch.int32).contiguous()
        return out

    @torch.inference_mode()
    def forward_commit(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        # DiffusionGemma commits through the target-verify forward (the main
        # model's causal forward writes the canvas KV); the DSpark commit
        # channel has no feature KV to seed.
        hidden = torch.zeros((0, self.hidden_size), dtype=torch.bfloat16)
        return PyModelOutputs(hidden)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        return self.forward_propose(inputs, fmha_impl)


__all__ = ["DiffusionGemmaProposer", "EntropyBoundDenoiser"]
