"""DiffusionGemma block-diffusion proposer (DSpark PROPOSE role).

One engine round = one full canvas (block_length*8 = 256 tokens by default):
the python forward_propose runs the entropy-bounded denoising loop over the
canvas (bidirectional decoder attention, read-only encoder KV), runs the EB
sampler exactly like the HF reference (temperature schedule 0.8 -> 0.4,
lowest-entropy acceptance under the mutual-information bound, full renoising,
adaptive stopping), and returns the final argmax canvas tokens through
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


class EntropyBoundDenoiser:
    """Reference-faithful EB sampler (transformers generation_diffusion_gemma):

    - initialize: uniform random canvas
    - per step: temperature t = t_min + (t_max - t_min) * step / max_steps;
      logits /= t; denoiser_canvas = multinomial(softmax(logits));
      accept lowest-entropy tokens while cumsum(sorted) - sorted <= bound;
      renoise everything else; final canvas = argmax of last logits.
    - adaptive stopping: mean entropy < confidence_threshold and argmax
      canvas stable across stability_threshold consecutive steps.
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

    def _accept_mask(self, logits: torch.Tensor) -> torch.Tensor:
        dist = torch.distributions.Categorical(logits=logits)
        token_entropy = dist.entropy()
        sorted_entropy, sorted_idx = torch.sort(token_entropy, dim=-1, descending=False)
        cumulative = torch.cumsum(sorted_entropy, dim=-1)
        sorted_mask = cumulative - sorted_entropy <= self.entropy_bound
        return torch.scatter(
            torch.zeros_like(sorted_mask), -1, sorted_idx, sorted_mask
        )

    def step(
        self,
        current_canvas: torch.Tensor,
        logits: torch.Tensor,
    ):
        """One accept/re noise step. Returns (new_canvas, accepted_mask)."""
        mask = self._accept_mask(logits)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        flat = probs.reshape(-1, probs.shape[-1])
        sampled = torch.multinomial(flat, 1, generator=self.generator)
        denoiser = sampled.squeeze(-1).view(current_canvas.shape)
        accepted = torch.where(mask, denoiser, current_canvas)
        random_canvas = self.initialize_canvas(current_canvas.shape[0], current_canvas.device)
        renoised = torch.where(mask, accepted, random_canvas)
        return renoised, mask

    def stop(
        self,
        argmax_canvas: torch.Tensor,
        logits: torch.Tensor,
        history: Optional[torch.Tensor],
    ):
        dist = torch.distributions.Categorical(logits=logits)
        entropy = dist.entropy()
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
        argmax_canvas = canvas
        history = None
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        embed_weight = main_model.embed_tokens.weight
        softcap = self.final_logit_softcapping
        last_hidden = None

        for step in range(self.max_denoising_steps):
            hidden = main_model.forward_canvas(
                inputs, canvas, flat_positions, self_conditioning_logits
            )
            last_hidden = hidden
            self._last_step = step
            logits = torch.nn.functional.linear(
                hidden.to(embed_weight.dtype), embed_weight
            ).to(torch.float32)
            logits = logits / softcap
            logits = torch.tanh(logits) * softcap
            logits = logits.view(batch, self.canvas_length, -1)
            # Reference: t = t_min + (t_max - t_min) * step / max_steps
            # (transformers TemperatureScheduleByDenoisingStep).
            temperature = denoiser.temperature(step)
            processed = logits / temperature
            new_argmax = torch.argmax(processed, dim=-1)
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
            finished = finished | denoiser.stop(new_argmax, processed, history)
            history = torch.cat([history[1:], new_argmax.unsqueeze(0)], dim=0)
            renoised, _mask = denoiser.step(frozen_canvas, processed)
            renoised = torch.where(finished[:, None], frozen_canvas, renoised)
            argmax_canvas = new_argmax
            canvas = renoised
            self_conditioning_logits = processed.to(embed_weight.dtype)
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
