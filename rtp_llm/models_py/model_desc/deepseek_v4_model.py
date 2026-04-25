"""DeepSeek-V4 integration into RTP-LLM's `GptModelBase` framework.

Wraps the standalone `V4Transformer` (rtp_llm/models_py/modules/dsv4/transformer.py)
so the C++ engine can drive it:
  - `forward(PyModelInputs) -> PyModelOutputs` returns [total_tokens, hidden_dim]
    pre-lm-head hidden states; engine applies lm_head + sampling externally.
  - Per-layer KV cache kept INSIDE the Attention modules (mock, register_buffer).
    Framework's BlockPool-backed KV cache is NOT wired yet (M4 work); this wrapper
    ignores the framework's kv_cache and bookkeeps its own `start_pos` from
    `attention_inputs.is_prefill` + sequence lengths.

Weight loading happens in `initialize()`: we bypass the framework's AtomicWeight
flow (V4-Flash has 67k+ tensors in ~250 packed FP4/FP8 shards) and directly call
`load_v4_safetensors` against the checkpoint path stored in `model_config`.
"""

import logging
import os
from typing import Any, Optional

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.dsv4.transformer import V4Args, V4Transformer
from rtp_llm.models_py.modules.dsv4.weight_loader import load_v4_safetensors
from rtp_llm.ops.compute_ops import PyModelInitResources, PyModelInputs, PyModelOutputs


def _args_from_model_config(mc: ModelConfig) -> V4Args:
    ac = mc.attn_config
    rc = ac.rope_config
    return V4Args(
        vocab_size=mc.vocab_size,
        dim=mc.hidden_size,
        n_heads=ac.head_num,
        n_layers=mc.num_layers,
        n_mtp_layers=0,  # MTP wiring lands later; main-layer inference first.
        q_lora_rank=ac.q_lora_rank,
        head_dim=ac.size_per_head,
        rope_head_dim=ac.rope_head_dim,
        o_groups=ac.o_groups,
        o_lora_rank=ac.o_lora_rank,
        window_size=ac.sliding_window,
        compress_ratios=list(ac.layer_compress_ratios)[: mc.num_layers],
        rope_theta=float(rc.base),
        compress_rope_theta=float(ac.compress_rope_theta),
        rope_factor=float(rc.scale) if rc.scale else 1.0,
        beta_fast=int(rc.factor2) if rc.factor2 else 32,
        beta_slow=int(rc.factor1) if rc.factor1 else 1,
        original_seq_len=int(rc.max_pos) if rc.max_pos else 0,
        index_n_heads=ac.indexer_head_num,
        index_head_dim=ac.indexer_head_dim,
        index_topk=ac.indexer_topk,
        moe_inter_dim=mc.inter_size // max(1, mc.moe_k or 1) if False else mc.inter_size // 1,
        n_routed_experts=mc.expert_num,
        n_shared_experts=1,
        n_activated_experts=mc.moe_k,
        score_func={0: "softmax", 1: "sigmoid", 2: "sqrtsoftplus"}[mc.scoring_func],
        route_scale=float(mc.routed_scaling_factor),
        swiglu_limit=float(mc.swiglu_limit),
        n_hash_layers=int(mc.num_hash_layers),
        hc_mult=int(mc.hc_mult),
        hc_sinkhorn_iters=int(mc.hc_sinkhorn_iters),
        hc_eps=float(mc.hc_eps),
        norm_eps=float(mc.layernorm_eps),
        max_batch_size=1,
        max_seq_len=min(int(mc.max_seq_len) or 4096, 4096),
    )


class DeepSeekV4Model(GptModelBase):
    """Framework-facing model: owns a V4Transformer, feeds framework IO into it."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config,
        weights: ModelWeights,
        moe_config,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config, parallelism_config, weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config, py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )

        # Build V4Transformer with matching args.
        args = _args_from_model_config(model_config)
        # MoE inter dim from V4 config: explicit (not inter_size which in RTP-LLM
        # is n_shared_experts * moe_intermediate_size for DeepSeek). Use moe_config
        # if available; else read from config's hidden_size-derived fallback.
        if moe_config is not None and getattr(moe_config, "moe_inter_padding_size", 0) > 0:
            args.moe_inter_dim = int(moe_config.moe_inter_padding_size)
        else:
            # V4-Flash = 2048. config.inter_size = n_shared * 2048 = 2048 (since n_shared=1).
            args.moe_inter_dim = int(model_config.inter_size) or args.moe_inter_dim

        logging.info(
            "[DeepSeekV4Model] V4Args: n_layers=%d n_heads=%d head_dim=%d q_lora=%d "
            "o_groups=%d n_experts=%d n_act=%d moe_inter=%d win=%d hc_mult=%d "
            "compress_ratios[:8]=%s score=%s route_scale=%g swiglu_limit=%g",
            args.n_layers, args.n_heads, args.head_dim, args.q_lora_rank,
            args.o_groups, args.n_routed_experts, args.n_activated_experts,
            args.moe_inter_dim, args.window_size, args.hc_mult,
            list(args.compress_ratios)[:8], args.score_func, args.route_scale,
            args.swiglu_limit,
        )
        self._v4_args = args

        # Build on meta first to avoid blowing CPU RAM during engine init;
        # materialize + ckpt load happens in `initialize()`.
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            self.v4 = V4Transformer(args)
        torch.set_default_dtype(prev_dtype)

        self._materialized = False
        self._ckpt_path: str = model_config.ckpt_path
        # Running cursor for prefill-then-decode — the only batching we support
        # until framework KV cache lands.
        self._running_pos: int = 0

        # Optional on-demand timeline capture. Set DSV4_PROFILE_TRACE=/path/trace.json
        # and touch /tmp/dsv4_profile_trigger to capture the NEXT forward only.
        self._profile_path = os.environ.get("DSV4_PROFILE_TRACE")
        self._profile_trigger = "/tmp/dsv4_profile_trigger"
        self._profile_done = False
        if self._profile_path:
            logging.info("[DeepSeekV4Model] timeline trigger: `touch %s` — "
                         "next forward captures to %s",
                         self._profile_trigger, self._profile_path)

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        # Called by the engine after construction and before forward.
        super().initialize(init_resource)
        if self._materialized:
            return True

        device = next(iter(self.weight.global_weights.values())).device if self.weight.global_weights else "cuda:0"
        device_str = str(device)
        logging.info("[DeepSeekV4Model] materializing on %s ...", device_str)
        self.v4 = self.v4.to_empty(device=device_str)

        # Recompute RoPE cache (meta-path buffers come back as zeros, not initialized).
        for layer in self.v4.layers:
            layer.attn.reset_rope_cache(device=device_str)

        logging.info("[DeepSeekV4Model] loading ckpt from %s ...", self._ckpt_path)
        loaded = load_v4_safetensors(
            self.v4, self._ckpt_path, dtype=torch.bfloat16, device=device_str,
            strict=False, verbose=False,
        )
        logging.info("[DeepSeekV4Model] loaded %d tensors", len(loaded))

        if torch.cuda.is_available() and device_str.startswith("cuda"):
            mem = torch.cuda.memory_allocated(torch.device(device_str)) / 1024**3
            logging.info("[DeepSeekV4Model] GPU mem after load: %.1f GB", mem)

        self._materialized = True
        self._running_pos = 0
        return True

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False) -> Any:
        """V4 uses its own sparse/compressed attention internally — no standard FMHA
        backend fits (64 Q heads × 1 KV head × head_dim=512 with CSA/HCA/SWA per-layer
        variants). Return None so the framework's fmha machinery is bypassed."""
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        # Framework expects flat [total_tokens] input_ids (cu_seqlens-packed batching).
        # We support B=1 only; extract as [1, S].
        input_ids: torch.Tensor = inputs.input_ids
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # [1, S]

        attn = inputs.attention_inputs
        is_prefill = bool(attn.is_prefill) if attn is not None else (input_ids.size(1) > 1)
        start_pos = 0 if is_prefill else self._running_pos

        # On-demand timeline capture for exactly one forward when trigger file exists.
        should_capture = (
            self._profile_path and not self._profile_done
            and os.path.exists(self._profile_trigger)
        )
        if should_capture:
            try:
                os.remove(self._profile_trigger)
            except OSError:
                pass
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
                with_stack=False,
            ) as prof:
                with torch.profiler.record_function(
                    f"V4_forward_prefill={is_prefill}_S={input_ids.size(1)}"
                ):
                    hidden = self.v4(input_ids, start_pos=start_pos, apply_lm_head=False)
                    torch.cuda.synchronize()
            prof.export_chrome_trace(self._profile_path)
            logging.info("[DeepSeekV4Model] timeline exported: %s (%d events)",
                         self._profile_path, len(prof.key_averages()))
            self._profile_done = True
        else:
            hidden = self.v4(input_ids, start_pos=start_pos, apply_lm_head=False)

        # Advance running position
        S = input_ids.size(1)
        if is_prefill:
            self._running_pos = S
        else:
            self._running_pos += S

        # Return flat [total_tokens, hidden_dim] for engine
        flat = hidden.reshape(-1, hidden.size(-1))
        return PyModelOutputs(flat)
