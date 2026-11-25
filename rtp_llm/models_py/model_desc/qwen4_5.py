"""Qwen 4.5-Flash (Qwen4_5ForCausalLM) PyTorch / Triton B300 Model Description.

Implements the complete 40-layer hybrid architecture verified against the
production ``1109-bf16-re-mxfp4-ple-fp8`` checkpoint:
  - 4-stream Hyper-Connections with ``hc_prev_mix=True``, per-branch GemmaRMSNorm+bias,
    Layer 0 head up-projection ``[5120, 128]``, Layer 39 ``emits_latent=False``, and
    final ``model.hyper_connection_mixer`` (no ``model.norm``).
  - Layer 1 (1-based ``ple_layer_ids=[2]``) Per-Layer N-gram Embedding (PLE) with
    128-shard FP8 E4M3 storage (CPU-UVA or GPU EP-sharded).
  - 3:1 hybrid attention: 30 Gated DeltaNet (GDN) layers + 10 MQA-1 dense Full
    Attention layers with YoCo KV/GDN sharing (Layer 19 -> 5 FullAttn consumers;
    Layer 20 -> 14 GDN consumers with O(1) Decode SSM state readout).
  - Layer 0 Dense MLP (12800) + Layers 1..39 Latent MoE (5120->2560->640 MXFP4
    routed experts top-10->2560->5120 + Shared Expert 1280), SiTU-GLU(4, 25),
    Expert Parallelism AllReduce in 2560-dim latent space.
  - Per-sublayer depthwise causal ShortConv (k=4, residual=True) on attention
    and FFN outputs, plus FFN output clamping to ``[-1000.0, 1000.0]``.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from rtp_llm.models_py.modules.qwen4_5.attention import Qwen4_5Attention
from rtp_llm.models_py.modules.qwen4_5.fast_comm import (
    get_qwen4_5_fast_all_reduce,
)
from rtp_llm.models_py.modules.qwen4_5.contracts import (
    GDNLayerState,
    Qwen4_5ArchConfig,
    Qwen4_5RuntimeState,
    SharedGDNInputs,
)
from rtp_llm.models_py.modules.qwen4_5.gdn import Qwen4_5GatedDeltaNet
from rtp_llm.models_py.modules.qwen4_5.hyper_connection import Qwen4_5HyperConnection
from rtp_llm.models_py.modules.qwen4_5.moe import Qwen4_5DenseMLP, Qwen4_5LatentMoE
from rtp_llm.models_py.modules.qwen4_5.mxfp4_ops import MXFP4_BLOCK_SIZE
from rtp_llm.models_py.modules.qwen4_5.ple import (
    DEFAULT_PLE_WEIGHT_SCALE,
    Qwen4_5PLELayer,
)
from rtp_llm.models_py.modules.qwen4_5.short_conv import Qwen4_5ShortConv
from rtp_llm.models_py.modules.qwen4_5.yoco import YoCoSharingPlan


def _resolve_tp_group_safe(tp_size: int) -> Any:
    if tp_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return None
    try:
        from rtp_llm.models_py.distributed import collective_torch

        if getattr(collective_torch, "_initialized", False):
            from rtp_llm.models_py.distributed.collective_torch import Group, _get_group

            return _get_group(Group.TP)
    except Exception:
        pass
    if dist.get_world_size() >= tp_size:
        return dist.group.WORLD
    return None


class Qwen4_5ProfileRecorder:
    """Lightweight CUDA-event category timer (zero overhead when unused)."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._pairs: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def reset(self) -> None:
        self._pairs.clear()

    @contextmanager
    def record(self, category: str) -> Iterator[None]:
        if not self.enabled or not torch.cuda.is_available():
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pairs.append((category, start, end))

    def summarize_ms(self) -> Dict[str, float]:
        if not self._pairs:
            return {}
        torch.cuda.synchronize()
        totals: Dict[str, float] = {}
        for cat, s, e in self._pairs:
            totals[cat] = totals.get(cat, 0.0) + float(s.elapsed_time(e))
        return totals


class Qwen4_5DecoderLayer(nn.Module):
    """Single decoder layer of Qwen 4.5-Flash (GDN or FullAttn + Dense or LatentMoE)."""

    def __init__(
        self,
        config: Qwen4_5ArchConfig,
        layer_idx: int,
        sharing_plan: YoCoSharingPlan,
        *,
        use_triton_mxfp4: bool = True,
        compute_mode: str = "bf16",
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        init_mxfp4_weights: bool = True,
        ep_size: int = 1,
        ep_rank: int = 0,
        ep_group: Any = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group: Any = None,
        fast_all_reduce: Any = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.layer_type = config.layer_types[layer_idx]
        self.is_consumer = sharing_plan.is_consumer(layer_idx)
        self.is_source = sharing_plan.is_source(layer_idx)
        self.source_layer_idx = sharing_plan.source_for(layer_idx)
        self.kv_cache_owner = sharing_plan.kv_cache_owner(layer_idx)
        self.tp_size = max(1, int(tp_size or 1))
        self.tp_rank = int(tp_rank or 0)
        self.tp_group = tp_group
        self.fast_all_reduce = fast_all_reduce

        is_first_layer = layer_idx == 0
        is_last_layer = layer_idx == (config.num_hidden_layers - 1)

        # 1. Optional PLE module (1-based ple_layer_ids -> 0-based layer_idx)
        self.has_ple = (layer_idx + 1) in config.ple_layer_ids
        if self.has_ple:
            self.ple: Optional[Qwen4_5PLELayer] = Qwen4_5PLELayer(
                hidden_size=config.hidden_size,
                hc_count=config.hc_count,
                ple_embed_dim=config.ple_embed_dim,
                ngram_size=config.ngram_size,
                heads_per_ngram=config.heads_per_ngram,
                conv_kernel_size=config.ple_conv_kernel_size,
                split_ngram_parts=config.split_ngram_parts,
                eos_token_id=config.eos_token_id,
                eps=config.rms_norm_eps,
                ngram_vocab_size_base=config.ngram_vocab_size_base,
                dtype=dtype,
                device=device,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                tp_group=tp_group,
                fast_all_reduce=fast_all_reduce,
            )
        else:
            self.ple = None

        # 2. Attention Hyper-Connection
        self.attn_hyper_connection = Qwen4_5HyperConnection(
            hidden_size=config.hidden_size,
            hc_count=config.hc_count,
            hc_lowrank=config.hc_lowrank,
            eps=config.rms_norm_eps,
            use_combine=True,
            prev_mix=config.hc_prev_mix,
            is_head=is_first_layer,
            emits_latent=True,
            dtype=dtype,
            device=device,
        )

        # 3. Attention block (GDN or dense MQA-1 Full Attention / QSA)
        if self.layer_type == "linear_attention":
            self.linear_attn: Optional[Qwen4_5GatedDeltaNet] = (
                Qwen4_5GatedDeltaNet.from_config(
                    config,
                    layer_idx,
                    sharing_plan=sharing_plan,
                    tp_size=self.tp_size,
                    tp_rank=self.tp_rank,
                    tp_group=tp_group,
                    fast_all_reduce=fast_all_reduce,
                ).to(device=device, dtype=dtype)
            )
            self.self_attn: Optional[Qwen4_5Attention] = None
            # Keep A_log in FP32 as required by GDN gating spec
            if self.linear_attn.A_log is not None:
                self.linear_attn.A_log.data = self.linear_attn.A_log.data.float()
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen4_5Attention.from_config(
                config,
                layer_idx,
                sharing_plan=sharing_plan,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                tp_group=tp_group,
                fast_all_reduce=fast_all_reduce,
            ).to(device=device, dtype=dtype)
            self.linear_attn = None
            # Keep RoPE inv_freq in FP32
            self.self_attn.inv_freq = self.self_attn.inv_freq.float()
        else:
            raise ValueError(f"Unsupported layer_type={self.layer_type!r} at layer {layer_idx}")

        # 4. Optional Attention output ShortConv (k=4, residual=True)
        attn_site = (
            "linear_attn_output_sconv"
            if self.layer_type == "linear_attention"
            else "full_attn_output_sconv"
        )
        if config.has_output_sconv(attn_site, is_consumer=self.is_consumer):
            self.attn_output_sconv: Optional[Qwen4_5ShortConv] = Qwen4_5ShortConv(
                channels=config.hidden_size,
                kernel_size=config.linear_conv_kernel_dim,
                residual=True,
                use_activation=config.output_sconv_use_activation,
                dtype=dtype,
                device=device,
            )
        else:
            self.attn_output_sconv = None

        # 5. MLP Hyper-Connection (last layer omits input_mix_weight_down -> emits_latent=False)
        self.mlp_hyper_connection = Qwen4_5HyperConnection(
            hidden_size=config.hidden_size,
            hc_count=config.hc_count,
            hc_lowrank=config.hc_lowrank,
            eps=config.rms_norm_eps,
            use_combine=True,
            prev_mix=config.hc_prev_mix,
            is_head=False,
            emits_latent=not is_last_layer,
            dtype=dtype,
            device=device,
        )

        # 6. FFN block: first_k_dense_replace Dense MLP, remaining Latent MoE
        self.is_moe = layer_idx >= config.first_k_dense_replace
        if self.is_moe:
            self.mlp: Union[Qwen4_5DenseMLP, Qwen4_5LatentMoE] = Qwen4_5LatentMoE.from_config(
                config,
                use_triton_mxfp4=use_triton_mxfp4,
                compute_mode=compute_mode,
                dtype=dtype,
                device=device,
                init_mxfp4_weights=init_mxfp4_weights,
                ep_size=ep_size,
                ep_rank=ep_rank,
                ep_group=ep_group,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                tp_group=tp_group,
                fast_all_reduce=fast_all_reduce,
            )
        else:
            self.mlp = Qwen4_5DenseMLP.from_config(
                config,
                is_shared_expert=False,
                use_triton=use_triton_mxfp4,
                dtype=dtype,
                device=device,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                tp_group=tp_group,
                fast_all_reduce=fast_all_reduce,
            )

        # 7. Optional FFN output ShortConv (k=4, residual=True)
        if config.has_output_sconv("ffn_output_sconv", is_consumer=self.is_consumer):
            self.ffn_output_sconv: Optional[Qwen4_5ShortConv] = Qwen4_5ShortConv(
                channels=config.hidden_size,
                kernel_size=config.linear_conv_kernel_dim,
                residual=True,
                use_activation=config.output_sconv_use_activation,
                dtype=dtype,
                device=device,
            )
        else:
            self.ffn_output_sconv = None

    @torch.no_grad()
    def prepare_weights(self, *, transpose_mxfp4: bool = False) -> None:
        if isinstance(self.mlp, Qwen4_5LatentMoE):
            self.mlp.prepare_weights(
                transpose_mxfp4=transpose_mxfp4, fuse_shared_gate_up=True
            )
        else:
            self.mlp.prepare_weights(fuse_gate_up=True)

    def forward(
        self,
        hyper_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        prev_latent: Optional[torch.Tensor],
        deferred: Optional[Tuple[torch.Tensor, torch.Tensor]],
        shared_gdn_by_source: Dict[int, SharedGDNInputs],
        shared_qsa_by_source: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        source_prefix_ssm: Optional[Dict[int, torch.Tensor]] = None,
        state: Optional[Qwen4_5RuntimeState] = None,
        positions: Optional[torch.Tensor] = None,
        seq_offset: Optional[int] = None,
        enable_yoco_decode: Optional[bool] = None,
        use_reference: bool = False,
        profiler: Optional[Qwen4_5ProfileRecorder] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Run one full decoder layer returning ``(hyper_hidden, prev_latent, deferred)``."""
        cfg = self.config
        lid = self.layer_idx

        # Step A: PLE injection (Layer 1 only) — flush deferred HC combine before PLE
        if self.ple is not None:
            if deferred is not None:
                if profiler is not None:
                    with profiler.record("hc"):
                        hyper_hidden = self.attn_hyper_connection.combine(
                            deferred[0], hyper_hidden, deferred[1]
                        )
                elif use_reference:
                    hyper_hidden = self.attn_hyper_connection.combine_ref(
                        deferred[0], hyper_hidden, deferred[1]
                    )
                else:
                    hyper_hidden = self.attn_hyper_connection.combine(
                        deferred[0], hyper_hidden, deferred[1]
                    )
                deferred = None

            ple_state = state.ple.get(lid) if state is not None else None
            if profiler is not None:
                with profiler.record("ple"):
                    hyper_hidden = self.ple(hyper_hidden, input_ids, state=ple_state)
            elif use_reference:
                hyper_hidden = self.ple.forward_ref(hyper_hidden, input_ids, state=ple_state)
            else:
                hyper_hidden = self.ple(hyper_hidden, input_ids, state=ple_state)

        # Step B: Attention Hyper-Connection (fused combine+norm+mix when deferred exists)
        if profiler is not None:
            with profiler.record("hc"):
                if deferred is None:
                    attn_in, hyper_hidden, attn_inj, prev_latent = (
                        self.attn_hyper_connection.mix_with_latent(hyper_hidden, prev_latent)
                    )
                else:
                    attn_in, hyper_hidden, attn_inj, prev_latent = (
                        self.attn_hyper_connection.combine_and_mix_with_latent(
                            deferred[0], hyper_hidden, deferred[1], prev_latent
                        )
                    )
        elif use_reference:
            if deferred is None:
                attn_in, hyper_hidden, attn_inj, prev_latent = (
                    self.attn_hyper_connection.mix_with_latent_ref(hyper_hidden, prev_latent)
                )
            else:
                attn_in, hyper_hidden, attn_inj, prev_latent = (
                    self.attn_hyper_connection.combine_and_mix_with_latent_ref(
                        deferred[0], hyper_hidden, deferred[1], prev_latent
                    )
                )
        else:
            if deferred is None:
                attn_in, hyper_hidden, attn_inj, prev_latent = (
                    self.attn_hyper_connection.mix_with_latent(hyper_hidden, prev_latent)
                )
            else:
                attn_in, hyper_hidden, attn_inj, prev_latent = (
                    self.attn_hyper_connection.combine_and_mix_with_latent(
                        deferred[0], hyper_hidden, deferred[1], prev_latent
                    )
                )
        assert attn_inj is not None

        # Step C: Attention core (GDN or Full Attention)
        if self.linear_attn is not None:
            gdn_state = state.gdn.get(lid) if state is not None else None
            if (
                self.is_source
                and source_prefix_ssm is not None
                and gdn_state is not None
                and gdn_state.ssm_state is not None
            ):
                source_prefix_ssm[lid] = gdn_state.ssm_state.clone()
            yoco_on = (
                cfg.enable_yoco_decode
                if enable_yoco_decode is None
                else bool(enable_yoco_decode)
            )
            gdn_call_state = gdn_state
            if (
                self.is_consumer
                and input_ids.shape[1] > 1
                and yoco_on
                and gdn_state is not None
                and gdn_state.ssm_state is None
                and source_prefix_ssm is not None
                and self.source_layer_idx in source_prefix_ssm
            ):
                # Reuse pre-suffix source SSM transiently so chunked/suffix prefill preserves prefix context
                gdn_call_state = GDNLayerState(
                    conv_state=gdn_state.conv_state,
                    ssm_state=source_prefix_ssm[self.source_layer_idx].clone(),  # type: ignore[index]
                )
            shared_in = (
                shared_gdn_by_source[self.source_layer_idx]
                if (self.is_consumer and self.source_layer_idx is not None)
                else None
            )
            cat = "gdn_consumer" if self.is_consumer else "gdn_normal"
            if profiler is not None:
                with profiler.record(cat):
                    attn_out, published = self.linear_attn(
                        attn_in,
                        state=gdn_call_state,
                        shared=shared_in,
                        enable_yoco_decode=enable_yoco_decode,
                    )
            else:
                attn_out, published = self.linear_attn(
                    attn_in,
                    state=gdn_call_state,
                    shared=shared_in,
                    enable_yoco_decode=enable_yoco_decode,
                )
            if published is not None:
                shared_gdn_by_source[lid] = published
        else:
            assert self.self_attn is not None
            kv_cache = state.full_attn_kv.get(self.kv_cache_owner) if state is not None else None
            k_sc = state.k_sconv.get(lid) if (state is not None and not self.is_consumer) else None
            v_sc = state.v_sconv.get(lid) if (state is not None and not self.is_consumer) else None
            qsa_kwargs: Dict[str, Any] = {}
            if self.self_attn.indexer is not None:
                if not self.is_consumer:
                    raw_prev = state.qsa_raw_keys.get(lid) if state is not None else None
                    past_raw = (
                        raw_prev if (raw_prev is not None and raw_prev.shape[1] > 0) else None
                    )
                    src_k_norm = self.self_attn.indexer.k_norm_gamma

                    def _record_raw(
                        raw_k: torch.Tensor,
                        _past: Optional[torch.Tensor] = past_raw,
                        _lid: int = lid,
                        _k_norm: torch.Tensor = src_k_norm,
                    ) -> None:
                        full_raw = (
                            torch.cat([_past, raw_k], dim=1) if _past is not None else raw_k
                        )
                        if state is not None:
                            state.qsa_raw_keys[_lid] = full_raw
                        if shared_qsa_by_source is not None:
                            shared_qsa_by_source[_lid] = (full_raw, _k_norm)

                    qsa_kwargs["qsa_past_raw_keys"] = past_raw
                    qsa_kwargs["qsa_on_raw_keys"] = _record_raw
                else:
                    src_lid = self.kv_cache_owner
                    src_raw = None
                    src_k_norm = None
                    if shared_qsa_by_source is not None and src_lid in shared_qsa_by_source:
                        src_raw, src_k_norm = shared_qsa_by_source[src_lid]
                    elif state is not None and src_lid in state.qsa_raw_keys:
                        src_raw = state.qsa_raw_keys[src_lid]
                    qsa_kwargs["qsa_source_raw_keys"] = src_raw
                    qsa_kwargs["qsa_source_k_norm_gamma"] = src_k_norm
            cat = "full_attn_consumer" if self.is_consumer else "full_attn_producer"
            if profiler is not None:
                with profiler.record(cat):
                    attn_out = self.self_attn(
                        attn_in,
                        positions=positions,
                        kv_cache=kv_cache,
                        k_sconv_state=k_sc,
                        v_sconv_state=v_sc,
                        seq_offset=seq_offset,
                        **qsa_kwargs,
                    )
            else:
                attn_out = self.self_attn(
                    attn_in,
                    positions=positions,
                    kv_cache=kv_cache,
                    k_sconv_state=k_sc,
                    v_sconv_state=v_sc,
                    seq_offset=seq_offset,
                    **qsa_kwargs,
                )

        # Step D: Attention output ShortConv (+ optional clip)
        if self.attn_output_sconv is not None:
            sc_state = state.attn_output_sconv.get(lid) if state is not None else None
            if profiler is not None:
                with profiler.record("short_conv"):
                    attn_out = self.attn_output_sconv(attn_out, state=sc_state)
            elif use_reference:
                attn_out = self.attn_output_sconv.forward_ref(attn_out, state=sc_state)
            else:
                attn_out = self.attn_output_sconv(attn_out, state=sc_state)
        if cfg.clip_attn_output is not None:
            attn_out = attn_out.clamp(-cfg.clip_attn_output, cfg.clip_attn_output)

        # Step E: MLP Hyper-Connection (fused combine of attn_out + norm + mix)
        if profiler is not None:
            with profiler.record("hc"):
                mlp_in, hyper_hidden, mlp_inj, prev_latent = (
                    self.mlp_hyper_connection.combine_and_mix_with_latent(
                        attn_out, hyper_hidden, attn_inj, prev_latent
                    )
                )
        elif use_reference:
            mlp_in, hyper_hidden, mlp_inj, prev_latent = (
                self.mlp_hyper_connection.combine_and_mix_with_latent_ref(
                    attn_out, hyper_hidden, attn_inj, prev_latent
                )
            )
        else:
            mlp_in, hyper_hidden, mlp_inj, prev_latent = (
                self.mlp_hyper_connection.combine_and_mix_with_latent(
                    attn_out, hyper_hidden, attn_inj, prev_latent
                )
            )
        assert mlp_inj is not None

        # Step F: Dense MLP (Layer 0) or Latent MoE (Layers 1..39)
        if isinstance(self.mlp, Qwen4_5LatentMoE):
            if profiler is not None:
                with profiler.record("latent_moe"):
                    mlp_out = self.mlp(mlp_in, use_triton_mxfp4=not use_reference)
            else:
                mlp_out = self.mlp(mlp_in, use_triton_mxfp4=not use_reference)
        else:
            if profiler is not None:
                with profiler.record("dense_mlp"):
                    mlp_out = self.mlp(mlp_in, use_triton=not use_reference)
            else:
                mlp_out = self.mlp(mlp_in, use_triton=not use_reference)

        # Step G: FFN output ShortConv + output clamping [-1000.0, 1000.0]
        if self.ffn_output_sconv is not None:
            sc_state = state.ffn_output_sconv.get(lid) if state is not None else None
            if profiler is not None:
                with profiler.record("short_conv"):
                    mlp_out = self.ffn_output_sconv(mlp_out, state=sc_state)
            elif use_reference:
                mlp_out = self.ffn_output_sconv.forward_ref(mlp_out, state=sc_state)
            else:
                mlp_out = self.ffn_output_sconv(mlp_out, state=sc_state)
        if cfg.clip_mlp_output is not None:
            mlp_out = mlp_out.clamp(-cfg.clip_mlp_output, cfg.clip_mlp_output)

        return hyper_hidden, prev_latent, (mlp_out, mlp_inj)


class Qwen4_5Model(nn.Module):
    """Full Qwen 4.5-Flash backbone (embeddings + N decoder layers + final HC mixer)."""

    def __init__(
        self,
        config: Qwen4_5ArchConfig,
        *,
        use_triton_mxfp4: bool = True,
        compute_mode: str = "bf16",
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        init_mxfp4_weights: bool = True,
        ep_size: int = 1,
        ep_rank: int = 0,
        ep_group: Any = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group: Any = None,
        fast_all_reduce: Any = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = max(1, int(tp_size or 1))
        self.tp_rank = int(tp_rank or 0)
        if not (0 <= self.tp_rank < self.tp_size):
            raise ValueError(f"tp_rank={self.tp_rank} out of range for tp_size={self.tp_size}")
        if tp_group is None and self.tp_size > 1:
            tp_group = _resolve_tp_group_safe(self.tp_size)
        self.tp_group = tp_group

        if (
            fast_all_reduce is None
            and self.tp_size > 1
            and tp_group is not None
            and device is not None
            and torch.device(device).type == "cuda"
            and dist.is_available()
            and dist.is_initialized()
        ):
            fast_all_reduce = get_qwen4_5_fast_all_reduce(
                tp_group,
                device=torch.device(device),
                dtype=dtype,
                hidden_dims=(int(config.hidden_size), int(config.routed_expert_hidden_size)),
            )
        self.fast_all_reduce = fast_all_reduce

        self.ep_size = int(ep_size)
        self.ep_rank = int(ep_rank)
        self.ep_group = ep_group
        self.sharing_plan = YoCoSharingPlan.from_config(config)

        # Target embed_tokens stays replicated [vocab_size, hidden_size]
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=dtype, device=device
        )
        self.layers = nn.ModuleList(
            [
                Qwen4_5DecoderLayer(
                    config,
                    layer_idx=i,
                    sharing_plan=self.sharing_plan,
                    use_triton_mxfp4=use_triton_mxfp4,
                    compute_mode=compute_mode,
                    dtype=dtype,
                    device=device,
                    init_mxfp4_weights=init_mxfp4_weights,
                    ep_size=ep_size,
                    ep_rank=ep_rank,
                    ep_group=ep_group,
                    tp_size=self.tp_size,
                    tp_rank=self.tp_rank,
                    tp_group=tp_group,
                    fast_all_reduce=fast_all_reduce,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        # Final mixer collapses 4-stream HC -> single hidden_size vector (NO model.norm!)
        self.hyper_connection_mixer = Qwen4_5HyperConnection(
            hidden_size=config.hidden_size,
            hc_count=config.hc_count,
            hc_lowrank=config.hc_lowrank,
            eps=config.rms_norm_eps,
            use_combine=False,
            prev_mix=False,
            is_head=False,
            emits_latent=True,
            dtype=dtype,
            device=device,
        )
        # Freeze inference parameters so Qwen4_5HyperConnection / PLE hit cached fused weights.
        self.requires_grad_(False)

    @torch.no_grad()
    def prepare_weights(self, *, transpose_mxfp4: bool = False) -> None:
        self.requires_grad_(False)
        for layer in self.layers:
            assert isinstance(layer, Qwen4_5DecoderLayer)
            layer.prepare_weights(transpose_mxfp4=transpose_mxfp4)
            for hc in (layer.attn_hyper_connection, layer.mlp_hyper_connection):
                if hc.emits_latent and hc.use_combine and hasattr(hc, "_get_fused_down_inject_weight"):
                    hc._get_fused_down_inject_weight()

    def create_runtime_state(
        self,
        batch_size: int,
        max_seq_len: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        enable_yoco_decode: Optional[bool] = None,
    ) -> Qwen4_5RuntimeState:
        """Allocate minimal YoCo-aware KV cache, GDN SSM, ShortConv, and PLE states."""
        dev = device if device is not None else self.embed_tokens.weight.device
        dt = dtype if dtype is not None else self.embed_tokens.weight.dtype
        yoco_on = (
            self.config.enable_yoco_decode
            if enable_yoco_decode is None
            else bool(enable_yoco_decode)
        )
        state = Qwen4_5RuntimeState(batch_size=batch_size, max_seq_len=max_seq_len)
        for layer in self.layers:
            assert isinstance(layer, Qwen4_5DecoderLayer)
            lid = layer.layer_idx
            if layer.ple is not None:
                state.ple[lid] = layer.ple.alloc_state(batch_size, device=dev, dtype=dt)
            if layer.linear_attn is not None:
                state.gdn[lid] = layer.linear_attn.create_state(
                    batch_size,
                    device=dev,
                    dtype=dt,
                    allocate_consumer_ssm=not yoco_on,
                )
            elif layer.self_attn is not None and not layer.is_consumer:
                kv = layer.self_attn.create_kv_cache(
                    batch_size, max_seq_len, device=dev, dtype=dt
                )
                if kv is not None:
                    state.full_attn_kv[lid] = kv
                k_sc, v_sc = layer.self_attn.create_sconv_states(
                    batch_size, device=dev, dtype=dt
                )
                if k_sc is not None:
                    state.k_sconv[lid] = k_sc
                if v_sc is not None:
                    state.v_sconv[lid] = v_sc
                if layer.self_attn.indexer is not None:
                    state.qsa_raw_keys[lid] = torch.empty(
                        (batch_size, 0, self.config.indexer_head_dim),
                        dtype=dt,
                        device=dev,
                    )
            if layer.attn_output_sconv is not None:
                state.attn_output_sconv[lid] = layer.attn_output_sconv.alloc_state(
                    batch_size, device=dev, dtype=dt
                )
            if layer.ffn_output_sconv is not None:
                state.ffn_output_sconv[lid] = layer.ffn_output_sconv.alloc_state(
                    batch_size, device=dev, dtype=dt
                )
        return state

    @torch.no_grad()
    def init_synthetic_weights(
        self,
        *,
        seed: int = 42,
        ple_mode: str = "gpu_sharded",
        ple_shard_rows: Optional[int] = None,
        ple_fp8: bool = True,
    ) -> None:
        """Populate well-conditioned synthetic weights deterministically across EP ranks.

        All replicated parameters use identical RNG streams across ranks, while
        MXFP4 routed experts and PLE shards are seeded per global index so
        ``EP=1``, ``EP=4``, and ``EP=8`` materialize the exact same mathematical
        model.
        """
        cfg = self.config
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype

        def _gen(offset: int) -> torch.Generator:
            g = torch.Generator(device=device)
            g.manual_seed(seed + offset)
            return g

        # Global embedding + final HC mixer
        self.embed_tokens.weight.normal_(0.0, 0.02, generator=_gen(1))
        for idx, p in enumerate(self.hyper_connection_mixer.parameters()):
            if p.ndim == 1:
                p.zero_()
            else:
                p.normal_(0.0, 0.02, generator=_gen(100 + idx))

        for layer in self.layers:
            assert isinstance(layer, Qwen4_5DecoderLayer)
            lid = layer.layer_idx
            base_seed = 10_000 + lid * 1_000

            def _slice_for_tp_param(p_name: str, full_t: torch.Tensor) -> torch.Tensor:
                if self.tp_size <= 1:
                    return full_t
                tp_s = self.tp_size
                tp_r = self.tp_rank
                if p_name == "self_attn.q_proj.weight":
                    local = full_t.shape[0] // tp_s
                    return full_t[tp_r * local : (tp_r + 1) * local]
                if p_name in ("self_attn.o_proj.weight", "mlp.down_proj.weight", "linear_attn.out_proj.weight"):
                    local = full_t.shape[1] // tp_s
                    return full_t[:, tp_r * local : (tp_r + 1) * local]
                if p_name in (
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "linear_attn.in_proj_qkvz.weight",
                    "linear_attn.in_proj_ba.weight",
                    "linear_attn.A_log",
                    "linear_attn.dt_bias",
                ):
                    local = full_t.shape[0] // tp_s
                    return full_t[tp_r * local : (tp_r + 1) * local]
                if p_name == "linear_attn.conv1d.weight":
                    if layer.is_consumer:
                        local = full_t.shape[0] // tp_s
                        return full_t[tp_r * local : (tp_r + 1) * local]
                    full_k = cfg.linear_num_key_heads * cfg.linear_key_head_dim
                    full_v = cfg.linear_num_value_heads * cfg.linear_value_head_dim
                    loc_k = full_k // tp_s
                    loc_v = full_v // tp_s
                    q_seg, k_seg, v_seg = torch.split(full_t, [full_k, full_k, full_v], dim=0)
                    return torch.cat(
                        [
                            q_seg[tp_r * loc_k : (tp_r + 1) * loc_k],
                            k_seg[tp_r * loc_k : (tp_r + 1) * loc_k],
                            v_seg[tp_r * loc_v : (tp_r + 1) * loc_v],
                        ],
                        dim=0,
                    )
                return full_t

            def _full_shape_for_tp_param(p_name: str, local_shape: Tuple[int, ...]) -> Tuple[int, ...]:
                if self.tp_size <= 1:
                    return local_shape
                tp_s = self.tp_size
                if p_name == "self_attn.q_proj.weight":
                    return (local_shape[0] * tp_s, *local_shape[1:])
                if p_name in ("self_attn.o_proj.weight", "mlp.down_proj.weight", "linear_attn.out_proj.weight"):
                    return (local_shape[0], local_shape[1] * tp_s, *local_shape[2:])
                if p_name in (
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "linear_attn.in_proj_qkvz.weight",
                    "linear_attn.in_proj_ba.weight",
                    "linear_attn.A_log",
                    "linear_attn.dt_bias",
                    "linear_attn.conv1d.weight",
                ):
                    return (local_shape[0] * tp_s, *local_shape[1:])
                return local_shape

            # Deterministic small-variance init (sliced identically across TP ranks)
            for p_idx, (name, param) in enumerate(layer.named_parameters()):
                g = _gen(base_seed + p_idx)
                full_shape = _full_shape_for_tp_param(name, tuple(param.shape))
                buf = (
                    param
                    if full_shape == tuple(param.shape)
                    else torch.empty(full_shape, dtype=param.dtype, device=param.device)
                )
                if name.endswith("A_log"):
                    # Match BF16 storage precision of real 1109 checkpoint (cast to FP32 at runtime)
                    buf.uniform_(-2.0, -0.5, generator=g)
                    buf.copy_(buf.to(torch.bfloat16).float())
                elif name.endswith("dt_bias"):
                    buf.uniform_(-1.0, 0.5, generator=g)
                elif "expert_bias" in name:
                    buf.uniform_(-0.05, 0.05, generator=g)
                elif buf.ndim == 1:
                    if "norm" in name and "linear_attn" in name:
                        buf.fill_(1.0)
                    else:
                        buf.zero_()
                elif buf.ndim == 3:
                    # Depthwise ShortConv / GDN conv1d weights
                    buf.normal_(0.0, 0.05, generator=g)
                elif name == "mlp.gate.weight":
                    # Realistic O(1) router logit scale so top-k routing is well-conditioned
                    std = 2.0 / math.sqrt(max(full_shape[-1], 1))
                    buf.normal_(0.0, std, generator=g)
                else:
                    std = min(0.02, 1.0 / math.sqrt(max(full_shape[-1], 1)))
                    buf.normal_(0.0, std, generator=g)
                if buf is not param:
                    param.copy_(_slice_for_tp_param(name, buf))

            # Deterministic per-expert MXFP4 routed weights (exact across EP sizes)
            if isinstance(layer.mlp, Qwen4_5LatentMoE):
                moe = layer.mlp
                assert (
                    moe.w13_u8 is not None
                    and moe.w13_scale is not None
                    and moe.w2_u8 is not None
                    and moe.w2_scale is not None
                )
                # Populate each local expert from its global expert id so EP1==EP4==EP8.
                # Exponent 121..123 keeps activations well-scaled across 40 layers.
                for local_e in range(moe.local_num_experts):
                    global_e = moe.ep_start + local_e
                    ge = _gen(500_000 + lid * 10_000 + global_e)
                    moe.w13_u8[local_e].random_(0, 256, generator=ge)
                    moe.w13_scale[local_e].random_(121, 124, generator=ge)
                    moe.w2_u8[local_e].random_(0, 256, generator=ge)
                    moe.w2_scale[local_e].random_(121, 124, generator=ge)

            # Deterministic PLE n-gram table shards
            if layer.ple is not None:
                ple = layer.ple
                total_shards = cfg.split_ngram_parts
                if ple_shard_rows is None:
                    align = math.lcm(
                        max(1, int(cfg.make_ngram_vocab_size_divisible_by)),
                        int(total_shards),
                    )
                    padded_rows = ((ple.total_vocab_size + align - 1) // align) * align
                    shard_rows = padded_rows // total_shards
                else:
                    shard_rows = int(ple_shard_rows)

                table_dtype = torch.float8_e4m3fn if ple_fp8 else dtype
                if ple_mode == "gpu_sharded" and self.ep_size > 1:
                    if total_shards % self.ep_size != 0:
                        raise ValueError(
                            f"split_ngram_parts={total_shards} not divisible by ep_size={self.ep_size}"
                        )
                    local_shards = total_shards // self.ep_size
                    shard_offset = self.ep_rank * local_shards
                    shards: List[torch.Tensor] = []
                    for s_local in range(local_shards):
                        s_global = shard_offset + s_local
                        gs = _gen(900_000 + lid * 1_000 + s_global)
                        if ple_fp8:
                            # Valid finite FP8 E4M3 codes (exclude NaN 0x7f/0xff)
                            raw = torch.randint(
                                0, 120, (shard_rows, ple.head_dim), dtype=torch.uint8, device=device, generator=gs
                            )
                            sign = torch.randint(
                                0, 2, (shard_rows, ple.head_dim), dtype=torch.uint8, device=device, generator=gs
                            ) << 7
                            raw |= sign
                            del sign
                            shards.append(raw.view(torch.float8_e4m3fn))
                        else:
                            shards.append(
                                torch.empty((shard_rows, ple.head_dim), dtype=dtype, device=device).normal_(
                                    0.0, 0.02, generator=gs
                                )
                            )
                    ple.set_ngram_table(
                        shards,
                        weight_scale=DEFAULT_PLE_WEIGHT_SCALE,
                        shard_offset=shard_offset,
                        total_shards=total_shards,
                        ep_group=self.ep_group,
                    )
                elif ple_mode in ("gpu_sharded", "gpu_full"):
                    shards = []
                    for s_global in range(total_shards):
                        gs = _gen(900_000 + lid * 1_000 + s_global)
                        if ple_fp8:
                            raw = torch.randint(
                                0, 120, (shard_rows, ple.head_dim), dtype=torch.uint8, device=device, generator=gs
                            )
                            sign = torch.randint(
                                0, 2, (shard_rows, ple.head_dim), dtype=torch.uint8, device=device, generator=gs
                            ) << 7
                            raw |= sign
                            del sign
                            shards.append(raw.view(torch.float8_e4m3fn))
                        else:
                            shards.append(
                                torch.empty((shard_rows, ple.head_dim), dtype=dtype, device=device).normal_(
                                    0.0, 0.02, generator=gs
                                )
                            )
                    ple.set_ngram_table(shards, weight_scale=DEFAULT_PLE_WEIGHT_SCALE)
                elif ple_mode == "cpu_uva":
                    cpu_gen = torch.Generator(device="cpu")
                    shards = []
                    for s_global in range(total_shards):
                        cpu_gen.manual_seed(seed + 900_000 + lid * 1_000 + s_global)
                        if ple_fp8:
                            raw = torch.randint(
                                0, 120, (shard_rows, ple.head_dim), dtype=torch.uint8, generator=cpu_gen
                            )
                            sign = torch.randint(
                                0, 2, (shard_rows, ple.head_dim), dtype=torch.uint8, generator=cpu_gen
                            ) << 7
                            raw |= sign
                            del sign
                            shards.append(raw.view(torch.float8_e4m3fn))
                        else:
                            shards.append(
                                torch.empty((shard_rows, ple.head_dim), dtype=dtype).normal_(
                                    0.0, 0.02, generator=cpu_gen
                                )
                            )
                    ple.set_ngram_table(
                        shards,
                        weight_scale=DEFAULT_PLE_WEIGHT_SCALE,
                        pin_cpu=True,
                        enable_cpu_uva=True,
                    )
                else:
                    raise ValueError(f"Unsupported ple_mode={ple_mode!r}")

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        state: Optional[Qwen4_5RuntimeState] = None,
        positions: Optional[torch.Tensor] = None,
        seq_offset: Optional[int] = None,
        enable_yoco_decode: Optional[bool] = None,
        use_reference: bool = False,
        profiler: Optional[Qwen4_5ProfileRecorder] = None,
    ) -> torch.Tensor:
        """Run full backbone forward returning collapsed ``[B, T, hidden_size]``."""
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids of shape [B, T], got {tuple(input_ids.shape)}")
        _, seq_len = input_ids.shape

        if profiler is not None:
            with profiler.record("embed"):
                emb = self.embed_tokens(input_ids)
                hyper_hidden = emb.repeat(1, 1, self.config.hc_count)
        else:
            emb = self.embed_tokens(input_ids)
            hyper_hidden = emb.repeat(1, 1, self.config.hc_count)

        prev_latent: Optional[torch.Tensor] = None
        deferred: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        shared_gdn_by_source: Dict[int, SharedGDNInputs] = {}
        shared_qsa_by_source: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        past_len = state.seq_len if state is not None else 0
        source_prefix_ssm: Optional[Dict[int, torch.Tensor]] = (
            {} if (state is not None and seq_len > 1 and past_len > 0) else None
        )

        for layer in self.layers:
            assert isinstance(layer, Qwen4_5DecoderLayer)
            hyper_hidden, prev_latent, deferred = layer(
                hyper_hidden,
                input_ids,
                prev_latent=prev_latent,
                deferred=deferred,
                shared_gdn_by_source=shared_gdn_by_source,
                shared_qsa_by_source=shared_qsa_by_source,
                source_prefix_ssm=source_prefix_ssm,
                state=state,
                positions=positions,
                seq_offset=seq_offset,
                enable_yoco_decode=enable_yoco_decode,
                use_reference=use_reference,
                profiler=profiler,
            )

        # Final HC combine + collapse to [B, T, hidden_size] (NO model.norm!)
        assert deferred is not None
        if profiler is not None:
            with profiler.record("final_hc"):
                hidden, _, _, _ = self.hyper_connection_mixer.combine_and_mix_with_latent(
                    deferred[0], hyper_hidden, deferred[1], prev_latent=None
                )
        elif use_reference:
            hidden, _, _, _ = self.hyper_connection_mixer.combine_and_mix_with_latent_ref(
                deferred[0], hyper_hidden, deferred[1], prev_latent=None
            )
        else:
            hidden, _, _, _ = self.hyper_connection_mixer.combine_and_mix_with_latent(
                deferred[0], hyper_hidden, deferred[1], prev_latent=None
            )

        if state is not None:
            state.seq_len += seq_len
        return hidden


class Qwen4_5ForCausalLM(nn.Module):
    """Causal LM wrapper around ``Qwen4_5Model`` with vocab-parallel ``lm_head``."""

    def __init__(
        self,
        config: Qwen4_5ArchConfig,
        *,
        use_triton_mxfp4: bool = True,
        compute_mode: str = "bf16",
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        init_mxfp4_weights: bool = True,
        ep_size: int = 1,
        ep_rank: int = 0,
        ep_group: Any = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group: Any = None,
        fast_all_reduce: Any = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen4_5Model(
            config,
            use_triton_mxfp4=use_triton_mxfp4,
            compute_mode=compute_mode,
            dtype=dtype,
            device=device,
            init_mxfp4_weights=init_mxfp4_weights,
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_group=ep_group,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            fast_all_reduce=fast_all_reduce,
        )
        self.tp_size = self.model.tp_size
        self.tp_rank = self.model.tp_rank
        self.tp_group = self.model.tp_group
        if config.vocab_size % self.tp_size != 0:
            raise ValueError(
                f"vocab_size ({config.vocab_size}) must be divisible by tp_size ({self.tp_size})"
            )
        self.local_vocab_size = config.vocab_size // self.tp_size
        self.lm_head = nn.Linear(
            config.hidden_size, self.local_vocab_size, bias=False, dtype=dtype, device=device
        )
        self.requires_grad_(False)

    @torch.no_grad()
    def prepare_weights(self, *, transpose_mxfp4: bool = False) -> None:
        self.model.prepare_weights(transpose_mxfp4=transpose_mxfp4)

    def create_runtime_state(
        self,
        batch_size: int,
        max_seq_len: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        enable_yoco_decode: Optional[bool] = None,
    ) -> Qwen4_5RuntimeState:
        return self.model.create_runtime_state(
            batch_size,
            max_seq_len,
            device=device,
            dtype=dtype,
            enable_yoco_decode=enable_yoco_decode,
        )

    @torch.no_grad()
    def init_synthetic_weights(
        self,
        *,
        seed: int = 42,
        ple_mode: str = "gpu_sharded",
        ple_shard_rows: Optional[int] = None,
        ple_fp8: bool = True,
    ) -> None:
        self.model.init_synthetic_weights(
            seed=seed,
            ple_mode=ple_mode,
            ple_shard_rows=ple_shard_rows,
            ple_fp8=ple_fp8,
        )
        g = torch.Generator(device=self.lm_head.weight.device)
        g.manual_seed(seed + 7_777_777)
        if self.tp_size > 1:
            full_lm = torch.empty(
                (self.config.vocab_size, self.config.hidden_size),
                dtype=self.lm_head.weight.dtype,
                device=self.lm_head.weight.device,
            )
            full_lm.normal_(0.0, 0.02, generator=g)
            s0 = self.tp_rank * self.local_vocab_size
            self.lm_head.weight.copy_(full_lm[s0 : s0 + self.local_vocab_size])
        else:
            self.lm_head.weight.normal_(0.0, 0.02, generator=g)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        state: Optional[Qwen4_5RuntimeState] = None,
        positions: Optional[torch.Tensor] = None,
        seq_offset: Optional[int] = None,
        enable_yoco_decode: Optional[bool] = None,
        logits_to_keep: Optional[int] = None,
        use_reference: bool = False,
        profiler: Optional[Qwen4_5ProfileRecorder] = None,
    ) -> torch.Tensor:
        hidden = self.model(
            input_ids,
            state=state,
            positions=positions,
            seq_offset=seq_offset,
            enable_yoco_decode=enable_yoco_decode,
            use_reference=use_reference,
            profiler=profiler,
        )
        if logits_to_keep is not None and logits_to_keep > 0:
            hidden = hidden[:, -logits_to_keep:, :]
        if profiler is not None:
            with profiler.record("lm_head"):
                logits = self.lm_head(hidden)
        else:
            logits = self.lm_head(hidden)
        if (
            self.tp_size > 1
            and self.tp_group is not None
            and dist.is_available()
            and dist.is_initialized()
        ):
            gathered = [torch.empty_like(logits) for _ in range(self.tp_size)]
            dist.all_gather(gathered, logits.contiguous(), group=self.tp_group)
            logits = torch.cat(gathered, dim=-1)
        return logits


__all__ = [
    "Qwen4_5DecoderLayer",
    "Qwen4_5ForCausalLM",
    "Qwen4_5Model",
    "Qwen4_5PagedStateBridge",
    "Qwen4_5ProfileRecorder",
    "Qwen4_5RTPModel",
]

def __getattr__(name: str) -> Any:
    if name in ("Qwen4_5RTPModel", "Qwen4_5PagedStateBridge"):
        from rtp_llm.models_py.model_desc import qwen4_5_runtime
        return getattr(qwen4_5_runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
